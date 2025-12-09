"""Compute comprehensive HOSER→LM-TAD mapping and collision statistics.

Outputs a JSON report suitable for paper-quality reporting and optionally
emits plots for density, heatmaps, and collision distributions.

Key capabilities (aligned to the request):
- Roads-per-cell distribution (min/max/mean/median and percentiles).
- Grid occupancy, dense-cell counts, and compression ratios.
- Candidate-set collision analysis (sampled) with severity bins and examples.
 - Candidate-set collision analysis (sampled) with severity bins and examples.
 - Optional simulated candidate sets (reachable neighbors from gene.py or k-NN)
     when no candidate file is available.
- Worst-case cells and example high-collision candidate sets.
- Information-loss metrics during generation.
- Optional plots: density histogram, spatial heatmap, collision histogram,
    scatter, and collision CDF.

Example:

    python tools/compute_conversion_stats.py \
        --trajectory-file eval_dir/generated/hoser_abnormal_od.csv \
        --roadmap-file data/porto_hoser/roadmap.geo \
        --dataset porto_hoser \
        --output-json eval_dir/eval_lmtad/analysis/conversion_stats.json \
        --plots-dir eval_dir/eval_lmtad/analysis/plots \
        --candidate-file path/to/candidate_sets.npz \
        --sample-size 1000

Notes:
- Candidate sets are optional; provide them to compute collision stats.
- If no candidate file is available, use --simulate-from-trajectories or
    --simulate-random to build synthetic candidate sets
    (reachable neighbors or k-NN via --sim-candidate-mode).
- Supports candidate formats: .npz with arrays `candidate_road_id` (N,T,K)
    and optional `candidate_len` (N,T); or CSV with `candidate_road_id` column
    of list/CSV strings (one candidate set per row).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from critics.grid_mapper import GridConfig, GridMapper
except ModuleNotFoundError:  # Fallback for direct script execution
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from critics.grid_mapper import GridConfig, GridMapper

logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "porto_hoser": {"grid_size": 0.001, "downsample_factor": 1},
    "beijing_hoser_reference": {"grid_size": 0.001, "downsample_factor": 1},
}


def empty_collision_section() -> Dict:
    """Template collision section used when no candidates are available."""

    return {
        "sample_size": 0,
        "collision_sets_pct": None,
        "mean_collision_rate": None,
        "mean_unique_cells": None,
        "severity_hist": {},
        "examples": [],
        "info_loss_pct": None,
        "collision_rates": [],
        "unique_cells_list": [],
        "total_candidates_list": [],
        "source": None,
        "candidate_mode": None,
    }


def load_reachable_dict(rel_file: Path, num_roads: int) -> Dict[int, List[int]]:
    """Load reachable road ids from roadmap.rel (origin_id -> destination_id)."""

    reachable: Dict[int, List[int]] = {i: [] for i in range(num_roads)}
    if not rel_file.exists():
        logger.warning(
            "relation file %s not found; reachable candidates disabled", rel_file
        )
        return reachable

    df = pd.read_csv(rel_file)
    if "origin_id" not in df.columns or "destination_id" not in df.columns:
        logger.warning("relation file %s missing origin_id/destination_id", rel_file)
        return reachable

    for o, d in zip(df["origin_id"], df["destination_id"]):
        try:
            reachable[int(o)].append(int(d))
        except Exception:
            continue
    return reachable


def build_knn_index(road_centroids: np.ndarray):
    """Build a BallTree on lat/lng for fast k-NN queries (haversine metric)."""

    from sklearn.neighbors import BallTree

    coords_rad = np.radians(road_centroids)
    tree = BallTree(coords_rad, metric="haversine")
    return tree, coords_rad


def knn_candidates(tree, coords_rad: np.ndarray, road_id: int, k: int) -> List[int]:
    """Return k nearest neighbor road ids (excluding self)."""

    if tree is None or coords_rad.size == 0:
        return []

    k_query = min(k + 1, coords_rad.shape[0])
    _, indices = tree.query(coords_rad[road_id].reshape(1, -1), k=k_query)
    result = [int(idx) for idx in indices[0].tolist() if int(idx) != road_id]
    return result[:k]


def extract_roads_from_trajectories(
    df: pd.DataFrame,
    rid_column: str,
    max_samples: int,
    seed: int,
    max_road_id: int,
) -> List[int]:
    """Flatten trajectory road ids with optional sampling and range guard."""

    roads: List[int] = []
    for row in df.itertuples():
        vals = parse_list_field(getattr(row, rid_column))
        if not vals:
            continue
        for v in vals:
            if isinstance(v, int) and 0 <= v < max_road_id:
                roads.append(v)

    if max_samples and len(roads) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(roads), size=max_samples, replace=False)
        roads = [roads[i] for i in idx]

    return roads


def build_simulated_candidate_sets(
    road_ids: List[int],
    candidate_mode: str,
    reachable: Dict[int, List[int]],
    tree,
    coords_rad: np.ndarray,
    k: int,
) -> List[List[int]]:
    """Create candidate sets for provided road ids using chosen mode."""

    candidates: List[List[int]] = []
    for road_id in road_ids:
        if candidate_mode == "reachable":
            cand = reachable.get(road_id, [])
        else:
            cand = knn_candidates(tree, coords_rad, road_id, k)
        if cand:
            candidates.append(cand)
    return candidates


def simulate_candidates_from_trajectories(
    df: pd.DataFrame,
    rid_column: str,
    candidate_mode: str,
    reachable: Dict[int, List[int]],
    tree,
    coords_rad: np.ndarray,
    k: int,
    max_samples: int,
    seed: int,
    num_roads: int,
) -> List[List[int]]:
    """Simulate candidate sets using real trajectories (option 1)."""

    roads = extract_roads_from_trajectories(
        df, rid_column, max_samples, seed, num_roads
    )
    return build_simulated_candidate_sets(
        roads, candidate_mode, reachable, tree, coords_rad, k
    )


def simulate_candidates_random(
    num_roads: int,
    iterations: int,
    candidate_mode: str,
    reachable: Dict[int, List[int]],
    tree,
    coords_rad: np.ndarray,
    k: int,
    seed: int,
) -> List[List[int]]:
    """Simulate candidate sets using random road picks (option 2)."""

    rng = np.random.default_rng(seed)
    roads = rng.integers(0, num_roads, size=iterations).tolist()
    return build_simulated_candidate_sets(
        roads, candidate_mode, reachable, tree, coords_rad, k
    )


def parse_list_field(value: object) -> Optional[List[int]]:
    """Parse a list-like cell (JSON list or comma-separated) into ints."""

    if pd.isna(value):
        return None
    if isinstance(value, list):
        try:
            return [int(v) for v in value]
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            return [int(v) for v in parsed]
        except Exception:
            return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except Exception:
        return None


def load_roadmap(roadmap_file: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load roadmap.geo and compute centroids and boundary."""

    schema = {"coordinates": str}
    roadmap = pd.read_csv(roadmap_file, dtype=schema)
    if "coordinates" not in roadmap.columns:
        raise ValueError("roadmap file missing 'coordinates' column")

    centroids: List[List[float]] = []
    min_lat, max_lat = float("inf"), float("-inf")
    min_lng, max_lng = float("inf"), float("-inf")

    for coords_str in roadmap["coordinates"]:
        try:
            coords = json.loads(coords_str)
        except Exception:
            continue
        if not coords:
            continue
        lats = [c[1] for c in coords]
        lngs = [c[0] for c in coords]
        centroids.append([sum(lats) / len(lats), sum(lngs) / len(lngs)])
        for lng, lat in coords:
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
            min_lng = min(min_lng, lng)
            max_lng = max(max_lng, lng)

    if not centroids:
        raise ValueError("no valid coordinates found in roadmap")

    boundary = {
        "min_lat": float(min_lat),
        "max_lat": float(max_lat),
        "min_lng": float(min_lng),
        "max_lng": float(max_lng),
    }
    return np.array(centroids, dtype=np.float64), boundary


def build_mapper(
    dataset: str, road_centroids: np.ndarray, boundary: Dict[str, float]
) -> Tuple[GridMapper, np.ndarray, GridConfig]:
    """Create GridMapper and mapping array for the given dataset."""

    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'")
    cfg = DATASET_CONFIGS[dataset]
    grid_cfg = GridConfig(
        min_lat=boundary["min_lat"],
        max_lat=boundary["max_lat"],
        min_lng=boundary["min_lng"],
        max_lng=boundary["max_lng"],
        grid_size=float(cfg["grid_size"]),
        downsample_factor=int(cfg["downsample_factor"]),
    )
    mapper = GridMapper(grid_cfg, road_centroids)
    road_to_token = mapper.map_all()
    return mapper, road_to_token, grid_cfg


def roads_per_cell_stats(token_counts: np.ndarray) -> Dict[str, float]:
    """Compute distribution stats for roads per grid cell."""

    occupied = token_counts[token_counts > 0]
    percentiles = [0, 50, 90, 95, 99, 100]
    pct_vals = (
        dict(zip(percentiles, np.percentile(occupied, percentiles)))
        if occupied.size
        else {p: 0.0 for p in percentiles}
    )
    stats = {
        "min": float(token_counts.min(initial=0)),
        "min_nonempty": float(occupied.min(initial=0)) if occupied.size else 0.0,
        "max": float(token_counts.max(initial=0)),
        "mean": float(token_counts.mean()) if token_counts.size else 0.0,
        "median": float(np.median(token_counts)) if token_counts.size else 0.0,
        "p90": float(pct_vals[90]),
        "p95": float(pct_vals[95]),
        "p99": float(pct_vals[99]),
        "occupied_cells": int(occupied.size),
        "empty_cells": int(token_counts.size - occupied.size),
    }
    return stats


def grid_occupancy_stats(
    token_counts: np.ndarray, grid_h: int, grid_w: int
) -> Dict[str, float]:
    """Compute occupancy and dense-cell counts."""

    total_cells = grid_h * grid_w
    occupied = int(np.sum(token_counts > 0))
    occupancy_rate = 100.0 * occupied / total_cells if total_cells else 0.0
    dense_gt5 = int(np.sum(token_counts > 5))
    dense_gt10 = int(np.sum(token_counts > 10))
    dense_gt20 = int(np.sum(token_counts > 20))
    return {
        "total_cells": int(total_cells),
        "occupied_cells": occupied,
        "occupancy_rate_pct": occupancy_rate,
        "dense_gt5": dense_gt5,
        "dense_gt10": dense_gt10,
        "dense_gt20": dense_gt20,
    }


def worst_cells(
    token_counts: np.ndarray,
    mapper: GridMapper,
    grid_cfg: GridConfig,
    top_k: int = 10,
) -> List[Dict[str, float]]:
    """Return top-k densest cells with locations and counts."""

    indices = np.argsort(token_counts)[::-1][:top_k]
    results: List[Dict[str, float]] = []
    effective_size = grid_cfg.grid_size * grid_cfg.downsample_factor
    for idx in indices:
        count = int(token_counts[idx])
        gi = int(idx // mapper.grid_w)
        gj = int(idx % mapper.grid_w)
        center_lat = grid_cfg.min_lat + (gi + 0.5) * effective_size
        center_lng = grid_cfg.min_lng + (gj + 0.5) * effective_size
        results.append(
            {
                "cell_index": int(idx),
                "grid_i": gi,
                "grid_j": gj,
                "count": count,
                "center_lat": center_lat,
                "center_lng": center_lng,
            }
        )
    return results


def trajectory_level_stats(
    df: pd.DataFrame, road_to_token: np.ndarray, rid_column: str
) -> Dict:
    """Aggregate mapping stats over trajectories."""

    total = len(df)
    parsed = 0
    skipped = 0
    total_input = 0
    total_mapped = 0
    per_frac: List[float] = []
    input_lengths: List[int] = []
    mapped_lengths: List[int] = []
    n = len(road_to_token)

    for row in df.itertuples():
        vals = parse_list_field(getattr(row, rid_column))
        if vals is None:
            skipped += 1
            continue
        parsed += 1
        total_input += len(vals)
        valid = [v for v in vals if isinstance(v, int) and 0 <= v < n]
        total_mapped += len(valid)
        input_lengths.append(len(vals))
        mapped_lengths.append(len(valid))
        per_frac.append(len(valid) / len(vals) if vals else 0.0)

    return {
        "num_trajectories": total,
        "parsed_trajectories": parsed,
        "skipped_trajectories": skipped,
        "total_input_roads": int(total_input),
        "total_mapped_roads": int(total_mapped),
        "overall_mapping_rate": float(total_mapped / total_input)
        if total_input
        else 0.0,
        "mean_input_length": float(np.mean(input_lengths)) if input_lengths else 0.0,
        "median_input_length": float(np.median(input_lengths))
        if input_lengths
        else 0.0,
        "mean_mapped_length": float(np.mean(mapped_lengths)) if mapped_lengths else 0.0,
        "median_mapped_length": float(np.median(mapped_lengths))
        if mapped_lengths
        else 0.0,
        "mean_per_traj_map_frac": float(np.mean(per_frac)) if per_frac else 0.0,
        "std_per_traj_map_frac": float(np.std(per_frac)) if per_frac else 0.0,
    }


def token_distribution_stats(token_counts: np.ndarray) -> Dict:
    """Compute entropy and related token stats."""

    total = int(token_counts.sum())
    probs = (
        token_counts / total if total > 0 else np.zeros_like(token_counts, dtype=float)
    )
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if nonzero.size else 0.0
    return {
        "num_cells": int(token_counts.size),
        "total_road_assignments": total,
        "nonempty_cells": int(np.sum(token_counts > 0)),
        "entropy": entropy,
        "max_count": int(token_counts.max(initial=0)),
    }


def compression_ratio(total_roads: int, occupied_cells: int) -> float:
    """Compute compression ratio (roads per occupied cell)."""

    if occupied_cells <= 0:
        return 0.0
    return float(total_roads / occupied_cells)


def load_candidate_sets(path: Path) -> List[Sequence[int]]:
    """Load candidate sets from npz/npy/CSV into a flat list of sequences."""

    if path.suffix == ".npz":
        data = np.load(path)
        if "candidate_road_id" not in data:
            raise ValueError("npz missing 'candidate_road_id'")
        cand = data["candidate_road_id"]
        if cand.ndim < 1:
            raise ValueError("candidate_road_id array must be at least 1-D")
        flat = cand.reshape(-1, cand.shape[-1]) if cand.ndim >= 2 else cand
        return [row.tolist() for row in flat]
    if path.suffix == ".npy":
        cand = np.load(path)
        if cand.ndim < 1:
            raise ValueError("candidate array must be at least 1-D")
        flat = cand.reshape(-1, cand.shape[-1]) if cand.ndim >= 2 else cand
        return [row.tolist() for row in flat]
    # CSV fallback
    df = pd.read_csv(path)
    if "candidate_road_id" not in df.columns:
        raise ValueError("CSV candidate file missing 'candidate_road_id'")
    return [parse_list_field(v) or [] for v in df["candidate_road_id"]]


def sample_candidate_sets(
    candidates: List[Sequence[int]], sample_size: int, seed: int = 42
) -> List[List[int]]:
    """Uniformly sample candidate sets."""

    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(
        len(candidates), size=min(sample_size, len(candidates)), replace=False
    )
    return [list(candidates[i]) for i in idx]


def collision_bins(rate: float) -> str:
    """Bucket collision rate into severity bins."""

    pct = 100.0 * rate
    if pct == 0:
        return "0%"
    if pct <= 10:
        return "1-10%"
    if pct <= 20:
        return "11-20%"
    if pct <= 30:
        return "21-30%"
    if pct <= 40:
        return "31-40%"
    if pct <= 50:
        return "41-50%"
    return ">50%"


def candidate_collision_analysis(
    candidates: List[Sequence[int]],
    road_to_token: np.ndarray,
    sample_size: int,
) -> Dict:
    """Compute collision metrics for candidate sets."""

    sampled = sample_candidate_sets(candidates, sample_size)
    if not sampled:
        return {
            "sample_size": 0,
            "collision_sets_pct": None,
            "mean_collision_rate": None,
            "mean_unique_cells": None,
            "severity_hist": {},
            "examples": [],
            "info_loss_pct": None,
            "collision_rates": [],
            "unique_cells_list": [],
            "total_candidates_list": [],
        }

    rates: List[float] = []
    unique_counts: List[int] = []
    total_sizes: List[int] = []
    severity: Dict[str, int] = {}
    examples: List[Dict] = []
    n_roads = len(road_to_token)

    for idx, cand in enumerate(sampled):
        valid = [r for r in cand if isinstance(r, int) and 0 <= r < n_roads]
        if not valid:
            continue
        tokens = [int(road_to_token[r]) for r in valid]
        uniq = len(set(tokens))
        total = len(tokens)
        rate = 1.0 - (uniq / total) if total else 0.0
        rates.append(rate)
        unique_counts.append(uniq)
        total_sizes.append(total)
        bucket = collision_bins(rate)
        severity[bucket] = severity.get(bucket, 0) + 1
        if rate > 0:
            examples.append(
                {
                    "example_index": idx,
                    "collision_rate": rate,
                    "total_candidates": total,
                    "unique_cells": uniq,
                    "road_ids": valid,
                    "grid_tokens": tokens,
                }
            )

    examples = sorted(examples, key=lambda e: e["collision_rate"], reverse=True)[:5]

    if not rates:
        return {
            "sample_size": len(sampled),
            "collision_sets_pct": 0.0,
            "mean_collision_rate": 0.0,
            "mean_unique_cells": float(np.mean(unique_counts))
            if unique_counts
            else 0.0,
            "severity_hist": severity,
            "examples": examples,
            "info_loss_pct": 0.0,
            "collision_rates": rates,
            "unique_cells_list": unique_counts,
            "total_candidates_list": total_sizes,
        }

    collision_pct = 100.0 * sum(r > 0 for r in rates) / len(rates)
    mean_rate = float(np.mean(rates))
    mean_unique = float(np.mean(unique_counts)) if unique_counts else 0.0
    info_loss = (
        float(
            np.mean(
                [
                    (total_sizes[i] / unique_counts[i]) - 1.0
                    for i in range(len(unique_counts))
                    if unique_counts[i] > 0
                ]
            )
        )
        if unique_counts
        else 0.0
    )

    return {
        "sample_size": len(sampled),
        "collision_sets_pct": collision_pct,
        "mean_collision_rate": mean_rate,
        "mean_unique_cells": mean_unique,
        "severity_hist": severity,
        "examples": examples,
        "info_loss_pct": info_loss * 100.0,
        "collision_rates": rates,
        "unique_cells_list": unique_counts,
        "total_candidates_list": total_sizes,
    }


def make_plots(
    token_counts: np.ndarray,
    mapper: GridMapper,
    collision_rates: List[float],
    unique_counts: List[int],
    total_sizes: List[int],
    plots_dir: Path,
) -> None:
    """Generate requested plots and save to plots_dir."""

    import matplotlib.pyplot as plt  # Local import to avoid hard dependency

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Roads per cell histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_counts[token_counts > 0], bins=50, color="steelblue")
    ax.set_xlabel("Roads per cell")
    ax.set_ylabel("Frequency")
    ax.set_title("Roads per Cell Histogram")
    fig.tight_layout()
    fig.savefig(plots_dir / "roads_per_cell_hist.png", dpi=150)
    plt.close(fig)

    # 2) Spatial heatmap
    grid = token_counts.reshape(mapper.grid_h, mapper.grid_w)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="Road count")
    ax.set_xlabel("grid_w")
    ax.set_ylabel("grid_h")
    ax.set_title("Road Density Heatmap")
    fig.tight_layout()
    fig.savefig(plots_dir / "road_density_heatmap.png", dpi=150)
    plt.close(fig)

    # 3) Collision rate distribution
    if collision_rates:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            [r * 100.0 for r in collision_rates],
            bins=30,
            color="indianred",
        )
        ax.set_xlabel("Collision rate (%)")
        ax.set_ylabel("Count")
        ax.set_title("Candidate Collision Rate Distribution")
        fig.tight_layout()
        fig.savefig(plots_dir / "collision_rate_hist.png", dpi=150)
        plt.close(fig)

        # 4) Candidate set scatter
        if unique_counts and total_sizes:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(total_sizes, unique_counts, alpha=0.5, s=10)
            ax.set_xlabel("Total candidates (k)")
            ax.set_ylabel("Unique grid cells")
            ax.set_title("Candidates vs Unique Cells")
            fig.tight_layout()
            fig.savefig(plots_dir / "candidate_scatter.png", dpi=150)
            plt.close(fig)

        # 5) CDF of collision rates
        sorted_rates = np.sort([r * 100.0 for r in collision_rates])
        y = np.linspace(0, 1, len(sorted_rates), endpoint=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sorted_rates, y)
        ax.set_xlabel("Collision rate (%)")
        ax.set_ylabel("CDF")
        ax.set_title("Collision Rate CDF")
        fig.tight_layout()
        fig.savefig(plots_dir / "collision_rate_cdf.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute conversion/mapping stats and candidate collision analysis"
        )
    )
    parser.add_argument(
        "--trajectory-file",
        type=Path,
        required=True,
        help="HOSER trajectories CSV (rid_list/gene_trace_road_id)",
    )
    parser.add_argument(
        "--roadmap-file", type=Path, required=True, help="roadmap.geo CSV"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        required=False,
        help="Optional candidate sets file (npz/npy/csv)",
    )
    parser.add_argument(
        "--rel-file",
        type=Path,
        required=False,
        help="Path to roadmap.rel (used for reachable-mode simulation)",
    )
    parser.add_argument(
        "--simulate-from-trajectories",
        action="store_true",
        help=(
            "Simulate candidate sets from trajectories when no candidate file is given"
        ),
    )
    parser.add_argument(
        "--simulate-random",
        action="store_true",
        help="Simulate candidate sets via random road picks",
    )
    parser.add_argument(
        "--sim-candidate-mode",
        type=str,
        choices=["reachable", "knn"],
        default="reachable",
        help=("Candidate generator: reachable neighbors (gene.py) or k-NN by distance"),
    )
    parser.add_argument(
        "--sim-k",
        type=int,
        default=64,
        help="k for k-NN simulation (ignored for reachable mode)",
    )
    parser.add_argument(
        "--sim-max-trajectories",
        type=int,
        default=5000,
        help="Max trajectory road occurrences to sample for simulation",
    )
    parser.add_argument(
        "--sim-random-iters",
        type=int,
        default=1000,
        help="Random simulation iterations",
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=42,
        help="Random seed for simulation sampling",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Candidate sets sampled for collision analysis",
    )
    parser.add_argument(
        "--output-json", type=Path, required=True, help="Where to save JSON report"
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        required=False,
        help="Directory to write plots (optional)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    road_centroids, boundary = load_roadmap(args.roadmap_file)
    mapper, road_to_token, grid_cfg = build_mapper(
        args.dataset, road_centroids, boundary
    )

    token_counts = np.bincount(road_to_token, minlength=mapper.grid_h * mapper.grid_w)

    roads_stats = roads_per_cell_stats(token_counts)
    occupancy_stats = grid_occupancy_stats(token_counts, mapper.grid_h, mapper.grid_w)
    worst = worst_cells(token_counts, mapper, grid_cfg, top_k=10)
    token_stats = token_distribution_stats(token_counts)
    comp_ratio = compression_ratio(
        len(road_to_token), occupancy_stats["occupied_cells"]
    )

    df = pd.read_csv(args.trajectory_file)
    if "rid_list" in df.columns:
        rid_col = "rid_list"
    elif "gene_trace_road_id" in df.columns:
        rid_col = "gene_trace_road_id"
    else:
        raise ValueError("Trajectory file missing 'rid_list' or 'gene_trace_road_id'")
    traj_stats = trajectory_level_stats(df, road_to_token, rid_col)

    collision_rates: List[float] = []
    unique_counts: List[int] = []
    total_sizes: List[int] = []
    collision_section: Dict = empty_collision_section()
    simulated_collision: Optional[Dict] = None

    # Primary source: explicit candidate file
    if args.candidate_file:
        candidates = load_candidate_sets(args.candidate_file)
        collision_section = candidate_collision_analysis(
            candidates, road_to_token, args.sample_size
        )
        collision_section.update({"source": "candidate_file", "candidate_mode": None})
        collision_rates = collision_section.get("collision_rates", [])
        unique_counts = collision_section.get("unique_cells_list", [])
        total_sizes = collision_section.get("total_candidates_list", [])

    # Optional simulation when no candidate file is provided or for comparison
    need_sim = args.simulate_from_trajectories or args.simulate_random
    if need_sim:
        rel_path = args.rel_file
        if rel_path is None:
            rel_path = args.roadmap_file.with_name("roadmap.rel")

        reachable: Dict[int, List[int]] = {i: [] for i in range(len(road_to_token))}
        knn_tree = None
        coords_rad = np.array([])

        if args.sim_candidate_mode == "reachable":
            reachable = load_reachable_dict(rel_path, len(road_to_token))
        else:
            knn_tree, coords_rad = build_knn_index(road_centroids)

        sim_candidates: List[List[int]] = []

        if args.simulate_from_trajectories:
            sim_candidates.extend(
                simulate_candidates_from_trajectories(
                    df,
                    rid_col,
                    args.sim_candidate_mode,
                    reachable,
                    knn_tree,
                    coords_rad,
                    args.sim_k,
                    args.sim_max_trajectories,
                    args.sim_seed,
                    len(road_to_token),
                )
            )

        if args.simulate_random:
            sim_candidates.extend(
                simulate_candidates_random(
                    len(road_to_token),
                    args.sim_random_iters,
                    args.sim_candidate_mode,
                    reachable,
                    knn_tree,
                    coords_rad,
                    args.sim_k,
                    args.sim_seed,
                )
            )

        if sim_candidates:
            simulated_collision = candidate_collision_analysis(
                sim_candidates, road_to_token, args.sample_size
            )
            simulated_collision.update(
                {
                    "source": "simulated",
                    "candidate_mode": args.sim_candidate_mode,
                }
            )

            # If no candidate file was supplied, use simulated stats as primary
            if collision_section["source"] is None:
                collision_section = simulated_collision
                collision_rates = collision_section.get("collision_rates", [])
                unique_counts = collision_section.get("unique_cells_list", [])
                total_sizes = collision_section.get("total_candidates_list", [])

    report = {
        "metadata": {
            "dataset": args.dataset,
            "trajectory_file": str(args.trajectory_file),
            "roadmap_file": str(args.roadmap_file),
            "candidate_file": (
                str(args.candidate_file) if args.candidate_file else None
            ),
            "sample_size": args.sample_size,
            "grid_dims": {"grid_h": mapper.grid_h, "grid_w": mapper.grid_w},
            "simulation": {
                "simulate_from_trajectories": args.simulate_from_trajectories,
                "simulate_random": args.simulate_random,
                "candidate_mode": args.sim_candidate_mode if need_sim else None,
                "sim_k": args.sim_k if need_sim else None,
            },
        },
        "roads_per_cell": {
            "distribution": roads_stats,
            "percentiles": {
                "p90": roads_stats["p90"],
                "p95": roads_stats["p95"],
                "p99": roads_stats["p99"],
            },
        },
        "grid_occupancy": occupancy_stats,
        "candidate_collision_analysis": collision_section,
        "worst_cases": {
            "max_cell": worst[0] if worst else None,
            "top10_cells": worst,
        },
        "information_loss": {
            "compression_ratio": comp_ratio,
            "generation_info_loss_pct": collision_section.get("info_loss_pct"),
        },
        "token_distribution": token_stats,
        "trajectory_stats": traj_stats,
    }

    if simulated_collision:
        report["simulated_candidate_collision"] = simulated_collision

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved conversion stats to {args.output_json}")

    if args.plots_dir:
        make_plots(
            token_counts,
            mapper,
            collision_rates,
            unique_counts,
            total_sizes,
            args.plots_dir,
        )


if __name__ == "__main__":
    main()
