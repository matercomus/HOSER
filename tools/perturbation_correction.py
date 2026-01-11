"""Perturbation correction evaluation (Phase B).

This evaluates whether a model's generated trajectories are closer to the clean
reference trajectory than to the perturbed (dirty) trajectory.

Data contract:
- Input CSV rows include a dirty `rid_list` and an `abnormality_info` column.
- For abnormal rows, `abnormality_info` is a Python-literal dict string with a
  `real` field that contains the clean `rid_list`.

Outputs:
- Writes summary JSON and per-row JSONL under
    `eval_dir/perturbation_correction/{model_type}/`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from fastdtw import fastdtw
from haversine import haversine

from evaluation import load_road_network
from gene import generate_trajectories_programmatic
from tools.abnormality_metadata import parse_abnormality_info, parse_rid_list

logger = logging.getLogger(__name__)


_TEACHER_CACHE: dict[
    tuple[str, str, str, int, float, str],
    tuple[Any, Any],
] = {}


@dataclass(frozen=True)
class PerturbationTeacherConfig:
    """Optional LM-TAD teacher config for perplexity triangulation."""

    lmtad_repo: Path
    lmtad_checkpoint: Path
    device: str
    batch_size: int = 128
    grid_size: float = 0.001


@dataclass(frozen=True)
class PerturbationCorrectionConfig:
    """Configuration for perturbation correction evaluation."""

    dataset: str
    eval_dir: Path
    project_root: Path

    perturbation_source_csv: Path

    # Optional dataset root override (absolute or relative to eval_dir).
    # Needed for eval workspaces where datasets live outside project_root/data/{dataset}.
    data_dir: Optional[Path] = None
    od_source: str = "train"
    max_entries: Optional[int] = None
    seed: int = 0

    use_astar: bool = False
    beam_width: int = 4

    cuda_device: int = 0

    force: bool = False

    teacher: Optional[PerturbationTeacherConfig] = None


@dataclass(frozen=True)
class AbnormalTrajectoryPair:
    """A single abnormal example containing clean/dirty reference trajectories."""

    traj_id: str
    dirty_road_ids: List[int]
    clean_road_ids: List[int]
    meta: Dict[str, Any]

    @property
    def od_pair(self) -> Tuple[int, int]:
        if len(self.clean_road_ids) < 2:
            raise ValueError("Clean trajectory too short to extract OD")
        return self.clean_road_ids[0], self.clean_road_ids[-1]


def _short_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:10]


def _parse_rid_list(value: Any) -> List[int]:
    """Backward-compatible wrapper for `tools.abnormality_metadata.parse_rid_list`."""

    return parse_rid_list(value)


def _parse_abnormality_info(value: str) -> Optional[Dict[str, Any]]:
    """Backward-compatible wrapper for `tools.abnormality_metadata.parse_abnormality_info`."""

    return parse_abnormality_info(value)


def iter_abnormal_pairs(
    source_csv: Path,
) -> Iterator[AbnormalTrajectoryPair]:
    """Stream abnormal rows from a perturbation CSV."""
    if not source_csv.exists():
        raise FileNotFoundError(f"Perturbation CSV not found: {source_csv}")

    with source_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "traj_id",
            "rid_list",
            "abnormality_info",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Perturbation CSV missing columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            ab = _parse_abnormality_info(row.get("abnormality_info", ""))
            if ab is None:
                continue

            real = ab.get("real")
            if not isinstance(real, dict):
                raise ValueError("abnormality_info['real'] must be a dict")

            dirty_road_ids = _parse_rid_list(row.get("rid_list"))
            clean_road_ids = _parse_rid_list(real.get("rid_list"))

            yield AbnormalTrajectoryPair(
                traj_id=str(row.get("traj_id", "")),
                dirty_road_ids=dirty_road_ids,
                clean_road_ids=clean_road_ids,
                meta={
                    "abnormality": ab,
                },
            )


def _reservoir_sample(
    items: Iterable[AbnormalTrajectoryPair],
    k: int,
    seed: int,
) -> List[AbnormalTrajectoryPair]:
    """Reservoir-sample k items from a stream."""
    rng = np.random.default_rng(seed)
    sample: List[AbnormalTrajectoryPair] = []

    for idx, item in enumerate(items):
        if idx < k:
            sample.append(item)
            continue

        j = int(rng.integers(0, idx + 1))
        if j < k:
            sample[j] = item

    return sample


def _build_road_gps(roadmap_geo: Path) -> Dict[int, Tuple[float, float]]:
    """Load road_id -> (lon, lat) mapping from roadmap.geo."""
    geo_df = load_road_network(str(roadmap_geo))
    geo_df = geo_df.set_index("road_id")

    road_gps: Dict[int, Tuple[float, float]] = {}
    for road_id, row in geo_df.iterrows():
        center_gps = row.get("center_gps")
        if center_gps is None:
            continue
        # center_gps comes from a Polars Series of tuples and may arrive as
        # tuple/list/np.ndarray depending on pandas conversion.
        if isinstance(center_gps, np.ndarray):
            center_gps = center_gps.tolist()

        if not isinstance(center_gps, (tuple, list)) or len(center_gps) != 2:
            continue

        lat, lon = center_gps
        if lat is None or lon is None:
            continue
        road_gps[int(road_id)] = (float(lon), float(lat))

    if not road_gps:
        raise RuntimeError("No valid road GPS centroids loaded")

    return road_gps


def _to_coord_traj(
    road_ids: Sequence[int],
    road_gps: Dict[int, Tuple[float, float]],
) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for rid in road_ids:
        gps = road_gps.get(int(rid))
        if gps is None:
            continue
        lon, lat = gps
        coords.append((lat, lon))
    return coords


def _dtw_km(
    traj_a: Sequence[int],
    traj_b: Sequence[int],
    road_gps: Dict[int, Tuple[float, float]],
) -> float:
    coords_a = _to_coord_traj(traj_a, road_gps)
    coords_b = _to_coord_traj(traj_b, road_gps)

    if len(coords_a) < 2 or len(coords_b) < 2:
        return float("inf")

    dist, _ = fastdtw(coords_a, coords_b, dist=haversine)
    return float(dist)


def _dtw_norm_km(
    traj_a: Sequence[int],
    traj_b: Sequence[int],
    road_gps: Dict[int, Tuple[float, float]],
) -> float:
    dist = _dtw_km(traj_a, traj_b, road_gps)
    if not np.isfinite(dist):
        return dist

    denom = (len(traj_a) + len(traj_b)) / 2.0
    if denom <= 0:
        return float("inf")
    return float(dist / denom)


def _load_generated_road_ids(generated_csv: Path) -> List[List[int]]:
    """Load gene_trace_road_id JSON lists in row order."""
    if not generated_csv.exists():
        raise FileNotFoundError(f"Generated CSV not found: {generated_csv}")

    road_ids: List[List[int]] = []
    with generated_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "gene_trace_road_id" not in (reader.fieldnames or []):
            raise ValueError(
                f"Generated CSV missing gene_trace_road_id column: {generated_csv}"
            )

        for row in reader:
            raw = row.get("gene_trace_road_id", "[]")
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("gene_trace_road_id was not a JSON list")
            road_ids.append([int(x) for x in parsed])

    return road_ids


def _maybe_load_teacher(
    teacher_cfg: PerturbationTeacherConfig,
    roadmap_geo: Path,
) -> Tuple[Any, Any]:
    """Load LM-TAD teacher and road_to_token mapping."""
    from critics.lmtad_teacher import LMTADTeacher
    from critics.grid_mapper import GridConfig, GridMapper
    import csv
    import json

    cache_key = (
        str(teacher_cfg.lmtad_repo),
        str(teacher_cfg.lmtad_checkpoint),
        str(teacher_cfg.device),
        int(teacher_cfg.batch_size),
        float(teacher_cfg.grid_size),
        str(roadmap_geo),
    )
    cached = _TEACHER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = LMTADTeacher(
        repo_path=str(teacher_cfg.lmtad_repo),
        ckpt_path=str(teacher_cfg.lmtad_checkpoint),
        device=teacher_cfg.device,
        dtype="float16",
        window=256,
    )

    # IMPORTANT: LM-TAD token ids depend on the exact grid boundary.
    # To keep tokenization consistent with teacher training and the project's
    # dataset-level LM-TAD evaluation (`tools/evaluate_dataset_with_lmtad.py`),
    # compute the grid boundary from *all* roadmap polyline points (not just
    # centroids), and compute per-road centroids as the mean of polyline points.
    min_lat, max_lat = float("inf"), float("-inf")
    min_lng, max_lng = float("inf"), float("-inf")
    centroids_by_id: Dict[int, Tuple[float, float]] = {}
    max_road_id = -1

    with roadmap_geo.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "coordinates" not in (reader.fieldnames or []):
            raise ValueError(
                f"Roadmap missing required column 'coordinates': {roadmap_geo}"
            )

        has_road_id = "road_id" in (reader.fieldnames or [])

        for row_idx, row in enumerate(reader):
            coords_str = row.get("coordinates")
            if coords_str is None:
                continue

            if has_road_id:
                rid_raw = row.get("road_id")
                if rid_raw is None:
                    continue
                try:
                    road_id = int(rid_raw)
                except (TypeError, ValueError):
                    continue
            else:
                # Most roadmap.geo exports in this repo do not include road_id;
                # they implicitly use the row index as road_id.
                road_id = int(row_idx)

            coords_str = str(coords_str)
            if not coords_str.strip():
                continue

            # roadmap.geo typically stores coordinates as JSON list of [lng, lat] pairs.
            try:
                coords = json.loads(coords_str)
            except json.JSONDecodeError:
                # Fallback: some older exports may use python-literal lists.
                coords = eval(coords_str)

            if not isinstance(coords, list) or not coords:
                continue

            try:
                lngs = [float(c[0]) for c in coords]
                lats = [float(c[1]) for c in coords]
            except (TypeError, ValueError, IndexError):
                continue

            if any(lat < -90 or lat > 90 for lat in lats) or any(
                lng < -180 or lng > 180 for lng in lngs
            ):
                continue

            centroid_lat = float(sum(lats) / len(lats))
            centroid_lng = float(sum(lngs) / len(lngs))
            centroids_by_id[road_id] = (centroid_lat, centroid_lng)
            max_road_id = max(max_road_id, road_id)

            # Boundary from *all* points.
            min_lat = min(min_lat, min(lats))
            max_lat = max(max_lat, max(lats))
            min_lng = min(min_lng, min(lngs))
            max_lng = max(max_lng, max(lngs))

    if max_road_id < 0 or not centroids_by_id:
        raise RuntimeError(f"Failed to parse road centroids from roadmap: {roadmap_geo}")

    if has_road_id:
        missing = [i for i in range(max_road_id + 1) if i not in centroids_by_id]
        if missing:
            preview = ",".join(str(i) for i in missing[:10])
            raise ValueError(
                "Roadmap road_id indices are not contiguous from 0..max; refusing to proceed to avoid road-id/index mismatch. "
                f"Missing: {len(missing)} (first: {preview}{'...' if len(missing) > 10 else ''})"
            )

    if not (min_lat < max_lat and min_lng < max_lng):
        raise ValueError(
            f"Invalid boundary computed from roadmap points: lat=[{min_lat}, {max_lat}], lng=[{min_lng}, {max_lng}]"
        )

    road_centroids_latlng = np.asarray(
        [centroids_by_id[i] for i in range(max_road_id + 1)], dtype=np.float64
    )
    grid_config = GridConfig(
        min_lat=float(min_lat),
        max_lat=float(max_lat),
        min_lng=float(min_lng),
        max_lng=float(max_lng),
        grid_size=teacher_cfg.grid_size,
    )

    mapper = GridMapper(boundary=grid_config, road_centroids=road_centroids_latlng, verify_hw=None)

    import torch

    road_to_token = torch.from_numpy(mapper.map_all()).to(teacher_cfg.device)
    _TEACHER_CACHE[cache_key] = (model, road_to_token)
    return model, road_to_token


def _teacher_perplexities(
    trajectories: List[List[int]],
    teacher_cfg: PerturbationTeacherConfig,
    roadmap_geo: Path,
) -> List[float]:
    from simple_evaluate_with_lmtad import evaluate_trajectories_direct

    model, road_to_token = _maybe_load_teacher(teacher_cfg, roadmap_geo)
    perplexities, _outliers, _seg = evaluate_trajectories_direct(
        trajectories=trajectories,
        model=model,
        road_to_token=road_to_token,
        device=teacher_cfg.device,
        batch_size=teacher_cfg.batch_size,
    )
    return [float(x) for x in perplexities]


def run_perturbation_correction(
    *,
    cfg: PerturbationCorrectionConfig,
    model_type: str,
    model_checkpoint: Path,
) -> Path:
    """Run perturbation correction evaluation for a single model.

    Returns:
        Path to the written summary JSON.
    """
    if cfg.od_source not in {"train", "test"}:
        raise ValueError("od_source must be 'train' or 'test'")

    resolved_data_dir: Optional[Path] = None
    if cfg.data_dir is not None:
        resolved_data_dir = Path(cfg.data_dir)
        if not resolved_data_dir.is_absolute():
            resolved_data_dir = (cfg.eval_dir / resolved_data_dir).resolve()
    else:
        resolved_data_dir = cfg.project_root / "data" / cfg.dataset

    roadmap_geo = resolved_data_dir / "roadmap.geo"
    if not roadmap_geo.exists():
        raise FileNotFoundError(f"Road network not found: {roadmap_geo}")

    out_dir = cfg.eval_dir / "perturbation_correction" / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Phase B: loading abnormal examples from %s", cfg.perturbation_source_csv
    )
    stream = iter_abnormal_pairs(cfg.perturbation_source_csv)

    if cfg.max_entries is not None:
        if cfg.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        pairs = _reservoir_sample(stream, cfg.max_entries, cfg.seed)
    else:
        pairs = list(stream)

    if not pairs:
        raise ValueError("No abnormal rows found in perturbation_source_csv")

    logger.info("Phase B: selected %d abnormal examples", len(pairs))

    od_pairs = [p.od_pair for p in pairs]

    search_method = "astar" if cfg.use_astar else "beam"
    sample_id = _short_hash(
        f"{cfg.perturbation_source_csv}|{cfg.max_entries}|{cfg.seed}|{search_method}"
    )

    gene_dir = cfg.eval_dir / "gene_perturbation" / cfg.dataset / f"seed{cfg.seed}"
    gene_dir.mkdir(parents=True, exist_ok=True)

    generated_csv = gene_dir / (f"perturb_{sample_id}_{model_type}_{cfg.od_source}.csv")

    if generated_csv.exists() and not cfg.force:
        logger.info("Reusing existing perturbation generation: %s", generated_csv)
    else:
        logger.info(
            "Generating %d trajectories for Phase B (%s)",
            len(od_pairs),
            search_method,
        )
        generate_trajectories_programmatic(
            dataset=cfg.dataset,
            model_path=str(model_checkpoint),
            od_source=cfg.od_source,
            seed=cfg.seed,
            num_gene=len(od_pairs),
            od_pairs=od_pairs,
            output_file=generated_csv,
            data_dir=str(resolved_data_dir),
            cuda_device=cfg.cuda_device,
            beam_search=not cfg.use_astar,
            beam_width=cfg.beam_width,
            enable_wandb=False,
            model_type=model_type,
        )

    predicted = _load_generated_road_ids(generated_csv)
    if len(predicted) != len(pairs):
        raise ValueError(
            "Mismatch: generated rows != abnormal examples "
            f"({len(predicted)} != {len(pairs)})"
        )

    road_gps = _build_road_gps(roadmap_geo)

    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"

    corrected = 0
    valid = 0
    invalid = 0

    dtw_clean_vals: List[float] = []
    dtw_dirty_vals: List[float] = []

    with rows_path.open("w") as f:
        for i, pair in enumerate(pairs):
            pred = predicted[i]
            dtw_clean = _dtw_km(pred, pair.clean_road_ids, road_gps)
            dtw_dirty = _dtw_km(pred, pair.dirty_road_ids, road_gps)
            dtw_clean_norm = _dtw_norm_km(pred, pair.clean_road_ids, road_gps)
            dtw_dirty_norm = _dtw_norm_km(pred, pair.dirty_road_ids, road_gps)

            if not np.isfinite(dtw_clean) or not np.isfinite(dtw_dirty):
                invalid += 1
                is_corrected = None
            else:
                valid += 1
                is_corrected = bool(dtw_clean < dtw_dirty)
                if is_corrected:
                    corrected += 1
                dtw_clean_vals.append(float(dtw_clean))
                dtw_dirty_vals.append(float(dtw_dirty))

            ab = pair.meta.get("abnormality", {})
            ab_type = ab.get("type")
            ab_level = ab.get("level")
            ab_strength = ab.get("strength")

            row_out = {
                "i": i,
                "traj_id": pair.traj_id,
                "od": list(pair.od_pair),
                "ab_type": ab_type,
                "ab_level": ab_level,
                "ab_strength": ab_strength,
                "dtw_to_clean_km": dtw_clean,
                "dtw_to_dirty_km": dtw_dirty,
                "dtw_to_clean_norm": dtw_clean_norm,
                "dtw_to_dirty_norm": dtw_dirty_norm,
                "corrected": is_corrected,
            }
            f.write(json.dumps(row_out) + "\n")

    rsr = float(corrected / valid) if valid > 0 else 0.0

    summary: Dict[str, Any] = {
        "model_type": model_type,
        "model_checkpoint": str(model_checkpoint),
        "dataset": cfg.dataset,
        "perturbation_source_csv": str(cfg.perturbation_source_csv),
        "od_source": cfg.od_source,
        "max_entries": cfg.max_entries,
        "seed": cfg.seed,
        "use_astar": cfg.use_astar,
        "beam_width": cfg.beam_width,
        "counts": {
            "total": len(pairs),
            "valid": valid,
            "invalid": invalid,
            "corrected": corrected,
        },
        "rsr": rsr,
        "dtw_km": {
            "mean_to_clean": float(np.mean(dtw_clean_vals)) if dtw_clean_vals else None,
            "mean_to_dirty": float(np.mean(dtw_dirty_vals)) if dtw_dirty_vals else None,
        },
        "artifacts": {
            "generated_csv": str(generated_csv),
            "rows_jsonl": str(rows_path),
        },
    }

    if cfg.teacher is not None:
        logger.info("Phase B: computing LM-TAD perplexities")
        pred_ppl = _teacher_perplexities(predicted, cfg.teacher, roadmap_geo)
        clean_ppl = _teacher_perplexities(
            [p.clean_road_ids for p in pairs],
            cfg.teacher,
            roadmap_geo,
        )
        dirty_ppl = _teacher_perplexities(
            [p.dirty_road_ids for p in pairs],
            cfg.teacher,
            roadmap_geo,
        )

        teacher_corrected = 0
        for g, c, d in zip(pred_ppl, clean_ppl, dirty_ppl):
            if abs(g - c) < abs(g - d):
                teacher_corrected += 1

        summary["teacher"] = {
            "device": cfg.teacher.device,
            "batch_size": cfg.teacher.batch_size,
            "mean_log_perplexity_generated": float(np.mean(pred_ppl)),
            "mean_log_perplexity_clean": float(np.mean(clean_ppl)),
            "mean_log_perplexity_dirty": float(np.mean(dirty_ppl)),
            "triangulation_rate": float(teacher_corrected / len(pairs)),
        }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase B: wrote %s", summary_path)
    return summary_path
