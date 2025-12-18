#!/usr/bin/env python3
"""Visualize dataset perturbations (real vs abnormal trajectories).

This tool is intentionally decoupled from evaluation workspaces (`eval_dir`) and
model-generated outputs. It operates directly on a dataset directory such as:
- `data/porto_hoser_abnormal_3`
- `data/Beijing_abnormal_3`

Input contract (for `--split train`):
- `<dataset_dir>/train.csv` contains an `abnormality_info` column.
- Abnormal rows have `abnormality_info` as a Python-literal dict string with a
  required `real` field storing the clean trajectory.

Outputs:
- Writes plots under `<dataset_dir>/viz/perturbations/<split>/<type>/<level>/<strength>/`.

Usage:
    uv run python -m tools.visualize_perturbations \
        --dataset-dir data/porto_hoser_abnormal_3 \
        --split train \
        --max-plots 30 \
        --per-group 3
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import polars as pl

from tools.abnormality_metadata import (
    CLEAN_MISSING_ALPHA,
    DIRTY_PERTURBED_COLOR,
    DIRTY_SHARED_COLOR,
    build_abnormality_metadata,
    parse_abnormality_info,
    parse_rid_list,
)
from tools.model_detection import get_model_color
from tools.trajectory_diff import AlignmentResult, align_trajectories

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VizConfig:
    """Configuration for perturbation visualization."""

    dataset_dir: Path
    split: str

    output_root: Path

    max_plots: int
    per_group: int
    seed: int

    include_types: Optional[Sequence[str]]
    include_levels: Optional[Sequence[str]]
    include_strengths: Optional[Sequence[str]]

    dpi: int = 250
    figsize: Tuple[int, int] = (12, 10)
    margin: float = 0.002


@dataclass(frozen=True)
class AbnormalExample:
    """A single abnormal training example containing clean+dirty trajectories."""

    traj_id: str
    dirty_road_ids: List[int]
    clean_road_ids: List[int]
    abnormality_info: Dict[str, Any]
    row_meta: Dict[str, Any]

    @property
    def od_pair(self) -> Tuple[int, int]:
        if len(self.clean_road_ids) < 2:
            raise ValueError("Clean trajectory too short")
        return self.clean_road_ids[0], self.clean_road_ids[-1]


class DatasetPaths:
    """Resolve common dataset paths from a dataset directory."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    @property
    def roadmap_geo(self) -> Path:
        return self.dataset_dir / "roadmap.geo"

    def split_csv(self, split: str) -> Path:
        return self.dataset_dir / f"{split}.csv"

    @property
    def output_root(self) -> Path:
        return self.dataset_dir / "viz" / "perturbations"


class RoadNetwork:
    """Road geometry lookup loaded from roadmap.geo."""

    def __init__(self, roadmap_geo: Path):
        self.roadmap_geo = roadmap_geo
        self._coords: Optional[Dict[int, List[Tuple[float, float]]]] = None

    def load(self) -> Dict[int, List[Tuple[float, float]]]:
        if self._coords is not None:
            return self._coords

        if not self.roadmap_geo.exists():
            raise FileNotFoundError(f"roadmap.geo not found: {self.roadmap_geo}")

        df = pl.read_csv(
            self.roadmap_geo,
            schema_overrides={
                "lanes": pl.Utf8,
                "oneway": pl.Utf8,
            },
        )

        id_col = "road_id" if "road_id" in df.columns else "geo_id"
        if id_col not in df.columns or "coordinates" not in df.columns:
            raise ValueError(
                f"Unexpected roadmap.geo schema (need {id_col}/coordinates): {self.roadmap_geo}"
            )

        coords: Dict[int, List[Tuple[float, float]]] = {}
        for row in df.iter_rows(named=True):
            rid_raw = row.get(id_col)
            if rid_raw is None:
                continue
            rid = int(rid_raw)
            raw = row.get("coordinates")
            if raw is None:
                continue
            try:
                parsed = ast.literal_eval(str(raw))
            except Exception:
                continue
            road_coords: List[Tuple[float, float]] = []
            for pair in parsed:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                lon, lat = pair
                road_coords.append((float(lon), float(lat)))
            if len(road_coords) >= 2:
                coords[rid] = road_coords

        if not coords:
            raise RuntimeError(f"No road coordinates loaded from {self.roadmap_geo}")

        self._coords = coords
        return coords


def iter_abnormal_examples(
    csv_path: Path,
    *,
    scan_limit: Optional[int] = None,
) -> Iterator[AbnormalExample]:
    """Stream abnormal examples from a dataset split CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header: {csv_path}")

        required = {"traj_id", "rid_list", "abnormality_info"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing required columns {sorted(missing)}: {csv_path} (cols={reader.fieldnames})"
            )

        for idx, row in enumerate(reader):
            if scan_limit is not None and idx >= scan_limit:
                break
            ab = parse_abnormality_info(row.get("abnormality_info"))
            if ab is None:
                continue

            real = ab.get("real")
            if not isinstance(real, dict):
                raise ValueError("abnormality_info['real'] must be a dict")

            dirty_road_ids = parse_rid_list(row.get("rid_list"))
            clean_road_ids = parse_rid_list(real.get("rid_list"))

            yield AbnormalExample(
                traj_id=str(row.get("traj_id", "")),
                dirty_road_ids=dirty_road_ids,
                clean_road_ids=clean_road_ids,
                abnormality_info=ab,
                row_meta={
                    "mm_id": row.get("mm_id"),
                    "user_id": row.get("user_id") or row.get("entity_id"),
                },
            )


def _stable_int_hash(text: str) -> int:
    """Stable integer hash for deterministic sampling (python hash is randomized)."""

    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) % 2**32
    return int(h)


def reservoir_sample_by_group(
    items: Iterable[AbnormalExample],
    *,
    per_group: int,
    seed: int,
    filters: "FilterConfig",
) -> Dict[str, List[AbnormalExample]]:
    """Reservoir-sample up to k items per abnormality group from a stream."""

    groups: Dict[str, List[AbnormalExample]] = {}
    seen: Dict[str, int] = {}

    import random

    for item in items:
        meta = build_abnormality_metadata(item.abnormality_info)
        if not filters.accept(meta):
            continue

        key = meta.group_key
        seen[key] = seen.get(key, 0) + 1
        bucket = groups.setdefault(key, [])

        rng = random.Random(seed + _stable_int_hash(key))
        count = seen[key]

        if len(bucket) < per_group:
            bucket.append(item)
            continue

        j = rng.randint(0, count - 1)
        if j < per_group:
            bucket[j] = item

    return groups


@dataclass(frozen=True)
class FilterConfig:
    include_types: Optional[Sequence[str]]
    include_levels: Optional[Sequence[str]]
    include_strengths: Optional[Sequence[str]]

    def accept(self, meta: Any) -> bool:
        type_token = str(meta.abnormal_type)
        level_token = str(meta.level)
        strength_token = str(meta.strength)

        if self.include_types and type_token not in self.include_types:
            return False
        if self.include_levels and level_token not in self.include_levels:
            return False
        if self.include_strengths and strength_token not in self.include_strengths:
            return False
        return True


class PerturbationPlotter:
    """Render a clean vs dirty overlay plot with colored segments."""

    def __init__(self, config: VizConfig, road_coords: Dict[int, List[Tuple[float, float]]]):
        self.config = config
        self.road_coords = road_coords
        self.real_color = get_model_color("real")
        # Visual layering: draw the real trajectory as a thicker underlay so it
        # remains visible even when the abnormal trajectory overlaps it.
        self._real_underlay_width = 7.0
        self._abnormal_width = 4.0

    def plot(
        self,
        *,
        example: AbnormalExample,
        alignment: AlignmentResult,
        out_path: Path,
    ) -> None:
        meta = build_abnormality_metadata(example.abnormality_info)
        origin, dest = example.od_pair

        clean_segments, clean_colors, clean_alphas = self._build_segments(
            example.clean_road_ids,
            alignment.clean_node_labels,
            shared_color=self.real_color,
            non_shared_color=self.real_color,
            non_shared_alpha=CLEAN_MISSING_ALPHA,
            non_shared_label="missing",
        )

        dirty_segments, dirty_colors, dirty_alphas = self._build_segments(
            example.dirty_road_ids,
            alignment.dirty_node_labels,
            shared_color=DIRTY_SHARED_COLOR,
            non_shared_color=DIRTY_PERTURBED_COLOR,
            non_shared_alpha=1.0,
            non_shared_label="perturbed",
        )

        if not clean_segments or not dirty_segments:
            raise ValueError("Insufficient coordinates to plot")

        fig, ax = plt.subplots(figsize=self.config.figsize, facecolor="white")
        ax.set_facecolor("white")

        # Plot clean (real) first, then dirty on top.
        self._add_line_collection(
            ax,
            clean_segments,
            clean_colors,
            clean_alphas,
            z=5,
            linewidth=self._real_underlay_width,
        )
        self._add_line_collection(
            ax,
            dirty_segments,
            dirty_colors,
            dirty_alphas,
            z=10,
            linewidth=self._abnormal_width,
        )

        # Start/end markers from clean trajectory.
        clean_start = clean_segments[0][0]
        clean_end = clean_segments[-1][1]
        ax.scatter(
            clean_start[0],
            clean_start[1],
            c=self.real_color,
            s=60,
            marker="o",
            zorder=20,
            edgecolors="black",
            linewidths=1.2,
        )
        ax.scatter(
            clean_end[0],
            clean_end[1],
            c=self.real_color,
            s=60,
            marker="s",
            zorder=20,
            edgecolors="black",
            linewidths=1.2,
        )

        # Bounds.
        all_points = [p for seg in clean_segments for p in seg] + [
            p for seg in dirty_segments for p in seg
        ]
        lons = [p[0] for p in all_points]
        lats = [p[1] for p in all_points]
        margin = self.config.margin
        ax.set_xlim(min(lons) - margin, max(lons) + margin)
        ax.set_ylim(min(lats) - margin, max(lats) + margin)

        title = (
            f"{self.config.split.upper()} • Perturbation Compare • {meta.display_name}"
            f" • OD: {origin} -> {dest} • traj_id={example.traj_id}"
        )
        ax.set_title(title, fontsize=14, pad=16)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")

        legend = self._build_legend(
            has_clean_shared="shared" in set(alignment.clean_node_labels),
            has_clean_missing="missing" in set(alignment.clean_node_labels),
            has_dirty_shared="shared" in set(alignment.dirty_node_labels),
            has_dirty_perturbed="perturbed" in set(alignment.dirty_node_labels),
        )
        ax.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=10,
            title="Legend",
            title_fontsize=11,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{out_path}.png", dpi=self.config.dpi, bbox_inches="tight")
        fig.savefig(f"{out_path}.pdf", dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

    def _add_line_collection(
        self,
        ax: Any,
        segments: List[List[Tuple[float, float]]],
        colors: List[str],
        alphas: List[float],
        *,
        z: int,
        linewidth: float,
    ) -> None:
        # Per-segment alpha requires setting RGBA; emulate by adjusting colors
        # through multiple collections (shared vs non-shared). For simplicity,
        # build two collections.
        #
        # This helper keeps the API simple: we split segments by alpha.
        by_alpha: Dict[float, List[int]] = {}
        for idx, a in enumerate(alphas):
            by_alpha.setdefault(float(a), []).append(idx)

        for alpha, indices in by_alpha.items():
            sub_segments = [segments[i] for i in indices]
            sub_colors = [colors[i] for i in indices]
            sub = LineCollection(
                sub_segments,
                colors=sub_colors,
                linewidths=float(linewidth),
                zorder=z,
                capstyle="round",
                joinstyle="round",
                alpha=alpha,
                linestyles="--" if alpha < 1.0 else "-",
            )
            ax.add_collection(sub)

    def _build_segments(
        self,
        road_ids: Sequence[int],
        road_labels: Sequence[str],
        *,
        shared_color: str,
        non_shared_color: str,
        non_shared_alpha: float,
        non_shared_label: str,
    ) -> Tuple[List[List[Tuple[float, float]]], List[str], List[float]]:
        if len(road_ids) != len(road_labels):
            raise ValueError("road_ids/labels length mismatch")

        segments: List[List[Tuple[float, float]]] = []
        colors: List[str] = []
        alphas: List[float] = []

        for rid, label in zip(road_ids, road_labels):
            coords = self.road_coords.get(int(rid))
            if not coords or len(coords) < 2:
                return [], [], []

            for a, b in zip(coords[:-1], coords[1:]):
                segments.append([a, b])
                if label == "shared":
                    colors.append(shared_color)
                    alphas.append(1.0)
                elif label == non_shared_label:
                    colors.append(non_shared_color)
                    alphas.append(float(non_shared_alpha))
                else:
                    # Defensive fallback.
                    colors.append(non_shared_color)
                    alphas.append(float(non_shared_alpha))

        return segments, colors, alphas

    def _build_legend(
        self,
        *,
        has_clean_shared: bool,
        has_clean_missing: bool,
        has_dirty_shared: bool,
        has_dirty_perturbed: bool,
    ) -> List[Any]:
        items: List[Any] = []

        if has_clean_shared:
            items.append(
                Line2D(
                    [0],
                    [0],
                    color=self.real_color,
                    linewidth=self._real_underlay_width,
                    label="Real (shared)",
                )
            )
        if has_clean_missing:
            items.append(
                Line2D(
                    [0],
                    [0],
                    color=self.real_color,
                    linewidth=self._real_underlay_width,
                    alpha=CLEAN_MISSING_ALPHA,
                    linestyle="--",
                    label="Real (missing vs abnormal)",
                )
            )
        if has_dirty_shared:
            items.append(
                Line2D(
                    [0],
                    [0],
                    color=DIRTY_SHARED_COLOR,
                    linewidth=self._abnormal_width,
                    label="Abnormal (same as real)",
                )
            )
        if has_dirty_perturbed:
            items.append(
                Line2D(
                    [0],
                    [0],
                    color=DIRTY_PERTURBED_COLOR,
                    linewidth=self._abnormal_width,
                    label="Abnormal (perturbed)",
                )
            )

        items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=9,
                markeredgecolor="black",
                markeredgewidth=1.2,
                label="Start",
                linestyle="",
            )
        )
        items.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="gray",
                markersize=9,
                markeredgecolor="black",
                markeredgewidth=1.2,
                label="End",
                linestyle="",
            )
        )

        return items


def _parse_csv_filter_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    tokens = [t.strip().lower() for t in value.split(",") if t.strip()]
    return tokens or None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Visualize real vs abnormal trajectories in an abnormal dataset directory"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory (e.g., data/porto_hoser_abnormal_3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split CSV to read (default: train)",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=30,
        help="Maximum number of plots to render (default: 30)",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=3,
        help="Reservoir samples per (type/level/strength) group (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed (default: 0)",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="Optional cap on how many CSV rows to scan (default: scan all)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated abnormality types to include (e.g. detour,route_switch)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated levels to include (e.g. low,medium,high)",
    )
    parser.add_argument(
        "--strengths",
        type=str,
        default=None,
        help="Comma-separated strengths to include (strong,weak)",
    )

    args = parser.parse_args()

    paths = DatasetPaths(args.dataset_dir)
    split_csv = paths.split_csv(args.split)
    out_root = paths.output_root

    cfg = VizConfig(
        dataset_dir=args.dataset_dir,
        split=args.split,
        output_root=out_root,
        max_plots=max(1, int(args.max_plots)),
        per_group=max(1, int(args.per_group)),
        seed=int(args.seed),
        include_types=_parse_csv_filter_list(args.types),
        include_levels=_parse_csv_filter_list(args.levels),
        include_strengths=_parse_csv_filter_list(args.strengths),
    )

    if not cfg.dataset_dir.exists():
        raise FileNotFoundError(f"dataset-dir not found: {cfg.dataset_dir}")

    road_coords = RoadNetwork(paths.roadmap_geo).load()

    examples = iter_abnormal_examples(split_csv, scan_limit=args.scan_limit)
    filters = FilterConfig(
        include_types=cfg.include_types,
        include_levels=cfg.include_levels,
        include_strengths=cfg.include_strengths,
    )

    grouped = reservoir_sample_by_group(
        examples,
        per_group=cfg.per_group,
        seed=cfg.seed,
        filters=filters,
    )

    # Flatten in stable order.
    group_keys = sorted(grouped.keys())
    to_plot: List[Tuple[str, AbnormalExample]] = []
    for key in group_keys:
        for item in grouped[key]:
            to_plot.append((key, item))

    if not to_plot:
        logger.warning("No abnormal examples selected (check filters/split)")
        return 0

    if len(to_plot) > cfg.max_plots:
        to_plot = to_plot[: cfg.max_plots]

    plotter = PerturbationPlotter(cfg, road_coords)

    rendered = 0
    for group_key, ex in to_plot:
        meta = build_abnormality_metadata(ex.abnormality_info)
        origin, dest = ex.od_pair

        alignment = align_trajectories(
            ex.clean_road_ids,
            ex.dirty_road_ids,
            abnormality_info=ex.abnormality_info,
        )

        out_dir = cfg.output_root / cfg.split / meta.group_key
        out_path = out_dir / f"{cfg.split}_traj{ex.traj_id}_origin{origin}_dest{dest}"

        try:
            plotter.plot(example=ex, alignment=alignment, out_path=out_path)
            rendered += 1
            logger.info("Saved %s", out_path)
        except Exception as e:
            logger.warning("Skipping traj_id=%s (%s)", ex.traj_id, e)

    logger.info("Rendered %d plot(s) under %s", rendered, cfg.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
