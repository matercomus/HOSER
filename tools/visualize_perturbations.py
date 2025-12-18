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
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D

import polars as pl

from tools.abnormality_metadata import (
    build_abnormality_metadata,
    parse_abnormality_info,
    parse_rid_list,
)
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

    # Styling (defaults intentionally avoid tools/model_detection.py palettes)
    real_color: str = "#1f77b4"  # matplotlib default blue
    abnormal_color: str = "#ff7f0e"  # matplotlib default orange
    real_linestyle: str = "-"  # solid
    abnormal_linestyle: str = "-"  # solid

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
    """Render a clean vs dirty comparison plot.

    This plot intentionally uses a *parallel offset* (ribbon) style so both the
    real and abnormal trajectories remain visible even when they overlap.
    """

    def __init__(self, config: VizConfig, road_coords: Dict[int, List[Tuple[float, float]]]):
        self.config = config
        self.road_coords = road_coords
        self.real_color = str(config.real_color)
        self._real_width = 4.0
        self._abnormal_width = 4.0
        self._abnormal_color = str(config.abnormal_color)
        self._real_linestyle = str(config.real_linestyle)
        self._abnormal_linestyle = str(config.abnormal_linestyle)

        # Pre-index road bounding boxes for fast per-plot filtering.
        self._road_bboxes: List[Tuple[List[Tuple[float, float]], Tuple[float, float, float, float]]] = []
        for coords in self.road_coords.values():
            if not coords or len(coords) < 2:
                continue
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            bbox = (min(lons), max(lons), min(lats), max(lats))
            self._road_bboxes.append((coords, bbox))

    def plot(
        self,
        *,
        example: AbnormalExample,
        alignment: AlignmentResult,
        out_path: Path,
    ) -> None:
        meta = build_abnormality_metadata(example.abnormality_info)
        origin, dest = example.od_pair

        clean_coords, clean_seg_road_idx = self._road_ids_to_polyline_with_segment_road_idx(
            example.clean_road_ids
        )
        dirty_coords, dirty_seg_road_idx = self._road_ids_to_polyline_with_segment_road_idx(
            example.dirty_road_ids
        )
        if len(clean_coords) < 2 or len(dirty_coords) < 2:
            raise ValueError("Insufficient coordinates to plot")

        clean_lons, clean_lats = zip(*clean_coords)
        dirty_lons, dirty_lats = zip(*dirty_coords)

        # Compute a small offset in degrees (dynamic with zoom level) and apply
        # symmetric offsets so the two curves sit side-by-side.
        offset_step = self._calculate_dynamic_offset_step(
            clean_lons,
            clean_lats,
            dirty_lons,
            dirty_lats,
            linewidth=max(self._real_width, self._abnormal_width),
            overlap_factor=0.99,
        )
        real_offset = -offset_step / 2
        abnormal_offset = offset_step / 2

        clean_lons_off, clean_lats_off = self._calculate_parallel_offset(
            list(clean_lons),
            list(clean_lats),
            real_offset,
        )
        dirty_lons_off, dirty_lats_off = self._calculate_parallel_offset(
            list(dirty_lons),
            list(dirty_lats),
            abnormal_offset,
        )

        lons = list(clean_lons_off) + list(dirty_lons_off)
        lats = list(clean_lats_off) + list(dirty_lats_off)
        bounds = (
            min(lons) - self.config.margin,
            max(lons) + self.config.margin,
            min(lats) - self.config.margin,
            max(lats) + self.config.margin,
        )

        fig, ax = plt.subplots(figsize=self.config.figsize, facecolor="white")
        ax.set_facecolor("white")

        # Plot road network underlay (light gray reference), filtered to bounds.
        self._plot_road_network(ax, bounds=bounds)

        # Plot abnormal trajectory segments: perturbed segments are fully opaque
        # and slightly darker; shared segments are lightly faded.
        self._plot_abnormal_with_segment_alpha(
            ax,
            lons=list(dirty_lons_off),
            lats=list(dirty_lats_off),
            seg_road_idx=dirty_seg_road_idx,
            dirty_node_labels=alignment.dirty_node_labels,
            shared_alpha=0.75,
            perturbed_alpha=1.0,
            zorder=10,
        )

        # Plot real trajectory segments: fade only the local window around the
        # abnormal region to help the abnormality stand out.
        fade_clean_road_idx = self._compute_clean_fade_road_idx(alignment)
        self._plot_real_with_local_fade(
            ax,
            lons=list(clean_lons_off),
            lats=list(clean_lats_off),
            seg_road_idx=clean_seg_road_idx,
            fade_road_idx=fade_clean_road_idx,
            faded_alpha=0.5,
            full_alpha=1.0,
            zorder=11,
        )

        # Start/end markers from (offset) clean trajectory.
        clean_start = (clean_lons_off[0], clean_lats_off[0])
        clean_end = (clean_lons_off[-1], clean_lats_off[-1])
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
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")

        legend = self._build_legend()
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

    def _road_ids_to_polyline_with_segment_road_idx(
        self, road_ids: Sequence[int]
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        """Build a polyline plus a per-segment road-index mapping.

        Returns:
            (points, seg_road_idx) where seg_road_idx has length len(points)-1
            and each entry is the index into `road_ids` for the segment
            connecting points[i] -> points[i+1].
        """

        points: List[Tuple[float, float]] = []
        seg_road_idx: List[int] = []

        for rid_idx, rid in enumerate(road_ids):
            seg = self.road_coords.get(int(rid))
            if not seg or len(seg) < 2:
                continue

            if not points:
                points.append(seg[0])

            start_at = 0
            if points[-1] == seg[0]:
                start_at = 1

            for pt in seg[start_at:]:
                if pt == points[-1]:
                    continue
                points.append(pt)
                seg_road_idx.append(int(rid_idx))

        return points, seg_road_idx

    def _compute_clean_fade_road_idx(self, alignment: AlignmentResult) -> "set[int]":
        """Return indices in the clean road-id sequence to fade.

        Prefer explicit missing regions (route_switch/perturb). For detours and
        insertions (no missing nodes), infer a local window using LCS alignment.
        """

        clean_len = len(alignment.clean_node_labels)
        if clean_len == 0:
            return set()

        missing = [
            i
            for i, label in enumerate(alignment.clean_node_labels)
            if label == "missing"
        ]
        if missing:
            start = max(0, min(missing) - 1)
            end = min(clean_len - 1, max(missing) + 1)
            return set(range(start, end + 1))

        perturbed_dirty = [
            i
            for i, label in enumerate(alignment.dirty_node_labels)
            if label == "perturbed"
        ]
        if not perturbed_dirty:
            return set()

        pairs = alignment.meta.get("lcs_pairs")
        if not isinstance(pairs, list) or not pairs:
            return set()

        dirty_to_clean: Dict[int, int] = {}
        shared_dirty: List[int] = []
        for entry in pairs:
            if (
                not isinstance(entry, (tuple, list))
                or len(entry) != 2
                or not isinstance(entry[0], int)
                or not isinstance(entry[1], int)
            ):
                continue
            clean_i, dirty_i = int(entry[0]), int(entry[1])
            dirty_to_clean[dirty_i] = clean_i
            shared_dirty.append(dirty_i)

        if not dirty_to_clean:
            return set()

        left = min(perturbed_dirty)
        right = max(perturbed_dirty)
        prev_shared = max((d for d in shared_dirty if d < left), default=None)
        next_shared = min((d for d in shared_dirty if d > right), default=None)

        anchors: List[int] = []
        if prev_shared is not None and prev_shared in dirty_to_clean:
            anchors.append(dirty_to_clean[prev_shared])
        if next_shared is not None and next_shared in dirty_to_clean:
            anchors.append(dirty_to_clean[next_shared])

        fade: "set[int]" = set()
        for idx in anchors:
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < clean_len:
                    fade.add(j)
        return fade

    def _plot_abnormal_with_segment_alpha(
        self,
        ax: Any,
        *,
        lons: List[float],
        lats: List[float],
        seg_road_idx: List[int],
        dirty_node_labels: Sequence[str],
        shared_alpha: float,
        perturbed_alpha: float,
        zorder: int,
    ) -> None:
        if len(lons) < 2 or len(lats) < 2:
            return

        if len(seg_road_idx) != len(lons) - 1:
            ax.plot(
                lons,
                lats,
                color=self._abnormal_color,
                linewidth=self._abnormal_width,
                linestyle=self._abnormal_linestyle,
                alpha=perturbed_alpha,
                zorder=zorder,
            )
            return

        emphasis = self._darken_color(self._abnormal_color, factor=0.75)
        segments: List[List[Tuple[float, float]]] = []
        colors: List[Tuple[float, float, float, float]] = []

        for i, rid_idx in enumerate(seg_road_idx):
            label = "perturbed"
            if 0 <= rid_idx < len(dirty_node_labels):
                label = str(dirty_node_labels[rid_idx])

            if label == "shared":
                color = to_rgba(self._abnormal_color, alpha=shared_alpha)
            else:
                color = to_rgba(emphasis, alpha=perturbed_alpha)

            segments.append([(lons[i], lats[i]), (lons[i + 1], lats[i + 1])])
            colors.append(color)

        ax.add_collection(
            LineCollection(
                segments,
                colors=colors,
                linewidths=self._abnormal_width,
                linestyles=self._abnormal_linestyle,
                zorder=zorder,
                capstyle="round",
                joinstyle="round",
            )
        )

    def _plot_real_with_local_fade(
        self,
        ax: Any,
        *,
        lons: List[float],
        lats: List[float],
        seg_road_idx: List[int],
        fade_road_idx: "set[int]",
        faded_alpha: float,
        full_alpha: float,
        zorder: int,
    ) -> None:
        if len(lons) < 2 or len(lats) < 2:
            return

        if len(seg_road_idx) != len(lons) - 1:
            ax.plot(
                lons,
                lats,
                color=self.real_color,
                linewidth=self._real_width,
                linestyle=self._real_linestyle,
                alpha=full_alpha,
                zorder=zorder,
            )
            return

        segments: List[List[Tuple[float, float]]] = []
        colors: List[Tuple[float, float, float, float]] = []

        for i, rid_idx in enumerate(seg_road_idx):
            alpha = full_alpha
            if rid_idx in fade_road_idx:
                alpha = faded_alpha

            segments.append([(lons[i], lats[i]), (lons[i + 1], lats[i + 1])])
            colors.append(to_rgba(self.real_color, alpha=alpha))

        ax.add_collection(
            LineCollection(
                segments,
                colors=colors,
                linewidths=self._real_width,
                linestyles=self._real_linestyle,
                zorder=zorder,
                capstyle="round",
                joinstyle="round",
            )
        )

    def _darken_color(self, color: str, *, factor: float) -> str:
        """Return a darker version of a color.

        factor=1.0 leaves the color unchanged, factor=0.0 returns black.
        """

        r, g, b = to_rgb(color)
        factor = max(0.0, min(1.0, float(factor)))
        return (r * factor, g * factor, b * factor)

    def _plot_road_network(
        self,
        ax: Any,
        *,
        bounds: Tuple[float, float, float, float],
    ) -> None:
        """Plot the road network as a light gray reference underlay."""
        min_lon, max_lon, min_lat, max_lat = bounds

        segments: List[List[Tuple[float, float]]] = []
        for coords, (rmin_lon, rmax_lon, rmin_lat, rmax_lat) in self._road_bboxes:
            # Fast bbox intersection test.
            if (
                rmax_lon < min_lon
                or rmin_lon > max_lon
                or rmax_lat < min_lat
                or rmin_lat > max_lat
            ):
                continue
            for a, b in zip(coords[:-1], coords[1:]):
                segments.append([a, b])

        if not segments:
            return

        # Use a single collection for performance.
        ax.add_collection(
            LineCollection(
                segments,
                colors="#CCCCCC",
                linewidths=0.5,
                alpha=0.6,
                linestyles="-",
                zorder=1,
                capstyle="round",
                joinstyle="round",
            )
        )

    def _calculate_parallel_offset(
        self,
        lons: List[float],
        lats: List[float],
        offset_distance: float,
    ) -> Tuple[List[float], List[float]]:
        """Calculate parallel curve coordinates using vector math."""
        if len(lons) < 2:
            return lons, lats

        import numpy as np

        points = np.column_stack([lons, lats])
        diffs = points[1:] - points[:-1]
        normals = np.column_stack([-diffs[:, 1], diffs[:, 0]])

        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1
        normals = normals / norms[:, None]

        vertex_normals = np.zeros_like(points)
        vertex_normals[0] = normals[0]
        vertex_normals[-1] = normals[-1]
        if len(points) > 2:
            vertex_normals[1:-1] = (normals[:-1] + normals[1:]) / 2

        v_norms = np.linalg.norm(vertex_normals, axis=1)
        v_norms[v_norms == 0] = 1
        vertex_normals = vertex_normals / v_norms[:, None]

        offset_points = points + vertex_normals * float(offset_distance)
        return offset_points[:, 0].tolist(), offset_points[:, 1].tolist()

    def _calculate_dynamic_offset_step(
        self,
        clean_lons: Sequence[float],
        clean_lats: Sequence[float],
        dirty_lons: Sequence[float],
        dirty_lats: Sequence[float],
        *,
        linewidth: float = 4.0,
        overlap_factor: float = 0.99,
    ) -> float:
        """Calculate an offset distance (in degrees) tuned to current zoom."""
        all_lons = list(clean_lons) + list(dirty_lons)
        all_lats = list(clean_lats) + list(dirty_lats)
        if not all_lons:
            return 0.0

        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)

        span_lon = max_lon - min_lon
        span_lat = max_lat - min_lat
        if span_lon == 0:
            span_lon = 1e-6
        if span_lat == 0:
            span_lat = 1e-6

        margin = self.config.margin
        span_lon += 2 * margin
        span_lat += 2 * margin

        fig_w_pts = self.config.figsize[0] * 72
        fig_h_pts = self.config.figsize[1] * 72
        scale_x = fig_w_pts / span_lon
        scale_y = fig_h_pts / span_lat
        scale = min(scale_x, scale_y)
        if scale <= 0:
            return 0.0

        offset_deg = (float(linewidth) * float(overlap_factor)) / scale
        return float(offset_deg)

    def _build_legend(self) -> List[Any]:
        return [
            Line2D(
                [0],
                [0],
                color=self.real_color,
                linewidth=self._real_width,
                linestyle=self._real_linestyle,
                label="Real trajectory",
            ),
            Line2D(
                [0],
                [0],
                color=self._abnormal_color,
                linewidth=self._abnormal_width,
                linestyle=self._abnormal_linestyle,
                label="Abnormal trajectory",
            ),
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
            ),
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
            ),
        ]


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

    parser.add_argument(
        "--real-color",
        type=str,
        default="#1f77b4",
        help="Color for real trajectory (default: matplotlib blue)",
    )
    parser.add_argument(
        "--abnormal-color",
        type=str,
        default="#ff7f0e",
        help="Color for abnormal trajectory (default: matplotlib orange)",
    )
    parser.add_argument(
        "--real-style",
        type=str,
        default="-",
        help="Line style for real trajectory (default: '-')",
    )
    parser.add_argument(
        "--abnormal-style",
        type=str,
        default="-",
        help="Line style for abnormal trajectory (default: '-')",
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
        real_color=str(args.real_color),
        abnormal_color=str(args.abnormal_color),
        real_linestyle=str(args.real_style),
        abnormal_linestyle=str(args.abnormal_style),
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
