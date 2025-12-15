#!/usr/bin/env python3
"""Compare the two HOSER→LM-TAD conversion methods.

This tool compares the *mapping inputs* that drive LM-TAD teacher scoring:

- Method A ("evaluate_dataset_with_lmtad"):
  - Road centroid: Shapely LineString centroid
  - Boundary: min/max over centroids

- Method B ("convert_to_lmtad_format"):
  - Road centroid: mean of all coordinate points
  - Boundary: min/max over *all* coordinate points

Because LM-TAD tokens are computed as:

    gi = floor((lat - min_lat) / grid_size)
    gj = floor((lng - min_lng) / grid_size)
    token = gi * grid_w + gj

small differences in boundary or centroid calculation can shift (gi, gj) and
therefore token IDs, destroying alignment with the teacher's learned token
semantics.

Outputs
-------
Writes the following into `--out-dir`:
- summary.json: machine-readable summary
- summary.md: human-readable summary
- per_road.csv: per-road centroid/token comparison (for common valid roads)

Example
-------
uv run python tools/compare_lmtad_conversion_methods.py \
  --roadmap-file data/Beijing/roadmap.geo \
  --out-dir tools_eval_lmtad/_conversion_compare/Beijing \
  --grid-size 0.001 \
  --max-roads 50000
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path so sibling packages (e.g. `critics`)
# are importable when this script is run from `tools/` or other working dirs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from critics.grid_mapper import GridConfig, GridMapper  # noqa: E402

try:
    from shapely.geometry import LineString
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: shapely. Install it in your env (uv/pyproject) "
        "or run this tool in the same environment used for LM-TAD evaluation."
    ) from exc


@dataclass(frozen=True)
class Boundary:
    """Geographic boundary in lat/lng."""

    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float


@dataclass(frozen=True)
class MethodResult:
    """Derived conversion metadata for one method."""

    name: str
    grid_size: float
    downsample_factor: int
    boundary: Boundary
    grid_h: int
    grid_w: int


@dataclass(frozen=True)
class ComparisonSummary:
    """Top-level summary for the conversion comparison."""

    roadmap_file: str
    num_roads_total: int
    num_roads_valid_common: int
    grid_size: float
    downsample_factor: int
    method_a: MethodResult
    method_b: MethodResult
    token_mismatch_count: int
    token_mismatch_rate: float
    centroid_mean_abs_delta_lat: float
    centroid_mean_abs_delta_lng: float
    centroid_max_abs_delta_lat: float
    centroid_max_abs_delta_lng: float


def _safe_parse_coords(value: Any) -> Optional[List[Tuple[float, float]]]:
    """Parse a roadmap `coordinates` field into a list of (lng, lat) tuples.

    Supports JSON strings and Python-literal strings.

    Returns None if parsing fails or the structure is invalid.
    """
    if value is None:
        return None

    if isinstance(value, list):
        obj = value
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                return None

    if not isinstance(obj, list) or not obj:
        return None

    out: List[Tuple[float, float]] = []
    for item in obj:
        if (
            not isinstance(item, (list, tuple))
            or len(item) < 2
            or not isinstance(item[0], (int, float))
            or not isinstance(item[1], (int, float))
        ):
            return None
        lng = float(item[0])
        lat = float(item[1])
        if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
            return None
        out.append((lng, lat))

    return out


def _centroid_method_a(coords: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid used by `simple_evaluate_with_lmtad.load_road_centroids`.

    Returns
    -------
    (lat, lng)
    """
    line = LineString(coords)
    c = line.centroid
    return float(c.y), float(c.x)


def _centroid_method_b(coords: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid used by `tools/convert_to_lmtad_format.extract_road_centroids`.

    Returns
    -------
    (lat, lng)
    """
    lats = [lat for _, lat in coords]
    lngs = [lng for lng, _ in coords]
    return float(sum(lats) / len(lats)), float(sum(lngs) / len(lngs))


def _boundary_from_centroids(centroids_latlng: np.ndarray) -> Boundary:
    """Compute min/max boundary from centroids (lat,lng)."""
    return Boundary(
        min_lat=float(np.min(centroids_latlng[:, 0])),
        max_lat=float(np.max(centroids_latlng[:, 0])),
        min_lng=float(np.min(centroids_latlng[:, 1])),
        max_lng=float(np.max(centroids_latlng[:, 1])),
    )


def _boundary_from_all_points(
    all_points_lnglat: Iterable[Tuple[float, float]],
) -> Boundary:
    """Compute min/max boundary from all polyline points."""
    min_lat = float("inf")
    max_lat = float("-inf")
    min_lng = float("inf")
    max_lng = float("-inf")

    seen = 0
    for lng, lat in all_points_lnglat:
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lng = min(min_lng, lng)
        max_lng = max(max_lng, lng)
        seen += 1

    if seen == 0:
        raise ValueError("No valid points to build boundary")

    if not (min_lat < max_lat and min_lng < max_lng):
        raise ValueError(
            f"Invalid boundary computed: lat=({min_lat},{max_lat}) lng=({min_lng},{max_lng})"
        )

    return Boundary(min_lat=min_lat, max_lat=max_lat, min_lng=min_lng, max_lng=max_lng)


def _grid_dims(
    boundary: Boundary, grid_size: float, downsample_factor: int
) -> Tuple[int, int]:
    """Compute (grid_h, grid_w) as in `critics.grid_mapper.GridMapper`."""
    lat_span = max(0.0, float(boundary.max_lat - boundary.min_lat))
    lng_span = max(0.0, float(boundary.max_lng - boundary.min_lng))

    grid_h = int(lat_span / grid_size) + 1
    grid_w = int(lng_span / grid_size) + 1

    if downsample_factor > 1:
        grid_h //= downsample_factor
        grid_w //= downsample_factor
        grid_h = max(grid_h, 1)
        grid_w = max(grid_w, 1)

    return grid_h, grid_w


def _compute_tokens(
    centroids_latlng: np.ndarray,
    boundary: Boundary,
    grid_size: float,
    downsample_factor: int,
) -> np.ndarray:
    """Compute road->token using `GridMapper` to match evaluation code."""
    cfg = GridConfig(
        min_lat=boundary.min_lat,
        max_lat=boundary.max_lat,
        min_lng=boundary.min_lng,
        max_lng=boundary.max_lng,
        grid_size=grid_size,
        downsample_factor=downsample_factor,
    )
    mapper = GridMapper(boundary=cfg, road_centroids=centroids_latlng, verify_hw=None)
    return mapper.map_all()


def compare_conversion_methods(
    roadmap_file: Path,
    out_dir: Path,
    grid_size: float = 0.001,
    downsample_factor: int = 1,
    max_roads: Optional[int] = None,
) -> ComparisonSummary:
    """Run the comparison and write results to disk."""
    if not roadmap_file.exists():
        raise FileNotFoundError(f"Roadmap file not found: {roadmap_file}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(roadmap_file)
    if "coordinates" not in df.columns:
        raise ValueError(
            f"Expected 'coordinates' column in {roadmap_file}, got {list(df.columns)}"
        )

    if max_roads is not None:
        df = df.head(int(max_roads))

    coords_by_idx: Dict[int, List[Tuple[float, float]]] = {}
    invalid_rows: List[int] = []

    for i, raw in enumerate(df["coordinates"].tolist()):
        coords = _safe_parse_coords(raw)
        if coords is None:
            invalid_rows.append(i)
            continue
        coords_by_idx[i] = coords

    if not coords_by_idx:
        raise ValueError(
            "No valid roads found in roadmap (failed to parse coordinates)"
        )

    common_idx = sorted(coords_by_idx.keys())

    # Compute centroids for the common valid roads
    cent_a: List[Tuple[float, float]] = []
    cent_b: List[Tuple[float, float]] = []
    points_all: List[Tuple[float, float]] = []

    for i in common_idx:
        coords = coords_by_idx[i]
        cent_a.append(_centroid_method_a(coords))
        cent_b.append(_centroid_method_b(coords))
        points_all.extend(coords)

    cent_a_arr = np.asarray(cent_a, dtype=np.float64)
    cent_b_arr = np.asarray(cent_b, dtype=np.float64)

    boundary_a = _boundary_from_centroids(cent_a_arr)
    boundary_b = _boundary_from_all_points(points_all)

    grid_h_a, grid_w_a = _grid_dims(
        boundary_a, grid_size=grid_size, downsample_factor=downsample_factor
    )
    grid_h_b, grid_w_b = _grid_dims(
        boundary_b, grid_size=grid_size, downsample_factor=downsample_factor
    )

    tokens_a = _compute_tokens(
        cent_a_arr, boundary_a, grid_size=grid_size, downsample_factor=downsample_factor
    )
    tokens_b = _compute_tokens(
        cent_b_arr, boundary_b, grid_size=grid_size, downsample_factor=downsample_factor
    )

    token_equal = tokens_a == tokens_b
    mismatch_count = int(np.sum(~token_equal))
    mismatch_rate = float(mismatch_count / max(1, len(token_equal)))

    dlat = np.abs(cent_a_arr[:, 0] - cent_b_arr[:, 0])
    dlng = np.abs(cent_a_arr[:, 1] - cent_b_arr[:, 1])

    summary = ComparisonSummary(
        roadmap_file=str(roadmap_file),
        num_roads_total=int(len(df)),
        num_roads_valid_common=int(len(common_idx)),
        grid_size=float(grid_size),
        downsample_factor=int(downsample_factor),
        method_a=MethodResult(
            name="evaluate_dataset_with_lmtad",
            grid_size=float(grid_size),
            downsample_factor=int(downsample_factor),
            boundary=boundary_a,
            grid_h=int(grid_h_a),
            grid_w=int(grid_w_a),
        ),
        method_b=MethodResult(
            name="convert_to_lmtad_format",
            grid_size=float(grid_size),
            downsample_factor=int(downsample_factor),
            boundary=boundary_b,
            grid_h=int(grid_h_b),
            grid_w=int(grid_w_b),
        ),
        token_mismatch_count=mismatch_count,
        token_mismatch_rate=mismatch_rate,
        centroid_mean_abs_delta_lat=float(np.mean(dlat)),
        centroid_mean_abs_delta_lng=float(np.mean(dlng)),
        centroid_max_abs_delta_lat=float(np.max(dlat)),
        centroid_max_abs_delta_lng=float(np.max(dlng)),
    )

    _write_outputs(
        out_dir=out_dir,
        summary=summary,
        road_idx=common_idx,
        cent_a=cent_a_arr,
        cent_b=cent_b_arr,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        invalid_rows=invalid_rows,
    )

    return summary


def _write_outputs(
    out_dir: Path,
    summary: ComparisonSummary,
    road_idx: Sequence[int],
    cent_a: np.ndarray,
    cent_b: np.ndarray,
    tokens_a: np.ndarray,
    tokens_b: np.ndarray,
    invalid_rows: Sequence[int],
) -> None:
    """Write `summary.json`, `summary.md`, and `per_road.csv`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    (out_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True)
    )

    # per_road.csv
    per_road_path = out_dir / "per_road.csv"
    with per_road_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "road_idx",
                "lat_a",
                "lng_a",
                "lat_b",
                "lng_b",
                "abs_dlat",
                "abs_dlng",
                "token_a",
                "token_b",
                "token_equal",
            ]
        )
        for i, (ra, rb, ta, tb) in enumerate(zip(cent_a, cent_b, tokens_a, tokens_b)):
            writer.writerow(
                [
                    int(road_idx[i]),
                    float(ra[0]),
                    float(ra[1]),
                    float(rb[0]),
                    float(rb[1]),
                    float(abs(ra[0] - rb[0])),
                    float(abs(ra[1] - rb[1])),
                    int(ta),
                    int(tb),
                    bool(int(ta) == int(tb)),
                ]
            )

    # invalid_rows.txt
    if invalid_rows:
        (out_dir / "invalid_rows.txt").write_text(
            "\n".join(str(int(i)) for i in invalid_rows) + "\n"
        )

    # summary.md
    md_lines: List[str] = []
    md_lines.append("# LM-TAD conversion method comparison")
    md_lines.append("")
    md_lines.append(f"Roadmap: `{summary.roadmap_file}`")
    md_lines.append(f"Total roads (limited): {summary.num_roads_total}")
    md_lines.append(f"Valid roads compared: {summary.num_roads_valid_common}")
    md_lines.append("")

    def fmt_b(b: Boundary) -> str:
        return f"lat=[{b.min_lat:.6f}, {b.max_lat:.6f}] lng=[{b.min_lng:.6f}, {b.max_lng:.6f}]"

    md_lines.append("## Method A: evaluate_dataset_with_lmtad")
    md_lines.append(f"- Boundary: {fmt_b(summary.method_a.boundary)}")
    md_lines.append(f"- Grid: {summary.method_a.grid_h} x {summary.method_a.grid_w}")
    md_lines.append("")

    md_lines.append("## Method B: convert_to_lmtad_format")
    md_lines.append(f"- Boundary: {fmt_b(summary.method_b.boundary)}")
    md_lines.append(f"- Grid: {summary.method_b.grid_h} x {summary.method_b.grid_w}")
    md_lines.append("")

    md_lines.append("## Differences")
    md_lines.append(
        f"- Token mismatches: {summary.token_mismatch_count} / {summary.num_roads_valid_common} ({summary.token_mismatch_rate:.2%})"
    )
    md_lines.append(
        "- Centroid |Δlat| mean/max: "
        f"{summary.centroid_mean_abs_delta_lat:.6e} / {summary.centroid_max_abs_delta_lat:.6e}"
    )
    md_lines.append(
        "- Centroid |Δlng| mean/max: "
        f"{summary.centroid_mean_abs_delta_lng:.6e} / {summary.centroid_max_abs_delta_lng:.6e}"
    )
    md_lines.append("")
    md_lines.append("Artifacts:")
    md_lines.append("- `summary.json`")
    md_lines.append("- `per_road.csv`")
    if invalid_rows:
        md_lines.append("- `invalid_rows.txt`")

    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compare HOSER→LM-TAD conversion methods (boundary/centroid/token)",
    )
    parser.add_argument(
        "--roadmap-file",
        type=Path,
        required=True,
        help="Path to roadmap.geo (CSV with a 'coordinates' column)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write outputs (summary.json, summary.md, per_road.csv)",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=0.001,
        help="Grid size used for tokenization (default: 0.001)",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Downsample factor (default: 1)",
    )
    parser.add_argument(
        "--max-roads",
        type=int,
        default=None,
        help="Optional limit on number of roads processed (debug/speed)",
    )

    args = parser.parse_args()

    if args.grid_size <= 0:
        raise SystemExit("--grid-size must be positive")
    if args.downsample_factor <= 0:
        raise SystemExit("--downsample-factor must be positive")

    summary = compare_conversion_methods(
        roadmap_file=args.roadmap_file,
        out_dir=args.out_dir,
        grid_size=args.grid_size,
        downsample_factor=args.downsample_factor,
        max_roads=args.max_roads,
    )

    print(f"Wrote outputs to: {args.out_dir}")
    print(f"Token mismatch rate: {summary.token_mismatch_rate:.2%}")
    print(
        "Grid dims: "
        f"A={summary.method_a.grid_h}x{summary.method_a.grid_w} "
        f"B={summary.method_b.grid_h}x{summary.method_b.grid_w}"
    )


if __name__ == "__main__":
    main()
