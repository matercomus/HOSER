#!/usr/bin/env python3
"""Non-linear coefficient (route directness) utilities.

This implements a simple route directness metric often used in transportation:

  non_linear_coefficient = (trajectory length) / (shortest-path length)

- Ideal is 1.0 (perfectly direct).
- Values >1 indicate detours / non-direct routing.

The supervisor note suggests comparing trajectory distance against the
shortest path via the road network.

This tool works with HOSER dataset folders containing:
- roadmap.geo (road geometries)
- roadmap.rel (road adjacency)

It can be used as an additional abnormality indicator alongside LM-TAD.
"""

from __future__ import annotations

import argparse
import ast
import csv
import heapq
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance (meters) between two lat/lon points."""

    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(r * c)


def _polyline_length_m(coords_lonlat: List[Tuple[float, float]]) -> float:
    """Return polyline length in meters for a list of (lon, lat) points."""
    if len(coords_lonlat) < 2:
        return 0.0
    dist = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords_lonlat[:-1], coords_lonlat[1:]):
        dist += _haversine_m(lat1, lon1, lat2, lon2)
    return float(dist)


def load_road_lengths_m(roadmap_geo: Path) -> Dict[int, float]:
    """Load per-road polyline lengths from a LibCity-style roadmap.geo."""

    if not roadmap_geo.exists():
        raise FileNotFoundError(f"roadmap.geo not found: {roadmap_geo}")

    import pandas as pd

    df = pd.read_csv(roadmap_geo)
    id_col = "road_id" if "road_id" in df.columns else "geo_id"
    if id_col not in df.columns or "coordinates" not in df.columns:
        raise ValueError(
            f"Unexpected roadmap.geo schema; need {id_col} and coordinates: {roadmap_geo}"
        )

    lengths: Dict[int, float] = {}
    for _, row in df.iterrows():
        rid = int(row[id_col])
        raw = row["coordinates"]
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue

        text = str(raw)
        try:
            coords = json.loads(text)
        except Exception:
            coords = ast.literal_eval(text)

        coords_lonlat: List[Tuple[float, float]] = []
        for pair in coords:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            lon, lat = pair
            coords_lonlat.append((float(lon), float(lat)))

        lengths[rid] = _polyline_length_m(coords_lonlat)

    if not lengths:
        raise RuntimeError(f"No road lengths parsed from: {roadmap_geo}")

    return lengths


def load_outgoing_edges(roadmap_rel: Path) -> Dict[int, List[int]]:
    """Load outgoing adjacency lists from a LibCity-style roadmap.rel."""
    if not roadmap_rel.exists():
        raise FileNotFoundError(f"roadmap.rel not found: {roadmap_rel}")

    import pandas as pd

    df = pd.read_csv(roadmap_rel)
    if "origin_id" not in df.columns or "destination_id" not in df.columns:
        raise ValueError(f"Unexpected roadmap.rel schema: {roadmap_rel}")

    outgoing: Dict[int, List[int]] = {}
    for o, d in zip(df["origin_id"].to_list(), df["destination_id"].to_list()):
        o_i = int(o)
        d_i = int(d)
        outgoing.setdefault(o_i, []).append(d_i)

    return outgoing


def shortest_path_length_m(
    *,
    start_road: int,
    end_road: int,
    outgoing: Dict[int, List[int]],
    road_len_m: Dict[int, float],
) -> Optional[float]:
    """Dijkstra on the road-transition graph.

    Nodes are road-ids. Edge (u -> v) cost is length(v), i.e. "cost to enter v".
    Total path length includes length(start) + sum(length(v) for subsequent v).
    """

    if start_road not in road_len_m or end_road not in road_len_m:
        return None

    if start_road == end_road:
        return float(road_len_m[start_road])

    dist: Dict[int, float] = {start_road: float(road_len_m[start_road])}
    heap: List[Tuple[float, int]] = [(dist[start_road], start_road)]

    while heap:
        cur, u = heapq.heappop(heap)
        if u == end_road:
            return float(cur)
        if cur != dist.get(u):
            continue

        for v in outgoing.get(u, []):
            w = float(road_len_m.get(v, 0.0))
            nxt = cur + w
            if nxt < dist.get(v, float("inf")):
                dist[v] = nxt
                heapq.heappush(heap, (nxt, v))

    return None


def trajectory_length_m(road_ids: List[int], road_len_m: Dict[int, float]) -> float:
    """Return trajectory length as sum of per-road polyline lengths."""
    return float(sum(float(road_len_m.get(int(r), 0.0)) for r in road_ids))


def non_linear_coefficient(
    road_ids: List[int],
    *,
    outgoing: Dict[int, List[int]],
    road_len_m: Dict[int, float],
) -> Optional[float]:
    """Compute trajectory_length / shortest_path_length for a road-id sequence."""
    if len(road_ids) < 2:
        return None

    start = int(road_ids[0])
    end = int(road_ids[-1])

    sp = shortest_path_length_m(
        start_road=start, end_road=end, outgoing=outgoing, road_len_m=road_len_m
    )
    if sp is None or sp <= 0:
        return None

    traj_len = trajectory_length_m(road_ids, road_len_m)
    if traj_len <= 0:
        return None

    return float(traj_len / sp)


def _parse_rid_list(text: str) -> List[int]:
    """Parse a HOSER `rid_list` cell into a list of road IDs.

    Supported formats:
    - Python literals: "[1, 2, 3]" or "[(1, 0), (2, 1)]"
    - Comma-separated: "1,2,3"
    """
    s = str(text).strip()
    if not s:
        return []
    if s.startswith("["):
        parsed = ast.literal_eval(s)
        if not parsed:
            return []
        first = parsed[0]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            return [int(x[0]) for x in parsed]
        return [int(x) for x in parsed]

    return [int(p.strip()) for p in s.split(",") if p.strip()]


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute non-linear coefficient over a dataset split"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-rows", type=int, default=20000)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    split_csv = dataset_dir / f"{args.split}.csv"

    road_len_m = load_road_lengths_m(dataset_dir / "roadmap.geo")
    outgoing = load_outgoing_edges(dataset_dir / "roadmap.rel")

    normals: List[float] = []
    abnormals: List[float] = []

    with split_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "rid_list" not in r.fieldnames:
            raise ValueError(f"Missing rid_list column: {split_csv}")

        for i, row in enumerate(r):
            if args.max_rows is not None and i >= int(args.max_rows):
                break
            rid_list = _parse_rid_list(row.get("rid_list") or "")
            c = non_linear_coefficient(rid_list, outgoing=outgoing, road_len_m=road_len_m)
            if c is None:
                continue

            ab_info = (row.get("abnormality_info") or "").strip().lower()
            if ab_info and ab_info != "normal":
                abnormals.append(c)
            else:
                normals.append(c)

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    print(
        f"{dataset_dir.name}/{args.split}: normals n={len(normals)} mean={_mean(normals):.3f} | "
        f"abnormals n={len(abnormals)} mean={_mean(abnormals):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
