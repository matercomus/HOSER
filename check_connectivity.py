from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick connectivity stats for a dataset's road network graph "
            "(roadmap.geo + roadmap.rel)."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/Beijing"),
        help=(
            "Dataset directory containing roadmap.geo and roadmap.rel "
            "(default: data/Beijing)."
        ),
    )
    parser.add_argument(
        "--rel",
        type=Path,
        default=None,
        help="Optional override path to roadmap.rel.",
    )
    parser.add_argument(
        "--geo",
        type=Path,
        default=None,
        help="Optional override path to roadmap.geo.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of example road IDs to print (default: 5).",
    )
    return parser.parse_args()


def _load_paths(
    dataset_dir: Path, rel: Path | None, geo: Path | None
) -> tuple[Path, Path]:
    rel_path = rel if rel is not None else dataset_dir / "roadmap.rel"
    geo_path = geo if geo is not None else dataset_dir / "roadmap.geo"
    if not rel_path.exists():
        raise FileNotFoundError(f"Missing roadmap.rel at: {rel_path}")
    if not geo_path.exists():
        raise FileNotFoundError(f"Missing roadmap.geo at: {geo_path}")
    return rel_path, geo_path


def _compute_connectivity(
    rel_df: pd.DataFrame, geo_df: pd.DataFrame
) -> dict[str, object]:
    """Compute basic connectivity statistics for a directed road graph.

    Notes:
        - "isolated" means a road ID present in geo but absent from both
          rel origin and rel destination columns.
        - "dead_end" means a road ID with zero outgoing edges.
          We compute this for (a) all geo roads, and (b) only roads that appear
          somewhere in rel (matching the original script's intent).
    """
    outgoing: defaultdict[int, int] = defaultdict(int)
    incoming: defaultdict[int, int] = defaultdict(int)

    for _, row in rel_df.iterrows():
        outgoing[int(row["origin_id"])] += 1
        incoming[int(row["destination_id"])] += 1

    all_roads = set(int(x) for x in geo_df["geo_id"].tolist())
    connected_roads = set(outgoing.keys()) | set(incoming.keys())

    isolated = all_roads - connected_roads
    dead_ends_connected = [r for r in connected_roads if outgoing[r] == 0]
    dead_ends_all = [r for r in all_roads if outgoing[r] == 0]
    with_outgoing = [r for r, c in outgoing.items() if c > 0]
    with_incoming = [r for r, c in incoming.items() if c > 0]

    return {
        "total_roads": len(all_roads),
        "total_edges": int(len(rel_df)),
        "connected_roads": len(connected_roads),
        "isolated": isolated,
        "dead_ends_connected": dead_ends_connected,
        "dead_ends_all": dead_ends_all,
        "roads_with_outgoing": len(with_outgoing),
        "roads_with_incoming": len(with_incoming),
    }


def main() -> None:
    args = _parse_args()
    rel_path, geo_path = _load_paths(args.dataset_dir, args.rel, args.geo)

    rel_df = pd.read_csv(rel_path)
    geo_df = pd.read_csv(geo_path)
    stats = _compute_connectivity(rel_df, geo_df)

    print(f"Dataset: {args.dataset_dir}")
    print(f"rel: {rel_path}")
    print(f"geo: {geo_path}")
    print(f"Total roads: {stats['total_roads']}")
    print(f"Total edges (rel rows): {stats['total_edges']}")
    print(f"Roads appearing in rel: {stats['connected_roads']}")
    print(f"Isolated roads (no rel in/out): {len(stats['isolated'])}")
    print(
        "Dead-end roads (0 outgoing): "
        f"{len(stats['dead_ends_connected'])} (among rel-connected roads), "
        f"{len(stats['dead_ends_all'])} (among all geo roads)"
    )
    print(f"Roads with outgoing: {stats['roads_with_outgoing']}")
    print(f"Roads with incoming: {stats['roads_with_incoming']}")

    if args.examples > 0:
        isolated = list(stats["isolated"])[: args.examples]
        dead_ends = list(stats["dead_ends_all"])[: args.examples]
        if isolated:
            print(f"\nExample isolated roads: {isolated}")
        if dead_ends:
            print(f"Example dead-end roads: {dead_ends}")


if __name__ == "__main__":
    main()
