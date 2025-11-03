#!/usr/bin/env python3
"""
Road Network Mapping Tool

Creates GPS-coordinate-based mapping between road networks of different datasets.
Uses haversine distance to find nearest road matches with comprehensive statistics
and validation for research purposes.

Usage:
    uv run python tools/map_road_networks.py \
        --source data/Beijing/roadmap.geo \
        --target data/BJUT_Beijing/roadmap.geo \
        --output road_mapping_beijing_to_bjut.json \
        --max-distance 50
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import polars as pl
import numpy as np
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two GPS coordinates in meters"""
    R = 6371000  # Earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def extract_road_centerpoint(coordinates_str: str) -> Tuple[float, float]:
    """Extract centerpoint (lat, lon) from LineString coordinates

    Args:
        coordinates_str: JSON string like "[[lon1, lat1], [lon2, lat2], ...]"

    Returns:
        (lat, lon) tuple of road centerpoint
    """
    try:
        coords = json.loads(coordinates_str)

        if not coords or len(coords) == 0:
            return None, None

        # Calculate centerpoint as average of all points
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]

        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        return center_lat, center_lon

    except (json.JSONDecodeError, IndexError, TypeError) as e:
        logger.warning(
            f"Failed to parse coordinates: {coordinates_str[:50]}... Error: {e}"
        )
        return None, None


def load_road_network_with_coords(geo_file: Path) -> pl.DataFrame:
    """Load road network and compute centerpoint coordinates

    Args:
        geo_file: Path to .geo file

    Returns:
        DataFrame with columns: geo_id, lat, lon, coordinates
    """
    logger.info(f"📂 Loading road network from {geo_file}")

    # Only read columns we need to avoid parsing errors in other columns
    df = pl.read_csv(geo_file, columns=["geo_id", "coordinates"])
    logger.info(f"  Loaded {len(df)} roads")

    # Extract centerpoints
    logger.info("  📐 Computing road centerpoints...")
    centerpoints = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        if idx % 5000 == 0 and idx > 0:
            logger.info(f"    Processed {idx:,} / {len(df):,} roads...")

        lat, lon = extract_road_centerpoint(row.get("coordinates", ""))
        centerpoints.append(
            {
                "geo_id": row["geo_id"],
                "lat": lat,
                "lon": lon,
                "coordinates": row.get("coordinates", ""),
            }
        )

    coords_df = pl.DataFrame(centerpoints)

    # Filter out invalid coordinates
    valid_df = coords_df.filter(
        pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
    )

    invalid_count = len(coords_df) - len(valid_df)
    if invalid_count > 0:
        logger.warning(f"  ⚠️  Dropped {invalid_count} roads with invalid coordinates")

    logger.info(f"  ✅ {len(valid_df)} roads with valid coordinates")

    return valid_df


def find_nearest_road_batch(
    source_roads: pl.DataFrame, target_roads: pl.DataFrame, max_distance_m: float = 50.0
) -> Dict:
    """Create road ID mapping using KD-tree nearest neighbor search (optimized)

    Uses scipy KD-tree for O(n log m) performance instead of O(n×m) brute force.
    Runtime: ~3 minutes instead of ~12 hours for 40k source × 87k target roads.

    Args:
        source_roads: Source road network DataFrame
        target_roads: Target road network DataFrame
        max_distance_m: Maximum distance threshold for matching

    Returns:
        Dictionary with mapping and statistics
    """
    logger.info("\n🗺️  Creating road network mapping...")
    logger.info(f"  Source roads: {len(source_roads):,}")
    logger.info(f"  Target roads: {len(target_roads):,}")
    logger.info(f"  Max distance: {max_distance_m}m")

    # Convert to numpy for faster computation
    source_coords = source_roads.select(["geo_id", "lat", "lon"]).to_numpy()
    target_coords = target_roads.select(["geo_id", "lat", "lon"]).to_numpy()

    # Build KD-tree spatial index for target roads (O(m log m) - very fast!)
    logger.info("  🏗️  Building spatial index (KD-tree)...")
    tree = cKDTree(target_coords[:, 1:3])  # Only lat/lon columns
    logger.info("  ✅ Spatial index built")

    # Query nearest neighbors for all source roads at once (O(n log m))
    logger.info("  🔍 Querying nearest neighbors...")
    euclidean_dists, nearest_indices = tree.query(
        source_coords[:, 1:3], k=1, workers=-1
    )
    logger.info("  ✅ Query complete")

    # Refine with exact haversine distances and apply threshold
    logger.info("  📏 Computing exact haversine distances...")
    mapping = {}
    distances = []
    unmapped_roads = []
    many_to_one = {}  # target_id -> [source_ids]

    for idx, (source_id, source_lat, source_lon) in enumerate(source_coords):
        if idx % 5000 == 0 and idx > 0:
            logger.info(f"    Processed {idx:,} / {len(source_coords):,} roads...")

        source_id = int(source_id)
        nearest_idx = nearest_indices[idx]
        target_id, target_lat, target_lon = target_coords[nearest_idx]
        target_id = int(target_id)

        # Compute exact haversine distance
        dist_m = haversine_distance(source_lat, source_lon, target_lat, target_lon)

        if dist_m <= max_distance_m:
            mapping[source_id] = target_id
            distances.append(dist_m)

            # Track many-to-one (multiple sources map to same target)
            if target_id not in many_to_one:
                many_to_one[target_id] = []
            many_to_one[target_id].append(source_id)
        else:
            unmapped_roads.append(
                {
                    "road_id": source_id,
                    "nearest_target_id": target_id,
                    "distance_m": float(dist_m),
                    "lat": float(source_lat),
                    "lon": float(source_lon),
                }
            )

    logger.info("  ✅ Mapping complete!")

    # Calculate statistics
    distances_arr = np.array(distances) if distances else np.array([0])

    stats = {
        "total_mapped": len(mapping),
        "total_unmapped": len(unmapped_roads),
        "mapping_rate_pct": (len(mapping) / len(source_coords) * 100)
        if len(source_coords) > 0
        else 0,
        "avg_distance_m": float(np.mean(distances_arr)) if len(distances) > 0 else 0,
        "median_distance_m": float(np.median(distances_arr))
        if len(distances) > 0
        else 0,
        "p95_distance_m": float(np.percentile(distances_arr, 95))
        if len(distances) > 0
        else 0,
        "max_distance_m": float(np.max(distances_arr)) if len(distances) > 0 else 0,
        "min_distance_m": float(np.min(distances_arr)) if len(distances) > 0 else 0,
    }

    # Distance distribution
    bins = [0, 10, 20, 30, 40, 50]
    dist_distribution = {}
    for i in range(len(bins) - 1):
        count = np.sum((distances_arr >= bins[i]) & (distances_arr < bins[i + 1]))
        dist_distribution[f"{bins[i]}-{bins[i + 1]}m"] = int(count)

    # Many-to-one count (multiple sources -> same target)
    many_to_one_cases = {k: v for k, v in many_to_one.items() if len(v) > 1}

    return {
        "mapping": mapping,
        "statistics": stats,
        "distance_distribution": dist_distribution,
        "many_to_one_count": len(many_to_one_cases),
        "many_to_one_max": max([len(v) for v in many_to_one.values()])
        if many_to_one
        else 0,
        "unmapped_roads": unmapped_roads,
    }


def save_comprehensive_output(
    result: Dict,
    source_dataset: str,
    target_dataset: str,
    source_geo: Path,
    target_geo: Path,
    max_distance: float,
    output_file: Path,
):
    """Save mapping and comprehensive statistics

    Args:
        result: Mapping result from find_nearest_road_batch
        source_dataset: Source dataset name
        target_dataset: Target dataset name
        source_geo: Source .geo file path
        target_geo: Target .geo file path
        max_distance: Max distance threshold used
        output_file: Output JSON file path
    """
    # Save main mapping file
    logger.info(f"\n💾 Saving mapping to {output_file}")
    with open(output_file, "w") as f:
        json.dump(result["mapping"], f, indent=2)

    # Save comprehensive stats
    stats_file = output_file.parent / f"{output_file.stem}_stats.json"

    stats_data = {
        "metadata": {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "source_geo_file": str(source_geo),
            "target_geo_file": str(target_geo),
            "timestamp": datetime.now().isoformat(),
            "mapping_method": "nearest_neighbor_haversine",
            "max_distance_threshold_m": max_distance,
        },
        "road_network_info": {
            "source_total_roads": len(result["mapping"])
            + len(result["unmapped_roads"]),
            "target_total_roads": result["statistics"]["total_mapped"]
            + result["statistics"]["total_unmapped"],
        },
        "mapping_quality": result["statistics"],
        "distance_distribution": result["distance_distribution"],
        "one_to_many_count": 0,  # Not applicable with nearest neighbor
        "many_to_one_count": result["many_to_one_count"],
        "many_to_one_max_sources": result["many_to_one_max"],
        "unmapped_roads_sample": result["unmapped_roads"][
            :50
        ],  # First 50 for debugging
        "unmapped_roads_total": len(result["unmapped_roads"]),
    }

    logger.info(f"💾 Saving statistics to {stats_file}")
    with open(stats_file, "w") as f:
        json.dump(stats_data, f, indent=2)

    # Print summary
    stats = result["statistics"]
    logger.info(f"\n{'=' * 70}")
    logger.info("📊 ROAD NETWORK MAPPING SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(
        f"Source: {source_dataset} ({stats['total_mapped'] + stats['total_unmapped']} roads)"
    )
    logger.info(f"Target: {target_dataset}")
    logger.info("\nMapping Quality:")
    logger.info(
        f"  ✅ Mapped:    {stats['total_mapped']:,} ({stats['mapping_rate_pct']:.1f}%)"
    )
    logger.info(f"  ❌ Unmapped:  {stats['total_unmapped']:,}")
    logger.info("\nDistance Metrics:")
    logger.info(f"  Average:  {stats['avg_distance_m']:.1f}m")
    logger.info(f"  Median:   {stats['median_distance_m']:.1f}m")
    logger.info(f"  95th %ile: {stats['p95_distance_m']:.1f}m")
    logger.info(f"  Max:      {stats['max_distance_m']:.1f}m")
    logger.info("\nMapping Patterns:")
    logger.info(
        f"  Many-to-one cases: {result['many_to_one_count']} (max {result['many_to_one_max']} sources)"
    )
    logger.info(f"{'=' * 70}")

    # Quality assessment
    if stats["mapping_rate_pct"] > 85 and stats["avg_distance_m"] < 15:
        logger.info("✅ Mapping Quality: GOOD")
    elif stats["mapping_rate_pct"] > 70 and stats["avg_distance_m"] < 30:
        logger.info("⚠️  Mapping Quality: FAIR")
    else:
        logger.info(
            "❌ Mapping Quality: POOR - Consider adjusting threshold or checking data"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create road network ID mapping between datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create Beijing → BJUT mapping
  uv run python tools/map_road_networks.py \\
    --source data/Beijing/roadmap.geo \\
    --target data/BJUT_Beijing/roadmap.geo \\
    --output road_mapping_beijing_to_bjut.json \\
    --max-distance 50

  # Create reverse mapping (BJUT → Beijing)
  uv run python tools/map_road_networks.py \\
    --source data/BJUT_Beijing/roadmap.geo \\
    --target data/Beijing/roadmap.geo \\
    --output road_mapping_bjut_to_beijing.json \\
    --max-distance 50
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source .geo file",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target .geo file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for mapping",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Maximum distance threshold in meters (default: 50)",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default=None,
        help="Source dataset name (auto-detected from path if not provided)",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        default=None,
        help="Target dataset name (auto-detected from path if not provided)",
    )

    args = parser.parse_args()

    # Validate inputs
    source_geo = Path(args.source)
    target_geo = Path(args.target)

    if not source_geo.exists():
        parser.error(f"Source .geo file not found: {source_geo}")
    if not target_geo.exists():
        parser.error(f"Target .geo file not found: {target_geo}")

    # Auto-detect dataset names from path
    source_dataset = args.source_dataset or source_geo.parent.name
    target_dataset = args.target_dataset or target_geo.parent.name

    logger.info(f"\n{'=' * 70}")
    logger.info(f"🗺️  ROAD NETWORK MAPPING: {source_dataset} → {target_dataset}")
    logger.info(f"{'=' * 70}")

    # Load road networks
    source_roads = load_road_network_with_coords(source_geo)
    target_roads = load_road_network_with_coords(target_geo)

    # Create mapping
    result = find_nearest_road_batch(
        source_roads=source_roads,
        target_roads=target_roads,
        max_distance_m=args.max_distance,
    )

    # Save outputs
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    save_comprehensive_output(
        result=result,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_geo=source_geo,
        target_geo=target_geo,
        max_distance=args.max_distance,
        output_file=output_file,
    )

    logger.info("\n✅ Mapping complete!")
    logger.info(f"  📄 Mapping: {output_file}")
    logger.info(
        f"  📊 Stats: {output_file.parent / (output_file.stem + '_stats.json')}"
    )


if __name__ == "__main__":
    main()
