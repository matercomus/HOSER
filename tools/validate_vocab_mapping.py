#!/usr/bin/env python3
"""Validate vocabulary mapping between HOSER and LM-TAD.

This script analyzes the mapping quality from HOSER road IDs to LM-TAD grid tokens
to ensure semantic equivalence is preserved during knowledge distillation.

Validation metrics:
- Mapping coverage: % of roads successfully mapped to grid tokens
- Unmapped tokens: Roads that fail to map (should be 0 with current implementation)
- Token frequency distribution: How many roads map to each grid cell
- Grid utilization: % of grid cells that receive at least one road

Usage:
    uv run python tools/validate_vocab_mapping.py --config config/Beijing.yaml
    uv run python tools/validate_vocab_mapping.py --config config/Porto.yaml --output validation_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl
import yaml

# Add project root to path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from critics.grid_mapper import GridConfig, GridMapper  # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_road_to_token_mapping(
    config: dict, data_dir: Path = None
) -> Tuple[np.ndarray, GridMapper]:
    """Build the road ID → grid token mapping.

    Returns:
        Tuple of (road_to_token array, GridMapper instance)
    """
    if data_dir is None:
        data_dir = Path(config.get("data_dir", ROOT / "data/Beijing"))

    # Load road network geometry file
    geo_path = data_dir / "roadmap.geo"
    if not geo_path.exists():
        raise FileNotFoundError(f"roadmap.geo not found at {geo_path}")

    # Load with Polars - only need coordinates column
    # Use infer_schema_length=0 to treat all columns as strings, we'll parse what we need
    geo = pl.read_csv(geo_path, infer_schema_length=0)

    # Parse coordinates column and compute centroids
    # Coordinates format: "[[lng1, lat1], [lng2, lat2], ...]"
    def compute_centroid(coords_str: str) -> Tuple[float, float]:
        """Parse coordinate string and compute centroid (lat, lng)."""
        coords = eval(coords_str)  # Parse string representation of list
        lats = [pt[1] for pt in coords]
        lngs = [pt[0] for pt in coords]
        return np.mean(lats), np.mean(lngs)

    # Compute centroids for all roads
    centroids = [compute_centroid(coords) for coords in geo["coordinates"].to_list()]
    road_centroids_lat = np.array([c[0] for c in centroids], dtype=np.float64)
    road_centroids_lng = np.array([c[1] for c in centroids], dtype=np.float64)

    # Get grid configuration from config file
    grid_size = float(config.get("distill", {}).get("grid_size", 0.001))
    downsample = int(config.get("distill", {}).get("downsample_factor", 1))

    # Compute boundaries from actual data
    coords_series = geo["coordinates"].to_list()
    all_coords = [eval(coords) for coords in coords_series]
    min_lng = min(min(pt[0] for pt in coords) for coords in all_coords)
    max_lng = max(max(pt[0] for pt in coords) for coords in all_coords)
    min_lat = min(min(pt[1] for pt in coords) for coords in all_coords)
    max_lat = max(max(pt[1] for pt in coords) for coords in all_coords)

    # Build grid config
    grid_cfg = GridConfig(
        min_lat=float(min_lat),
        max_lat=float(max_lat),
        min_lng=float(min_lng),
        max_lng=float(max_lng),
        grid_size=grid_size,
        downsample_factor=downsample,
    )

    # Create mapper and compute mapping
    mapper = GridMapper(
        grid_cfg, np.stack([road_centroids_lat, road_centroids_lng], axis=1)
    )
    road_to_token = mapper.map_all()

    return road_to_token.astype(np.int64), mapper


def validate_mapping(
    road_to_token: np.ndarray, mapper: GridMapper, dataset_name: str
) -> Dict:
    """Compute validation metrics for the vocabulary mapping.

    Returns:
        Dictionary containing validation metrics and analysis
    """
    num_roads = len(road_to_token)
    grid_h, grid_w = mapper.grid_h, mapper.grid_w
    total_grid_cells = grid_h * grid_w

    # Basic coverage metrics
    unique_tokens = np.unique(road_to_token)
    num_unique_tokens = len(unique_tokens)

    # Check for invalid tokens (should not happen with current implementation)
    max_valid_token = total_grid_cells - 1
    invalid_tokens = road_to_token[
        (road_to_token < 0) | (road_to_token > max_valid_token)
    ]
    num_invalid = len(invalid_tokens)

    # Token frequency distribution
    token_counts = np.bincount(road_to_token, minlength=total_grid_cells)
    occupied_cells = np.sum(token_counts > 0)

    # Statistical analysis
    roads_per_cell_mean = (
        token_counts[token_counts > 0].mean() if occupied_cells > 0 else 0
    )
    roads_per_cell_std = (
        token_counts[token_counts > 0].std() if occupied_cells > 0 else 0
    )
    roads_per_cell_max = token_counts.max()
    roads_per_cell_min = (
        token_counts[token_counts > 0].min() if occupied_cells > 0 else 0
    )

    # Find cells with high road density (potential bottlenecks)
    high_density_threshold = roads_per_cell_mean + 2 * roads_per_cell_std
    high_density_cells = np.where(token_counts > high_density_threshold)[0]

    # Find empty cells (unused grid positions)
    empty_cells = np.where(token_counts == 0)[0]

    # Compute coverage percentages
    mapping_coverage = 100.0 * (num_roads - num_invalid) / num_roads
    grid_utilization = 100.0 * occupied_cells / total_grid_cells

    results = {
        "dataset": dataset_name,
        "num_roads": int(num_roads),
        "grid_dimensions": {"height": int(grid_h), "width": int(grid_w)},
        "total_grid_cells": int(total_grid_cells),
        "mapping_coverage": {
            "valid_mappings": int(num_roads - num_invalid),
            "invalid_mappings": int(num_invalid),
            "coverage_percent": float(mapping_coverage),
        },
        "grid_utilization": {
            "occupied_cells": int(occupied_cells),
            "empty_cells": int(len(empty_cells)),
            "utilization_percent": float(grid_utilization),
        },
        "token_distribution": {
            "unique_tokens_used": int(num_unique_tokens),
            "roads_per_cell_mean": float(roads_per_cell_mean),
            "roads_per_cell_std": float(roads_per_cell_std),
            "roads_per_cell_min": int(roads_per_cell_min),
            "roads_per_cell_max": int(roads_per_cell_max),
        },
        "high_density_cells": {
            "count": int(len(high_density_cells)),
            "threshold": float(high_density_threshold),
            "cell_ids": high_density_cells[:10].tolist(),  # Top 10
            "roads_in_densest": int(roads_per_cell_max),
        },
        "validation_status": {
            "passed": num_invalid == 0 and mapping_coverage >= 99.0,
            "warnings": [],
        },
    }

    # Add warnings
    if num_invalid > 0:
        results["validation_status"]["warnings"].append(
            f"Found {num_invalid} invalid token mappings"
        )
    if mapping_coverage < 99.0:
        results["validation_status"]["warnings"].append(
            f"Mapping coverage {mapping_coverage:.2f}% is below 99%"
        )
    if grid_utilization < 50.0:
        results["validation_status"]["warnings"].append(
            f"Grid utilization {grid_utilization:.2f}% is low (many empty cells)"
        )
    if roads_per_cell_max > 100:
        results["validation_status"]["warnings"].append(
            f"Densest cell has {roads_per_cell_max} roads (potential information loss)"
        )

    return results


def print_summary(results: Dict) -> None:
    """Print human-readable summary of validation results."""
    print(f"\n{'=' * 70}")
    print(f"Vocabulary Mapping Validation: {results['dataset']}")
    print(f"{'=' * 70}\n")

    print("📊 Dataset Statistics:")
    print(f"  Roads in network: {results['num_roads']:,}")
    print(
        f"  Grid dimensions: {results['grid_dimensions']['height']} × {results['grid_dimensions']['width']}"
    )
    print(f"  Total grid cells: {results['total_grid_cells']:,}\n")

    print("✅ Mapping Coverage:")
    cov = results["mapping_coverage"]
    print(f"  Valid mappings: {cov['valid_mappings']:,} / {results['num_roads']:,}")
    print(f"  Invalid mappings: {cov['invalid_mappings']}")
    print(f"  Coverage: {cov['coverage_percent']:.2f}%\n")

    print("🎯 Grid Utilization:")
    util = results["grid_utilization"]
    print(
        f"  Occupied cells: {util['occupied_cells']:,} / {results['total_grid_cells']:,}"
    )
    print(f"  Empty cells: {util['empty_cells']:,}")
    print(f"  Utilization: {util['utilization_percent']:.2f}%\n")

    print("📈 Token Distribution:")
    dist = results["token_distribution"]
    print(f"  Unique tokens used: {dist['unique_tokens_used']:,}")
    print(
        f"  Roads per cell: {dist['roads_per_cell_mean']:.1f} ± {dist['roads_per_cell_std']:.1f}"
    )
    print(f"  Range: [{dist['roads_per_cell_min']}, {dist['roads_per_cell_max']}]\n")

    print("⚠️  High Density Cells:")
    hd = results["high_density_cells"]
    print(f"  Cells above threshold: {hd['count']}")
    print(f"  Density threshold: {hd['threshold']:.1f} roads/cell")
    print(f"  Densest cell: {hd['roads_in_densest']} roads\n")

    status = results["validation_status"]
    if status["passed"]:
        print("✅ Validation PASSED")
    else:
        print("❌ Validation FAILED")

    if status["warnings"]:
        print("\nWarnings:")
        for warning in status["warnings"]:
            print(f"  ⚠️  {warning}")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate vocabulary mapping for HOSER→LM-TAD distillation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dataset config YAML (e.g., config/Beijing.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save JSON results",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dataset_name = args.config.stem  # Beijing, Porto, etc.

    # Build mapping
    print(f"Building vocabulary mapping for {dataset_name}...")
    road_to_token, mapper = build_road_to_token_mapping(config)

    # Validate
    print("Computing validation metrics...")
    results = validate_mapping(road_to_token, mapper, dataset_name)

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
