#!/usr/bin/env python3
"""
CLI Wrapper for Abnormal Trajectory Analysis

This script provides a command-line interface for running abnormal trajectory detection
on real trajectory data. It loads trajectories from a CSV file, applies detection algorithms,
and saves detailed results to JSON files.

Usage:
    uv run python tools/analyze_abnormal.py \
        --real_file data/BJUT_Beijing/test.csv \
        --dataset BJUT_Beijing \
        --config config/abnormal_detection.yaml \
        --output_dir abnormal/BJUT_Beijing

The script will create:
    - detection_results.json: Full detection results
    - statistics_by_category.json: Summary statistics
    - samples/*.json: Example trajectories for each category (if enabled)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.detect_abnormal_trajectories import (
    AbnormalTrajectoryDetector,
    AbnormalityConfig,
)

# Use functions from evaluation.py for data loading
from evaluation import load_trajectories, load_road_network

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_abnormal_analysis(
    real_file: Path, dataset: str, config_path: Path, output_dir: Path
) -> Dict[str, Any]:
    """Run abnormal trajectory analysis on real data.

    Args:
        real_file: Path to real trajectory CSV file
        dataset: Dataset name (used to locate road network files)
        config_path: Path to abnormal detection configuration YAML
        output_dir: Directory to save results

    Returns:
        Dictionary with detection results and statistics
    """
    logger.info("🔍 Starting abnormal trajectory analysis...")
    logger.info(f"📂 Dataset: {dataset}")
    logger.info(f"📂 Real data: {real_file}")
    logger.info(f"⚙️  Config: {config_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Output directory: {output_dir}")

    # Load configuration
    logger.info("⚙️  Loading configuration...")
    config = AbnormalityConfig.from_yaml(config_path)

    # Determine data directory from dataset name
    data_dir = Path(f"data/{dataset}")
    if not data_dir.exists():
        # Try alternative paths
        data_dir = Path(f"../data/{dataset}")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found for dataset: {dataset}")

    # Load road network
    geo_path = data_dir / "roadmap.geo"
    rel_path = data_dir / "roadmap.rel"

    if not geo_path.exists():
        raise FileNotFoundError(f"Road network geometry not found: {geo_path}")
    if not rel_path.exists():
        raise FileNotFoundError(f"Road network relations not found: {rel_path}")

    logger.info("📂 Loading road network...")
    geo_df = load_road_network(str(geo_path))

    # Load relations separately
    rel_df = pl.read_csv(str(rel_path))

    # Load real trajectories
    if not real_file.exists():
        raise FileNotFoundError(f"Real trajectory file not found: {real_file}")

    logger.info("📂 Loading real trajectories...")
    # Get max road_id from geo_df for validation
    max_road_id = geo_df["road_id"].max()
    trajectories = load_trajectories(
        str(real_file), is_real_data=True, max_road_id=max_road_id
    )

    logger.info(f"✅ Loaded {len(trajectories)} trajectories")

    # Convert trajectories to DataFrame format expected by detector
    # trajectories is list of trajectories, where each trajectory is list of (road_id, timestamp) tuples
    traj_data = []
    for traj_idx, trajectory in enumerate(trajectories):
        for road_id, timestamp in trajectory:
            traj_data.append(
                {"traj_id": traj_idx, "road_id": road_id, "timestamp": timestamp}
            )

    trajectories_df = pl.DataFrame(traj_data)
    logger.info(f"📊 Total trajectory points: {len(trajectories_df):,}")

    # Initialize detector
    logger.info("🔧 Initializing abnormal trajectory detector...")
    detector = AbnormalTrajectoryDetector(config, geo_df, rel_df)

    # Run detection
    logger.info("🔍 Running abnormality detection...")
    results = detector.detect_abnormal_trajectories(trajectories_df)

    # Save detection results
    logger.info("💾 Saving detection results...")

    # Save full detection results
    results_file = output_dir / "detection_results.json"
    with open(results_file, "w") as f:
        # Convert results to JSON-serializable format
        json_results = {
            "dataset": dataset,
            "config_file": str(config_path),
            "total_trajectories": results["total_trajectories"],
            "abnormal_indices": {
                category: list(indices)
                for category, indices in results["abnormal_indices"].items()
            },
            "statistics": results["statistics"],
        }
        json.dump(json_results, f, indent=2)
    logger.info(f"✅ Saved detection results to {results_file}")

    # Save statistics summary
    stats_file = output_dir / "statistics_by_category.json"
    with open(stats_file, "w") as f:
        json.dump(results["statistics"], f, indent=2)
    logger.info(f"✅ Saved statistics to {stats_file}")

    # Save trajectory samples if enabled
    if config.analysis_config.get("save_trajectory_samples", False):
        logger.info("💾 Saving trajectory samples...")
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        max_samples = config.analysis_config.get("max_samples_per_category", 50)

        for category, indices in results["abnormal_indices"].items():
            if not indices:
                continue

            # Get sample of trajectories for this category
            sample_indices = indices[: min(len(indices), max_samples)]

            category_samples = []
            for idx in sample_indices:
                # Find the analysis result for this trajectory
                traj_result = next(
                    (r for r in results["all_results"] if r["traj_idx"] == idx), None
                )
                if traj_result:
                    category_samples.append(traj_result)

            # Save samples for this category
            sample_file = samples_dir / f"{category}_samples.json"
            with open(sample_file, "w") as f:
                json.dump(category_samples, f, indent=2)

            logger.info(
                f"  ✅ Saved {len(category_samples)} {category} samples to {sample_file}"
            )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 ABNORMAL TRAJECTORY DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Total trajectories analyzed: {results['total_trajectories']:,}")
    logger.info("\nAbnormalities detected:")

    for category, stats in results["statistics"].items():
        count = stats["count"]
        percentage = stats["percentage"]
        logger.info(f"  • {category}: {count:,} ({percentage:.2f}%)")

    logger.info("=" * 60)
    logger.info(f"💾 Results saved to: {output_dir}")
    logger.info("✅ Analysis complete!")

    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze trajectories for abnormal patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze BJUT_Beijing test trajectories
  uv run python tools/analyze_abnormal.py \\
      --real_file data/BJUT_Beijing/test.csv \\
      --dataset BJUT_Beijing \\
      --config config/abnormal_detection.yaml \\
      --output_dir abnormal/BJUT_Beijing

  # Analyze with custom output directory
  uv run python tools/analyze_abnormal.py \\
      --real_file data/BJUT_Beijing/val.csv \\
      --dataset BJUT_Beijing \\
      --config config/abnormal_detection.yaml \\
      --output_dir abnormal/BJUT_Beijing_val
        """,
    )

    parser.add_argument(
        "--real_file",
        type=Path,
        required=True,
        help="Path to real trajectory CSV file (test.csv, val.csv, etc.)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., BJUT_Beijing, Beijing)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to abnormal detection configuration YAML file",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save detection results and samples",
    )

    args = parser.parse_args()

    try:
        run_abnormal_analysis(
            real_file=args.real_file,
            dataset=args.dataset,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
