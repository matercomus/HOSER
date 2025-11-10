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
from typing import Dict, Any, List, Optional
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.detect_abnormal_trajectories import (
    AbnormalTrajectoryDetector,
    AbnormalityConfig,
)

# Wang et al. 2018 statistical detector
from tools.detect_abnormal_statistical import (
    BaselineStatistics,
    WangConfig,
    WangStatisticalDetector,
)

# Translation quality filtering
from tools.filter_translated_by_quality import (
    load_quality_report,
    filter_trajectories_by_quality,
)

# Use functions from evaluation.py for data loading
from evaluation import load_trajectories, load_road_network

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ensure_json_serializable(obj):
    """Recursively convert object to JSON-serializable types

    Handles numpy types, booleans, and nested structures.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, str):
        return obj
    elif obj is None:
        return None
    else:
        # Last resort: convert to string
        return str(obj)


def _get_detection_method(config_path: Path) -> str:
    """Extract detection method from config file.

    Args:
        config_path: Path to configuration YAML

    Returns:
        Detection method: "threshold", "wang_statistical", or "both"
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    detection = config.get("detection", {})
    if isinstance(detection, dict):
        return detection.get("method", "z_score")  # Default to threshold method
    return "z_score"  # Fallback


def _load_baselines_for_dataset(
    dataset: str, config_path: Optional[Path] = None
) -> Optional[BaselineStatistics]:
    """Load baseline statistics for a dataset.

    Args:
        dataset: Dataset name (e.g., "Beijing", "BJUT_Beijing")
        config_path: Optional path to config file to read baselines path from

    Returns:
        BaselineStatistics object or None if not found
    """
    import yaml

    # Try to get baselines path from config if provided
    baselines_path = None
    baseline_dataset = dataset

    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            baseline_dataset = config.get("baseline_dataset", dataset)
            baselines_config = config.get("baselines", {})
            baselines_path_str = (
                baselines_config.get("path") if baselines_config else None
            )

            if baselines_path_str:
                # Use explicit path from config
                baselines_path = Path(baselines_path_str)
                if not baselines_path.is_absolute():
                    # Relative to project root
                    script_dir = Path(__file__).parent
                    project_root = script_dir.parent
                    baselines_path = project_root / baselines_path

    # If not specified in config, use default auto-detect logic
    if baselines_path is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        baselines_path = (
            project_root
            / "baselines"
            / f"baselines_{baseline_dataset.lower().replace(' ', '_')}.json"
        )

    if not baselines_path.exists():
        logger.warning(f"⚠️  Baselines not found: {baselines_path}")
        logger.warning(
            "   To use statistical detection, run: "
            f"uv run python tools/compute_trajectory_baselines.py --dataset {baseline_dataset}"
        )
        return None

    logger.info(f"✅ Found baselines: {baselines_path.name}")
    return BaselineStatistics(baselines_path)


def _prepare_trajectories_for_wang(
    trajectories: List, geo_df: pl.DataFrame
) -> pl.DataFrame:
    """Convert trajectory list to DataFrame format for Wang detector.

    Args:
        trajectories: List of trajectories (each is list of (road_id, timestamp) tuples)
        geo_df: Road network geometry (not used here but for consistency)

    Returns:
        Polars DataFrame with columns: traj_id, road_ids, timestamps
    """
    from datetime import datetime

    traj_data = []
    for traj_idx, trajectory in enumerate(trajectories):
        if not trajectory:
            continue

        road_ids = [road_id for road_id, _ in trajectory]
        timestamps_raw = [timestamp for _, timestamp in trajectory]

        # Convert timestamps to integers (seconds since epoch)
        # Handle both datetime objects and integer/string formats
        timestamps = []
        for ts in timestamps_raw:
            if isinstance(ts, datetime):
                # Convert datetime to seconds since epoch
                timestamps.append(int(ts.timestamp()))
            elif isinstance(ts, (int, float)):
                # Already numeric (seconds)
                timestamps.append(int(ts))
            elif isinstance(ts, str):
                # Parse string to datetime then to seconds
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                    timestamps.append(int(dt.timestamp()))
                except ValueError:
                    # Try ISO format
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(int(dt.timestamp()))
            else:
                # Try to convert to timedelta and get total_seconds
                if hasattr(ts, "total_seconds"):
                    timestamps.append(int(ts.total_seconds()))
                else:
                    logger.warning(f"Unknown timestamp type: {type(ts)} for {ts}")
                    continue

        traj_data.append(
            {"traj_id": traj_idx, "road_ids": road_ids, "timestamps": timestamps}
        )

    return pl.DataFrame(traj_data)


def _convert_wang_results_to_standard_format(wang_results: Dict) -> Dict:
    """Convert Wang detector results to standard format for compatibility.

    Args:
        wang_results: Results from WangStatisticalDetector

    Returns:
        Results in standard format compatible with threshold-based detector
    """
    # Convert pattern counts to abnormal indices
    abnormal_indices = {}

    # Map Wang patterns to threshold categories (approximate)
    # Abp2 (temporal delay) → unusual_duration + suspicious_stops
    # Abp3 (route deviation) → detour + circuitous
    # Abp4 (both) → all of the above

    # Extract abnormal trajectory IDs by pattern
    abnormal_by_pattern = {}
    # Also create a mapping of traj_id -> details for sample saving
    abnormal_details_map = {}
    for traj in wang_results.get("abnormal_trajectories", []):
        pattern = traj["pattern"]
        traj_id = traj["traj_id"]

        if pattern not in abnormal_by_pattern:
            abnormal_by_pattern[pattern] = []
        abnormal_by_pattern[pattern].append(traj_id)

        # Store details for this trajectory
        abnormal_details_map[traj_id] = {
            "traj_idx": traj_id,
            "pattern": pattern,
            "details": traj.get("details", {}),
        }

    # Map to standard categories
    abnormal_indices["wang_temporal_delay"] = abnormal_by_pattern.get(
        "Abp2_temporal_delay", []
    )
    abnormal_indices["wang_route_deviation"] = abnormal_by_pattern.get(
        "Abp3_route_deviation", []
    )
    abnormal_indices["wang_both_deviations"] = abnormal_by_pattern.get(
        "Abp4_both_deviations", []
    )

    # Compute statistics
    total = wang_results["analysis_metadata"]["trajectories_analyzed"]
    statistics = {}

    for category, indices in abnormal_indices.items():
        count = len(indices)
        percentage = (count / total * 100) if total > 0 else 0
        statistics[category] = {"count": count, "percentage": percentage}

    return {
        "total_trajectories": total,
        "abnormal_indices": abnormal_indices,
        "statistics": statistics,
        "pattern_counts": wang_results.get("pattern_counts", {}),
        "wang_metadata": wang_results.get("analysis_metadata", {}),
        "wang_abnormal_details": abnormal_details_map,  # For sample saving
    }


def _save_comparison_results(
    output_dir: Path,
    results_threshold: Dict,
    results_wang: Optional[Dict],
    dataset: str,
    config_path: Path,
):
    """Save comparison results from both detection methods.

    Args:
        output_dir: Output directory
        results_threshold: Results from threshold-based detection
        results_wang: Results from Wang statistical detection (or None if not available)
        dataset: Dataset name
        config_path: Config file path
    """
    # Save threshold results
    threshold_file = output_dir / "detection_results_threshold.json"
    with open(threshold_file, "w") as f:
        json_results = {
            "method": "threshold_based",
            "dataset": dataset,
            "config_file": str(config_path),
            "total_trajectories": results_threshold["total_trajectories"],
            "abnormal_indices": {
                category: list(indices)
                for category, indices in results_threshold["abnormal_indices"].items()
            },
            "statistics": results_threshold["statistics"],
        }
        json.dump(ensure_json_serializable(json_results), f, indent=2)
    logger.info(f"  ✅ Saved threshold results to {threshold_file}")

    # Save Wang results if available
    if results_wang:
        wang_file = output_dir / "detection_results_wang.json"
        with open(wang_file, "w") as f:
            json_results = {
                "method": "wang_statistical",
                "dataset": dataset,
                "config_file": str(config_path),
                "total_trajectories": results_wang["total_trajectories"],
                "abnormal_indices": {
                    category: list(indices)
                    for category, indices in results_wang["abnormal_indices"].items()
                },
                "statistics": results_wang["statistics"],
                "pattern_counts": results_wang.get("pattern_counts", {}),
                "wang_metadata": results_wang.get("wang_metadata", {}),
            }
            json.dump(ensure_json_serializable(json_results), f, indent=2)
        logger.info(f"  ✅ Saved Wang results to {wang_file}")

        # Create comparison summary
        comparison_file = output_dir / "method_comparison.json"
        comparison = {
            "dataset": dataset,
            "config_file": str(config_path),
            "threshold_method": {
                "total_abnormal": sum(
                    len(indices)
                    for indices in results_threshold["abnormal_indices"].values()
                ),
                "abnormal_rate": sum(
                    stats["percentage"]
                    for stats in results_threshold["statistics"].values()
                ),
                "categories": list(results_threshold["statistics"].keys()),
            },
            "wang_method": {
                "total_abnormal": (
                    results_wang["total_trajectories"]
                    - results_wang.get("pattern_counts", {}).get("Abp1_normal", 0)
                ),
                "abnormal_rate": results_wang.get(
                    "abnormal_rate", 0
                ),  # Direct from Wang results
                "pattern_counts": results_wang.get("pattern_counts", {}),
            },
        }
        with open(comparison_file, "w") as f:
            json.dump(ensure_json_serializable(comparison), f, indent=2)
        logger.info(f"  ✅ Saved comparison summary to {comparison_file}")
    else:
        logger.warning("  ⚠️  Wang results not available (baselines missing)")


def _apply_quality_filtering(
    real_file: Path, config_path: Path, output_dir: Path
) -> Optional[Path]:
    """Apply translation quality filtering if enabled in config.

    Args:
        real_file: Original trajectory file
        config_path: Config file path
        output_dir: Output directory for filtered file

    Returns:
        Path to filtered file or None if filtering not applied
    """
    import yaml

    # Load config to check if filtering is enabled
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    filtering_config = config.get("translation_filtering", {})

    if not filtering_config.get("enabled", False):
        return None

    min_rate = filtering_config.get("min_translation_rate", 95.0)
    require_quality = filtering_config.get("require_quality_file", True)

    logger.info(f"🔍 Translation quality filtering enabled (min rate: {min_rate}%)")

    # Look for quality file
    quality_file = real_file.parent / f"{real_file.stem}_quality.json"

    if not quality_file.exists():
        if require_quality:
            logger.error(f"❌ Quality file required but not found: {quality_file}")
            raise FileNotFoundError(
                f"Translation quality file not found: {quality_file}. "
                "Run translation with quality tracking or disable filtering."
            )
        else:
            logger.warning(
                f"⚠️  Quality file not found: {quality_file} - Skipping filtering"
            )
            return None

    # Load quality and filter
    quality = load_quality_report(quality_file)

    # Create filtered file in output dir
    filtered_file = output_dir / f"{real_file.stem}_filtered.csv"

    filter_stats = filter_trajectories_by_quality(
        input_file=real_file,
        quality=quality,
        min_translation_rate=min_rate,
        output_file=filtered_file,
    )

    logger.info(
        f"✅ Filtered to {filter_stats['retention_rate_pct']:.1f}% of trajectories"
    )

    return filtered_file


def run_abnormal_analysis(
    real_file: Path,
    dataset: str,
    config_path: Path,
    output_dir: Path,
    is_real_data: bool = True,
) -> Dict[str, Any]:
    """Run abnormal trajectory analysis on trajectory data.

    Args:
        real_file: Path to trajectory CSV file (real or generated)
        dataset: Dataset name (used to locate road network files)
        config_path: Path to abnormal detection configuration YAML
        output_dir: Directory to save results
        is_real_data: True for real data format, False for generated data format

    Returns:
        Dictionary with detection results and statistics
    """
    logger.info("🔍 Starting abnormal trajectory analysis...")
    logger.info(f"📂 Dataset: {dataset}")
    logger.info(f"📂 Input file: {real_file}")
    logger.info(f"⚙️  Config: {config_path}")

    # Apply quality filtering if enabled (before loading trajectories)
    filtered_file = _apply_quality_filtering(real_file, config_path, output_dir)
    if filtered_file:
        logger.info(f"📊 Using filtered file: {filtered_file}")
        analysis_file = filtered_file
    else:
        analysis_file = real_file

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Output directory: {output_dir}")

    # Determine detection method from config
    detection_method = _get_detection_method(config_path)
    logger.info(f"🔍 Detection method: {detection_method}")

    # Load configuration (needed for either method)
    logger.info("⚙️  Loading configuration...")

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

    # Load trajectories (from filtered file if quality filtering was applied)
    if not analysis_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {analysis_file}")

    data_type = "real" if is_real_data else "generated"
    logger.info(f"📂 Loading {data_type} trajectories...")
    # Get max road_id from geo_df for validation
    max_road_id = geo_df["road_id"].max()
    trajectories = load_trajectories(
        str(analysis_file), is_real_data=is_real_data, max_road_id=max_road_id
    )

    logger.info(f"✅ Loaded {len(trajectories)} trajectories")

    # Route to appropriate detector based on method
    if detection_method == "wang_statistical":
        # Wang et al. 2018 statistical detection
        logger.info("📊 Using Wang et al. 2018 statistical detection")

        # Load baselines (pass config_path to read baselines path from config)
        baselines = _load_baselines_for_dataset(dataset, config_path)
        if baselines is None:
            logger.error("❌ Baselines required for statistical detection")
            raise FileNotFoundError(
                f"Baseline statistics not found for {dataset}. "
                "Run: uv run python tools/compute_trajectory_baselines.py"
            )

        # Load Wang config
        wang_config = WangConfig.from_yaml(config_path)

        # Prepare trajectories for Wang detector
        trajectories_df = _prepare_trajectories_for_wang(trajectories, geo_df)
        logger.info(f"📊 Prepared {len(trajectories_df)} trajectories for analysis")

        # Initialize and run Wang detector
        detector = WangStatisticalDetector(baselines, wang_config, geo_df)
        results = detector.detect_abnormal_trajectories(trajectories_df)

        # Convert Wang results to compatible format
        results = _convert_wang_results_to_standard_format(results)

    elif detection_method == "both":
        # Run both methods and compare
        logger.info("🔄 Running BOTH detection methods for comparison")

        # Method 1: Threshold-based
        logger.info("  Method 1: Threshold-based detection")
        config = AbnormalityConfig.from_yaml(config_path)
        traj_data = []
        for traj_idx, trajectory in enumerate(trajectories):
            for road_id, timestamp in trajectory:
                traj_data.append(
                    {"traj_id": traj_idx, "road_id": road_id, "timestamp": timestamp}
                )
        trajectories_df_threshold = pl.DataFrame(traj_data)
        detector_threshold = AbnormalTrajectoryDetector(config, geo_df, rel_df)
        results_threshold = detector_threshold.detect_abnormal_trajectories(
            trajectories_df_threshold
        )

        # Method 2: Wang statistical
        logger.info("  Method 2: Wang statistical detection")
        baselines = _load_baselines_for_dataset(dataset, config_path)
        if baselines:
            wang_config = WangConfig.from_yaml(config_path)
            trajectories_df_wang = _prepare_trajectories_for_wang(trajectories, geo_df)
            detector_wang = WangStatisticalDetector(baselines, wang_config, geo_df)
            results_wang = detector_wang.detect_abnormal_trajectories(
                trajectories_df_wang
            )
            results_wang = _convert_wang_results_to_standard_format(results_wang)
        else:
            logger.warning("  ⚠️  Skipping Wang detection (baselines not found)")
            results_wang = None

        # Save both results
        _save_comparison_results(
            output_dir, results_threshold, results_wang, dataset, config_path
        )

        # Return threshold results as primary for backward compatibility
        results = results_threshold

    else:
        # Default: Threshold-based detection (z_score)
        logger.info("📊 Using threshold-based detection (z-score)")

        # Load threshold config
        config = AbnormalityConfig.from_yaml(config_path)

        # Convert trajectories to DataFrame format expected by threshold detector
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
        json.dump(ensure_json_serializable(json_results), f, indent=2)
    logger.info(f"✅ Saved detection results to {results_file}")

    # Save statistics summary
    stats_file = output_dir / "statistics_by_category.json"
    with open(stats_file, "w") as f:
        json.dump(ensure_json_serializable(results["statistics"]), f, indent=2)
    logger.info(f"✅ Saved statistics to {stats_file}")

    # Save trajectory samples if enabled
    # Load config for analysis settings (needed for all methods)
    import yaml

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    analysis_config = config_dict.get("analysis", {})

    if analysis_config.get("save_trajectory_samples", False):
        logger.info("💾 Saving trajectory samples...")
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        max_samples = analysis_config.get("max_samples_per_category", 50)

        for category, indices in results["abnormal_indices"].items():
            if not indices:
                continue

            # Get sample of trajectories for this category
            sample_indices = indices[: min(len(indices), max_samples)]

            category_samples = []
            for idx in sample_indices:
                # Find the analysis result for this trajectory
                # Check if all_results exists (threshold-based detector)
                if "all_results" in results:
                    traj_result = next(
                        (r for r in results["all_results"] if r["traj_idx"] == idx),
                        None,
                    )
                    if traj_result:
                        category_samples.append(traj_result)
                # For Wang detector, use abnormal_trajectories if available
                elif "wang_abnormal_details" in results:
                    # Find in Wang abnormal details map
                    wang_details = results.get("wang_abnormal_details", {})
                    if idx in wang_details:
                        category_samples.append(wang_details[idx])
                    else:
                        # Fallback if not found in details
                        category_samples.append(
                            {
                                "traj_idx": idx,
                                "category": category,
                                "pattern": "unknown",
                                "note": "Wang statistical detection - trajectory details not available",
                            }
                        )
                else:
                    # Fallback: create minimal sample entry
                    category_samples.append({"traj_idx": idx, "category": category})

            # Save samples for this category
            sample_file = samples_dir / f"{category}_samples.json"
            with open(sample_file, "w") as f:
                json.dump(ensure_json_serializable(category_samples), f, indent=2)

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

    parser.add_argument(
        "--is-generated",
        action="store_true",
        help="Set this flag if analyzing generated trajectories (uses gene_trace_* columns)",
    )

    args = parser.parse_args()

    try:
        run_abnormal_analysis(
            real_file=args.real_file,
            dataset=args.dataset,
            config_path=args.config,
            output_dir=args.output_dir,
            is_real_data=not args.is_generated,
        )
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
