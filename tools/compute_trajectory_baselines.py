#!/usr/bin/env python3
"""
Compute OD-Pair Baseline Statistics for Statistical Abnormality Detection

Implements baseline computation from Wang et al. 2018 methodology.
Computes mean/std/quantiles for route length, travel time, and speed
on a per-OD-pair basis from real trajectory data.

IMPORTANT: By default, baselines are computed using ONLY training data
to avoid data leakage. This ensures valid evaluation on test data.

Usage:
    # Compute baselines (train-only, recommended):
    uv run python tools/compute_trajectory_baselines.py --dataset Beijing
    uv run python tools/compute_trajectory_baselines.py --dataset BJUT_Beijing
    uv run python tools/compute_trajectory_baselines.py --dataset porto_hoser

    # Legacy mode (includes test data, causes data leakage):
    uv run python tools/compute_trajectory_baselines.py --dataset Beijing --include-test
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_real_trajectories(
    dataset: str, data_dir: Path, train_only: bool = True
) -> pl.DataFrame:
    """Load real trajectories for baseline computation

    Args:
        dataset: Dataset name
        data_dir: Path to data directory
        train_only: If True, load only training data to avoid data leakage.
                   If False, combine train + test (legacy behavior, not recommended)

    Returns:
        DataFrame with real trajectories (train-only by default for valid baselines)
    """
    logger.info(f"📂 Loading real trajectories for {dataset}")

    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")

    logger.info(f"  Loading train data from {train_file}")
    train_df = pl.read_csv(train_file)
    logger.info(f"  ✅ Loaded {len(train_df)} train trajectory points")

    if train_only:
        logger.info("  ✅ Using train-only data (no data leakage)")
        return train_df
    else:
        # Legacy behavior: combine train + test (causes data leakage!)
        logger.warning("  ⚠️  WARNING: Combining train + test data causes DATA LEAKAGE!")
        logger.warning("  ⚠️  Test set statistics will leak into baseline computation.")
        logger.warning("  ⚠️  Use train_only=True for valid evaluation.")

        if test_file.exists():
            logger.info(f"  Loading test data from {test_file}")
            test_df = pl.read_csv(test_file)
            logger.info(f"  ✅ Loaded {len(test_df)} test trajectory points")

            # Combine
            combined_df = pl.concat([train_df, test_df])
            logger.info(f"  📊 Combined: {len(combined_df)} total points")
        else:
            logger.warning("  ⚠️  Test file not found, using train only")
            combined_df = train_df

        return combined_df


def compute_trajectory_metrics(row: dict, road_network: pl.DataFrame = None) -> dict:
    """Compute length, time, and speed metrics for a trajectory row

    Args:
        row: Single row dict with rid_list, time_list columns
        road_network: Optional road network for accurate length calculation

    Returns:
        Dict with route_length_m, duration_sec, avg_speed_kmh, origin, destination
    """
    # Parse road IDs and timestamps
    road_ids_str = row["rid_list"]
    timestamps_str = row["time_list"]

    road_ids = [int(r) for r in str(road_ids_str).split(",")]
    timestamps = [
        datetime.strptime(t.strip('"'), "%Y-%m-%dT%H:%M:%SZ")
        for t in str(timestamps_str).split(",")
    ]

    if len(road_ids) == 0 or len(timestamps) == 0:
        return None

    # Route length (simplified: count road segments × average segment length)
    # TODO: Use actual road lengths from .geo file for accuracy
    route_length_m = len(road_ids) * 100  # Rough estimate: 100m per segment

    # Travel time
    duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()

    # Average speed
    avg_speed_kmh = (
        (route_length_m / 1000) / (duration_sec / 3600) if duration_sec > 0 else 0
    )

    return {
        "origin": road_ids[0],
        "destination": road_ids[-1],
        "route_length_m": route_length_m,
        "duration_sec": duration_sec,
        "avg_speed_kmh": avg_speed_kmh,
        "num_points": len(road_ids),
    }


def compute_od_baselines(trajectories_df: pl.DataFrame) -> dict:
    """Compute baseline statistics per OD pair

    Args:
        trajectories_df: Training trajectories (should be train-only to avoid data leakage)

    Returns:
        Dict with baselines per OD pair and global statistics

    Note:
        To avoid data leakage, this should only receive TRAINING data.
        Test set should be evaluated against baselines computed from train set only.
    """
    logger.info("\n📊 Computing OD-pair baseline statistics...")

    # Beijing format: one row per trajectory with rid_list and time_list
    logger.info(f"  Processing {len(trajectories_df)} trajectory rows")

    # Compute metrics for each trajectory
    logger.info("  Computing trajectory metrics...")
    traj_metrics = []

    for idx, row in enumerate(trajectories_df.iter_rows(named=True)):
        if idx % 100000 == 0 and idx > 0:
            logger.info(
                f"    Processed {idx:,} / {len(trajectories_df):,} trajectories..."
            )

        try:
            metrics = compute_trajectory_metrics(row)
            if metrics:
                traj_metrics.append(metrics)
        except Exception as e:
            logger.debug(f"    Skipping trajectory {idx}: {e}")
            continue

    metrics_df = pl.DataFrame(traj_metrics)
    logger.info(f"  ✅ Computed metrics for {len(metrics_df)} trajectories")

    # Compute baselines per OD pair
    logger.info("\n  📈 Computing per-OD-pair statistics...")
    od_baselines = {}

    od_groups = metrics_df.group_by(["origin", "destination"]).agg(
        [
            pl.col("route_length_m").mean().alias("mean_length_m"),
            pl.col("route_length_m").std().alias("std_length_m"),
            pl.col("route_length_m").min().alias("min_length_m"),
            pl.col("route_length_m").median().alias("p50_length_m"),
            pl.col("route_length_m").quantile(0.95).alias("p95_length_m"),
            pl.col("duration_sec").mean().alias("mean_duration_sec"),
            pl.col("duration_sec").std().alias("std_duration_sec"),
            pl.col("duration_sec").min().alias("min_duration_sec"),
            pl.col("duration_sec").median().alias("p50_duration_sec"),
            pl.col("duration_sec").quantile(0.95).alias("p95_duration_sec"),
            pl.col("avg_speed_kmh").mean().alias("mean_speed_kmh"),
            pl.col("avg_speed_kmh").std().alias("std_speed_kmh"),
            pl.col("avg_speed_kmh").max().alias("max_speed_kmh"),
            pl.len().alias("n_samples"),
        ]
    )

    for row in od_groups.iter_rows(named=True):
        od_key = f"({row['origin']}, {row['destination']})"
        od_baselines[od_key] = {
            "mean_length_m": float(row["mean_length_m"])
            if row["mean_length_m"] is not None
            else 0,
            "std_length_m": float(row["std_length_m"])
            if row["std_length_m"] is not None
            else 0,
            "min_length_m": float(row["min_length_m"])
            if row["min_length_m"] is not None
            else 0,
            "p50_length_m": float(row["p50_length_m"])
            if row["p50_length_m"] is not None
            else 0,
            "p95_length_m": float(row["p95_length_m"])
            if row["p95_length_m"] is not None
            else 0,
            "mean_duration_sec": float(row["mean_duration_sec"])
            if row["mean_duration_sec"] is not None
            else 0,
            "std_duration_sec": float(row["std_duration_sec"])
            if row["std_duration_sec"] is not None
            else 0,
            "min_duration_sec": float(row["min_duration_sec"])
            if row["min_duration_sec"] is not None
            else 0,
            "p50_duration_sec": float(row["p50_duration_sec"])
            if row["p50_duration_sec"] is not None
            else 0,
            "p95_duration_sec": float(row["p95_duration_sec"])
            if row["p95_duration_sec"] is not None
            else 0,
            "mean_speed_kmh": float(row["mean_speed_kmh"])
            if row["mean_speed_kmh"] is not None
            else 0,
            "std_speed_kmh": float(row["std_speed_kmh"])
            if row["std_speed_kmh"] is not None
            else 0,
            "max_speed_kmh": float(row["max_speed_kmh"])
            if row["max_speed_kmh"] is not None
            else 0,
            "n_samples": int(row["n_samples"]),
        }

    logger.info(f"  ✅ Computed baselines for {len(od_baselines)} OD pairs")

    # Global statistics
    global_stats = {
        "mean_length_m": float(metrics_df["route_length_m"].mean()),
        "std_length_m": float(metrics_df["route_length_m"].std()),
        "mean_duration_sec": float(metrics_df["duration_sec"].mean()),
        "std_duration_sec": float(metrics_df["duration_sec"].std()),
        "mean_speed_kmh": float(metrics_df["avg_speed_kmh"].mean()),
        "std_speed_kmh": float(metrics_df["avg_speed_kmh"].std()),
    }

    # Coverage statistics
    od_pairs_with_min_samples = sum(
        1 for b in od_baselines.values() if b["n_samples"] >= 5
    )
    coverage_pct = (
        (od_pairs_with_min_samples / len(od_baselines) * 100)
        if len(od_baselines) > 0
        else 0
    )

    return {
        "od_pair_baselines": od_baselines,
        "global_statistics": global_stats,
        "coverage": {
            "total_trajectories": len(metrics_df),
            "total_od_pairs": len(od_baselines),
            "od_pairs_with_min_5_samples": od_pairs_with_min_samples,
            "coverage_pct": coverage_pct,
        },
    }


def save_baselines(baselines: dict, dataset: str, output_dir: Path):
    """Save baseline statistics with metadata"""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "dataset": dataset,
            "computed_at": datetime.now().isoformat(),
            "baseline_source": "real_train+test_combined",
            "methodology": "Wang et al. 2018 - ISPRS Int. J. Geo-Inf. 7(1), 25",
        },
        **baselines,
    }

    output_file = output_dir / f"baselines_{dataset.lower().replace(' ', '_')}.json"

    logger.info(f"\n💾 Saving baselines to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("✅ Baselines saved!")
    logger.info(f"\n{'=' * 70}")
    logger.info("📊 BASELINE STATISTICS SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Total trajectories: {baselines['coverage']['total_trajectories']:,}")
    logger.info(f"Total OD pairs: {baselines['coverage']['total_od_pairs']:,}")
    logger.info(
        f"OD pairs with ≥5 samples: {baselines['coverage']['od_pairs_with_min_5_samples']:,} ({baselines['coverage']['coverage_pct']:.1f}%)"
    )
    logger.info("\nGlobal Statistics:")
    logger.info(
        f"  Mean route length: {baselines['global_statistics']['mean_length_m']:.1f}m"
    )
    logger.info(
        f"  Mean duration: {baselines['global_statistics']['mean_duration_sec']:.1f}s"
    )
    logger.info(
        f"  Mean speed: {baselines['global_statistics']['mean_speed_kmh']:.1f} km/h"
    )
    logger.info(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute OD-pair baseline statistics for statistical abnormality detection",
        epilog="""
Examples:
  # Compute Beijing baselines
  uv run python tools/compute_trajectory_baselines.py --dataset Beijing
  
  # Compute BJUT baselines
  uv run python tools/compute_trajectory_baselines.py --dataset BJUT_Beijing
  
  # Compute Porto baselines
  uv run python tools/compute_trajectory_baselines.py --dataset porto_hoser
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (Beijing, BJUT_Beijing, porto_hoser, etc.)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: data/{dataset})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines",
        help="Output directory for baseline files (default: baselines/)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include test data in baseline computation (NOT RECOMMENDED: causes data leakage!)",
    )

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(f"data/{args.dataset}")
        if not data_dir.exists():
            data_dir = Path(f"../data/{args.dataset}")

    if not data_dir.exists():
        parser.error(f"Data directory not found: {data_dir}")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"🎯 COMPUTING BASELINES: {args.dataset}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Warn if including test data
    if args.include_test:
        logger.warning("\n⚠️  WARNING: Including test data in baseline computation!")
        logger.warning(
            "⚠️  This causes DATA LEAKAGE and invalidates evaluation results!"
        )
        logger.warning("⚠️  Use --include-test only for legacy compatibility.\n")

    # Load trajectories (train-only by default to avoid data leakage)
    trajectories_df = load_real_trajectories(
        args.dataset, data_dir, train_only=not args.include_test
    )

    # Compute baselines
    baselines = compute_od_baselines(trajectories_df)

    # Save results
    output_dir = Path(args.output_dir)
    save_baselines(baselines, args.dataset, output_dir)

    logger.info("\n✅ Baseline computation complete!")


if __name__ == "__main__":
    main()
