#!/usr/bin/env python3
"""
Translation Quality Filter

Filters translated trajectory files based on per-trajectory translation quality.
Useful for creating high-quality subsets for more reliable abnormality detection
and evaluation.

Usage:
    uv run python tools/filter_translated_by_quality.py \
        --input gene_translated/BJUT/distilled_train.csv \
        --quality gene_translated/BJUT/distilled_train_quality.json \
        --min-rate 95.0 \
        --output gene_translated/BJUT/distilled_train_highquality.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_quality_report(quality_file: Path) -> Dict:
    """Load per-trajectory quality report.

    Args:
        quality_file: Path to quality JSON file

    Returns:
        Quality report dictionary
    """
    logger.info(f"📂 Loading quality report from {quality_file}")
    with open(quality_file, "r") as f:
        quality = json.load(f)

    logger.info(
        f"  ✅ Loaded quality data for {quality['total_trajectories']:,} trajectories"
    )

    return quality


def filter_trajectories_by_quality(
    input_file: Path,
    quality: Dict,
    min_translation_rate: float = 95.0,
    output_file: Path = None,
) -> Dict:
    """Filter trajectories based on translation quality.

    Args:
        input_file: Translated trajectory CSV file
        quality: Quality report dictionary
        min_translation_rate: Minimum translation rate percentage (default: 95%)
        output_file: Output file path (optional)

    Returns:
        Dictionary with filtering statistics
    """
    logger.info(f"\n🔍 Filtering trajectories (min rate: {min_translation_rate}%)")

    # Load trajectory data
    df = pl.read_csv(input_file)
    logger.info(f"  Loaded {len(df)} trajectory points")

    # Build set of high-quality trajectory indices
    high_quality_indices: Set[int] = set()

    for traj in quality["trajectories"]:
        if traj["translation_rate_pct"] >= min_translation_rate:
            high_quality_indices.add(traj["source_index"])

    logger.info(
        f"  Found {len(high_quality_indices):,} trajectories meeting quality threshold"
    )

    # Filter DataFrame by source_index
    filtered_df = df.filter(pl.col("source_index").is_in(high_quality_indices))

    logger.info(f"  Filtered to {len(filtered_df)} trajectory points")

    # Calculate statistics
    original_traj_count = quality["total_trajectories"]
    filtered_traj_count = len(high_quality_indices)
    retention_rate = (
        (filtered_traj_count / original_traj_count * 100)
        if original_traj_count > 0
        else 0
    )

    stats = {
        "input_file": str(input_file),
        "quality_file": str(quality.get("file", "unknown")),
        "min_translation_rate_pct": min_translation_rate,
        "original_trajectories": original_traj_count,
        "filtered_trajectories": filtered_traj_count,
        "retention_rate_pct": retention_rate,
        "original_points": len(df),
        "filtered_points": len(filtered_df),
        "points_retention_rate_pct": (
            (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
        ),
    }

    # Save if output file specified
    if output_file:
        logger.info(f"  💾 Saving high-quality trajectories to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.write_csv(output_file)

        # Save filtering stats
        stats_file = output_file.parent / f"{output_file.stem}_filter_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"  📊 Filter statistics saved to {stats_file}")
        stats["output_file"] = str(output_file)

    # Print summary
    logger.info(f"\n{'=' * 70}")
    logger.info("📊 QUALITY FILTERING SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"Minimum translation rate: {min_translation_rate}%")
    logger.info("\nTrajectories:")
    logger.info(f"  Original:  {original_traj_count:,}")
    logger.info(f"  Filtered:  {filtered_traj_count:,}")
    logger.info(f"  Retention: {retention_rate:.1f}%")
    logger.info("\nTrajectory points:")
    logger.info(f"  Original:  {len(df):,}")
    logger.info(f"  Filtered:  {len(filtered_df):,}")
    logger.info(f"  Retention: {stats['points_retention_rate_pct']:.1f}%")
    logger.info(f"{'=' * 70}")

    return stats


def batch_filter_directory(
    input_dir: Path,
    min_translation_rate: float = 95.0,
    output_suffix: str = "_highquality",
):
    """Filter all translated files in a directory.

    Args:
        input_dir: Directory containing translated files and quality reports
        min_translation_rate: Minimum translation rate percentage
        output_suffix: Suffix to add to filtered files
    """
    logger.info(f"🔍 Batch filtering directory: {input_dir}")
    logger.info(f"  Minimum translation rate: {min_translation_rate}%")

    # Find all quality files
    quality_files = list(input_dir.glob("*_quality.json"))

    if not quality_files:
        logger.error(f"❌ No quality files found in {input_dir}")
        return

    logger.info(f"  Found {len(quality_files)} quality files")

    all_stats = []

    for quality_file in quality_files:
        # Determine corresponding CSV file
        base_name = quality_file.stem.replace("_quality", "")
        csv_file = input_dir / f"{base_name}.csv"

        if not csv_file.exists():
            logger.warning(f"  ⚠️  Skipping {base_name}: CSV file not found")
            continue

        # Output file
        output_file = input_dir / f"{base_name}{output_suffix}.csv"

        # Load quality
        quality = load_quality_report(quality_file)

        # Filter
        stats = filter_trajectories_by_quality(
            input_file=csv_file,
            quality=quality,
            min_translation_rate=min_translation_rate,
            output_file=output_file,
        )

        all_stats.append(stats)

    # Save batch summary
    batch_summary = {
        "input_directory": str(input_dir),
        "min_translation_rate_pct": min_translation_rate,
        "files_processed": len(all_stats),
        "overall_stats": {
            "total_original_trajectories": sum(
                s["original_trajectories"] for s in all_stats
            ),
            "total_filtered_trajectories": sum(
                s["filtered_trajectories"] for s in all_stats
            ),
            "average_retention_rate_pct": (
                sum(s["retention_rate_pct"] for s in all_stats) / len(all_stats)
                if all_stats
                else 0
            ),
        },
        "files": all_stats,
    }

    summary_file = input_dir / "batch_filter_summary.json"
    with open(summary_file, "w") as f:
        json.dump(batch_summary, f, indent=2)

    logger.info("\n✅ Batch filtering complete!")
    logger.info(f"📊 Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter translated trajectories by translation quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter single file
  uv run python tools/filter_translated_by_quality.py \\
    --input gene_translated/BJUT/distilled_train.csv \\
    --quality gene_translated/BJUT/distilled_train_quality.json \\
    --min-rate 95.0 \\
    --output gene_translated/BJUT/distilled_train_highquality.csv

  # Batch filter entire directory
  uv run python tools/filter_translated_by_quality.py \\
    --batch-dir gene_translated/BJUT \\
    --min-rate 95.0
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input translated trajectory CSV file",
    )
    parser.add_argument(
        "--quality",
        type=str,
        help="Quality report JSON file",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=95.0,
        help="Minimum translation rate percentage (default: 95.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filtered trajectory CSV file",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        help="Process all files in directory (alternative to --input/--quality)",
    )

    args = parser.parse_args()

    # Batch mode
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            parser.error(f"Batch directory not found: {batch_dir}")

        batch_filter_directory(
            input_dir=batch_dir,
            min_translation_rate=args.min_rate,
        )
        return

    # Single file mode
    if not args.input or not args.quality:
        parser.error("--input and --quality required (or use --batch-dir)")

    input_file = Path(args.input)
    quality_file = Path(args.quality)

    if not input_file.exists():
        parser.error(f"Input file not found: {input_file}")
    if not quality_file.exists():
        parser.error(f"Quality file not found: {quality_file}")

    output_file = Path(args.output) if args.output else None

    # Load quality
    quality = load_quality_report(quality_file)

    # Filter
    stats = filter_trajectories_by_quality(
        input_file=input_file,
        quality=quality,
        min_translation_rate=args.min_rate,
        output_file=output_file,
    )

    # Quality assessment
    if stats["retention_rate_pct"] > 80:
        logger.info("✅ Retention Rate: EXCELLENT (>80%)")
    elif stats["retention_rate_pct"] > 60:
        logger.info("✅ Retention Rate: GOOD (>60%)")
    elif stats["retention_rate_pct"] > 40:
        logger.info("⚠️  Retention Rate: FAIR (>40%)")
    else:
        logger.warning("❌ Retention Rate: POOR (<40%) - Consider lowering threshold")

    logger.info("\n✅ Filtering complete!")


if __name__ == "__main__":
    main()
