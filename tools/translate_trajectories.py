#!/usr/bin/env python3
"""
Trajectory Road ID Translator

Translates generated trajectory files from source road network IDs to target
road network IDs using a pre-computed mapping. Tracks translation quality
and saves comprehensive statistics for debugging and validation.

Usage:
    uv run python tools/translate_trajectories.py \
        --input gene/Beijing/seed42/distilled_train.csv \
        --mapping road_mapping_beijing_to_bjut.json \
        --output gene_translated/BJUT/distilled_train.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import polars as pl
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_mapping(mapping_file: Path) -> Dict[int, int]:
    """Load road ID mapping from JSON file"""
    logger.info(f"📂 Loading road ID mapping from {mapping_file}")
    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    # Convert string keys to ints
    mapping_int = {int(k): int(v) for k, v in mapping.items()}
    logger.info(f"  ✅ Loaded {len(mapping_int):,} road ID mappings")

    return mapping_int


def translate_trajectory_file(
    input_file: Path,
    mapping: Dict[int, int],
    output_file: Path,
) -> Dict:
    """Translate road IDs in a generated trajectory CSV file

    Args:
        input_file: Generated trajectory CSV file
        mapping: Road ID mapping dict
        output_file: Output file path

    Returns:
        Dictionary with translation statistics
    """
    logger.info(f"\n🔄 Translating {input_file.name}")

    # Load trajectory data
    df = pl.read_csv(input_file)
    logger.info(f"  Loaded {len(df)} trajectory points")

    # Check required columns
    required_cols = ["gene_trace_road_id", "gene_trace_datetime"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse road IDs (comma-separated string)
    logger.info("  🔍 Parsing road ID sequences...")

    translation_stats = {
        "trajectories_processed": 0,
        "total_road_points": 0,
        "points_translated": 0,
        "points_unmapped": 0,
        "trajectories_fully_translated": 0,
        "trajectories_with_gaps": 0,
        "trajectories_failed": 0,
        "unmapped_roads_encountered": Counter(),
    }

    translated_rows = []

    # Group by trajectory
    trajectory_groups = df.group_by(
        ["origin_road_id", "destination_road_id", "source_index"]
    ).agg(pl.all())

    logger.info(f"  Processing {len(trajectory_groups)} trajectories...")

    for idx, row in enumerate(trajectory_groups.iter_rows(named=True)):
        if idx % 500 == 0 and idx > 0:
            logger.info(
                f"    Translated {idx:,} / {len(trajectory_groups):,} trajectories..."
            )

        translation_stats["trajectories_processed"] += 1

        # Parse road ID sequence (handle JSON array format)
        road_ids_str = str(row["gene_trace_road_id"][0])

        # Check if it's a JSON array string
        if road_ids_str.startswith("["):
            road_ids = json.loads(road_ids_str)
        else:
            # Fallback: comma-separated string
            road_ids = [int(r) for r in road_ids_str.split(",")]

        translation_stats["total_road_points"] += len(road_ids)

        # Translate each road ID
        translated_ids = []
        failed_count = 0

        for road_id in road_ids:
            if road_id in mapping:
                translated_ids.append(mapping[road_id])
                translation_stats["points_translated"] += 1
            else:
                # Road not in mapping - keep original and track
                translated_ids.append(road_id)
                translation_stats["points_unmapped"] += 1
                translation_stats["unmapped_roads_encountered"][road_id] += 1
                failed_count += 1

        # Classify trajectory translation quality
        if failed_count == 0:
            translation_stats["trajectories_fully_translated"] += 1
        elif failed_count < len(road_ids):
            translation_stats["trajectories_with_gaps"] += 1
        else:
            translation_stats["trajectories_failed"] += 1

        # Create translated row
        translated_row = {
            "origin_road_id": mapping.get(
                row["origin_road_id"][0], row["origin_road_id"][0]
            ),
            "destination_road_id": mapping.get(
                row["destination_road_id"][0], row["destination_road_id"][0]
            ),
            "source_index": row["source_index"][0],
            "source_origin_time": row["source_origin_time"][0],
            "gene_trace_road_id": ",".join(map(str, translated_ids)),
            "gene_trace_datetime": row["gene_trace_datetime"][0],
        }

        translated_rows.append(translated_row)

    # Create output DataFrame
    translated_df = pl.DataFrame(translated_rows)

    # Save translated file
    logger.info(f"  💾 Saving translated trajectories to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    translated_df.write_csv(output_file)

    # Calculate final statistics
    translation_stats["translation_rate_pct"] = (
        translation_stats["points_translated"]
        / translation_stats["total_road_points"]
        * 100
        if translation_stats["total_road_points"] > 0
        else 0
    )

    # Get top unmapped roads
    top_unmapped = [
        {"road_id": road_id, "occurrences": count}
        for road_id, count in translation_stats[
            "unmapped_roads_encountered"
        ].most_common(10)
    ]

    # Prepare stats output (remove Counter object)
    stats_output = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "mapping_file": str(input_file.parent.parent / "road_mapping_*"),
        "trajectories_processed": translation_stats["trajectories_processed"],
        "total_road_points": translation_stats["total_road_points"],
        "translation_results": {
            "points_translated": translation_stats["points_translated"],
            "points_unmapped": translation_stats["points_unmapped"],
            "translation_rate_pct": translation_stats["translation_rate_pct"],
            "trajectories_fully_translated": translation_stats[
                "trajectories_fully_translated"
            ],
            "trajectories_with_gaps": translation_stats["trajectories_with_gaps"],
            "trajectories_failed": translation_stats["trajectories_failed"],
        },
        "unmapped_roads_encountered": {
            "unique_count": len(translation_stats["unmapped_roads_encountered"]),
            "total_occurrences": translation_stats["points_unmapped"],
            "top_10_unmapped": top_unmapped,
        },
    }

    # Save translation stats
    stats_file = output_file.parent / f"{output_file.stem}_translation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats_output, f, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 70}")
    logger.info("📊 TRANSLATION SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"Trajectories: {translation_stats['trajectories_processed']:,}")
    logger.info(f"Road points:  {translation_stats['total_road_points']:,}")
    logger.info("\nTranslation Quality:")
    logger.info(
        f"  ✅ Translated: {translation_stats['points_translated']:,} ({translation_stats['translation_rate_pct']:.1f}%)"
    )
    logger.info(f"  ❌ Unmapped:   {translation_stats['points_unmapped']:,}")
    logger.info("\nTrajectory Quality:")
    logger.info(
        f"  ✅ Fully translated:  {translation_stats['trajectories_fully_translated']:,}"
    )
    logger.info(
        f"  ⚠️  With gaps:         {translation_stats['trajectories_with_gaps']:,}"
    )
    logger.info(f"  ❌ Failed:            {translation_stats['trajectories_failed']:,}")
    logger.info(f"{'=' * 70}")
    logger.info(f"📊 Statistics saved to {stats_file}")

    return stats_output


def main():
    parser = argparse.ArgumentParser(
        description="Translate trajectory road IDs using mapping file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate single file
  uv run python tools/translate_trajectories.py \\
    --input gene/Beijing/seed42/distilled_train.csv \\
    --mapping road_mapping_beijing_to_bjut.json \\
    --output gene_translated/BJUT/distilled_train.csv

  # Translate multiple files (will process all matching)
  uv run python tools/translate_trajectories.py \\
    --input gene/Beijing/seed42/*.csv \\
    --mapping road_mapping_beijing_to_bjut.json \\
    --output-dir gene_translated/BJUT/
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input generated trajectory CSV file",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="Road ID mapping JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output translated trajectory CSV file",
    )

    args = parser.parse_args()

    # Validate inputs
    input_file = Path(args.input)
    mapping_file = Path(args.mapping)
    output_file = Path(args.output)

    if not input_file.exists():
        parser.error(f"Input file not found: {input_file}")
    if not mapping_file.exists():
        parser.error(f"Mapping file not found: {mapping_file}")

    # Load mapping
    mapping = load_mapping(mapping_file)

    # Translate
    stats = translate_trajectory_file(
        input_file=input_file,
        mapping=mapping,
        output_file=output_file,
    )

    # Quality check
    if stats["translation_results"]["translation_rate_pct"] > 95:
        logger.info("✅ Translation Quality: EXCELLENT")
    elif stats["translation_results"]["translation_rate_pct"] > 85:
        logger.info("✅ Translation Quality: GOOD")
    elif stats["translation_results"]["translation_rate_pct"] > 70:
        logger.info("⚠️  Translation Quality: FAIR")
    else:
        logger.error("❌ Translation Quality: POOR - Many unmapped roads!")

    logger.info("\n✅ Translation complete!")


if __name__ == "__main__":
    main()
