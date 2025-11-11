#!/usr/bin/env python3
"""
Extract Abnormal OD Pairs from Detection Results

This script parses abnormal trajectory detection results and extracts the
origin-destination pairs that are associated with abnormal patterns. This
creates a targeted test set for evaluating model performance on edge cases.

Usage:
    uv run python tools/extract_abnormal_od_pairs.py \
        --detection-results abnormal/BJUT_Beijing/train/real_data/detection_results.json \
        --real-data data/BJUT_Beijing/train.csv \
        --output abnormal_od_pairs.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_detection_results(results_file: Path) -> Dict:
    """Load abnormal trajectory detection results from JSON file"""
    logger.info(f"📂 Loading detection results from {results_file}")
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def extract_od_pairs_from_trajectories(
    real_data_file: Path, abnormal_traj_ids: List[int]
) -> List[Tuple[int, int]]:
    """Extract origin-destination pairs from abnormal trajectory IDs

    Args:
        real_data_file: Path to real trajectory CSV file
        abnormal_traj_ids: List of trajectory IDs marked as abnormal

    Returns:
        List of (origin, destination) tuples
    """
    logger.info(f"📂 Loading real data from {real_data_file}")

    # First, read a sample to detect the format
    sample_df = pl.read_csv(real_data_file, n_rows=1)

    # Detect format: road_id (point-based) vs rid_list (trajectory-based)
    if "road_id" in sample_df.columns:
        # Point-based format: each row is a point in a trajectory
        df = pl.read_csv(
            real_data_file,
            columns=["traj_id", "road_id"],
        )

        logger.info(f"✅ Loaded {len(df)} trajectory points (road_id format)")

        # Filter to abnormal trajectories
        abnormal_df = df.filter(pl.col("traj_id").is_in(abnormal_traj_ids))

        # Group by trajectory and get first/last road_id (origin/destination)
        od_pairs = (
            abnormal_df.group_by("traj_id")
            .agg(
                [
                    pl.col("road_id").first().alias("origin"),
                    pl.col("road_id").last().alias("destination"),
                ]
            )
            .select(["origin", "destination"])
        )

        # Convert to list of tuples
        od_list = [
            (row["origin"], row["destination"])
            for row in od_pairs.iter_rows(named=True)
        ]
    elif "rid_list" in sample_df.columns:
        # Trajectory-based format: each row is a complete trajectory
        df = pl.read_csv(
            real_data_file,
            columns=["traj_id", "rid_list"],
        )

        logger.info(f"✅ Loaded {len(df)} trajectories (rid_list format)")

        # Filter to abnormal trajectories
        abnormal_df = df.filter(pl.col("traj_id").is_in(abnormal_traj_ids))

        # Extract first and last road ID from each rid_list
        od_list = []
        for row in abnormal_df.iter_rows(named=True):
            try:
                # Parse rid_list string (e.g., "[1, 2, 3, 4]" or "1,2,3,4")
                rid_list_str = row["rid_list"]
                if isinstance(rid_list_str, str):
                    # Remove brackets if present and split
                    rid_list_str = rid_list_str.strip("[]")
                    rid_list = [
                        int(x.strip()) for x in rid_list_str.split(",") if x.strip()
                    ]
                else:
                    # Already a list
                    rid_list = list(rid_list_str)

                if len(rid_list) >= 2:
                    origin = rid_list[0]
                    destination = rid_list[-1]
                    od_list.append((origin, destination))
            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"⚠️  Failed to parse rid_list for traj_id {row['traj_id']}: {e}"
                )
                continue
    else:
        raise ValueError(
            f"Unknown data format: expected 'road_id' or 'rid_list' column. "
            f"Found columns: {sample_df.columns}"
        )

    # Deduplicate
    od_set = set(od_list)

    logger.info(f"🔍 Extracted {len(od_list)} OD pairs ({len(od_set)} unique)")

    return list(od_set)


def extract_abnormal_od_pairs(
    detection_results_files: List[Path],
    real_data_files: List[Path],
    dataset_name: str,
) -> Dict:
    """Extract abnormal OD pairs from multiple detection result files

    Args:
        detection_results_files: List of detection result JSON files
        real_data_files: List of corresponding real data CSV files
        dataset_name: Name of the dataset

    Returns:
        Dictionary with OD pairs categorized by abnormality type
    """
    logger.info(f"\n🔍 Extracting abnormal OD pairs from {dataset_name}")

    # Store OD pairs by category (dynamically populated from detection results)
    od_pairs_by_category = {}

    all_abnormal_traj_ids = set()

    # Process each detection results file
    for det_file, data_file in zip(detection_results_files, real_data_files):
        logger.info(f"\n  Processing: {det_file.name}")

        # Load detection results
        results = load_detection_results(det_file)
        abnormal_indices = results.get("abnormal_indices", {})

        # Process each category
        for category, traj_ids in abnormal_indices.items():
            if not traj_ids:
                continue

            # Initialize category if not seen before
            if category not in od_pairs_by_category:
                od_pairs_by_category[category] = set()

            logger.info(f"    {category}: {len(traj_ids)} abnormal trajectories")

            # Extract OD pairs for this category
            od_pairs = extract_od_pairs_from_trajectories(data_file, traj_ids)
            od_pairs_by_category[category].update(od_pairs)
            all_abnormal_traj_ids.update(traj_ids)

    # Convert sets to lists for JSON serialization
    od_pairs_output = {
        category: sorted(list(pairs))
        for category, pairs in od_pairs_by_category.items()
    }

    # Summary statistics
    total_unique_od_pairs = len(
        set().union(*[set(pairs) for pairs in od_pairs_output.values()])
    )

    logger.info("\n📊 Summary:")
    logger.info(f"  Total abnormal trajectories: {len(all_abnormal_traj_ids)}")
    logger.info(f"  Total unique OD pairs: {total_unique_od_pairs}")
    for category, pairs in od_pairs_output.items():
        logger.info(f"    {category}: {len(pairs)} OD pairs")

    return {
        "dataset": dataset_name,
        "total_abnormal_trajectories": len(all_abnormal_traj_ids),
        "total_unique_od_pairs": total_unique_od_pairs,
        "od_pairs_by_category": od_pairs_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract abnormal OD pairs from detection results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  uv run python tools/extract_abnormal_od_pairs.py \\
    --detection-results abnormal/BJUT_Beijing/train/real_data/detection_results.json \\
    --real-data data/BJUT_Beijing/train.csv \\
    --output abnormal_od_pairs.json

  # Multiple files (train + test)
  uv run python tools/extract_abnormal_od_pairs.py \\
    --detection-results abnormal/BJUT_Beijing/train/real_data/detection_results.json \\
                        abnormal/BJUT_Beijing/test/real_data/detection_results.json \\
    --real-data data/BJUT_Beijing/train.csv \\
                data/BJUT_Beijing/test.csv \\
    --dataset BJUT_Beijing \\
    --output abnormal_od_pairs_bjut.json
        """,
    )

    parser.add_argument(
        "--detection-results",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to detection results JSON file(s)",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to real trajectory CSV file(s)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BJUT_Beijing",
        help="Dataset name (default: BJUT_Beijing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.detection_results) != len(args.real_data):
        parser.error(
            "Number of detection-results files must match number of real-data files"
        )

    detection_files = [Path(f) for f in args.detection_results]
    data_files = [Path(f) for f in args.real_data]

    # Check files exist
    for f in detection_files + data_files:
        if not f.exists():
            parser.error(f"File not found: {f}")

    # Extract OD pairs
    result = extract_abnormal_od_pairs(
        detection_files,
        data_files,
        args.dataset,
    )

    # Save to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n✅ Saved abnormal OD pairs to {output_path}")


def extract_and_save_abnormal_od_pairs(
    detection_results_files: List[Path],
    real_data_files: List[Path],
    dataset_name: str,
    output_file: Path,
) -> Path:
    """
    Extract abnormal OD pairs and save to file (programmatic interface).

    Args:
        detection_results_files: List of detection result JSON files
        real_data_files: List of corresponding real data CSV files
        dataset_name: Name of the dataset
        output_file: Path to output JSON file

    Returns:
        Path to the saved output file

    Example:
        >>> from pathlib import Path
        >>> from tools.extract_abnormal_od_pairs import extract_and_save_abnormal_od_pairs
        >>>
        >>> output = extract_and_save_abnormal_od_pairs(
        ...     detection_results_files=[
        ...         Path("abnormal/train/real_data/detection_results.json"),
        ...         Path("abnormal/test/real_data/detection_results.json")
        ...     ],
        ...     real_data_files=[
        ...         Path("data/train.csv"),
        ...         Path("data/test.csv")
        ...     ],
        ...     dataset_name="porto_hoser",
        ...     output_file=Path("abnormal_od_pairs.json")
        ... )
    """
    # Validate inputs
    assert len(detection_results_files) == len(real_data_files), (
        f"Mismatch: {len(detection_results_files)} detection files vs {len(real_data_files)} data files"
    )

    for f in detection_results_files + real_data_files:
        assert f.exists(), f"File not found: {f}"

    # Extract OD pairs
    result = extract_abnormal_od_pairs(
        detection_results_files,
        real_data_files,
        dataset_name,
    )

    # Save to output file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n✅ Saved abnormal OD pairs to {output_file}")

    return output_file


if __name__ == "__main__":
    main()
