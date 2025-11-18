#!/usr/bin/env python3
"""
Extract Spatial Abnormal OD Pairs from LM-TAD Source Evaluation

This script extracts origin-destination pairs from LM-TAD-identified spatial
abnormalities (route switch and detour outliers) in the source dataset evaluation.

Usage:
    uv run python tools/extract_lmtad_spatial_abnormal_od.py \\
        --tsv-file /path/to/ckpt_best_outliers_*.tsv \\
        --dataset porto_hoser \\
        --output abnormal_od_pairs_lmtad_spatial_porto_hoser.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# EOS token ID used in LM-TAD trajectories
EOS_TOKEN = 6165


def parse_trajectory_from_tsv(trajectory_str: str) -> List[int]:
    """Parse trajectory from TSV trajectory column (JSON array string)

    Args:
        trajectory_str: String representation of trajectory (e.g., "[74, 74, 208, ..., 6165]")

    Returns:
        List of road IDs
    """
    try:
        # Handle both string representation and actual JSON
        if isinstance(trajectory_str, str):
            # Remove whitespace and parse as JSON
            trajectory_str = trajectory_str.strip()
            if trajectory_str.startswith("["):
                trajectory = json.loads(trajectory_str)
            else:
                # Try parsing as comma-separated values
                trajectory = [
                    int(x.strip()) for x in trajectory_str.split(",") if x.strip()
                ]
        else:
            # Already a list
            trajectory = list(trajectory_str)

        return [int(x) for x in trajectory if x is not None]
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(
            f"Failed to parse trajectory: {trajectory_str[:50]}... Error: {e}"
        )
        return []


def extract_od_from_trajectory(road_ids: List[int]) -> Tuple[int, int]:
    """Extract origin and destination from trajectory

    Args:
        road_ids: List of road IDs (last element should be EOS token 6165)

    Returns:
        Tuple of (origin, destination)
    """
    if not road_ids:
        raise ValueError("Empty trajectory")

    # Remove EOS token if present
    if road_ids and road_ids[-1] == EOS_TOKEN:
        road_ids = road_ids[:-1]

    if len(road_ids) < 1:
        raise ValueError("Trajectory has no road IDs (only EOS token)")

    # Origin is first road ID, destination is last road ID
    origin = road_ids[0]
    destination = road_ids[-1]

    return (origin, destination)


def extract_spatial_abnormal_od_pairs(
    tsv_file: Path, dataset: str, source_eval_dir: Path
) -> Dict:
    """Extract OD pairs from LM-TAD-identified spatial outliers

    Args:
        tsv_file: Path to LM-TAD evaluation TSV file (or directory to process all TSV files)
        dataset: Dataset name
        source_eval_dir: Path to source evaluation directory (for metadata)

    Returns:
        Dictionary with extracted OD pairs and metadata
    """
    # If tsv_file is a directory, process all TSV files in it
    if tsv_file.is_dir():
        tsv_files = sorted(tsv_file.glob("ckpt_best_outliers_*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"No TSV files found in {tsv_file}")
        logger.info(f"📂 Found {len(tsv_files)} TSV files, processing all...")
    else:
        tsv_files = [tsv_file]

    # Process all TSV files and combine results
    all_route_switch_od_pairs = set()
    all_detour_od_pairs = set()
    total_spatial_abnormal = 0
    total_route_switch = 0
    total_detour = 0
    total_failed = 0
    processed_configs = []

    for tsv_file_path in tsv_files:
        logger.info(f"📂 Reading TSV file: {tsv_file_path.name}")

        # Read TSV file
        try:
            df = pd.read_csv(tsv_file_path, sep="\t")
        except Exception as e:
            logger.error(f"Failed to read TSV file {tsv_file_path}: {e}")
            continue

        logger.info(f"✅ Loaded {len(df)} trajectories from {tsv_file_path.name}")

        # Filter for spatial abnormalities
        # Handle both formats: "route switch"/"detour" and "route switch outlier"/"detour outlier"
        spatial_outliers = df[
            df["outlier"].isin(
                ["route switch", "detour", "route switch outlier", "detour outlier"]
            )
        ].copy()

        if len(spatial_outliers) == 0:
            logger.warning(f"No spatial outliers found in {tsv_file_path.name}")
            continue

        route_switch_mask = spatial_outliers["outlier"].isin(
            ["route switch", "route switch outlier"]
        )
        detour_mask = spatial_outliers["outlier"].isin(["detour", "detour outlier"])
        logger.info(
            f"🔍 Found {len(spatial_outliers)} spatial abnormal trajectories "
            f"(route switch: {len(spatial_outliers[route_switch_mask])}, "
            f"detour: {len(spatial_outliers[detour_mask])})"
        )

        # Extract OD pairs from this file
        file_route_switch_count = 0
        file_detour_count = 0
        file_failed_count = 0

        for idx, row in spatial_outliers.iterrows():
            outlier_type = row["outlier"]
            trajectory_str = row["trajectory"]

            try:
                # Parse trajectory
                road_ids = parse_trajectory_from_tsv(trajectory_str)

                if not road_ids:
                    file_failed_count += 1
                    continue

                # Extract OD pair
                od_pair = extract_od_from_trajectory(road_ids)

                # Add to appropriate category
                # Normalize outlier type (handle both "route switch" and "route switch outlier")
                if outlier_type in ["route switch", "route switch outlier"]:
                    all_route_switch_od_pairs.add(od_pair)
                    file_route_switch_count += 1
                elif outlier_type in ["detour", "detour outlier"]:
                    all_detour_od_pairs.add(od_pair)
                    file_detour_count += 1

            except Exception as e:
                logger.warning(
                    f"Failed to extract OD from trajectory {idx} in {tsv_file_path.name}: {e}"
                )
                file_failed_count += 1
                continue

        # Accumulate totals
        total_spatial_abnormal += len(spatial_outliers)
        total_route_switch += file_route_switch_count
        total_detour += file_detour_count
        total_failed += file_failed_count

        # Extract config name from filename
        config_name = tsv_file_path.stem.replace("ckpt_best_outliers_config_", "")
        processed_configs.append(config_name)

        logger.info(
            f"  ✅ Extracted from {tsv_file_path.name}: "
            f"{file_route_switch_count} route switch, {file_detour_count} detour OD pairs"
        )

    # Combine all results
    if len(all_route_switch_od_pairs) == 0 and len(all_detour_od_pairs) == 0:
        logger.warning("No spatial outliers found in any TSV file")
        return {
            "dataset": dataset,
            "source": "lmtad",
            "lmtad_config": ",".join(processed_configs)
            if processed_configs
            else "none",
            "total_spatial_abnormal_trajectories": 0,
            "total_unique_od_pairs": 0,
            "od_pairs_by_type": {"route_switch": [], "detour": []},
            "metadata": {
                "source_eval_dir": str(source_eval_dir),
                "route_switch_count": 0,
                "detour_count": 0,
                "processed_tsv_files": processed_configs,
            },
        }

    # Convert sets to sorted lists for JSON serialization
    od_pairs_by_type = {
        "route_switch": sorted(list(all_route_switch_od_pairs)),
        "detour": sorted(list(all_detour_od_pairs)),
    }

    total_unique_od_pairs = len(od_pairs_by_type["route_switch"]) + len(
        od_pairs_by_type["detour"]
    )

    result = {
        "dataset": dataset,
        "source": "lmtad",
        "lmtad_config": ",".join(processed_configs),
        "total_spatial_abnormal_trajectories": total_spatial_abnormal,
        "total_unique_od_pairs": total_unique_od_pairs,
        "od_pairs_by_type": od_pairs_by_type,
        "metadata": {
            "source_eval_dir": str(source_eval_dir),
            "processed_tsv_files": processed_configs,
            "num_tsv_files": len(tsv_files),
            "route_switch_count": total_route_switch,
            "detour_count": total_detour,
            "failed_extraction_count": total_failed,
        },
    }

    logger.info(f"\n✅ Combined extraction from {len(tsv_files)} TSV files:")
    logger.info(f"   Total spatial abnormal trajectories: {total_spatial_abnormal}")
    logger.info(f"   Total unique OD pairs: {total_unique_od_pairs}")
    logger.info(f"   Route switch OD pairs: {len(od_pairs_by_type['route_switch'])}")
    logger.info(f"   Detour OD pairs: {len(od_pairs_by_type['detour'])}")

    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract spatial abnormal OD pairs from LM-TAD source evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from Porto evaluation
  uv run python tools/extract_lmtad_spatial_abnormal_od.py \\
    --tsv-file /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/.../eval/ckpt_best_outliers_config_ratio_0.05_level_3_prob_0.3.tsv \\
    --dataset porto_hoser \\
    --output abnormal_od_pairs_lmtad_spatial_porto_hoser.json

  # Auto-detect source eval directory
  uv run python tools/extract_lmtad_spatial_abnormal_od.py \\
    --tsv-file path/to/outliers.tsv \\
    --dataset porto_hoser \\
    --source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/.../eval \\
    --output abnormal_od_pairs_lmtad_spatial_porto_hoser.json
        """,
    )

    parser.add_argument(
        "--tsv-file",
        type=Path,
        required=True,
        help="Path to LM-TAD evaluation TSV file (ckpt_best_outliers_*.tsv)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., porto_hoser, Beijing)",
    )
    parser.add_argument(
        "--source-eval-dir",
        type=Path,
        required=True,
        help="Path to LM-TAD source evaluation directory (for metadata)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.tsv_file.exists():
        logger.error(f"TSV file/directory not found: {args.tsv_file}")
        return 1

    if not args.source_eval_dir.exists():
        logger.error(f"Source eval directory not found: {args.source_eval_dir}")
        return 1

    # If tsv_file is a file, validate it's a TSV file
    if args.tsv_file.is_file() and not args.tsv_file.name.endswith(".tsv"):
        logger.warning(f"File {args.tsv_file} doesn't appear to be a TSV file")

    # Extract OD pairs
    try:
        result = extract_spatial_abnormal_od_pairs(
            tsv_file=args.tsv_file,
            dataset=args.dataset,
            source_eval_dir=args.source_eval_dir,
        )

        # Save to JSON
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"✅ Saved OD pairs to {args.output}")
        logger.info(f"   Total unique OD pairs: {result['total_unique_od_pairs']}")
        logger.info(
            f"   Route switch: {len(result['od_pairs_by_type']['route_switch'])}"
        )
        logger.info(f"   Detour: {len(result['od_pairs_by_type']['detour'])}")

        return 0

    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
