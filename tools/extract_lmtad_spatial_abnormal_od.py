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
from typing import Dict, List, Tuple, Set

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
        tsv_file: Path to LM-TAD evaluation TSV file
        dataset: Dataset name
        source_eval_dir: Path to source evaluation directory (for metadata)

    Returns:
        Dictionary with extracted OD pairs and metadata
    """
    logger.info(f"📂 Reading TSV file: {tsv_file}")

    # Read TSV file
    try:
        df = pd.read_csv(tsv_file, sep="\t")
    except Exception as e:
        logger.error(f"Failed to read TSV file: {e}")
        raise

    logger.info(f"✅ Loaded {len(df)} trajectories from TSV")

    # Filter for spatial abnormalities
    spatial_outliers = df[df["outlier"].isin(["route switch", "detour"])].copy()

    if len(spatial_outliers) == 0:
        logger.warning("No spatial outliers found in TSV file")
        return {
            "dataset": dataset,
            "source": "lmtad",
            "lmtad_config": tsv_file.stem.replace("ckpt_best_outliers_config_", ""),
            "total_spatial_abnormal_trajectories": 0,
            "total_unique_od_pairs": 0,
            "od_pairs_by_type": {"route_switch": [], "detour": []},
            "metadata": {
                "source_eval_dir": str(source_eval_dir),
                "route_switch_count": 0,
                "detour_count": 0,
            },
        }

    logger.info(
        f"🔍 Found {len(spatial_outliers)} spatial abnormal trajectories "
        f"(route switch: {len(spatial_outliers[spatial_outliers['outlier'] == 'route switch'])}, "
        f"detour: {len(spatial_outliers[spatial_outliers['outlier'] == 'detour'])})"
    )

    # Extract OD pairs by type
    od_pairs_by_type: Dict[str, Set[Tuple[int, int]]] = {
        "route_switch": set(),
        "detour": set(),
    }

    route_switch_count = 0
    detour_count = 0
    failed_count = 0

    for idx, row in spatial_outliers.iterrows():
        outlier_type = row["outlier"]
        trajectory_str = row["trajectory"]

        try:
            # Parse trajectory
            road_ids = parse_trajectory_from_tsv(trajectory_str)

            if not road_ids:
                failed_count += 1
                continue

            # Extract OD pair
            od_pair = extract_od_from_trajectory(road_ids)

            # Add to appropriate category
            if outlier_type == "route switch":
                od_pairs_by_type["route_switch"].add(od_pair)
                route_switch_count += 1
            elif outlier_type == "detour":
                od_pairs_by_type["detour"].add(od_pair)
                detour_count += 1

        except Exception as e:
            logger.warning(f"Failed to extract OD from trajectory {idx}: {e}")
            failed_count += 1
            continue

    if failed_count > 0:
        logger.warning(f"⚠️  Failed to extract OD from {failed_count} trajectories")

    # Convert sets to lists for JSON serialization
    od_pairs_output = {
        "route_switch": sorted(list(od_pairs_by_type["route_switch"])),
        "detour": sorted(list(od_pairs_by_type["detour"])),
    }

    total_unique_od_pairs = len(od_pairs_by_type["route_switch"]) + len(
        od_pairs_by_type["detour"]
    )

    # Extract config name from filename
    config_name = tsv_file.stem.replace("ckpt_best_outliers_config_", "")

    result = {
        "dataset": dataset,
        "source": "lmtad",
        "lmtad_config": config_name,
        "total_spatial_abnormal_trajectories": len(spatial_outliers),
        "total_unique_od_pairs": total_unique_od_pairs,
        "od_pairs_by_type": od_pairs_output,
        "metadata": {
            "source_eval_dir": str(source_eval_dir),
            "route_switch_count": route_switch_count,
            "detour_count": detour_count,
            "failed_extraction_count": failed_count,
        },
    }

    logger.info(f"✅ Extracted {total_unique_od_pairs} unique OD pairs:")
    logger.info(f"   Route switch: {len(od_pairs_by_type['route_switch'])}")
    logger.info(f"   Detour: {len(od_pairs_by_type['detour'])}")

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
        logger.error(f"TSV file not found: {args.tsv_file}")
        return 1

    if not args.source_eval_dir.exists():
        logger.error(f"Source eval directory not found: {args.source_eval_dir}")
        return 1

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
