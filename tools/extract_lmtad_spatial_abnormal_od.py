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
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# Add parent directory to path for imports when run as script
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from tools.convert_to_lmtad_format import (  # noqa: E402
    extract_road_centroids,
    create_grid_mapper,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# EOS token ID used in LM-TAD trajectories
EOS_TOKEN = 6165


def build_reverse_grid_mapping(
    dataset: str,
    data_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[int, List[int]]], Optional[int]]:
    """Build mapping from grid token to list of road IDs and find EOS token"""
    # Map dataset name to config name
    config_dataset = dataset
    if dataset == "Beijing":
        config_dataset = "beijing_hoser_reference"

    # Find roadmap file
    roadmap_path = None
    if data_dir is not None:
        candidate = Path(data_dir) / "roadmap.geo"
        if candidate.exists():
            roadmap_path = candidate

    if roadmap_path is None:
        roadmap_path = Path("data") / dataset / "roadmap.geo"
    if not roadmap_path.exists():
        # Try alternative path for porto_hoser
        if dataset == "porto_hoser":
            roadmap_path = Path("data") / "porto_hoser" / "roadmap.geo"

    if not roadmap_path.exists():
        logger.warning(
            f"Roadmap not found at {roadmap_path}, cannot build grid mapping. Assuming IDs are road IDs."
        )
        return None, None

    logger.info(f"Building reverse grid mapping from {roadmap_path}...")

    try:
        # Extract centroids
        road_centroids, boundary = extract_road_centroids(roadmap_path)

        # Create mapper
        mapper, vocab = create_grid_mapper(config_dataset, road_centroids, boundary)

        # Inspect vocab
        logger.debug(f"Vocab size: {len(vocab)}")
        logger.debug(f"First 10 vocab keys: {list(vocab.keys())[:10]}")

        # Find EOS token
        eos_token = None
        for token, tid in vocab.items():
            if token in ["<eos>", "EOS", "[EOS]", "</s>"]:
                eos_token = tid
                break

        if eos_token is None:
            if dataset == "Beijing":
                eos_token = 51661
                logger.info(f"Using hardcoded EOS token for Beijing: {eos_token}")
            elif dataset == "porto_hoser":
                eos_token = 6165
                logger.info(f"Using hardcoded EOS token for Porto: {eos_token}")
            else:
                logger.warning("EOS token not found in vocab and no hardcoded default")
        else:
            logger.info(f"Found EOS token in vocab: {eos_token}")

        # Get forward mapping: road_id -> grid_token
        road_to_grid = mapper.map_all()

        # Build reverse mapping
        grid_to_roads = {}
        for road_id, grid_token in enumerate(road_to_grid):
            grid_token = int(grid_token)
            if grid_token not in grid_to_roads:
                grid_to_roads[grid_token] = []
            grid_to_roads[grid_token].append(road_id)

        logger.info(f"✅ Built reverse mapping for {len(grid_to_roads)} grid tokens")
        return grid_to_roads, eos_token
    except Exception as e:
        logger.error(f"Failed to build grid mapping: {e}")
        return None, None


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


def extract_od_from_trajectory(
    road_ids: List[int], eos_token: int = EOS_TOKEN
) -> Tuple[int, int]:
    """Extract origin and destination from trajectory

    Args:
        road_ids: List of road IDs (last element should be EOS token)
        eos_token: EOS token ID (default: global EOS_TOKEN)

    Returns:
        Tuple of (origin, destination)
    """
    if not road_ids:
        raise ValueError("Empty trajectory")

    # Remove EOS token if present
    if road_ids and road_ids[-1] == eos_token:
        road_ids = road_ids[:-1]

    if len(road_ids) < 1:
        raise ValueError("Trajectory has no road IDs (only EOS token)")

    # Origin is first road ID, destination is last road ID
    origin = road_ids[0]
    destination = road_ids[-1]

    return (origin, destination)


def extract_spatial_abnormal_od_pairs(
    tsv_file: Path,
    dataset: str,
    source_eval_dir: Path,
    data_dir: Optional[Path] = None,
    grid_to_roads: Optional[Dict[int, List[int]]] = None,
    eos_token: Optional[int] = None,
) -> Dict:
    """Extract OD pairs from LM-TAD-identified spatial outliers

    Args:
        tsv_file: Path to LM-TAD evaluation TSV file (or directory to process all TSV files)
        dataset: Dataset name
        source_eval_dir: Path to source evaluation directory (for metadata)
        grid_to_roads: Optional mapping from grid token to list of road IDs
        eos_token: Optional EOS token ID

    Returns:
        Dictionary with extracted OD pairs and metadata
    """
    # Use default EOS token if not provided
    if eos_token is None:
        eos_token = EOS_TOKEN

    # Auto-build grid mapping if not provided
    if grid_to_roads is None:
        grid_to_roads, inferred_eos = build_reverse_grid_mapping(
            dataset=dataset,
            data_dir=data_dir,
        )
        if eos_token == EOS_TOKEN and inferred_eos is not None:
            eos_token = inferred_eos

    if tsv_file.is_dir():
        # Prefer the canonical ckpt_best_outliers_* pattern used by LM-TAD
        tsv_files = sorted(tsv_file.glob("ckpt_best_outliers_*.tsv"))

        # Fallback: some LM-TAD evaluation exports use different naming
        # (e.g., final_model_outliers_config_*.tsv). Try broader patterns.
        if not tsv_files:
            logger.info(
                "🔎 No files matching 'ckpt_best_outliers_*.tsv' found, trying '*outliers*.tsv'"
            )
            tsv_files = sorted(tsv_file.glob("*outliers*.tsv"))

        if not tsv_files:
            logger.info("🔎 No '*outliers*.tsv' files found, trying any '*.tsv' files")
            tsv_files = sorted(tsv_file.glob("*.tsv"))

        if not tsv_files:
            raise FileNotFoundError(f"No TSV files found in {tsv_file}")

        logger.info(f"📂 Found {len(tsv_files)} TSV files, processing all...")
    else:
        tsv_files = [tsv_file]

    # Process all TSV files and combine results
    all_route_switch_od_pairs = set()
    all_detour_od_pairs = set()
    total_spatial_abnormal = 0
    total_failed = 0
    processed_configs = []
    per_file_stats = []  # Track statistics per file

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
        file_route_switch_od_pairs = set()  # Track unique OD pairs per file
        file_detour_od_pairs = set()  # Track unique OD pairs per file
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
                od_pair_tokens = extract_od_from_trajectory(
                    road_ids, eos_token=eos_token
                )

                # Map tokens to road IDs if mapping exists
                if grid_to_roads:
                    origin_token, dest_token = od_pair_tokens

                    if origin_token not in grid_to_roads:
                        if file_failed_count < 5:  # Log first few failures
                            logger.debug(f"Origin token {origin_token} not in grid map")
                        file_failed_count += 1
                        continue

                    if dest_token not in grid_to_roads:
                        if file_failed_count < 5:
                            logger.debug(
                                f"Destination token {dest_token} not in grid map"
                            )
                        file_failed_count += 1
                        continue

                    # Pick first road ID for each token
                    # Ideally we would pick the one closest to the center, but first is fine for now
                    origin_road = grid_to_roads[origin_token][0]
                    dest_road = grid_to_roads[dest_token][0]

                    od_pair = (origin_road, dest_road)
                else:
                    od_pair = od_pair_tokens

                # Add to appropriate category
                # Normalize outlier type (handle both "route switch" and "route switch outlier")
                if outlier_type in ["route switch", "route switch outlier"]:
                    all_route_switch_od_pairs.add(od_pair)
                    file_route_switch_od_pairs.add(od_pair)
                elif outlier_type in ["detour", "detour outlier"]:
                    all_detour_od_pairs.add(od_pair)
                    file_detour_od_pairs.add(od_pair)

            except Exception as e:
                logger.warning(
                    f"Failed to extract OD from trajectory {idx} in {tsv_file_path.name}: {e}"
                )
                file_failed_count += 1
                continue

        # Accumulate totals
        total_spatial_abnormal += len(spatial_outliers)
        total_failed += file_failed_count

        # Extract config name from filename
        config_name = tsv_file_path.stem.replace("ckpt_best_outliers_config_", "")
        processed_configs.append(config_name)

        # Track per-file statistics
        file_route_switch_trajectories = len(spatial_outliers[route_switch_mask])
        file_detour_trajectories = len(spatial_outliers[detour_mask])
        per_file_stats.append(
            {
                "tsv_file": tsv_file_path.name,
                "config": config_name,
                "total_trajectories": len(df),
                "spatial_abnormal_trajectories": len(spatial_outliers),
                "route_switch_trajectories": file_route_switch_trajectories,
                "route_switch_rate": (
                    file_route_switch_trajectories / len(df) * 100 if len(df) > 0 else 0
                ),
                "detour_trajectories": file_detour_trajectories,
                "detour_rate": (
                    file_detour_trajectories / len(df) * 100 if len(df) > 0 else 0
                ),
                "route_switch_od_pairs": len(file_route_switch_od_pairs),
                "detour_od_pairs": len(file_detour_od_pairs),
                "failed_extractions": file_failed_count,
            }
        )

        logger.info(
            f"  ✅ Extracted from {tsv_file_path.name}: "
            f"{len(file_route_switch_od_pairs)} route switch, {len(file_detour_od_pairs)} detour OD pairs"
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
                "processed_tsv_files": processed_configs,
                "num_tsv_files": len(tsv_files),
                "route_switch_od_pairs": 0,
                "detour_od_pairs": 0,
                "failed_extraction_count": total_failed,
            },
            "per_file_statistics": per_file_stats,  # May be empty if no outliers found
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
            "route_switch_od_pairs": len(all_route_switch_od_pairs),
            "detour_od_pairs": len(all_detour_od_pairs),
            "failed_extraction_count": total_failed,
        },
        "per_file_statistics": per_file_stats,  # Statistics per TSV file for baseline comparison
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
    --tsv-file /home/mka299/LMTAD/code/results/LMTAD/porto_hoser/.../eval/ckpt_best_outliers_config_ratio_0.05_level_3_prob_0.3.tsv \\
    --dataset porto_hoser \\
    --output abnormal_od_pairs_lmtad_spatial_porto_hoser.json

  # Auto-detect source eval directory
  uv run python tools/extract_lmtad_spatial_abnormal_od.py \\
    --tsv-file path/to/outliers.tsv \\
    --dataset porto_hoser \\
    --source-eval-dir /home/mka299/LMTAD/code/results/LMTAD/porto_hoser/.../eval \\
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

    # Build reverse grid mapping
    grid_to_roads, eos_token = build_reverse_grid_mapping(args.dataset)

    # Extract OD pairs
    try:
        result = extract_spatial_abnormal_od_pairs(
            tsv_file=args.tsv_file,
            dataset=args.dataset,
            source_eval_dir=args.source_eval_dir,
            grid_to_roads=grid_to_roads,
            eos_token=eos_token,
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
