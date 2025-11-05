#!/usr/bin/env python3
"""
Generate Trajectories for Abnormal OD Pairs

This script generates trajectories for specific origin-destination pairs that
were identified as abnormal in the cross-dataset analysis. This tests whether
models can handle challenging edge cases.

Usage:
    uv run python tools/generate_abnormal_od.py \
        --od-pairs abnormal_od_pairs_bjut.json \
        --model-dir hoser-distill-optuna-6/models \
        --output-dir gene_abnormal/Beijing/seed42 \
        --num-traj 100 \
        --seed 42
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gene import generate_trajectories_programmatic

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_abnormal_od_pairs(od_pairs_file: Path) -> dict:
    """Load abnormal OD pairs from JSON file"""
    logger.info(f"📂 Loading abnormal OD pairs from {od_pairs_file}")
    with open(od_pairs_file, "r") as f:
        data = json.load(f)
    return data


def create_od_pair_list(
    od_pairs_data: dict, max_pairs_per_category: int = None
) -> List[Tuple[int, int]]:
    """Create flat list of OD pairs from categorized data

    Args:
        od_pairs_data: Dictionary with OD pairs by category
        max_pairs_per_category: Optional limit on pairs per category

    Returns:
        List of (origin, destination) tuples
    """
    all_pairs = []
    od_pairs_by_cat = od_pairs_data.get("od_pairs_by_category", {})

    for category, pairs in od_pairs_by_cat.items():
        if not pairs:
            continue

        # Limit pairs per category if specified
        if max_pairs_per_category:
            pairs = pairs[:max_pairs_per_category]

        logger.info(f"  {category}: {len(pairs)} OD pairs")
        all_pairs.extend(pairs)

    # Deduplicate while preserving order
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        pair_tuple = tuple(pair)
        if pair_tuple not in seen:
            seen.add(pair_tuple)
            unique_pairs.append(pair_tuple)

    logger.info(f"✅ Total unique OD pairs: {len(unique_pairs)}")
    return unique_pairs


def generate_for_abnormal_od_pairs(
    od_pairs: List[Tuple[int, int]],
    model_dir: Path,
    output_dir: Path,
    dataset: str,
    num_traj_per_od: int,
    seed: int,
    cuda_device: int = 0,
):
    """Generate trajectories for abnormal OD pairs using all models

    Args:
        od_pairs: List of (origin, destination) tuples
        model_dir: Directory containing model files
        output_dir: Directory to save generated trajectories
        dataset: Dataset name (e.g., "Beijing")
        num_traj_per_od: Number of trajectories to generate per OD pair
        seed: Random seed
        cuda_device: CUDA device ID
    """
    logger.info(f"\n🤖 Generating trajectories for {len(od_pairs)} abnormal OD pairs")
    logger.info(f"  {num_traj_per_od} trajectories per OD pair")
    logger.info(f"  Total trajectories to generate: {len(od_pairs) * num_traj_per_od}")

    # Find all model files
    model_files = list(model_dir.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    logger.info(f"\n📦 Found {len(model_files)} models:")
    for mf in model_files:
        logger.info(f"  - {mf.name}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each model
    for model_file in model_files:
        model_name = model_file.stem
        logger.info(f"\n{'=' * 70}")
        logger.info(f"🚀 Generating with model: {model_name}")
        logger.info(f"{'=' * 70}")

        try:
            # Prepare OD pairs in format expected by gene.py
            # Replicate each OD pair num_traj_per_od times
            od_list_expanded = []
            for origin, dest in od_pairs:
                for _ in range(num_traj_per_od):
                    od_list_expanded.append((origin, dest))

            logger.info(f"  Generating {len(od_list_expanded)} trajectories...")

            # Call generation function
            output_file = output_dir / f"{model_name}_abnormal_od.csv"

            result = generate_trajectories_programmatic(
                model_path=str(model_file),
                dataset=dataset,
                num_generate=len(od_list_expanded),
                od_list=od_list_expanded,  # Use specific OD pairs
                output_file=str(output_file),
                seed=seed,
                cuda_device=cuda_device,
                beam_search=True,  # Use beam search by default (4-width)
                beam_width=4,
            )

            if result.get("output_file"):
                traj_count = result.get("num_generated", 0)
                logger.info(f"  ✅ Generated {traj_count} trajectories")
                logger.info(f"  💾 Saved to {result['output_file']}")
            else:
                logger.error("  ❌ Generation failed: No output file produced")

        except Exception as e:
            logger.error(f"  ❌ Error generating with {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info(f"\n✅ Abnormal OD generation complete! Results in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectories for abnormal OD pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all abnormal OD pairs
  uv run python tools/generate_abnormal_od.py \\
    --od-pairs abnormal_od_pairs_bjut.json \\
    --model-dir hoser-distill-optuna-6/models \\
    --output-dir gene_abnormal/Beijing/seed42 \\
    --num-traj 50 \\
    --seed 42

  # Limit to 10 pairs per category
  uv run python tools/generate_abnormal_od.py \\
    --od-pairs abnormal_od_pairs_bjut.json \\
    --model-dir hoser-distill-optuna-6/models \\
    --output-dir gene_abnormal/Beijing/seed42 \\
    --num-traj 100 \\
    --max-pairs 10 \\
    --seed 42
        """,
    )

    parser.add_argument(
        "--od-pairs",
        type=str,
        required=True,
        help="Path to abnormal OD pairs JSON file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model files (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated trajectories",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        help="Dataset name for road network (default: Beijing)",
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=100,
        help="Number of trajectories to generate per OD pair (default: 100)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum OD pairs per category (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device ID (default: 0)",
    )

    args = parser.parse_args()

    # Load OD pairs
    od_pairs_file = Path(args.od_pairs)
    if not od_pairs_file.exists():
        parser.error(f"OD pairs file not found: {od_pairs_file}")

    od_pairs_data = load_abnormal_od_pairs(od_pairs_file)

    # Create flat list of OD pairs
    od_pairs = create_od_pair_list(od_pairs_data, args.max_pairs)

    if not od_pairs:
        logger.error("❌ No OD pairs found in input file")
        return

    # Validate model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        parser.error(f"Model directory not found: {model_dir}")

    # Generate trajectories
    output_dir = Path(args.output_dir)

    generate_for_abnormal_od_pairs(
        od_pairs=od_pairs,
        model_dir=model_dir,
        output_dir=output_dir,
        dataset=args.dataset,
        num_traj_per_od=args.num_traj,
        seed=args.seed,
        cuda_device=args.cuda,
    )


if __name__ == "__main__":
    main()
