#!/usr/bin/env python3
"""
Generate Trajectories for LM-TAD Spatial Abnormal OD Pairs

This script generates trajectories for spatial abnormal OD pairs extracted from
LM-TAD source evaluation, using HOSER models.

Usage:
    uv run python tools/generate_lmtad_spatial_abnormal_trajectories.py \\
        --od-pairs-file abnormal_od_pairs_lmtad_spatial_porto_hoser.json \\
        --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \\
        --dataset porto_hoser \\
        --seed 42 \\
        --num-trajectories-per-od 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gene import generate_trajectories_programmatic  # noqa: E402
from tools.model_detection import detect_model_files  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_od_pairs(od_pairs_file: Path) -> Dict:
    """Load OD pairs from JSON file

    Args:
        od_pairs_file: Path to OD pairs JSON file

    Returns:
        Dictionary with OD pairs by type
    """
    logger.info(f"📂 Loading OD pairs from {od_pairs_file}")
    with open(od_pairs_file, "r") as f:
        data = json.load(f)

    return data


def find_models(eval_dir: Path, dataset: str) -> List[Tuple[str, Path]]:
    """Find all model files in evaluation directory

    Args:
        eval_dir: Evaluation directory
        dataset: Dataset name

    Returns:
        List of (model_name, model_path) tuples
    """
    models_dir = eval_dir / "models"
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return []

    # Use model detection utility
    model_files = detect_model_files(
        models_dir,
        pattern="*.pth",
        require_model=True,
        require_od_type=False,
        recursive=False,
    )

    models = []
    for model_file in model_files:
        models.append((model_file.model, model_file.path))

    logger.info(f"✅ Found {len(models)} models: {', '.join([m[0] for m in models])}")
    return models


def generate_spatial_abnormal_trajectories(
    od_pairs_file: Path,
    eval_dir: Path,
    dataset: str,
    models: List[str],
    seed: int,
    num_traj_per_od: int = 50,
    cuda_device: int = 0,
    beam_search: bool = False,
    beam_width: int = 4,
) -> None:
    """Generate trajectories for spatial abnormal OD pairs

    Args:
        od_pairs_file: Path to OD pairs JSON file
        eval_dir: Evaluation directory
        dataset: Dataset name
        models: List of model names to generate for (if empty, auto-detect all)
        seed: Random seed
        num_traj_per_od: Number of trajectories to generate per OD pair
        cuda_device: CUDA device index
        beam_search: Use beam search (True) or A* search (False, default)
        beam_width: Beam width for beam search
    """
    # Load OD pairs
    od_pairs_data = load_od_pairs(od_pairs_file)

    # Combine all OD pairs from both types
    # Convert lists to tuples since JSON arrays are deserialized as lists
    # but sets require hashable types (tuples)
    all_od_pairs = []
    for od_type, pairs in od_pairs_data.get("od_pairs_by_type", {}).items():
        logger.info(f"  {od_type}: {len(pairs)} OD pairs")
        # Convert lists to tuples for hashability
        normalized_pairs = [
            tuple(pair) if isinstance(pair, list) else pair for pair in pairs
        ]
        all_od_pairs.extend(normalized_pairs)

    if not all_od_pairs:
        logger.warning("No OD pairs found in file")
        return

    # Deduplicate (now safe since all pairs are tuples)
    unique_od_pairs = list(set(all_od_pairs))
    logger.info(f"✅ Total unique OD pairs: {len(unique_od_pairs)}")

    # Find models
    if not models:
        model_list = find_models(eval_dir, dataset)
        if not model_list:
            logger.error("No models found")
            return
    else:
        models_dir = eval_dir / "models"
        model_list = []
        for model_name in models:
            model_file = detect_model_files(
                models_dir,
                pattern=f"*{model_name}*.pth",
                require_model=True,
                require_od_type=False,
                recursive=False,
            )
            if model_file:
                model_list.append((model_name, model_file[0].path))
            else:
                logger.warning(f"Model {model_name} not found, skipping")

    if not model_list:
        logger.error("No models to generate for")
        return

    # Output directory
    output_dir = eval_dir / "gene_abnormal_lmtad_spatial" / dataset / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand OD pairs (repeat each pair num_traj_per_od times)
    od_list_expanded = []
    for origin, dest in unique_od_pairs:
        for _ in range(num_traj_per_od):
            od_list_expanded.append((origin, dest))

    logger.info(f"📊 Will generate {len(od_list_expanded)} trajectories total")

    # Generate for each model
    for model_name, model_path in model_list:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"🚀 Generating with {model_name}")
        logger.info(f"{'=' * 70}")

        output_file = output_dir / f"{model_name}_spatial_abnormal.csv"

        # Check if already exists
        if output_file.exists():
            logger.info(f"  ⏭️  File already exists: {output_file.name}, skipping")
            continue

        try:
            result = generate_trajectories_programmatic(
                dataset=dataset,
                model_path=str(model_path),
                od_pairs=od_list_expanded,
                output_file=str(output_file),
                seed=seed,
                cuda_device=cuda_device,
                beam_search=beam_search,
                beam_width=beam_width,
                enable_wandb=False,
                wandb_project=None,
                wandb_run_name=None,
                wandb_tags=None,
                model_type=model_name,
            )

            if result.get("output_file"):
                traj_count = result.get("num_generated", 0)
                logger.info(
                    f"  ✅ Generated {traj_count} trajectories → {result['output_file']}"
                )
            else:
                logger.error("  ❌ Generation failed: No output file produced")

        except Exception as e:
            logger.error(f"  ❌ Error generating with {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info(f"\n✅ Generation complete! Results in {output_dir}/")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate trajectories for LM-TAD spatial abnormal OD pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all models
  uv run python tools/generate_lmtad_spatial_abnormal_trajectories.py \\
    --od-pairs-file abnormal_od_pairs_lmtad_spatial_porto_hoser.json \\
    --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \\
    --dataset porto_hoser \\
    --seed 42

  # Generate for specific models
  uv run python tools/generate_lmtad_spatial_abnormal_trajectories.py \\
    --od-pairs-file abnormal_od_pairs_lmtad_spatial_porto_hoser.json \\
    --eval-dir eval_dir \\
    --dataset porto_hoser \\
    --models vanilla,distill_phase1 \\
    --seed 42
        """,
    )

    parser.add_argument(
        "--od-pairs-file",
        type=Path,
        required=True,
        help="Path to OD pairs JSON file",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., porto_hoser, Beijing)",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names (auto-detect all if not provided)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-trajectories-per-od",
        type=int,
        default=50,
        help="Number of trajectories to generate per OD pair (default: 50)",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--beam-search",
        action="store_true",
        default=False,
        help="Use beam search (default: False, uses A* search)",
    )
    parser.add_argument(
        "--no-beam-search",
        dest="beam_search",
        action="store_false",
        help="Use A* search instead of beam search (default)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for beam search (default: 4)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.od_pairs_file.exists():
        logger.error(f"OD pairs file not found: {args.od_pairs_file}")
        return 1

    if not args.eval_dir.exists():
        logger.error(f"Evaluation directory not found: {args.eval_dir}")
        return 1

    # Parse models list
    models = []
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    # Generate trajectories
    try:
        generate_spatial_abnormal_trajectories(
            od_pairs_file=args.od_pairs_file,
            eval_dir=args.eval_dir,
            dataset=args.dataset,
            models=models,
            seed=args.seed,
            num_traj_per_od=args.num_trajectories_per_od,
            cuda_device=args.cuda_device,
            beam_search=args.beam_search,
            beam_width=args.beam_width,
        )
        return 0

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
