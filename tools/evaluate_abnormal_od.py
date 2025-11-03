#!/usr/bin/env python3
"""
Evaluate Model Performance on Abnormal OD Pairs

This script evaluates how well models perform on trajectories generated for
abnormal OD pairs, comparing against real abnormal trajectories from the
cross-dataset.

Usage:
    uv run python tools/evaluate_abnormal_od.py \
        --generated-dir gene_abnormal/Beijing/seed42 \
        --real-abnormal-file data/BJUT_Beijing/train.csv \
        --abnormal-od-pairs abnormal_od_pairs_bjut.json \
        --output-dir eval_abnormal/Beijing
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import evaluate_trajectories_programmatic
from tools.analyze_abnormal import run_abnormal_analysis

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


def evaluate_abnormal_od_trajectories(
    generated_dir: Path,
    real_abnormal_file: Path,
    abnormal_od_data: dict,
    output_dir: Path,
    dataset: str,
):
    """Evaluate generated trajectories for abnormal OD pairs

    Args:
        generated_dir: Directory with generated trajectories
        real_abnormal_file: Real data file with abnormal trajectories
        abnormal_od_data: Dictionary with abnormal OD pairs by category
        output_dir: Directory to save evaluation results
        dataset: Dataset name
    """
    logger.info("\n🔍 Evaluating Abnormal OD Performance")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Generated trajectories: {generated_dir}")
    logger.info(f"  Real abnormal data: {real_abnormal_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all generated files
    generated_files = list(generated_dir.glob("*_abnormal_od.csv"))
    if not generated_files:
        raise FileNotFoundError(f"No generated files found in {generated_dir}")

    logger.info(f"\n📦 Found {len(generated_files)} generated files:")
    for gf in generated_files:
        logger.info(f"  - {gf.name}")

    # Results storage
    all_results = {}

    # Evaluate each model
    for gen_file in generated_files:
        model_name = gen_file.stem.replace("_abnormal_od", "")
        logger.info(f"\n{'=' * 70}")
        logger.info(f"📊 Evaluating model: {model_name}")
        logger.info(f"{'=' * 70}")

        try:
            # Step 1: Run abnormal detection on generated trajectories
            logger.info("\n  🔍 Step 1: Detecting abnormalities in generated data...")
            detection_output = output_dir / model_name / "detection"
            detection_output.mkdir(parents=True, exist_ok=True)

            detection_results = run_abnormal_analysis(
                real_file=gen_file,
                dataset=dataset,
                config_path=Path("config/abnormal_detection.yaml"),
                output_dir=detection_output,
            )

            # Calculate abnormality rates by category
            abnormal_by_category = {}
            total_traj = detection_results.get("total_trajectories", 0)

            for category, indices in detection_results.get(
                "abnormal_indices", {}
            ).items():
                count = len(indices)
                rate = (count / total_traj * 100) if total_traj > 0 else 0
                abnormal_by_category[category] = {
                    "count": count,
                    "rate": rate,
                }
                logger.info(f"    {category}: {count}/{total_traj} ({rate:.2f}%)")

            # Step 2: Run trajectory evaluation metrics
            logger.info("\n  📏 Step 2: Computing trajectory similarity metrics...")
            eval_output = output_dir / model_name / "metrics"
            eval_output.mkdir(parents=True, exist_ok=True)

            eval_results = evaluate_trajectories_programmatic(
                real_file=str(real_abnormal_file),
                generated_file=str(gen_file),
                dataset=dataset,
                output_dir=str(eval_output),
            )

            # Extract key metrics
            metrics = {}
            if eval_results.get("status") == "success":
                eval_data = eval_results.get("results", {})
                metrics = {
                    "edr": eval_data.get("edr", 0.0),
                    "dtw": eval_data.get("dtw", 0.0),
                    "hausdorff": eval_data.get("hausdorff", 0.0),
                }
                logger.info(f"    EDR: {metrics['edr']:.4f}")
                logger.info(f"    DTW: {metrics['dtw']:.4f}")
                logger.info(f"    Hausdorff: {metrics['hausdorff']:.4f}")
            else:
                logger.warning(
                    f"    ⚠️  Evaluation failed: {eval_results.get('message')}"
                )

            # Compile results
            model_results = {
                "model": model_name,
                "generated_file": str(gen_file),
                "total_trajectories": total_traj,
                "abnormality_detection": abnormal_by_category,
                "similarity_metrics": metrics,
            }

            all_results[model_name] = model_results

            # Save individual model results
            model_result_file = output_dir / model_name / "abnormal_od_evaluation.json"
            with open(model_result_file, "w") as f:
                json.dump(model_results, f, indent=2)

            logger.info(f"\n  ✅ Saved results to {model_result_file}")

        except Exception as e:
            logger.error(f"  ❌ Error evaluating {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Step 3: Generate comparison report
    logger.info(f"\n{'=' * 70}")
    logger.info("📊 Comparison Report: Abnormal OD Performance")
    logger.info(f"{'=' * 70}")

    # Compare abnormality rates across models
    logger.info("\n  Abnormality Reproduction Rates:")
    for model_name, results in all_results.items():
        total_abnormal = sum(
            cat_data["count"] for cat_data in results["abnormality_detection"].values()
        )
        total_traj = results["total_trajectories"]
        overall_rate = (total_abnormal / total_traj * 100) if total_traj > 0 else 0
        logger.info(
            f"    {model_name:15s}: {total_abnormal:4d}/{total_traj:4d} ({overall_rate:5.2f}%)"
        )

    # Compare similarity metrics
    logger.info("\n  Trajectory Similarity (vs real abnormal data):")
    for model_name, results in all_results.items():
        metrics = results["similarity_metrics"]
        logger.info(
            f"    {model_name:15s}: EDR={metrics.get('edr', 0):.4f} | "
            f"DTW={metrics.get('dtw', 0):.4f} | "
            f"Hausdorff={metrics.get('hausdorff', 0):.4f}"
        )

    # Save comprehensive comparison report
    comparison_report = {
        "dataset": dataset,
        "real_abnormal_file": str(real_abnormal_file),
        "abnormal_od_pairs": abnormal_od_data.get("total_unique_od_pairs", 0),
        "model_results": all_results,
    }

    comparison_file = output_dir / "comparison_report.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"\n✅ Saved comparison report to {comparison_file}")
    logger.info(f"✅ Evaluation complete! Results in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on abnormal OD pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate generated abnormal OD trajectories
  uv run python tools/evaluate_abnormal_od.py \\
    --generated-dir gene_abnormal/Beijing/seed42 \\
    --real-abnormal-file data/BJUT_Beijing/train.csv \\
    --abnormal-od-pairs abnormal_od_pairs_bjut.json \\
    --output-dir eval_abnormal/Beijing
        """,
    )

    parser.add_argument(
        "--generated-dir",
        type=str,
        required=True,
        help="Directory containing generated abnormal OD trajectories",
    )
    parser.add_argument(
        "--real-abnormal-file",
        type=str,
        required=True,
        help="Real data file with abnormal trajectories (for comparison)",
    )
    parser.add_argument(
        "--abnormal-od-pairs",
        type=str,
        required=True,
        help="Path to abnormal OD pairs JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        help="Dataset name (default: Beijing)",
    )

    args = parser.parse_args()

    # Validate inputs
    generated_dir = Path(args.generated_dir)
    if not generated_dir.exists():
        parser.error(f"Generated directory not found: {generated_dir}")

    real_abnormal_file = Path(args.real_abnormal_file)
    if not real_abnormal_file.exists():
        parser.error(f"Real abnormal file not found: {real_abnormal_file}")

    od_pairs_file = Path(args.abnormal_od_pairs)
    if not od_pairs_file.exists():
        parser.error(f"OD pairs file not found: {od_pairs_file}")

    # Load abnormal OD pairs data
    abnormal_od_data = load_abnormal_od_pairs(od_pairs_file)

    # Run evaluation
    output_dir = Path(args.output_dir)

    evaluate_abnormal_od_trajectories(
        generated_dir=generated_dir,
        real_abnormal_file=real_abnormal_file,
        abnormal_od_data=abnormal_od_data,
        output_dir=output_dir,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
