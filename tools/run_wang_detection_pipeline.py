#!/usr/bin/env python3
"""
Wang Statistical Detection Pipeline - Complete Workflow

This script runs the full Wang statistical abnormality detection pipeline on an
evaluation directory, including real data, generated models, aggregation, and
visualization generation.

Usage:
    # Run on evaluation directory
    uv run python tools/run_wang_detection_pipeline.py \\
        --eval-dir hoser-distill-optuna-porto-eval-xyz \\
        --dataset porto_hoser

    # Skip visualization generation (faster)
    uv run python tools/run_wang_detection_pipeline.py \\
        --eval-dir eval_dir \\
        --dataset porto_hoser \\
        --skip-visualization

    # Only run specific steps
    uv run python tools/run_wang_detection_pipeline.py \\
        --eval-dir eval_dir \\
        --dataset porto_hoser \\
        --skip-real \\
        --skip-generated
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports when run as script
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import shared model detection utility (after path setup)
from tools.model_detection import detect_model_files  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status

    Args:
        cmd: Command to run as list of strings
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"{'=' * 70}")
    logger.info(f"Step: {description}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed: {e}")
        return False


def find_generated_models(eval_dir: Path, dataset: str) -> Dict[str, Path]:
    """Find all generated model files using shared detection utility

    Args:
        eval_dir: Evaluation directory
        dataset: Dataset name

    Returns:
        Dict mapping model_name -> file path for test split files
    """
    gene_dir = eval_dir / "gene" / dataset / "seed42"

    if not gene_dir.exists():
        logger.warning(f"Gene directory not found: {gene_dir}")
        return {}

    # Use shared model detection utility
    model_files = detect_model_files(
        gene_dir,
        pattern="*_test.csv",  # Only test files for Wang detection
        require_model=True,
        require_od_type=True,
        recursive=False,
    )

    # Build dict of model_name -> file path
    models = {}
    for model_file in model_files:
        if model_file.od_type == "test":
            models[model_file.model] = model_file.path

    logger.info(
        f"Found {len(models)} generated models: {', '.join(sorted(models.keys()))}"
    )
    return models


def run_wang_detection_pipeline(
    eval_dir: Path,
    dataset: str,
    skip_real: bool = False,
    skip_generated: bool = False,
    skip_aggregation: bool = False,
    skip_visualization: bool = False,
) -> bool:
    """Run complete Wang detection pipeline

    Args:
        eval_dir: Evaluation directory path
        dataset: Dataset name (e.g., porto_hoser, Beijing)
        skip_real: Skip real data detection
        skip_generated: Skip generated data detection
        skip_aggregation: Skip result aggregation
        skip_visualization: Skip visualization generation

    Returns:
        True if all steps successful, False otherwise
    """
    project_root = Path(__file__).parent.parent
    eval_dir = Path(eval_dir)

    if not eval_dir.is_absolute():
        eval_dir = project_root / eval_dir

    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        return False

    # Paths
    config_file = eval_dir / "config" / "abnormal_detection_statistical.yaml"
    abnormal_dir = eval_dir / "abnormal" / dataset
    data_dir = project_root / "data" / dataset

    # Check prerequisites
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        logger.info(
            "Please ensure config/abnormal_detection_statistical.yaml exists in the evaluation directory"
        )
        return False

    logger.info(f"\n{'=' * 70}")
    logger.info("Wang Statistical Detection Pipeline")
    logger.info(f"{'=' * 70}")
    logger.info(f"Evaluation directory: {eval_dir}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Config: {config_file}")
    logger.info(f"{'=' * 70}\n")

    success_count = 0
    total_steps = 0
    failed_steps = []

    # Step 1: Real data detection (test split)
    if not skip_real:
        total_steps += 1
        real_test_file = data_dir / "test.csv"
        if real_test_file.exists():
            cmd = [
                "uv",
                "run",
                "python",
                str(project_root / "tools" / "analyze_abnormal.py"),
                "--real_file",
                str(real_test_file),
                "--dataset",
                dataset,
                "--config",
                str(config_file),
                "--output_dir",
                str(abnormal_dir / "test" / "real_data"),
            ]
            if run_command(cmd, "Real data detection (test split)"):
                success_count += 1
            else:
                failed_steps.append("Real data (test)")
        else:
            logger.warning(f"Real test file not found: {real_test_file}, skipping")

    # Step 2: Real data detection (train split)
    if not skip_real:
        total_steps += 1
        real_train_file = data_dir / "train.csv"
        if real_train_file.exists():
            cmd = [
                "uv",
                "run",
                "python",
                str(project_root / "tools" / "analyze_abnormal.py"),
                "--real_file",
                str(real_train_file),
                "--dataset",
                dataset,
                "--config",
                str(config_file),
                "--output_dir",
                str(abnormal_dir / "train" / "real_data"),
            ]
            if run_command(cmd, "Real data detection (train split)"):
                success_count += 1
            else:
                failed_steps.append("Real data (train)")
        else:
            logger.warning(f"Real train file not found: {real_train_file}, skipping")

    # Step 3: Generated models detection
    if not skip_generated:
        models = find_generated_models(eval_dir, dataset)
        if models:
            for model_name, model_file in models.items():
                total_steps += 1

                if model_file.exists():
                    cmd = [
                        "uv",
                        "run",
                        "python",
                        str(project_root / "tools" / "analyze_abnormal.py"),
                        "--real_file",
                        str(model_file),
                        "--dataset",
                        dataset,
                        "--config",
                        str(config_file),
                        "--output_dir",
                        str(abnormal_dir / "test" / "generated" / model_name),
                    ]
                    if run_command(cmd, f"Generated model detection: {model_name}"):
                        success_count += 1
                    else:
                        failed_steps.append(f"Generated: {model_name}")
                else:
                    logger.warning(f"Model file not found: {model_file}, skipping")
        else:
            logger.warning("No generated models found, skipping generated detection")

    # Step 4: Aggregate results
    if not skip_aggregation:
        total_steps += 1
        output_file = eval_dir / "wang_results_aggregated.json"
        cmd = [
            "uv",
            "run",
            "python",
            str(project_root / "tools" / "analyze_wang_results.py"),
            "--eval-dir",
            str(eval_dir),
            "--output",
            str(output_file),
        ]
        if run_command(cmd, "Aggregate Wang results"):
            success_count += 1
        else:
            failed_steps.append("Aggregation")

    # Step 5: Generate visualizations
    if not skip_visualization:
        total_steps += 1
        results_file = eval_dir / "wang_results_aggregated.json"
        output_dir = eval_dir / "figures" / "wang_abnormality"

        if results_file.exists():
            cmd = [
                "uv",
                "run",
                "python",
                str(project_root / "tools" / "visualize_wang_results.py"),
                "--input",
                str(results_file),
                "--output-dir",
                str(output_dir),
            ]
            if run_command(cmd, "Generate visualizations"):
                success_count += 1
            else:
                failed_steps.append("Visualization")
        else:
            logger.warning(
                f"Results file not found: {results_file}, skipping visualization"
            )

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("Pipeline Summary")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_steps)}")

    if failed_steps:
        logger.error("\n❌ Failed steps:")
        for step in failed_steps:
            logger.error(f"  - {step}")
        return False
    else:
        logger.info("\n✅ All pipeline steps completed successfully!")
        logger.info("\nResults saved to:")
        logger.info(f"  - Aggregated data: {eval_dir / 'wang_results_aggregated.json'}")
        logger.info(f"  - Visualizations: {eval_dir / 'figures' / 'wang_abnormality'}")
        return True


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run complete Wang statistical detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on Porto evaluation
  uv run python tools/run_wang_detection_pipeline.py \\
    --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \\
    --dataset porto_hoser

  # Skip visualization (faster)
  uv run python tools/run_wang_detection_pipeline.py \\
    --eval-dir eval_dir \\
    --dataset Beijing \\
    --skip-visualization

  # Only aggregate and visualize (detection already done)
  uv run python tools/run_wang_detection_pipeline.py \\
    --eval-dir eval_dir \\
    --dataset porto_hoser \\
    --skip-real \\
    --skip-generated

Pipeline Steps:
  1. Real data detection (test split)
  2. Real data detection (train split)
  3. Generated models detection (all models in gene/dataset/seed42/)
  4. Aggregate results into JSON
  5. Generate visualizations (PNG + SVG)

Prerequisites:
  - config/abnormal_detection_statistical.yaml must exist in eval directory
  - baselines/baselines_{dataset}.json must exist in project root
  - data/{dataset}/test.csv and train.csv for real data detection
  - gene/{dataset}/seed42/*_trajectories.csv for generated models
        """,
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory path (relative to project root or absolute)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., porto_hoser, Beijing, BJUT_Beijing)",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Skip real data detection",
    )
    parser.add_argument(
        "--skip-generated",
        action="store_true",
        help="Skip generated models detection",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip result aggregation",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation",
    )

    args = parser.parse_args()

    try:
        success = run_wang_detection_pipeline(
            eval_dir=args.eval_dir,
            dataset=args.dataset,
            skip_real=args.skip_real,
            skip_generated=args.skip_generated,
            skip_aggregation=args.skip_aggregation,
            skip_visualization=args.skip_visualization,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
