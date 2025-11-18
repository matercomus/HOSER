#!/usr/bin/env python3
"""
LM-TAD Spatial Abnormality Detection Pipeline - Complete Workflow

This script runs the full LM-TAD spatial abnormality evaluation pipeline, including
OD pair extraction, trajectory generation, evaluation, aggregation, and visualization.

Usage:
    # Run on evaluation directory (checkpoint auto-detected from eval dir)
    uv run python tools/run_lmtad_spatial_pipeline.py \\
        --eval-dir hoser-distill-optuna-porto-eval-xyz \\
        --dataset porto_hoser \\
        --lmtad-source-eval-dir /path/to/lmtad/eval \\
        --lmtad-checkpoint /path/to/ckpt_best.pt

    # Skip generation (use existing trajectories)
    uv run python tools/run_lmtad_spatial_pipeline.py \\
        --eval-dir eval_dir \\
        --dataset porto_hoser \\
        --lmtad-source-eval-dir /path/to/lmtad/eval \\
        --lmtad-checkpoint /path/to/ckpt_best.pt \\
        --skip-generation

    # Only aggregate and visualize (evaluation already done)
    uv run python tools/run_lmtad_spatial_pipeline.py \\
        --eval-dir eval_dir \\
        --dataset porto_hoser \\
        --lmtad-source-eval-dir /path/to/lmtad/eval \\
        --lmtad-checkpoint /path/to/ckpt_best.pt \\
        --skip-extraction \\
        --skip-generation \\
        --skip-evaluation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports when run as script
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import programmatic interfaces (after path setup)
from tools.analyze_lmtad_spatial_results import (  # noqa: E402
    aggregate_lmtad_spatial_results,
    ensure_json_serializable,
)
from tools.evaluate_lmtad_spatial_abnormal import evaluate_spatial_abnormal_trajectories  # noqa: E402
from tools.extract_lmtad_spatial_abnormal_od import extract_spatial_abnormal_od_pairs  # noqa: E402
from tools.generate_lmtad_spatial_abnormal_trajectories import (  # noqa: E402
    generate_spatial_abnormal_trajectories,
)
from tools.visualize_lmtad_spatial_results import (  # noqa: E402
    load_aggregated_results,
    plot_model_rankings_spatial,
    plot_perplexity_distribution_spatial,
    plot_route_switch_vs_detour_breakdown,
    plot_spatial_abnormality_rates_comparison,
    plot_statistical_significance_spatial,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_lmtad_tsv_file(source_eval_dir: Path) -> Path:
    """Find LM-TAD evaluation TSV file

    Args:
        source_eval_dir: Path to LM-TAD source evaluation directory

    Returns:
        Path to TSV file
    """
    tsv_files = list(source_eval_dir.glob("ckpt_best_outliers_*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in {source_eval_dir}")

    # Use first available (or most recent)
    return sorted(tsv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def find_generated_models(eval_dir: Path, dataset: str, seed: int) -> Dict[str, Path]:
    """Find all generated spatial abnormal trajectory files

    Args:
        eval_dir: Evaluation directory
        dataset: Dataset name
        seed: Random seed

    Returns:
        Dict mapping model_name -> file path
    """
    gene_dir = eval_dir / "gene_abnormal_lmtad_spatial" / dataset / f"seed{seed}"

    if not gene_dir.exists():
        logger.warning(f"Gene directory not found: {gene_dir}")
        return {}

    # Find all spatial abnormal CSV files
    csv_files = list(gene_dir.glob("*_spatial_abnormal.csv"))
    if not csv_files:
        logger.warning(f"No spatial abnormal trajectory files found in {gene_dir}")
        return {}

    # Extract model names from filenames
    models = {}
    for csv_file in csv_files:
        # Filename format: {model}_spatial_abnormal.csv
        model_name = csv_file.stem.replace("_spatial_abnormal", "")
        models[model_name] = csv_file

    logger.info(
        f"Found {len(models)} generated models: {', '.join(sorted(models.keys()))}"
    )
    return models


def run_lmtad_spatial_pipeline(
    eval_dir: Path,
    dataset: str,
    lmtad_source_eval_dir: Path,
    lmtad_checkpoint: Path,
    skip_extraction: bool = False,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    skip_aggregation: bool = False,
    skip_visualization: bool = False,
    seed: int = 42,
    num_traj_per_od: int = 20,
    max_od_pairs: int = 250,
    lmtad_repo: Path | None = None,
    force: bool = False,
) -> bool:
    """Run complete LM-TAD spatial abnormality evaluation pipeline

    Args:
        eval_dir: Evaluation directory path
        dataset: Dataset name (e.g., porto_hoser, Beijing)
        lmtad_source_eval_dir: Path to LM-TAD source evaluation directory
        lmtad_checkpoint: Path to LM-TAD checkpoint file
        skip_extraction: Skip OD pair extraction
        skip_generation: Skip trajectory generation
        skip_evaluation: Skip LM-TAD evaluation
        skip_aggregation: Skip result aggregation
        skip_visualization: Skip visualization generation
        seed: Random seed for generation
        num_traj_per_od: Number of trajectories per OD pair (default: 20, target: ~5,000 total)
        max_od_pairs: Maximum number of OD pairs to sample (default: 250, uses stratified sampling)
        lmtad_repo: Path to LM-TAD repository root (auto-detected from checkpoint if None)
        force: Force rerun even if output files exist (default: False)

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

    if not lmtad_source_eval_dir.exists():
        logger.error(f"LM-TAD source eval directory not found: {lmtad_source_eval_dir}")
        return False

    if not lmtad_checkpoint.exists():
        logger.error(f"LM-TAD checkpoint not found: {lmtad_checkpoint}")
        return False

    logger.info(f"\n{'=' * 70}")
    logger.info("LM-TAD Spatial Abnormality Detection Pipeline")
    logger.info(f"{'=' * 70}")
    logger.info(f"Evaluation directory: {eval_dir}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"LM-TAD source eval: {lmtad_source_eval_dir}")
    logger.info(f"LM-TAD checkpoint: {lmtad_checkpoint}")
    logger.info(f"{'=' * 70}\n")

    success_count = 0
    total_steps = 0
    failed_steps = []

    # Step 1: Extract spatial abnormal OD pairs
    if not skip_extraction:
        total_steps += 1
        output_file = eval_dir / f"abnormal_od_pairs_lmtad_spatial_{dataset}.json"

        # Check if already exists and has OD pairs
        should_extract = True
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    existing_data = json.load(f)
                total_od_pairs = existing_data.get("total_unique_od_pairs", 0)
                if total_od_pairs > 0 and not force:
                    logger.info(
                        f"  ⏭️  OD pairs file already exists with {total_od_pairs} OD pairs: {output_file.name}, skipping"
                    )
                    should_extract = False
                elif force:
                    logger.info(
                        f"  🔄 Force flag set, re-extracting OD pairs (existing file has {total_od_pairs} OD pairs)"
                    )
                else:
                    logger.warning(
                        "  ⚠️  OD pairs file exists but has 0 OD pairs, re-extracting..."
                    )
            except Exception as e:
                logger.warning(
                    f"  ⚠️  Failed to read existing OD pairs file: {e}, re-extracting..."
                )

        if should_extract:
            logger.info(f"{'=' * 70}")
            logger.info("Step: Extract spatial abnormal OD pairs")
            logger.info(f"{'=' * 70}")
            try:
                result = extract_spatial_abnormal_od_pairs(
                    tsv_file=lmtad_source_eval_dir,  # Pass directory to process all TSV files
                    dataset=dataset,
                    source_eval_dir=lmtad_source_eval_dir,
                )
                # Save results
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(
                    "✅ Extract spatial abnormal OD pairs completed successfully"
                )
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Extract spatial abnormal OD pairs failed: {e}")
                failed_steps.append("OD pair extraction")
    else:
        logger.info("⏭️  Skipping OD pair extraction")

    # Step 2: Generate trajectories for spatial abnormal OD pairs
    if not skip_generation:
        total_steps += 1
        od_pairs_file = eval_dir / f"abnormal_od_pairs_lmtad_spatial_{dataset}.json"

        if not od_pairs_file.exists():
            logger.warning(
                f"OD pairs file not found: {od_pairs_file}, skipping generation"
            )
        else:
            gene_dir = (
                eval_dir / "gene_abnormal_lmtad_spatial" / dataset / f"seed{seed}"
            )
            existing_files = (
                list(gene_dir.glob("*_spatial_abnormal.csv"))
                if gene_dir.exists()
                else []
            )

            if existing_files and not force:
                logger.info(
                    f"  ⏭️  Trajectories already generated in {gene_dir}, skipping"
                )
            else:
                # Need to generate (either no files exist, or force is True)
                if force and existing_files:
                    logger.info(
                        f"  🔄 Force flag set, regenerating trajectories in {gene_dir}"
                    )
                    # Remove existing trajectory files
                    for csv_file in existing_files:
                        csv_file.unlink()
                        logger.debug(f"  Removed {csv_file.name}")

                logger.info(f"{'=' * 70}")
                logger.info("Step: Generate trajectories for spatial abnormal OD pairs")
                logger.info(f"{'=' * 70}")
                try:
                    generate_spatial_abnormal_trajectories(
                        od_pairs_file=od_pairs_file,
                        eval_dir=eval_dir,
                        dataset=dataset,
                        models=[],  # Auto-detect all models
                        seed=seed,
                        num_traj_per_od=num_traj_per_od,
                        max_od_pairs=max_od_pairs,
                        stratified_sampling=True,  # Use stratified sampling to maintain ratio
                        cuda_device=0,
                        beam_search=False,  # Use A* by default
                        beam_width=4,
                    )
                    logger.info(
                        "✅ Generate trajectories for spatial abnormal OD pairs completed successfully"
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(
                        f"❌ Generate trajectories for spatial abnormal OD pairs failed: {e}"
                    )
                    failed_steps.append("Trajectory generation")
    else:
        logger.info("⏭️  Skipping trajectory generation")

    # Step 3: Evaluate generated trajectories with LM-TAD
    if not skip_evaluation:
        models = find_generated_models(eval_dir, dataset, seed)
        if models:
            for model_name, trajectory_file in models.items():
                total_steps += 1

                if trajectory_file.exists():
                    output_file = (
                        eval_dir
                        / "eval_lmtad_spatial"
                        / dataset
                        / f"{model_name}_spatial_evaluation.json"
                    )

                    # Check if already exists
                    if output_file.exists() and not force:
                        logger.info(
                            f"  ⏭️  Evaluation result already exists: {output_file.name}, skipping"
                        )
                        success_count += 1
                    elif force and output_file.exists():
                        logger.info(
                            f"  🔄 Force flag set, re-evaluating: {output_file.name}"
                        )
                        output_file.unlink()  # Remove existing file
                        # Fall through to evaluation
                    if not output_file.exists():
                        logger.info(f"{'=' * 70}")
                        logger.info(
                            f"Step: Evaluate spatial abnormal trajectories: {model_name}"
                        )
                        logger.info(f"{'=' * 70}")
                        try:
                            result = evaluate_spatial_abnormal_trajectories(
                                trajectory_file=trajectory_file,
                                lmtad_checkpoint=lmtad_checkpoint,
                                source_eval_dir=lmtad_source_eval_dir,
                                dataset=dataset,
                                device="cuda:0",
                                batch_size=128,
                                lmtad_repo=lmtad_repo,
                            )
                            # Save results
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_file, "w") as f:
                                json.dump(result, f, indent=2)
                            logger.info(
                                f"✅ Evaluate spatial abnormal trajectories: {model_name} completed successfully"
                            )
                            success_count += 1
                        except Exception as e:
                            logger.error(
                                f"❌ Evaluate spatial abnormal trajectories: {model_name} failed: {e}"
                            )
                            failed_steps.append(f"Evaluation: {model_name}")
                else:
                    logger.warning(
                        f"Trajectory file not found: {trajectory_file}, skipping"
                    )
        else:
            logger.warning("No generated models found, skipping evaluation")
    else:
        logger.info("⏭️  Skipping evaluation")

    # Step 4: Aggregate results
    if not skip_aggregation:
        total_steps += 1
        output_file = (
            eval_dir
            / "analysis_abnormal"
            / dataset
            / "lmtad_spatial_results_aggregated.json"
        )

        # Check if already exists
        if output_file.exists() and not force:
            logger.info(
                f"  ⏭️  Aggregated results already exist: {output_file.name}, skipping"
            )
            success_count += 1
        elif force and output_file.exists():
            logger.info(
                f"  🔄 Force flag set, re-aggregating results (removing {output_file.name})"
            )
            output_file.unlink()
            # Fall through to aggregation
        if not output_file.exists():
            logger.info(f"{'=' * 70}")
            logger.info("Step: Aggregate LM-TAD spatial results")
            logger.info(f"{'=' * 70}")
            try:
                result = aggregate_lmtad_spatial_results(
                    eval_dir=eval_dir,
                    dataset=dataset,
                    source_eval_dir=lmtad_source_eval_dir,
                )
                # Save results (ensure JSON serializable for extra safety)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(ensure_json_serializable(result), f, indent=2)
                logger.info(
                    "✅ Aggregate LM-TAD spatial results completed successfully"
                )
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Aggregate LM-TAD spatial results failed: {e}")
                failed_steps.append("Aggregation")
    else:
        logger.info("⏭️  Skipping aggregation")

    # Step 5: Generate visualizations
    if not skip_visualization:
        total_steps += 1
        results_file = (
            eval_dir
            / "analysis_abnormal"
            / dataset
            / "lmtad_spatial_results_aggregated.json"
        )
        output_dir = eval_dir / "figures" / "lmtad_spatial_abnormality" / dataset

        if results_file.exists():
            logger.info(f"{'=' * 70}")
            logger.info("Step: Generate visualizations")
            logger.info(f"{'=' * 70}")
            try:
                results = load_aggregated_results(results_file)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate all plots
                plot_spatial_abnormality_rates_comparison(results, output_dir, dataset)
                plot_route_switch_vs_detour_breakdown(results, output_dir, dataset)
                plot_model_rankings_spatial(results, output_dir, dataset)
                plot_statistical_significance_spatial(results, output_dir, dataset)
                plot_perplexity_distribution_spatial(results, output_dir, dataset)

                logger.info("✅ Generate visualizations completed successfully")
                logger.info(f"Visualizations saved to {output_dir}/")
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Generate visualizations failed: {e}")
                failed_steps.append("Visualization")
        else:
            logger.warning(
                f"Results file not found: {results_file}, skipping visualization"
            )
    else:
        logger.info("⏭️  Skipping visualization")

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
        logger.info(
            f"  - Aggregated data: {eval_dir / 'analysis_abnormal' / dataset / 'lmtad_spatial_results_aggregated.json'}"
        )
        logger.info(
            f"  - Visualizations: {eval_dir / 'figures' / 'lmtad_spatial_abnormality' / dataset}"
        )
        return True


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run complete LM-TAD spatial abnormality detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on Porto evaluation
  uv run python tools/run_lmtad_spatial_pipeline.py \\
    --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \\
    --dataset porto_hoser \\
    --lmtad-source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval \\
    --lmtad-checkpoint /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt

  # Skip generation (use existing trajectories)
  uv run python tools/run_lmtad_spatial_pipeline.py \\
    --eval-dir eval_dir \\
    --dataset porto_hoser \\
    --lmtad-source-eval-dir /path/to/eval \\
    --lmtad-checkpoint /path/to/ckpt_best.pt \\
    --skip-generation

  # Only aggregate and visualize (evaluation already done)
  uv run python tools/run_lmtad_spatial_pipeline.py \\
    --eval-dir eval_dir \\
    --dataset porto_hoser \\
    --lmtad-source-eval-dir /path/to/eval \\
    --lmtad-checkpoint /path/to/ckpt_best.pt \\
    --skip-extraction \\
    --skip-generation \\
    --skip-evaluation

Pipeline Steps:
  1. Extract spatial abnormal OD pairs from LM-TAD source evaluation
  2. Generate trajectories for these OD pairs (all models)
  3. Evaluate generated trajectories with LM-TAD (classify spatial types)
  4. Aggregate results into JSON
  5. Generate visualizations (PNG + SVG)

Prerequisites:
  - LM-TAD source evaluation directory with TSV files
  - LM-TAD checkpoint file
  - Evaluation directory with models/ directory
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
        "--lmtad-source-eval-dir",
        type=Path,
        required=True,
        help="Path to LM-TAD source evaluation directory",
    )
    parser.add_argument(
        "--lmtad-checkpoint",
        type=Path,
        required=True,
        help="Path to LM-TAD checkpoint file",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip OD pair extraction",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip trajectory generation",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip LM-TAD evaluation",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)",
    )
    parser.add_argument(
        "--num-trajectories-per-od",
        type=int,
        default=20,
        help="Number of trajectories to generate per OD pair (default: 20, target: ~5,000 total)",
    )
    parser.add_argument(
        "--max-od-pairs",
        type=int,
        default=250,
        help="Maximum number of OD pairs to sample (default: 250, uses stratified sampling)",
    )
    parser.add_argument(
        "--lmtad-repo",
        type=Path,
        default=None,
        help="Path to LM-TAD repository root (auto-detected from checkpoint if not provided)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if output files exist (regenerates trajectories and re-evaluates)",
    )

    args = parser.parse_args()

    try:
        success = run_lmtad_spatial_pipeline(
            eval_dir=args.eval_dir,
            dataset=args.dataset,
            lmtad_source_eval_dir=args.lmtad_source_eval_dir,
            lmtad_checkpoint=args.lmtad_checkpoint,
            skip_extraction=args.skip_extraction,
            skip_generation=args.skip_generation,
            skip_evaluation=args.skip_evaluation,
            skip_aggregation=args.skip_aggregation,
            skip_visualization=args.skip_visualization,
            seed=args.seed,
            num_traj_per_od=args.num_trajectories_per_od,
            max_od_pairs=args.max_od_pairs,
            lmtad_repo=args.lmtad_repo,
            force=args.force,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
