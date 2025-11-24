#!/usr/bin/env python3
"""
LM-TAD Perplexity-Based Evaluation Pipeline - Complete Workflow

This script runs the full LM-TAD perplexity-based evaluation pipeline, including
OD pair extraction, trajectory generation, perplexity evaluation, aggregation, and visualization.

The pipeline evaluates generated trajectories using LM-TAD teacher model perplexity scoring,
with support for per-road-segment perplexity analysis and cross-model comparison on the
same OD pairs.

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
import tools.analyze_lmtad_spatial_results as analyze_lmtad_spatial_results  # noqa: E402
from tools.evaluate_lmtad_spatial_abnormal import evaluate_spatial_abnormal_trajectories  # noqa: E402
from tools.extract_lmtad_spatial_abnormal_od import extract_spatial_abnormal_od_pairs  # noqa: E402
from tools.generate_lmtad_spatial_abnormal_trajectories import (  # noqa: E402
    generate_spatial_abnormal_trajectories,
)
import tools.visualize_lmtad_spatial_results as viz  # noqa: E402

# Backwards compatibility: expose helper aliases used by tests
load_aggregated_results = viz.load_aggregated_results

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

    # Use model_detection to robustly extract model names from filenames
    from tools.model_detection import detect_model_files  # noqa: E402

    detected = detect_model_files(gene_dir, pattern="*_spatial_abnormal.csv")

    models = {}
    # Build mapping model_name -> file (ModelFile.filename preserves original name)
    for mf in detected:
        models[mf.model_name] = gene_dir / mf.filename

    # If both plain base model and seeded variants exist (e.g., 'vanilla' and 'vanilla_seed42'),
    # prefer the seeded variants and drop the plain base to avoid duplicate/ambiguous aggregation
    bases_with_seeds = set()
    for name in list(models.keys()):
        # detect seeded pattern
        if "_seed" in name:
            base = name.split("_seed")[0]
            bases_with_seeds.add(base)

    if bases_with_seeds:
        for base in bases_with_seeds:
            if base in models:
                logger.info(
                    f"Dropping plain model '{base}' in generated models because seeded variants exist"
                )
                models.pop(base, None)

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
    max_duplicate_ratio: float = 1.0,
    force: bool = False,
    eval_config: Dict | None = None,
) -> bool:
    """Run complete LM-TAD perplexity-based evaluation pipeline

    This pipeline evaluates generated trajectories using LM-TAD teacher model perplexity scoring.
    It supports per-road-segment perplexity analysis and cross-model comparison on the same OD pairs.

    Args:
        eval_dir: Evaluation directory path
        dataset: Dataset name (e.g., porto_hoser, Beijing)
        lmtad_source_eval_dir: Path to LM-TAD source evaluation directory
        lmtad_checkpoint: Path to LM-TAD checkpoint file
        skip_extraction: Skip OD pair extraction
        skip_generation: Skip trajectory generation
        skip_evaluation: Skip LM-TAD perplexity evaluation
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

    # Canonical OD pairs file for this evaluation (may be created in extraction step)
    od_pairs_file = eval_dir / f"abnormal_od_pairs_lmtad_spatial_{dataset}.json"

    if not lmtad_source_eval_dir.exists():
        logger.error(f"LM-TAD source eval directory not found: {lmtad_source_eval_dir}")
        return False

    if not lmtad_checkpoint.exists():
        logger.error(f"LM-TAD checkpoint not found: {lmtad_checkpoint}")
        return False

    logger.info(f"\n{'=' * 70}")
    logger.info("LM-TAD Perplexity-Based Evaluation Pipeline")
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
            logger.info("Step: Extract OD pairs for perplexity-based evaluation")
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
                logger.info(
                    "Step: Generate trajectories for perplexity-based evaluation"
                )
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
                        "✅ Generate trajectories for perplexity-based evaluation completed successfully"
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(
                        f"❌ Generate trajectories for perplexity-based evaluation failed: {e}"
                    )
                    failed_steps.append("Trajectory generation")
    else:
        logger.info("⏭️  Skipping trajectory generation")

    # Step 3: Evaluate generated trajectories with LM-TAD
    if not skip_evaluation:
        models = find_generated_models(eval_dir, dataset, seed)
        if models:
            # Precompute a road_id -> token mapping once for this dataset so we can
            # pass it into the evaluation function (avoids re-computing per model).
            try:
                from tools.convert_to_lmtad_format import extract_road_centroids  # noqa: E402
                from critics.grid_mapper import GridMapper, GridConfig  # noqa: E402

                data_dir = Path(__file__).parent.parent / "data" / dataset
                roadmap_file = data_dir / "roadmap.geo"
                if not roadmap_file.exists():
                    roadmap_file = Path("data") / dataset / "roadmap.geo"

                road_to_token_override = None
                # First, attempt to auto-load a canonical mapping file from the
                # evaluation workspace or data directory. This allows users to
                # provide a precomputed `road_to_token.json` without requiring
                # the pipeline to recompute mappings from the roadmap.
                try:
                    from critics.mapping_utils import load_road_to_token_mapping  # noqa: E402

                    candidate_paths = [
                        eval_dir / "road_to_token.json",
                        eval_dir / "analysis_abnormal" / dataset / "road_to_token.json",
                        Path(__file__).parent.parent
                        / "data"
                        / dataset
                        / "road_to_token.json",
                    ]
                    for cand in candidate_paths:
                        if cand.exists():
                            try:
                                road_to_token_override = load_road_to_token_mapping(
                                    cand
                                )
                                logger.info(
                                    "🔁 Pipeline: auto-loaded road_id->token mapping from %s",
                                    cand,
                                )
                                break
                            except Exception as e:
                                logger.warning(
                                    "Pipeline: failed to parse mapping file %s: %s",
                                    cand,
                                    e,
                                )
                except Exception:
                    # Conservative: if the mapping_utils import fails for any reason,
                    # fall back to precompute below.
                    pass

                # If we didn't find/parse a canonical mapping file, fall back to
                # computing the mapping from the dataset roadmap (existing behavior).
                if road_to_token_override is None:
                    if roadmap_file.exists():
                        road_centroids, boundary_from_roadmap = extract_road_centroids(
                            roadmap_file
                        )
                        grid_config = GridConfig(
                            min_lat=boundary_from_roadmap["min_lat"],
                            max_lat=boundary_from_roadmap["max_lat"],
                            min_lng=boundary_from_roadmap["min_lng"],
                            max_lng=boundary_from_roadmap["max_lng"],
                            grid_size=0.001,
                            downsample_factor=1,
                        )
                        mapper = GridMapper(
                            boundary=grid_config,
                            road_centroids=road_centroids,
                            verify_hw=None,
                        )
                        road_to_token_override = mapper.map_all()
                        logger.info(
                            "🔁 Pipeline: precomputed road_id->token mapping for dataset %s",
                            dataset,
                        )
                    else:
                        logger.warning(
                            "Pipeline: roadmap file not found; skipping precompute mapping"
                        )
            except Exception as e:
                logger.warning(
                    "Pipeline: failed to precompute road->token mapping: %s", e
                )
                road_to_token_override = None

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
                            f"Step: Evaluate trajectories with LM-TAD perplexity: {model_name}"
                        )
                        logger.info(f"{'=' * 70}")
                        try:
                            # Forward the canonical OD pairs file into evaluation when available
                            od_pairs_arg = (
                                od_pairs_file if od_pairs_file.exists() else None
                            )
                            result = evaluate_spatial_abnormal_trajectories(
                                trajectory_file=trajectory_file,
                                lmtad_checkpoint=lmtad_checkpoint,
                                source_eval_dir=lmtad_source_eval_dir,
                                dataset=dataset,
                                device="cuda:0",
                                batch_size=128,
                                lmtad_repo=lmtad_repo,
                                eval_config=eval_config,
                                max_duplicate_ratio=max_duplicate_ratio,
                                road_to_token_override=road_to_token_override,
                                od_pairs_file=od_pairs_arg,
                            )
                            # Save results
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_file, "w") as f:
                                json.dump(result, f, indent=2)
                            logger.info(
                                f"✅ Evaluate trajectories with LM-TAD perplexity: {model_name} completed successfully"
                            )
                            logger.info(
                                "   📊 Capturing per-road-segment perplexity for detailed analysis"
                            )
                            success_count += 1
                        except Exception as e:
                            logger.error(
                                f"❌ Evaluate trajectories with LM-TAD perplexity: {model_name} failed: {e}"
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
            logger.info("Step: Aggregate LM-TAD perplexity-based results")
            logger.info(f"{'=' * 70}")
            try:
                # For backward compatibility (tests patch this symbol), call the
                # wrapper `aggregate_lmtad_spatial_results` which delegates to the
                # new aggregator internally.
                result = analyze_lmtad_spatial_results.aggregate_lmtad_spatial_results(
                    eval_dir=eval_dir,
                    dataset=dataset,
                    source_eval_dir=lmtad_source_eval_dir,
                )
                # Save results (ensure JSON serializable for extra safety)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(
                        analyze_lmtad_spatial_results.ensure_json_serializable(result),
                        f,
                        indent=2,
                    )
                logger.info(
                    "✅ Aggregate LM-TAD perplexity-based results completed successfully"
                )
                success_count += 1
            except Exception as e:
                logger.error(
                    f"❌ Aggregate LM-TAD perplexity-based results failed: {e}"
                )
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

                # Generate all plots (perplexity-based approach)
                # Call the backward-compatible wrappers (old names) so tests that
                # patch those functions will observe the calls. The wrappers
                # themselves delegate to the new plotting functions.
                viz.plot_spatial_abnormality_rates_comparison(
                    results, output_dir, dataset
                )
                viz.plot_route_switch_vs_detour_breakdown(results, output_dir, dataset)
                viz.plot_perplexity_distribution_comparison(
                    results, output_dir, dataset
                )
                viz.plot_model_rankings_spatial(results, output_dir, dataset)
                viz.plot_statistical_significance_spatial(results, output_dir, dataset)

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
    logger.info("Pipeline Summary - Perplexity-Based Evaluation")
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
        logger.info("\nPerplexity-based evaluation results saved to:")
        logger.info(
            f"  - Aggregated data: {eval_dir / 'analysis_abnormal' / dataset / 'lmtad_spatial_results_aggregated.json'}"
        )
        logger.info(
            f"  - Visualizations: {eval_dir / 'figures' / 'lmtad_spatial_abnormality' / dataset}"
        )
        logger.info("\nKey Features Evaluated:")
        logger.info("  - Overall trajectory perplexity (lower = better)")
        logger.info("  - Per-road-segment perplexity (identifies problematic segments)")
        logger.info("  - Cross-model comparison on same OD pairs")
        logger.info("  - Statistical significance of perplexity differences")
        return True


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run complete LM-TAD perplexity-based evaluation pipeline",
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
  1. Extract OD pairs from LM-TAD source evaluation for perplexity-based analysis
  2. Generate trajectories for these OD pairs (all models)
  3. Evaluate generated trajectories with LM-TAD teacher model (perplexity scoring)
  4. Aggregate perplexity results and cross-model comparisons into JSON
  5. Generate visualizations (PNG + SVG) with perplexity distributions

Key Features:
  - Per-road-segment perplexity analysis
  - Cross-model comparison on same OD pairs
  - No source-label classification (pure perplexity-based)
  - Statistical significance testing of perplexity distributions

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
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Path to evaluation configuration YAML file (optional)",
    )
    parser.add_argument(
        "--lmtad-max-duplicate-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum duplicate ratio allowed for trajectories during validation (default: 1.0). "
            "Set to a value < 1.0 (e.g., 0.1) to enable duplicate checks; 1.0 disables the duplicate check."
        ),
    )

    args = parser.parse_args()

    # Load evaluation config if provided
    eval_config = None
    if args.eval_config:
        if not args.eval_config.exists():
            logger.error(f"Evaluation config file not found: {args.eval_config}")
            return 1
        try:
            import yaml

            with open(args.eval_config, "r") as f:
                eval_config = yaml.safe_load(f)
            logger.info(f"📋 Loaded evaluation config from: {args.eval_config}")
        except Exception as e:
            logger.error(f"Failed to load evaluation config: {e}")
            return 1

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
            eval_config=eval_config,
            max_duplicate_ratio=args.lmtad_max_duplicate_ratio,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
