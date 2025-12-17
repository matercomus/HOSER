#!/usr/bin/env python3
"""
Evaluate normal generated trajectories (from gene/) with LM-TAD teacher model.

This script:
1. Finds all generated trajectory CSV files in gene/porto_hoser/seed42/
2. Converts each from HOSER road IDs to LM-TAD grid tokens
3. Evaluates with LM-TAD teacher to compute perplexity scores
4. Classifies outliers and generates comparative analysis across models/seeds
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.convert_to_lmtad_format import convert_hoser_to_lmtad_format
from tools.evaluate_with_lmtad import evaluate_with_lmtad
from tools.model_detection import parse_model_components

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_generated_trajectory_files(gene_dir: Path) -> List[Path]:
    """
    Find all generated trajectory CSV files.

    Args:
        gene_dir: Base gene directory (e.g., eval_dir/gene/porto_hoser/seed42/)

    Returns:
        List of CSV file paths sorted by name
    """
    csv_files = []

    # Pattern: YYYY-MM-DD_HH-MM-SS_{model}_{seed}_{split}.csv
    # Example: 2025-01-26_12-34-56_vanilla_42_train.csv
    for csv_file in gene_dir.glob("*.csv"):
        # Skip performance JSON files
        if csv_file.stem.endswith("_perf"):
            continue
        csv_files.append(csv_file)

    csv_files.sort()
    logger.info(f"Found {len(csv_files)} generated trajectory files in {gene_dir}")
    return csv_files


def parse_filename(csv_file: Path) -> Dict[str, str]:
    """
    Parse model info from generated trajectory filename.

    Args:
        csv_file: Path to CSV file

    Returns:
        Dict with 'model_type', 'seed', 'split', 'timestamp' keys
    """
    # Pattern: YYYY-MM-DD_HH-MM-SS_{model}_{split}.csv (seed 42)
    # Pattern: YYYY-MM-DD_HH-MM-SS_{model}_seed{N}_{split}.csv (seed 43, 44)
    stem = csv_file.stem
    parts = stem.split("_")

    # Extract timestamp
    timestamp = f"{parts[0]}_{parts[1]}"  # YYYY-MM-DD_HH-MM-SS

    # Extract split (train or test) - always last component
    split = parts[-1]

    # Reconstruct model name (everything between timestamp and split)
    # Join all parts between timestamp and split
    model_parts = parts[2:-1]  # Skip timestamp (0,1) and split (last)
    model_name_with_seed = "_".join(model_parts)

    # Use existing model detection utilities
    components = parse_model_components(model_name_with_seed)
    model_type = components["base_model"]
    seed = components["seed"] if components["seed"] else "seed42"

    # Clean up seed format
    if seed and seed.startswith("seed"):
        seed = seed[4:]  # Remove "seed" prefix to get just the number

    return {
        "model_type": model_type,
        "seed": seed,
        "split": split,
        "timestamp": timestamp,
        "filename": csv_file.name,
    }


def convert_all_trajectories(
    csv_files: List[Path],
    roadmap_file: Path,
    vocab_file: Path,
    dataset: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Convert all HOSER trajectory files to LM-TAD grid format.

    Args:
        csv_files: List of HOSER trajectory CSV files
        roadmap_file: Path to roadmap.geo file
        vocab_file: Path to LM-TAD vocab.json
        dataset: Dataset name (e.g., 'porto_hoser')
        output_dir: Directory to save converted files

    Returns:
        Dict mapping original file path to converted file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_files = {}

    logger.info(f"Converting {len(csv_files)} trajectory files to LM-TAD format...")

    for i, csv_file in enumerate(csv_files, 1):
        file_info = parse_filename(csv_file)
        logger.info(
            f"  [{i}/{len(csv_files)}] Converting {file_info['model_type']}/"
            f"seed{file_info['seed']}/{file_info['split']}..."
        )

        # Create subdirectory for organization
        model_dir = output_dir / file_info["model_type"] / f"seed{file_info['seed']}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Output filename
        lmtad_csv = model_dir / f"{file_info['split']}_lmtad_format.csv"

        # Convert
        try:
            convert_hoser_to_lmtad_format(
                trajectory_file=csv_file,
                roadmap_file=roadmap_file,
                output_file=lmtad_csv,
                vocab_file=vocab_file,
                dataset=dataset,
            )
            converted_files[str(csv_file)] = lmtad_csv
            logger.info(f"      ✓ Saved to {lmtad_csv}")
        except Exception as e:
            logger.error(f"      ✗ Failed to convert {csv_file}: {e}")
            continue

    logger.info(f"✅ Converted {len(converted_files)}/{len(csv_files)} files")
    return converted_files


def evaluate_all_trajectories(
    converted_files: Dict[str, Path],
    vocab_file: Path,
    lmtad_checkpoint: Path,
    lmtad_repo_path: Path,
    dataset: str,
    output_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 128,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate all converted trajectories with LM-TAD teacher model.

    Args:
        converted_files: Dict mapping original to converted file paths
        vocab_file: Path to LM-TAD vocab.json
        lmtad_checkpoint: Path to teacher model checkpoint
        lmtad_repo_path: Path to LM-TAD repository
        dataset: Dataset name
        output_dir: Directory to save evaluation results
        device: CUDA device
        batch_size: Batch size for evaluation

    Returns:
        Dict mapping file identifier to evaluation results DataFrame
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    logger.info(
        f"Evaluating {len(converted_files)} trajectory files with LM-TAD teacher..."
    )

    for i, (orig_path, lmtad_csv) in enumerate(converted_files.items(), 1):
        orig_file = Path(orig_path)
        file_info = parse_filename(orig_file)

        logger.info(
            f"  [{i}/{len(converted_files)}] Evaluating {file_info['model_type']}/"
            f"seed{file_info['seed']}/{file_info['split']}..."
        )

        # Create output directory for this file
        model_output_dir = (
            output_dir / file_info["model_type"] / f"seed{file_info['seed']}"
        )
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        try:
            results_df = evaluate_with_lmtad(
                trajectory_file=lmtad_csv,
                vocab_file=vocab_file,
                lmtad_checkpoint=lmtad_checkpoint,
                lmtad_repo_path=lmtad_repo_path,
                dataset=dataset,
                output_dir=model_output_dir / file_info["split"],
                device=device,
                batch_size=batch_size,
            )

            # Store results
            file_key = f"{file_info['model_type']}_seed{file_info['seed']}_{file_info['split']}"
            all_results[file_key] = results_df

            # Log summary
            outlier_rate = results_df["is_outlier"].mean()
            mean_ppl = results_df["perplexity"].mean()
            median_ppl = results_df["perplexity"].median()

            logger.info(f"      ✓ Outlier rate: {outlier_rate:.2%}")
            logger.info(
                f"      ✓ Perplexity: mean={mean_ppl:.4f}, median={median_ppl:.4f}"
            )

        except Exception as e:
            logger.error(f"      ✗ Failed to evaluate {lmtad_csv}: {e}")
            continue

    logger.info(f"✅ Evaluated {len(all_results)}/{len(converted_files)} files")
    return all_results


def generate_comparative_analysis(
    all_results: Dict[str, pd.DataFrame], output_dir: Path
) -> Dict:
    """
    Generate comparative analysis across models, seeds, and splits.

    Args:
        all_results: Dict mapping file keys to evaluation DataFrames
        output_dir: Directory to save analysis results

    Returns:
        Dictionary with comparative metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating comparative analysis...")

    # Aggregate metrics
    summary = []

    for file_key, results_df in all_results.items():
        # Parse file key
        parts = file_key.split("_")
        model_type = "_".join(parts[:-2])  # Handle distill_phase1, distill_phase2
        seed = parts[-2].replace("seed", "")
        split = parts[-1]

        # Compute statistics
        metrics = {
            "model_type": model_type,
            "seed": seed,
            "split": split,
            "num_trajectories": len(results_df),
            "outlier_rate": float(results_df["is_outlier"].mean()),
            "mean_perplexity": float(results_df["perplexity"].mean()),
            "median_perplexity": float(results_df["perplexity"].median()),
            "std_perplexity": float(results_df["perplexity"].std()),
            "min_perplexity": float(results_df["perplexity"].min()),
            "max_perplexity": float(results_df["perplexity"].max()),
            "q25_perplexity": float(results_df["perplexity"].quantile(0.25)),
            "q75_perplexity": float(results_df["perplexity"].quantile(0.75)),
        }
        summary.append(metrics)

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)

    # Save summary
    summary_csv = output_dir / "comparative_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"✅ Saved summary to {summary_csv}")

    # Group by model type for comparison
    model_comparison = (
        summary_df.groupby("model_type")
        .agg(
            {
                "outlier_rate": ["mean", "std"],
                "mean_perplexity": ["mean", "std"],
                "median_perplexity": ["mean", "std"],
            }
        )
        .round(4)
    )

    model_comparison_file = output_dir / "model_type_comparison.csv"
    model_comparison.to_csv(model_comparison_file)
    logger.info(f"✅ Saved model comparison to {model_comparison_file}")

    # Group by seed for stability analysis
    seed_comparison = (
        summary_df.groupby("seed")
        .agg(
            {
                "outlier_rate": ["mean", "std"],
                "mean_perplexity": ["mean", "std"],
                "median_perplexity": ["mean", "std"],
            }
        )
        .round(4)
    )

    seed_comparison_file = output_dir / "seed_stability_comparison.csv"
    seed_comparison.to_csv(seed_comparison_file)
    logger.info(f"✅ Saved seed comparison to {seed_comparison_file}")

    # Create analysis dict
    analysis = {
        "total_files_evaluated": len(all_results),
        "total_trajectories": int(summary_df["num_trajectories"].sum()),
        "overall_outlier_rate": float(summary_df["outlier_rate"].mean()),
        "overall_mean_perplexity": float(summary_df["mean_perplexity"].mean()),
        "model_ranking": model_comparison["mean_perplexity"]["mean"]
        .sort_values()
        .to_dict(),
        "seed_stability": {
            "outlier_rate_std": float(seed_comparison["outlier_rate"]["std"].mean()),
            "perplexity_std": float(seed_comparison["mean_perplexity"]["std"].mean()),
        },
    }

    # Save analysis JSON
    analysis_json = output_dir / "analysis_summary.json"
    with open(analysis_json, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"✅ Saved analysis to {analysis_json}")

    return analysis


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate generated trajectories with LM-TAD teacher model"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory (e.g., hoser-distill-optuna-porto-eval-*)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="porto_hoser",
        help="Dataset name (default: porto_hoser)",
    )
    parser.add_argument(
        "--lmtad-checkpoint",
        type=Path,
        required=True,
        help="Path to LM-TAD teacher checkpoint",
    )
    parser.add_argument(
        "--lmtad-repo",
        type=Path,
        default=Path("/home/mka299/LMTAD/code"),
        help="Path to LM-TAD repository",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {args.eval_dir}")
    if not args.lmtad_checkpoint.exists():
        raise FileNotFoundError(f"LM-TAD checkpoint not found: {args.lmtad_checkpoint}")
    if not args.lmtad_repo.exists():
        raise FileNotFoundError(f"LM-TAD repo not found: {args.lmtad_repo}")

    logger.info("=" * 80)
    logger.info("LM-TAD Evaluation of Generated Trajectories")
    logger.info("=" * 80)
    logger.info(f"Eval directory: {args.eval_dir}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"LM-TAD checkpoint: {args.lmtad_checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 80)

    # Setup paths
    gene_dir = args.eval_dir / "gene" / args.dataset / "seed42"
    data_dir = Path("data") / args.dataset
    roadmap_file = data_dir / "roadmap.geo"
    vocab_file = Path("/home/mka299/LMTAD/data") / args.dataset / "vocab.json"

    # Output directories
    converted_dir = args.eval_dir / "eval_lmtad" / args.dataset / "converted"
    results_dir = args.eval_dir / "eval_lmtad" / args.dataset / "results"
    analysis_dir = args.eval_dir / "eval_lmtad" / args.dataset / "analysis"

    # Validate required files
    if not roadmap_file.exists():
        raise FileNotFoundError(f"Roadmap file not found: {roadmap_file}")
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

    # Step 1: Find trajectory files
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Finding Generated Trajectory Files")
    logger.info("=" * 80)
    csv_files = find_generated_trajectory_files(gene_dir)

    if not csv_files:
        logger.error(f"No trajectory files found in {gene_dir}")
        return

    # Step 2: Convert to LM-TAD format
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Converting HOSER → LM-TAD Format")
    logger.info("=" * 80)
    converted_files = convert_all_trajectories(
        csv_files=csv_files,
        roadmap_file=roadmap_file,
        vocab_file=vocab_file,
        dataset=args.dataset,
        output_dir=converted_dir,
    )

    if not converted_files:
        logger.error("No files converted successfully")
        return

    # Step 3: Evaluate with LM-TAD
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Evaluating with LM-TAD Teacher Model")
    logger.info("=" * 80)
    all_results = evaluate_all_trajectories(
        converted_files=converted_files,
        vocab_file=vocab_file,
        lmtad_checkpoint=args.lmtad_checkpoint,
        lmtad_repo_path=args.lmtad_repo,
        dataset=args.dataset,
        output_dir=results_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    if not all_results:
        logger.error("No files evaluated successfully")
        return

    # Step 4: Comparative analysis
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Generating Comparative Analysis")
    logger.info("=" * 80)
    analysis = generate_comparative_analysis(
        all_results=all_results, output_dir=analysis_dir
    )

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total files evaluated: {analysis['total_files_evaluated']}")
    logger.info(f"Total trajectories: {analysis['total_trajectories']:,}")
    logger.info(f"Overall outlier rate: {analysis['overall_outlier_rate']:.2%}")
    logger.info(f"Overall mean perplexity: {analysis['overall_mean_perplexity']:.4f}")
    logger.info("\nModel ranking (by mean perplexity, lower is better):")
    for rank, (model, ppl) in enumerate(analysis["model_ranking"].items(), 1):
        logger.info(f"  {rank}. {model}: {ppl:.4f}")
    logger.info(f"\nResults saved to: {args.eval_dir / 'eval_lmtad' / args.dataset}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
