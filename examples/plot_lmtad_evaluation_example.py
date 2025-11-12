#!/usr/bin/env python3
"""
Example: LM-TAD Evaluation Plotting

This script demonstrates how to use the plot_lmtad_evaluation module to generate
comprehensive visualizations for LM-TAD teacher model evaluation results.

Usage:
    # Create sample evaluation results (for demo purposes)
    uv run python examples/plot_lmtad_evaluation_example.py --create-sample-data

    # Generate plots from existing results
    uv run python examples/plot_lmtad_evaluation_example.py \\
        --real-results eval_lmtad/porto_hoser/real_evaluation_results.json \\
        --generated-results eval_lmtad/porto_hoser/generated_evaluation_results.json \\
        --output-dir figures/lmtad/porto_hoser \\
        --dataset porto_hoser
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import tools module
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.plot_lmtad_evaluation import (
    LMTADPlotConfig,
    plot_lmtad_evaluation_from_files,
)


def create_sample_evaluation_data(output_dir: Path, dataset: str = "sample_dataset"):
    """Create sample evaluation data for demonstration purposes

    Args:
        output_dir: Directory to save sample data files
        dataset: Dataset name
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate realistic sample data
    np.random.seed(42)

    # Real trajectories - lower perplexity (better fit to model)
    real_perplexities = np.random.lognormal(0.8, 0.4, 1000)
    real_lengths = np.random.randint(5, 60, 1000)

    real_results = {
        "total_trajectories": 1000,
        "outlier_rate": 0.12,
        "normal_trajectory_rate": 0.88,
        "mean_perplexity": float(np.mean(real_perplexities)),
        "std_perplexity": float(np.std(real_perplexities)),
        "perplexity_values": [float(x) for x in real_perplexities],
        "trajectory_lengths": [int(x) for x in real_lengths],
        "evaluation_config": {
            "model": "LM-TAD Teacher",
            "dataset": dataset,
            "perplexity_threshold": 3.0,
        },
    }

    # Generated trajectories - slightly higher perplexity (less natural)
    gen_perplexities = np.random.lognormal(0.95, 0.45, 800)
    gen_lengths = np.random.randint(5, 60, 800)

    generated_results = {
        "total_trajectories": 800,
        "outlier_rate": 0.18,
        "normal_trajectory_rate": 0.82,
        "mean_perplexity": float(np.mean(gen_perplexities)),
        "std_perplexity": float(np.std(gen_perplexities)),
        "perplexity_values": [float(x) for x in gen_perplexities],
        "trajectory_lengths": [int(x) for x in gen_lengths],
        "evaluation_config": {
            "model": "HOSER Student",
            "dataset": dataset,
            "perplexity_threshold": 3.0,
        },
    }

    # Save to JSON files
    real_file = output_dir / "real_evaluation_results.json"
    gen_file = output_dir / "generated_evaluation_results.json"

    with open(real_file, "w") as f:
        json.dump(real_results, f, indent=2)
    print(f"Created sample real results: {real_file}")

    with open(gen_file, "w") as f:
        json.dump(generated_results, f, indent=2)
    print(f"Created sample generated results: {gen_file}")

    return real_file, gen_file


def main():
    parser = argparse.ArgumentParser(
        description="Example: Generate LM-TAD evaluation plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample evaluation data for demonstration",
    )
    parser.add_argument(
        "--real-results",
        type=Path,
        help="Path to real trajectory evaluation results JSON",
    )
    parser.add_argument(
        "--generated-results",
        type=Path,
        help="Path to generated trajectory evaluation results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/lmtad_example"),
        help="Output directory for plots (default: figures/lmtad_example)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample_dataset",
        help="Dataset name (default: sample_dataset)",
    )

    args = parser.parse_args()

    if args.create_sample_data:
        print("\n=== Creating Sample Evaluation Data ===\n")
        sample_dir = Path("eval_lmtad_sample")
        real_file, gen_file = create_sample_evaluation_data(sample_dir, args.dataset)

        print("\n=== Generating Plots from Sample Data ===\n")
        plot_files = plot_lmtad_evaluation_from_files(
            real_results_file=real_file,
            generated_results_file=gen_file,
            output_dir=args.output_dir,
            dataset=args.dataset,
        )

    elif args.real_results and args.generated_results:
        print("\n=== Generating Plots from Provided Results ===\n")

        # Optional: use custom configuration
        config = LMTADPlotConfig(
            figure_size=(14, 8),
            dpi=300,
            perplexity_bins=40,
        )

        plot_files = plot_lmtad_evaluation_from_files(
            real_results_file=args.real_results,
            generated_results_file=args.generated_results,
            output_dir=args.output_dir,
            dataset=args.dataset,
            config=config,
        )

    else:
        parser.error(
            "Either --create-sample-data or both --real-results and "
            "--generated-results must be provided"
        )

    print("\n=== Summary ===\n")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {len(plot_files)} visualization outputs:")
    for plot_name, plot_path in sorted(plot_files.items()):
        print(f"  - {plot_name}: {plot_path}")

    print("\nVisualization types:")
    print("  1. Outlier rate comparison (bar chart)")
    print("  2. Perplexity distributions (histograms, box plots, CDF)")
    print("  3. Normal trajectory rates (stacked bar chart)")
    print("  4. Perplexity scatter (correlation with trajectory length)")
    print("  5. Summary tables (LaTeX and Markdown)")

    print("\nDone!")


if __name__ == "__main__":
    main()
