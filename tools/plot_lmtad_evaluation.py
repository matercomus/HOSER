#!/usr/bin/env python3
"""
Plot LMTAD Evaluation Results

This module provides comprehensive plotting functionality for LM-TAD teacher model evaluation.
Generates visualizations for outlier detection rates, perplexity distributions, normal trajectory
rates, and detailed statistical comparisons between real and generated trajectories.

Usage (Module):
    from tools.plot_lmtad_evaluation import plot_lmtad_evaluation_from_files

    plot_lmtad_evaluation_from_files(
        real_results_file=Path("eval_lmtad/porto_hoser/real_evaluation_results.json"),
        generated_results_file=Path("eval_lmtad/porto_hoser/generated_evaluation_results.json"),
        output_dir=Path("figures/lmtad/porto_hoser"),
        dataset="porto_hoser"
    )

Usage (CLI):
    uv run python tools/plot_lmtad_evaluation.py \\
        --real-results eval_lmtad/porto_hoser/real_evaluation_results.json \\
        --generated-results eval_lmtad/porto_hoser/generated_evaluation_results.json \\
        --output-dir figures/lmtad/porto_hoser \\
        --dataset porto_hoser
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set publication-quality plot defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "font.family": "sans-serif",
    }
)

# Use seaborn color palette for consistency
sns.set_palette("husl")


@dataclass
class LMTADPlotConfig:
    """Configuration for LMTAD evaluation plotting"""

    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    perplexity_bins: int = 50
    scatter_alpha: float = 0.5
    scatter_size: int = 20


def load_evaluation_results(results_file: Path) -> Dict[str, Any]:
    """Load LMTAD evaluation results JSON with fail-fast assertions

    Args:
        results_file: Path to evaluation results JSON file

    Returns:
        Dictionary containing evaluation results data

    Raises:
        AssertionError: If file doesn't exist or required keys are missing
    """
    assert results_file.exists(), f"Results file not found: {results_file}"

    logger.info(f"Loading evaluation results from {results_file}")

    with open(results_file, "r") as f:
        results_data = json.load(f)

    # Validate required keys
    required_keys = ["total_trajectories", "outlier_rate", "normal_trajectory_rate"]
    for key in required_keys:
        assert key in results_data, (
            f"Results file must contain '{key}' key. "
            f"Found keys: {list(results_data.keys())}"
        )

    logger.info(
        f"Loaded results: {results_data['total_trajectories']} trajectories, "
        f"outlier_rate={results_data['outlier_rate']:.2%}, "
        f"normal_rate={results_data['normal_trajectory_rate']:.2%}"
    )
    return results_data


def plot_outlier_rate_comparison(
    real_results: Dict[str, Any],
    generated_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[LMTADPlotConfig] = None,
) -> None:
    """Plot outlier rate comparison between real and generated trajectories

    Args:
        real_results: Dictionary containing real trajectory evaluation results
        generated_results: Dictionary containing generated trajectory evaluation results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If required keys are missing
    """
    assert "outlier_rate" in real_results, "Real results missing 'outlier_rate'"
    assert "outlier_rate" in generated_results, (
        "Generated results missing 'outlier_rate'"
    )

    if config is None:
        config = LMTADPlotConfig()

    logger.info("Plotting outlier rate comparison...")

    # Extract outlier rates
    real_outlier_rate = real_results["outlier_rate"] * 100  # Convert to percentage
    gen_outlier_rate = generated_results["outlier_rate"] * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot comparison
    categories = ["Real Trajectories", "Generated Trajectories"]
    outlier_rates = [real_outlier_rate, gen_outlier_rate]
    colors = ["#3498db", "#e74c3c"]

    bars = ax.bar(categories, outlier_rates, color=colors, alpha=0.8, width=0.6)

    # Add value labels on bars
    for bar, rate in zip(bars, outlier_rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Calculate difference
    diff = gen_outlier_rate - real_outlier_rate
    diff_text = f"Difference: {diff:+.2f}%"
    ax.text(
        0.5,
        0.95,
        diff_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
    )

    ax.set_ylabel("Outlier Rate (%)", fontsize=12)
    ax.set_title(
        f"LM-TAD Outlier Rate Comparison - {dataset}\n"
        f"(Trajectories with perplexity > threshold)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(outlier_rates) * 1.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_file}")


def plot_perplexity_distributions(
    real_results: Dict[str, Any],
    generated_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[LMTADPlotConfig] = None,
) -> None:
    """Plot perplexity distributions for real vs generated trajectories

    Args:
        real_results: Dictionary containing real trajectory evaluation results
        generated_results: Dictionary containing generated trajectory evaluation results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If required keys are missing
    """
    assert "perplexity_values" in real_results or "mean_perplexity" in real_results, (
        "Real results missing perplexity data"
    )
    assert (
        "perplexity_values" in generated_results
        or "mean_perplexity" in generated_results
    ), "Generated results missing perplexity data"

    if config is None:
        config = LMTADPlotConfig()

    logger.info("Plotting perplexity distributions...")

    # Extract perplexity values (use arrays if available, otherwise create from stats)
    real_perplexities = real_results.get("perplexity_values", [])
    gen_perplexities = generated_results.get("perplexity_values", [])

    # Get summary statistics
    real_mean = real_results.get(
        "mean_perplexity", np.mean(real_perplexities) if real_perplexities else 0
    )
    gen_mean = generated_results.get(
        "mean_perplexity", np.mean(gen_perplexities) if gen_perplexities else 0
    )
    real_std = real_results.get(
        "std_perplexity", np.std(real_perplexities) if real_perplexities else 0
    )
    gen_std = generated_results.get(
        "std_perplexity", np.std(gen_perplexities) if gen_perplexities else 0
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overlapping histograms
    ax1 = axes[0, 0]
    if real_perplexities and gen_perplexities:
        # Determine common bins for comparison
        all_perplexities = np.concatenate([real_perplexities, gen_perplexities])
        bins = np.linspace(
            np.percentile(all_perplexities, 1),
            np.percentile(all_perplexities, 99),
            config.perplexity_bins,
        )

        ax1.hist(
            real_perplexities,
            bins=bins,
            alpha=0.6,
            label="Real",
            color="#3498db",
            edgecolor="black",
            density=True,
        )
        ax1.hist(
            gen_perplexities,
            bins=bins,
            alpha=0.6,
            label="Generated",
            color="#e74c3c",
            edgecolor="black",
            density=True,
        )
    ax1.set_xlabel("Perplexity")
    ax1.set_ylabel("Density")
    ax1.set_title("Perplexity Distribution Comparison", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Box plots
    ax2 = axes[0, 1]
    if real_perplexities and gen_perplexities:
        box_data = [real_perplexities, gen_perplexities]
        bp = ax2.boxplot(
            box_data,
            tick_labels=["Real", "Generated"],
            patch_artist=True,
            notch=True,
            widths=0.5,
        )
        # Color the boxes
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity Box Plot Comparison", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    if real_perplexities and gen_perplexities:
        real_sorted = np.sort(real_perplexities)
        gen_sorted = np.sort(gen_perplexities)
        real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)

        ax3.plot(real_sorted, real_cdf, label="Real", color="#3498db", linewidth=2)
        ax3.plot(gen_sorted, gen_cdf, label="Generated", color="#e74c3c", linewidth=2)
    ax3.set_xlabel("Perplexity")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("Cumulative Distribution Function", fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_stats = [
        ["Metric", "Real", "Generated", "Difference"],
        [
            "Mean",
            f"{real_mean:.3f}",
            f"{gen_mean:.3f}",
            f"{gen_mean - real_mean:+.3f}",
        ],
        ["Std Dev", f"{real_std:.3f}", f"{gen_std:.3f}", f"{gen_std - real_std:+.3f}"],
        [
            "Outlier Rate",
            f"{real_results['outlier_rate']:.2%}",
            f"{generated_results['outlier_rate']:.2%}",
            f"{generated_results['outlier_rate'] - real_results['outlier_rate']:+.2%}",
        ],
    ]

    table = ax4.table(
        cellText=summary_stats,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, 4):
        for j in range(4):
            table[(i, j)].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    ax4.set_title("Summary Statistics", fontweight="bold", pad=20)

    plt.suptitle(
        f"Perplexity Analysis - {dataset}", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_file}")


def plot_normal_trajectory_rates(
    real_results: Dict[str, Any],
    generated_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[LMTADPlotConfig] = None,
) -> None:
    """Plot normal trajectory rates (non-outlier) comparison

    Args:
        real_results: Dictionary containing real trajectory evaluation results
        generated_results: Dictionary containing generated trajectory evaluation results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If required keys are missing
    """
    assert "normal_trajectory_rate" in real_results, (
        "Real results missing 'normal_trajectory_rate'"
    )
    assert "normal_trajectory_rate" in generated_results, (
        "Generated results missing 'normal_trajectory_rate'"
    )

    if config is None:
        config = LMTADPlotConfig()

    logger.info("Plotting normal trajectory rates...")

    # Extract rates
    real_normal_rate = real_results["normal_trajectory_rate"] * 100
    gen_normal_rate = generated_results["normal_trajectory_rate"] * 100

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Real Trajectories", "Generated Trajectories"]
    normal_rates = [real_normal_rate, gen_normal_rate]
    outlier_rates = [
        real_results["outlier_rate"] * 100,
        generated_results["outlier_rate"] * 100,
    ]

    x = np.arange(len(categories))
    width = 0.6

    # Stacked bars
    bars1 = ax.bar(x, normal_rates, width, label="Normal", color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(
        x,
        outlier_rates,
        width,
        bottom=normal_rates,
        label="Outlier",
        color="#e74c3c",
        alpha=0.8,
    )

    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Normal rate label
        ax.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height() / 2,
            f"{normal_rates[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        # Outlier rate label
        if outlier_rates[i] > 5:  # Only show if significant
            ax.text(
                bar2.get_x() + bar2.get_width() / 2,
                normal_rates[i] + outlier_rates[i] / 2,
                f"{outlier_rates[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(
        f"Trajectory Classification Rates - {dataset}\n(Normal vs Outlier by LM-TAD)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_file}")


def plot_perplexity_scatter(
    real_results: Dict[str, Any],
    generated_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[LMTADPlotConfig] = None,
) -> None:
    """Plot perplexity scatter plot with trajectory length correlation

    Args:
        real_results: Dictionary containing real trajectory evaluation results
        generated_results: Dictionary containing generated trajectory evaluation results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If required data is missing
    """
    if config is None:
        config = LMTADPlotConfig()

    logger.info("Plotting perplexity scatter with trajectory length...")

    # Extract perplexity and length data
    real_perplexities = real_results.get("perplexity_values", [])
    gen_perplexities = generated_results.get("perplexity_values", [])
    real_lengths = real_results.get("trajectory_lengths", [])
    gen_lengths = generated_results.get("trajectory_lengths", [])

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Real trajectories scatter
    ax1 = axes[0]
    if real_perplexities and real_lengths:
        assert len(real_perplexities) == len(real_lengths), (
            f"Mismatched lengths: perplexity={len(real_perplexities)}, "
            f"lengths={len(real_lengths)}"
        )
        sc1 = ax1.scatter(
            real_lengths,
            real_perplexities,
            alpha=config.scatter_alpha,
            s=config.scatter_size,
            c=real_perplexities,
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )
        plt.colorbar(sc1, ax=ax1, label="Perplexity")

        # Add correlation coefficient
        if len(real_perplexities) > 1:
            corr = np.corrcoef(real_lengths, real_perplexities)[0, 1]
            ax1.text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
    ax1.set_xlabel("Trajectory Length (# road segments)")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Real Trajectories", fontweight="bold")
    ax1.grid(alpha=0.3)

    # 2. Generated trajectories scatter
    ax2 = axes[1]
    if gen_perplexities and gen_lengths:
        assert len(gen_perplexities) == len(gen_lengths), (
            f"Mismatched lengths: perplexity={len(gen_perplexities)}, "
            f"lengths={len(gen_lengths)}"
        )
        sc2 = ax2.scatter(
            gen_lengths,
            gen_perplexities,
            alpha=config.scatter_alpha,
            s=config.scatter_size,
            c=gen_perplexities,
            cmap="plasma",
            edgecolors="black",
            linewidth=0.5,
        )
        plt.colorbar(sc2, ax=ax2, label="Perplexity")

        # Add correlation coefficient
        if len(gen_perplexities) > 1:
            corr = np.corrcoef(gen_lengths, gen_perplexities)[0, 1]
            ax2.text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
    ax2.set_xlabel("Trajectory Length (# road segments)")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Generated Trajectories", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Perplexity vs Trajectory Length - {dataset}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_file}")


def create_lmtad_summary_table(
    real_results: Dict[str, Any],
    generated_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
) -> None:
    """Create LaTeX and Markdown summary tables for LMTAD evaluation

    Args:
        real_results: Dictionary containing real trajectory evaluation results
        generated_results: Dictionary containing generated trajectory evaluation results
        output_file: Path to save the table (without extension, will create .tex and .md)
        dataset: Dataset name

    Raises:
        AssertionError: If required keys are missing
    """
    logger.info("Creating LMTAD summary tables...")

    # Extract key metrics
    real_count = real_results["total_trajectories"]
    gen_count = generated_results["total_trajectories"]

    real_outlier_rate = real_results["outlier_rate"] * 100
    gen_outlier_rate = generated_results["outlier_rate"] * 100

    real_normal_rate = real_results["normal_trajectory_rate"] * 100
    gen_normal_rate = generated_results["normal_trajectory_rate"] * 100

    real_mean_perp = real_results.get("mean_perplexity", 0)
    gen_mean_perp = generated_results.get("mean_perplexity", 0)

    real_std_perp = real_results.get("std_perplexity", 0)
    gen_std_perp = generated_results.get("std_perplexity", 0)

    # Create LaTeX table
    latex_content = f"""% LM-TAD Evaluation Summary for {dataset}
% Generated by plot_lmtad_evaluation.py

\\begin{{table}}[htbp]
\\centering
\\caption{{LM-TAD Evaluation Results - {dataset}}}
\\label{{tab:lmtad_{dataset.replace("_", "-")}}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Real}} & \\textbf{{Generated}} & \\textbf{{Difference}} \\\\
\\midrule
Trajectories & {real_count:,} & {gen_count:,} & -- \\\\
Mean Perplexity & {real_mean_perp:.3f} & {gen_mean_perp:.3f} & {gen_mean_perp - real_mean_perp:+.3f} \\\\
Std Perplexity & {real_std_perp:.3f} & {gen_std_perp:.3f} & {gen_std_perp - real_std_perp:+.3f} \\\\
Outlier Rate (\\%) & {real_outlier_rate:.2f} & {gen_outlier_rate:.2f} & {gen_outlier_rate - real_outlier_rate:+.2f} \\\\
Normal Rate (\\%) & {real_normal_rate:.2f} & {gen_normal_rate:.2f} & {gen_normal_rate - real_normal_rate:+.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Create Markdown table
    markdown_content = f"""# LM-TAD Evaluation Summary - {dataset}

| Metric | Real | Generated | Difference |
|--------|------|-----------|------------|
| Trajectories | {real_count:,} | {gen_count:,} | -- |
| Mean Perplexity | {real_mean_perp:.3f} | {gen_mean_perp:.3f} | {gen_mean_perp - real_mean_perp:+.3f} |
| Std Perplexity | {real_std_perp:.3f} | {gen_std_perp:.3f} | {gen_std_perp - real_std_perp:+.3f} |
| Outlier Rate (%) | {real_outlier_rate:.2f} | {gen_outlier_rate:.2f} | {gen_outlier_rate - real_outlier_rate:+.2f} |
| Normal Rate (%) | {real_normal_rate:.2f} | {gen_normal_rate:.2f} | {gen_normal_rate - real_normal_rate:+.2f} |

## Summary

- **Dataset**: {dataset}
- **Real Trajectories**: {real_count:,}
- **Generated Trajectories**: {gen_count:,}
- **Outlier Rate Difference**: {gen_outlier_rate - real_outlier_rate:+.2f}%
- **Mean Perplexity Difference**: {gen_mean_perp - real_mean_perp:+.3f}

Generated by `tools/plot_lmtad_evaluation.py`
"""

    # Save files
    latex_file = output_file.with_suffix(".tex")
    markdown_file = output_file.with_suffix(".md")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(latex_file, "w") as f:
        f.write(latex_content)
    logger.info(f"Saved LaTeX table: {latex_file}")

    with open(markdown_file, "w") as f:
        f.write(markdown_content)
    logger.info(f"Saved Markdown table: {markdown_file}")


def plot_lmtad_evaluation_from_files(
    real_results_file: Path,
    generated_results_file: Path,
    output_dir: Path,
    dataset: str,
    config: Optional[LMTADPlotConfig] = None,
) -> Dict[str, Path]:
    """Main entry point: generate all LMTAD evaluation plots from results files

    Args:
        real_results_file: Path to real trajectory evaluation results JSON
        generated_results_file: Path to generated trajectory evaluation results JSON
        output_dir: Directory to save all plots
        dataset: Dataset name
        config: Optional plotting configuration

    Returns:
        Dictionary mapping plot names to their file paths

    Raises:
        AssertionError: If required files don't exist
    """
    assert real_results_file.exists(), (
        f"Real results file not found: {real_results_file}"
    )
    assert generated_results_file.exists(), (
        f"Generated results file not found: {generated_results_file}"
    )
    assert output_dir.parent.exists(), (
        f"Output directory parent does not exist: {output_dir.parent}"
    )

    if config is None:
        config = LMTADPlotConfig()

    logger.info("=" * 80)
    logger.info("LM-TAD Evaluation Plots Generation")
    logger.info("=" * 80)

    # Load evaluation results
    real_results = load_evaluation_results(real_results_file)
    generated_results = load_evaluation_results(generated_results_file)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    plot_files = {}

    # 1. Outlier rate comparison
    plot_files["outlier_rate_comparison"] = output_dir / "outlier_rate_comparison.png"
    plot_outlier_rate_comparison(
        real_results,
        generated_results,
        plot_files["outlier_rate_comparison"],
        dataset,
        config,
    )

    # 2. Perplexity distributions
    plot_files["perplexity_distributions"] = output_dir / "perplexity_distributions.png"
    plot_perplexity_distributions(
        real_results,
        generated_results,
        plot_files["perplexity_distributions"],
        dataset,
        config,
    )

    # 3. Normal trajectory rates
    plot_files["normal_trajectory_rates"] = output_dir / "normal_trajectory_rates.png"
    plot_normal_trajectory_rates(
        real_results,
        generated_results,
        plot_files["normal_trajectory_rates"],
        dataset,
        config,
    )

    # 4. Perplexity scatter (if trajectory length data available)
    if (
        "trajectory_lengths" in real_results
        and "trajectory_lengths" in generated_results
    ):
        plot_files["perplexity_scatter"] = output_dir / "perplexity_scatter.png"
        plot_perplexity_scatter(
            real_results,
            generated_results,
            plot_files["perplexity_scatter"],
            dataset,
            config,
        )

    # 5. Summary table
    plot_files["summary_table"] = output_dir / "lmtad_summary_table"
    create_lmtad_summary_table(
        real_results, generated_results, plot_files["summary_table"], dataset
    )

    logger.info(f"Generated {len(plot_files)} plots and tables in {output_dir}")
    return plot_files


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate LM-TAD evaluation plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all LM-TAD evaluation plots
  uv run python tools/plot_lmtad_evaluation.py \\
    --real-results eval_lmtad/porto_hoser/real_evaluation_results.json \\
    --generated-results eval_lmtad/porto_hoser/generated_evaluation_results.json \\
    --output-dir figures/lmtad/porto_hoser \\
    --dataset porto_hoser
        """,
    )

    parser.add_argument(
        "--real-results",
        type=Path,
        required=True,
        help="Path to real trajectory evaluation results JSON",
    )
    parser.add_argument(
        "--generated-results",
        type=Path,
        required=True,
        help="Path to generated trajectory evaluation results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'porto_hoser', 'BJUT_Beijing')",
    )
    parser.add_argument(
        "--perplexity-bins",
        type=int,
        default=50,
        help="Number of bins for perplexity histograms (default: 50)",
    )

    args = parser.parse_args()

    # Fail-fast assertions
    assert args.real_results.exists(), (
        f"Real results file not found: {args.real_results}"
    )
    assert args.generated_results.exists(), (
        f"Generated results file not found: {args.generated_results}"
    )
    assert args.output_dir.parent.exists(), (
        f"Output directory parent does not exist: {args.output_dir.parent}"
    )

    config = LMTADPlotConfig(perplexity_bins=args.perplexity_bins)

    try:
        plot_files = plot_lmtad_evaluation_from_files(
            real_results_file=args.real_results,
            generated_results_file=args.generated_results,
            output_dir=args.output_dir,
            dataset=args.dataset,
            config=config,
        )

        print("\nLM-TAD evaluation plots generated successfully!")
        print(f"Output directory: {args.output_dir}")
        print(f"Generated {len(plot_files)} plots and tables")

    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        raise


if __name__ == "__main__":
    main()
