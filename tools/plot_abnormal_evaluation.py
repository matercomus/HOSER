#!/usr/bin/env python3
"""
Plot Abnormal OD Evaluation Results

This module provides plotting functionality for generated trajectory evaluation results.
Plots include abnormality reproduction rates, similarity metrics, category distributions,
and metrics comparison heatmaps.

Usage (Module):
    from tools.plot_abnormal_evaluation import plot_evaluation_from_files
    
    plot_evaluation_from_files(
        comparison_report_file=Path("eval_abnormal/porto_hoser/comparison_report.json"),
        output_dir=Path("figures/abnormal_od/porto_hoser"),
        dataset="porto_hoser"
    )

Usage (CLI):
    uv run python tools/plot_abnormal_evaluation.py \\
        --comparison-report eval_abnormal/porto_hoser/comparison_report.json \\
        --output-dir figures/abnormal_od/porto_hoser \\
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
from matplotlib.patches import Patch

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


@dataclass
class EvaluationPlotConfig:
    """Configuration for evaluation plotting"""

    figure_size: Tuple[int, int] = (12, 6)
    dpi: int = 300


def load_comparison_report(comparison_file: Path) -> Dict[str, Any]:
    """Load comparison report JSON with fail-fast assertions

    Args:
        comparison_file: Path to comparison_report.json file

    Returns:
        Dictionary containing comparison report data

    Raises:
        AssertionError: If file doesn't exist or required keys are missing
    """
    assert comparison_file.exists(), f"Comparison report not found: {comparison_file}"

    logger.info(f"📂 Loading comparison report from {comparison_file}")

    with open(comparison_file, "r") as f:
        report_data = json.load(f)

    assert "model_results" in report_data, (
        f"Comparison report must contain 'model_results' key. "
        f"Found keys: {list(report_data.keys())}"
    )

    assert report_data["model_results"], (
        "Comparison report contains empty 'model_results'. No model results to plot."
    )

    logger.info(
        f"✅ Loaded comparison report with {len(report_data['model_results'])} models"
    )
    return report_data


def plot_abnormality_reproduction_rates(
    model_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[EvaluationPlotConfig] = None,
) -> None:
    """Plot abnormality reproduction rates across models

    Args:
        model_results: Dictionary mapping model names to their results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If model_results is empty
    """
    assert model_results, "Model results cannot be empty"

    if config is None:
        config = EvaluationPlotConfig()

    logger.info("📊 Plotting abnormality reproduction rates...")

    models = []
    rates = []
    counts = []

    for model_name, results in sorted(model_results.items()):
        total_abnormal = sum(
            cat_data["count"] for cat_data in results["abnormality_detection"].values()
        )
        total_traj = results["total_trajectories"]
        rate = (total_abnormal / total_traj * 100) if total_traj > 0 else 0

        models.append(model_name)
        rates.append(rate)
        counts.append(total_abnormal)

    # Create plot
    fig, ax = plt.subplots(figsize=config.figure_size)

    # Bar colors - blue for distilled, red for vanilla
    colors = ["#3498db" if "distill" in m.lower() else "#e74c3c" for m in models]

    bars = ax.barh(models, rates, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, rate, count, total) in enumerate(
        zip(
            bars,
            rates,
            counts,
            [model_results[m]["total_trajectories"] for m in models],
        )
    ):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}% ({count}/{total})",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Abnormality Reproduction Rate (%)", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        f"Abnormality Reproduction Rates - {dataset}\n"
        f"(Generated trajectories reproducing abnormal patterns)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Add legend
    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.8, label="Distilled Models"),
        Patch(facecolor="#e74c3c", alpha=0.8, label="Vanilla Models"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_similarity_metrics(
    model_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[EvaluationPlotConfig] = None,
) -> None:
    """Plot similarity metrics comparison across models

    Args:
        model_results: Dictionary mapping model names to their results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If model_results is empty
    """
    assert model_results, "Model results cannot be empty"

    if config is None:
        config = EvaluationPlotConfig()

    logger.info("📊 Plotting similarity metrics...")

    models = list(sorted(model_results.keys()))

    edr_scores = []
    dtw_scores = []
    hausdorff_scores = []

    for model_name in models:
        metrics = model_results[model_name]["similarity_metrics"]
        edr_scores.append(metrics.get("edr", 0))
        dtw_scores.append(metrics.get("dtw", 0))
        hausdorff_scores.append(metrics.get("hausdorff", 0))

    # Create grouped bar plot
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(
        x - width, edr_scores, width, label="EDR", color="#2ecc71", alpha=0.8
    )
    bars2 = ax.bar(x, dtw_scores, width, label="DTW", color="#3498db", alpha=0.8)
    bars3 = ax.bar(
        x + width,
        hausdorff_scores,
        width,
        label="Hausdorff",
        color="#e74c3c",
        alpha=0.8,
    )

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_ylabel("Similarity Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        f"Trajectory Similarity Metrics - {dataset}\n"
        f"(Lower is better - distance from real abnormal trajectories)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_abnormality_by_category(
    model_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[EvaluationPlotConfig] = None,
) -> None:
    """Plot abnormality detection by category across models

    Args:
        model_results: Dictionary mapping model names to their results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If model_results is empty
    """
    assert model_results, "Model results cannot be empty"

    if config is None:
        config = EvaluationPlotConfig()

    logger.info("📊 Plotting abnormality by category...")

    # Collect all categories
    all_categories = set()
    for results in model_results.values():
        all_categories.update(results["abnormality_detection"].keys())

    categories = sorted(all_categories)
    models = list(sorted(model_results.keys()))

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data
    category_data = {cat: [] for cat in categories}
    for model_name in models:
        abnormal_by_cat = model_results[model_name]["abnormality_detection"]
        for cat in categories:
            count = abnormal_by_cat.get(cat, {}).get("count", 0)
            category_data[cat].append(count)

    # Color scheme for categories
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

    # Create stacked bars
    bottom = np.zeros(len(models))
    for i, cat in enumerate(categories):
        ax.bar(
            models,
            category_data[cat],
            bottom=bottom,
            label=cat,
            color=colors[i],
            alpha=0.8,
        )
        bottom += np.array(category_data[cat])

    ax.set_ylabel("Abnormal Trajectory Count", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        f"Abnormality Detection by Category - {dataset}\n"
        f"(Distribution of abnormal patterns in generated trajectories)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_metrics_comparison_heatmap(
    model_results: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[EvaluationPlotConfig] = None,
) -> None:
    """Create heatmap comparing all metrics across models

    Args:
        model_results: Dictionary mapping model names to their results
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If model_results is empty
    """
    assert model_results, "Model results cannot be empty"

    if config is None:
        config = EvaluationPlotConfig()

    logger.info("📊 Plotting metrics comparison heatmap...")

    models = list(sorted(model_results.keys()))

    # Collect metrics
    metrics_data = []
    metric_names = ["EDR", "DTW", "Hausdorff", "Abnormality Rate (%)"]

    for model_name in models:
        results = model_results[model_name]
        metrics = results["similarity_metrics"]

        total_abnormal = sum(
            cat_data["count"] for cat_data in results["abnormality_detection"].values()
        )
        total_traj = results["total_trajectories"]
        abnormal_rate = (total_abnormal / total_traj * 100) if total_traj > 0 else 0

        metrics_data.append(
            [
                metrics.get("edr", 0),
                metrics.get("dtw", 0),
                metrics.get("hausdorff", 0),
                abnormal_rate,
            ]
        )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.5 + 2))

    # Normalize each column separately for better visualization
    metrics_array = np.array(metrics_data)
    normalized_data = np.zeros_like(metrics_array)
    for i in range(metrics_array.shape[1]):
        col = metrics_array[:, i]
        if col.max() > 0:
            normalized_data[:, i] = col / col.max()

    im = ax.imshow(normalized_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metric_names)
    ax.set_yticklabels(models)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values in cells
    for i in range(len(models)):
        for j in range(len(metric_names)):
            ax.text(
                j,
                i,
                f"{metrics_array[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title(
        f"Metrics Comparison Heatmap - {dataset}\n"
        f"(Normalized by column, darker = worse)",
        fontsize=14,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Score (0=best, 1=worst)", rotation=270, labelpad=20)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_evaluation_from_files(
    comparison_report_file: Path,
    output_dir: Path,
    dataset: str,
    config: Optional[EvaluationPlotConfig] = None,
) -> Dict[str, Path]:
    """Main entry point: generate all evaluation plots from comparison report

    Args:
        comparison_report_file: Path to comparison_report.json
        output_dir: Directory to save all plots
        dataset: Dataset name
        config: Optional plotting configuration

    Returns:
        Dictionary mapping plot names to their file paths

    Raises:
        AssertionError: If required files don't exist
    """
    assert comparison_report_file.exists(), (
        f"Comparison report not found: {comparison_report_file}"
    )
    assert output_dir.parent.exists(), (
        f"Output directory parent does not exist: {output_dir.parent}"
    )

    if config is None:
        config = EvaluationPlotConfig()

    logger.info("=" * 80)
    logger.info("📊 Generating Evaluation Plots")
    logger.info("=" * 80)

    # Load comparison report
    report_data = load_comparison_report(comparison_report_file)
    model_results = report_data["model_results"]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    plot_files = {}

    plot_files["abnormality_reproduction_rates"] = (
        output_dir / "abnormality_reproduction_rates.png"
    )
    plot_abnormality_reproduction_rates(
        model_results, plot_files["abnormality_reproduction_rates"], dataset, config
    )

    plot_files["similarity_metrics_comparison"] = (
        output_dir / "similarity_metrics_comparison.png"
    )
    plot_similarity_metrics(
        model_results, plot_files["similarity_metrics_comparison"], dataset, config
    )

    plot_files["abnormality_by_category"] = output_dir / "abnormality_by_category.png"
    plot_abnormality_by_category(
        model_results, plot_files["abnormality_by_category"], dataset, config
    )

    plot_files["metrics_heatmap"] = output_dir / "metrics_heatmap.png"
    plot_metrics_comparison_heatmap(
        model_results, plot_files["metrics_heatmap"], dataset, config
    )

    logger.info(f"✅ Generated {len(plot_files)} evaluation plots in {output_dir}")
    return plot_files


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots for abnormal OD workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all evaluation plots
  uv run python tools/plot_abnormal_evaluation.py \\
    --comparison-report eval_abnormal/porto_hoser/comparison_report.json \\
    --output-dir figures/abnormal_od/porto_hoser \\
    --dataset porto_hoser
        """,
    )

    parser.add_argument(
        "--comparison-report",
        type=Path,
        required=True,
        help="Path to comparison_report.json file",
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

    args = parser.parse_args()

    # Fail-fast assertions
    assert args.comparison_report.exists(), (
        f"Comparison report not found: {args.comparison_report}"
    )
    assert args.output_dir.parent.exists(), (
        f"Output directory parent does not exist: {args.output_dir.parent}"
    )

    try:
        plot_files = plot_evaluation_from_files(
            comparison_report_file=args.comparison_report,
            output_dir=args.output_dir,
            dataset=args.dataset,
        )

        print("\n✅ Evaluation plots generated successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📊 Generated {len(plot_files)} plots")

    except Exception as e:
        logger.error(f"❌ Plotting failed: {e}")
        raise


if __name__ == "__main__":
    main()
