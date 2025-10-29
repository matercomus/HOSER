"""
Metric comparison plots for scenario analysis.

Plots:
- #1: Scenario Metrics Heatmap (6-panel)
- #2: Train OD Scenario Comparison (grouped bar)
- #3: Test OD Scenario Comparison (grouped bar)
- #8: Metric Sensitivity Grid (3×3)
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data_loader import (
    get_metric_value,
    classify_models,
    get_model_colors,
    get_model_labels,
    get_available_scenarios,
    get_available_metrics,
    get_metric_display_labels,
)

logger = logging.getLogger(__name__)

# Style configuration
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def plot_all(data: Dict, output_dir: Path, dpi: int = 300):
    """Generate all metrics plots"""
    logger.info("  📈 Metrics plots...")

    plot_scenario_metrics_heatmap(data, output_dir, dpi)
    plot_od_scenario_comparison(data, "train", output_dir, dpi)
    plot_od_scenario_comparison(data, "test", output_dir, dpi)
    plot_metric_sensitivity(data, output_dir, dpi)


def plot_scenario_metrics_heatmap(data: Dict, output_dir: Path, dpi: int):
    """Plot #1: 6-panel heatmap of all metrics × scenarios × models"""
    logger.info("    1. Scenario metrics heatmap")

    # Dynamic extraction
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    model_labels_dict = get_model_labels(data, "train")

    scenarios = get_available_scenarios(data, "train")
    metrics = get_available_metrics(data, "train")

    if not metrics:
        logger.warning("No metrics found for heatmap, skipping plot")
        return

    # Use up to 6 metrics for the 6-panel plot
    metrics = metrics[:6]
    metric_labels = get_metric_display_labels(metrics)

    # Calculate grid layout based on number of metrics
    n_metrics = len(metrics)
    n_rows = (n_metrics + 2) // 3  # Ceiling division for rows
    n_cols = min(3, n_metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle(
        "Scenario Performance Metrics: All Models",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Flatten axes for easier indexing
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Build matrix: scenarios × models
        matrix = np.zeros((len(scenarios), len(models)))
        for i, scen in enumerate(scenarios):
            for j, model in enumerate(models):
                val = get_metric_value(data, "train", model, scen, metric)
                matrix[i, j] = val if val is not None else np.nan

        # Normalize for coloring (lower is better)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=[model_labels_dict[m] for m in models],
            yticklabels=[s.replace("_", " ").title() for s in scenarios],
            ax=ax,
            cbar_kws={"label": "Value"},
            vmin=np.nanmin(matrix),
            vmax=np.nanmax(matrix),
        )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused subplots if any
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = output_dir / "scenario_metrics_heatmap"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_od_scenario_comparison(data: Dict, od_source: str, output_dir: Path, dpi: int):
    """Plots #2-3: Grouped bar charts comparing Distance JSD across scenarios"""
    logger.info(
        f"    {2 if od_source == 'train' else 3}. {od_source.upper()} OD comparison"
    )

    # Dynamic extraction
    vanilla_models, distilled_models = classify_models(data, od_source)
    models = sorted(vanilla_models + distilled_models)
    model_colors_dict = get_model_colors(data, od_source)
    model_labels_dict = get_model_labels(data, od_source)

    scenarios = get_available_scenarios(data, od_source)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    width = 0.8 / len(models)  # Dynamic width based on number of models

    for i, model in enumerate(models):
        values = []
        for s in scenarios:
            val = get_metric_value(data, od_source, model, s, "Distance_JSD")
            values.append(val if val is not None else 0)

        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=model_labels_dict[model],
            color=model_colors_dict[model],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance JSD", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{od_source.upper()} OD: Distance JSD by Scenario",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(
        [s.replace("_", " ").title() for s in scenarios], rotation=45, ha="right"
    )
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    output_path = output_dir / f"{od_source}_od_scenario_comparison"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_metric_sensitivity(data: Dict, output_dir: Path, dpi: int):
    """Plot #8: 3×3 grid showing metric sensitivity to scenario types"""
    logger.info("    8. Metric sensitivity grid")

    # Dynamic extraction
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    model_colors_dict = get_model_colors(data, "train")
    model_labels_dict = get_model_labels(data, "train")

    # Get all scenarios and dynamically categorize
    all_scenarios = get_available_scenarios(data, "train")

    temporal_scenarios = [
        s
        for s in all_scenarios
        if any(kw in s for kw in ["peak", "weekday", "weekend"])
    ]
    spatial_scenarios = [
        s for s in all_scenarios if any(kw in s for kw in ["center", "suburban"])
    ]
    trip_type_scenarios = [
        s for s in all_scenarios if any(kw in s for kw in ["to_", "from_", "within_"])
    ]

    scenario_groups = []
    if temporal_scenarios:
        scenario_groups.append((temporal_scenarios, "Temporal Scenarios"))
    if spatial_scenarios:
        scenario_groups.append((spatial_scenarios, "Spatial Scenarios"))
    if trip_type_scenarios:
        scenario_groups.append((trip_type_scenarios, "Trip Types"))

    if not scenario_groups:
        logger.warning(
            "No scenario groups found for sensitivity analysis, skipping plot"
        )
        return

    # Get available metrics, use first 3 for 3-row layout
    all_metrics = get_available_metrics(data, "train")
    if not all_metrics:
        logger.warning("No metrics found for sensitivity analysis, skipping plot")
        return

    metrics = all_metrics[:3]
    metric_labels = get_metric_display_labels(metrics)

    # Create grid based on actual dimensions
    n_rows = len(metrics)
    n_cols = len(scenario_groups)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    fig.suptitle(
        "Metric Sensitivity by Scenario Type", fontsize=16, fontweight="bold", y=0.995
    )

    # Flatten axes for easier indexing if needed
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for row, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for col, (scenarios, group_label) in enumerate(scenario_groups):
            ax = axes[row][col]

            available_scenarios = scenarios  # Already filtered during categorization

            if not available_scenarios:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title(f"{label}\n{group_label}", fontsize=10)
                continue

            x = np.arange(len(available_scenarios))

            for model in models:
                values = []
                for s in available_scenarios:
                    val = get_metric_value(data, "train", model, s, metric)
                    values.append(val if val is not None else 0)

                ax.plot(
                    x,
                    values,
                    marker="o",
                    label=model_labels_dict[model],
                    color=model_colors_dict[model],
                    linewidth=2,
                    markersize=6,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [s.replace("_", " ").title() for s in available_scenarios],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3, linestyle="--")

            if row == 0:
                ax.set_title(group_label, fontsize=10, fontweight="bold")

            if row == 0 and col == 2:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = output_dir / "metric_sensitivity_by_scenario"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
