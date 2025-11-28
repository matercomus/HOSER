"""
Advanced analysis plots.

Plots:
- #9: Duration Ceiling Effect (box plots)
- #10: Spatial Metrics Differentiation (scatter with zones)
- #11: Scenario Variance Analysis (range plot)
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

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def plot_all(data: Dict, output_dir: Path, dpi: int = 300):
    """Generate all analysis plots"""
    logger.info("  📊 Analysis plots...")

    plot_duration_ceiling_effect(data, output_dir, dpi)
    plot_metric_sensitivity_by_scenario(data, output_dir, dpi)
    plot_variance_analysis(data, output_dir, dpi)


def plot_duration_ceiling_effect(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #9: Box plots showing duration JSD ceiling effect"""
    logger.info("    9. Duration ceiling effect")

    # Dynamic extraction
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    model_labels_dict = get_model_labels(data, "train")

    # Get all scenarios and group them dynamically
    all_scenarios = get_available_scenarios(data, "train")

    scenario_groups = {
        "Temporal": [
            s
            for s in all_scenarios
            if any(kw in s for kw in ["peak", "weekday", "weekend"])
        ],
        "Spatial": [
            s for s in all_scenarios if any(kw in s for kw in ["center", "suburban"])
        ],
        "Trip Type": [
            s
            for s in all_scenarios
            if any(kw in s for kw in ["to_", "from_", "within_"])
        ],
    }

    # Filter out empty groups
    scenario_groups = {k: v for k, v in scenario_groups.items() if v}

    fig, ax = plt.subplots(figsize=(12, 6))

    all_data = []
    all_labels = []
    positions = []
    pos = 0

    for group_name, scenarios in scenario_groups.items():
        for model in models:
            values = []
            for s in scenarios:
                val = get_metric_value(data, "train", model, s, "Duration_JSD")
                if val is not None:
                    values.append(val)

            if values:
                all_data.append(values)
                all_labels.append(f"{group_name}\n{model_labels_dict[model]}")
                positions.append(pos)
                pos += 1

        pos += 0.5  # Gap between groups

    # Create box plots
    ax.boxplot(
        all_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=6, alpha=0.5),
    )

    # Overlay individual points
    for i, (values, pos) in enumerate(zip(all_data, positions)):
        x = np.random.normal(pos, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.4, s=30, color="navy")

    # Horizontal line at "excellent" threshold
    ax.axhline(
        y=0.020,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Excellent (<0.020)",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Duration JSD", fontsize=11, fontweight="bold")
    ax.set_title(
        "Duration JSD Ceiling Effect Across Scenario Types",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(0, 0.05)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    output_path = output_dir / "duration_ceiling_effect"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_metric_sensitivity_by_scenario(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #10: Scatter plot showing spatial differentiation"""
    logger.info("    10. Spatial differentiation")

    # Implementation of spatial differentiation / metric sensitivity
    # For now, we'll create a placeholder implementation to avoid errors
    # Ideally this should plot metrics against some spatial characteristic

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5,
        0.5,
        "Metric Sensitivity by Scenario\n(Implementation Pending)",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.set_title("Metric Sensitivity by Scenario", fontsize=16)

    output_path = output_dir / "metric_sensitivity_by_scenario"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_scenario_difficulty_ranking(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot Scenario difficulty ranking by performance"""
    logger.info("    Scenario difficulty ranking")

    # Dynamic extraction
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    scenarios = get_available_scenarios(data, "train")

    if not scenarios:
        return

    # Calculate average performance across all models for Distance_JSD
    scenario_scores = {}
    for s in scenarios:
        scores = []
        for m in models:
            val = get_metric_value(data, "train", m, s, "Distance_JSD")
            if val is not None:
                scores.append(val)
        if scores:
            scenario_scores[s] = np.mean(scores)

    # Sort scenarios by difficulty (higher JSD = harder)
    sorted_scenarios = sorted(scenario_scores.items(), key=lambda x: x[1], reverse=True)

    if not sorted_scenarios:
        return

    names, values = zip(*sorted_scenarios)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(names))

    ax.barh(y_pos, values, align="center", color="skyblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ").title() for n in names])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Average Distance JSD (Higher = More Difficult)")
    ax.set_title("Scenario Difficulty Ranking (Based on Distance JSD)")

    plt.tight_layout()

    output_path = output_dir / "scenario_difficulty_ranking"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_variance_analysis(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #11: Range plot showing variance across scenarios"""
    logger.info("    11. Scenario variance analysis")

    # Dynamic extraction
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    model_colors_dict = get_model_colors(data, "train")
    model_labels_dict = get_model_labels(data, "train")

    scenarios = get_available_scenarios(data, "train")
    metrics = get_available_metrics(data, "train")

    if not metrics:
        logger.warning("No metrics found for variance analysis, skipping plot")
        return

    # Use up to 6 metrics for display
    metrics = metrics[:6]
    metric_labels = get_metric_display_labels(metrics)

    fig, ax = plt.subplots(figsize=(14, 7))

    metric_centers = []
    pos = 0

    for metric, label in zip(metrics, metric_labels):
        metric_start = pos

        for model in models:
            values = []
            for s in scenarios:
                val = get_metric_value(data, "train", model, s, metric)
                if val is not None:
                    values.append(val)

            if values:
                # Normalize to 0-1 for this metric
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    norm_values = [(v - min_val) / (max_val - min_val) for v in values]
                else:
                    norm_values = [0.5] * len(values)

                mean_val = np.mean(norm_values)
                min_norm = min(norm_values)
                max_norm = max(norm_values)
                std_val = np.std(norm_values)
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0

                # Plot range
                ax.plot(
                    [pos, pos],
                    [min_norm, max_norm],
                    color=model_colors_dict[model],
                    linewidth=3,
                    alpha=0.7,
                )
                ax.scatter(
                    pos,
                    mean_val,
                    s=150,
                    color=model_colors_dict[model],
                    zorder=5,
                    edgecolors="black",
                    linewidths=1.5,
                )

                # Add CV annotation
                ax.text(
                    pos,
                    max_norm + 0.03,
                    f"{cv:.1f}%",
                    ha="center",
                    fontsize=8,
                    color=model_colors_dict[model],
                    fontweight="bold",
                )

                pos += 0.3

        # Store center position for this metric
        metric_center = (metric_start + pos - 0.3) / 2
        metric_centers.append(metric_center)

        pos += 0.5  # Gap between metrics

    # Create legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=model_colors_dict[m],
            markersize=10,
            label=model_labels_dict[m],
            markeredgecolor="black",
        )
        for m in models
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95, fontsize=10)

    ax.set_xticks(metric_centers)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Normalized Value (0-1)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Scenario Variance Analysis: Min/Mean/Max Across All Scenarios",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    output_path = output_dir / "scenario_variance_analysis"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
