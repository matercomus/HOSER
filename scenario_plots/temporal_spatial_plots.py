"""
Temporal and spatial scenario analysis plots.

Plots:
- #4: Temporal Scenarios Comparison (3-panel line plot)
- #5: Spatial Complexity Analysis (2-panel figure)
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
)

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def plot_all(data: Dict, output_dir: Path, dpi: int = 300):
    """Generate all temporal/spatial plots"""
    logger.info("  ⏰ Temporal/Spatial plots...")

    plot_temporal_scenarios_comparison(data, output_dir, dpi)
    plot_spatial_complexity_analysis(data, output_dir, dpi)


def plot_temporal_scenarios_comparison(data: Dict, output_dir: Path, dpi: int):
    """Plot #4: 3-row line plot comparing temporal scenarios"""
    logger.info("    4. Temporal scenarios comparison")

    # Dynamic extraction of models and scenarios
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    model_colors = get_model_colors(data, "train")
    model_labels = get_model_labels(data, "train")

    # Get all scenarios and filter to temporal ones
    all_scenarios = get_available_scenarios(data, "train")
    temporal_scenarios = [
        s
        for s in all_scenarios
        if any(kw in s for kw in ["peak", "weekday", "weekend"])
    ]

    if not temporal_scenarios:
        logger.warning("No temporal scenarios found, skipping plot")
        return

    metrics = ["Distance_JSD", "Duration_JSD", "Radius_JSD"]
    metric_labels = ["Distance JSD", "Duration JSD", "Radius of Gyration JSD"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(
        "Temporal Scenario Performance Comparison", fontsize=16, fontweight="bold"
    )

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        x = np.arange(len(temporal_scenarios))

        for model in models:
            values = []
            for s in temporal_scenarios:
                val = get_metric_value(data, "train", model, s, metric)
                values.append(val if val is not None else np.nan)

            ax.plot(
                x,
                values,
                marker="o",
                label=model_labels[model],
                color=model_colors[model],
                linewidth=2.5,
                markersize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in temporal_scenarios])
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(loc="best", framealpha=0.95)

        # Highlight key insight
        if metric == "Duration_JSD":
            ax.axhline(y=0.02, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
            ax.text(
                0.02,
                0.02,
                "Excellent threshold",
                transform=ax.transData,
                fontsize=8,
                va="bottom",
                color="gray",
            )

    axes[-1].set_xlabel("Temporal Scenario", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / "temporal_scenarios_comparison"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_spatial_complexity_analysis(data: Dict, output_dir: Path, dpi: int):
    """Plot #5: 2-panel spatial analysis (bar chart + scatter)"""
    logger.info("    5. Spatial complexity analysis")

    # Dynamic extraction of models and scenarios
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    model_colors = get_model_colors(data, "train")
    model_labels = get_model_labels(data, "train")

    # Get all scenarios and filter to spatial ones
    all_scenarios = get_available_scenarios(data, "train")
    spatial_scenarios = [
        s for s in all_scenarios if any(kw in s for kw in ["center", "suburban"])
    ]

    if not spatial_scenarios:
        logger.warning("No spatial scenarios found, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Spatial Complexity Analysis", fontsize=16, fontweight="bold")

    # Panel 1: Grouped bar chart (Distance + Radius JSD)
    ax1 = axes[0]
    x = np.arange(len(spatial_scenarios))
    width = 0.12

    for i, model in enumerate(models):
        # Distance JSD
        dist_vals = [
            get_metric_value(data, "train", model, s, "Distance_JSD")
            for s in spatial_scenarios
        ]
        ax1.bar(
            x + i * width,
            dist_vals,
            width,
            label=f"{model_labels[model]} (Dist)",
            color=model_colors[model],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Radius JSD (offset)
        radius_vals = [
            get_metric_value(data, "train", model, s, "Radius_JSD")
            for s in spatial_scenarios
        ]
        ax1.bar(
            x + (i + len(models)) * width,
            radius_vals,
            width,
            label=f"{model_labels[model]} (Radius)",
            color=model_colors[model],
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
            hatch="//",
        )

    ax1.set_xlabel("Spatial Scenario", fontsize=11, fontweight="bold")
    ax1.set_ylabel("JSD Value", fontsize=11, fontweight="bold")
    ax1.set_title("Distance vs Radius JSD", fontsize=12, fontweight="bold")
    ax1.set_xticks(x + (len(models) - 1) * width / 2 + len(models) * width / 2)
    ax1.set_xticklabels([s.replace("_", " ").title() for s in spatial_scenarios])
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9, ncol=2)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 2: Scatter plot (avg trip length vs Distance JSD)
    ax2 = axes[1]

    for model in models:
        distances = []
        jsds = []
        labels_list = []

        for scenario in spatial_scenarios:
            dist_mean = get_metric_value(
                data, "train", model, scenario, "Distance_gen_mean"
            )
            jsd = get_metric_value(data, "train", model, scenario, "Distance_JSD")

            if dist_mean is not None and jsd is not None:
                distances.append(dist_mean)
                jsds.append(jsd)
                labels_list.append(scenario)

        ax2.scatter(
            distances,
            jsds,
            s=150,
            label=model_labels[model],
            color=model_colors[model],
            alpha=0.7,
            edgecolors="black",
            linewidth=1.5,
        )

        # Add labels
        for x, y, lbl in zip(distances, jsds, labels_list):
            ax2.annotate(
                lbl.replace("_", " ").title(),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax2.set_xlabel("Average Trip Distance (km)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Distance JSD", fontsize=11, fontweight="bold")
    ax2.set_title("Trip Length vs Distribution Quality", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", framealpha=0.95)
    ax2.grid(alpha=0.3, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / "spatial_scenarios_analysis"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
