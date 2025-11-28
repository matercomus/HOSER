"""
Robustness and difficulty analysis plots.

Plots:
- #6: Seed Robustness Across Scenarios (6-panel bars)
- #7: Scenario Difficulty Ranking (horizontal bars)
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
)

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def plot_all(data: Dict, output_dir: Path, dpi: int = 300):
    """Generate all robustness plots"""
    logger.info("  🔬 Robustness plots...")

    plot_scenario_robustness(data, output_dir, dpi)
    plot_cross_scenario_stability(data, output_dir, dpi)


def plot_scenario_robustness(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #6: 6-panel bar charts comparing seeds"""
    logger.info("    6. Seed robustness across scenarios")

    # Dynamic extraction of models
    vanilla_models, distilled_models = classify_models(data, "train")

    if len(distilled_models) < 2:
        logger.warning(
            "Need at least 2 distilled models for seed robustness comparison, skipping plot"
        )
        return

    # Use first two distilled models (whatever their names/seeds)
    model1 = distilled_models[0]
    model2 = distilled_models[1]
    model_labels_dict = get_model_labels(data, "train")
    model_colors_dict = get_model_colors(data, "train")

    # Get all available scenarios
    scenarios = get_available_scenarios(data, "train")

    # Get all available metrics
    metrics = get_available_metrics(data, "train")

    if not metrics:
        logger.warning("No metrics found for seed robustness comparison, skipping plot")
        return

    # Use up to 6 metrics for the 6-panel plot
    metrics = metrics[:6]

    # Generate metric labels dynamically
    from .data_loader import get_metric_display_labels

    metric_labels = get_metric_display_labels(metrics)

    # Calculate grid layout based on number of metrics
    n_metrics = len(metrics)
    n_rows = (n_metrics + 2) // 3  # Ceiling division for rows
    n_cols = min(3, n_metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle(
        f"Seed Robustness: {model_labels_dict[model1]} vs {model_labels_dict[model2]}",
        fontsize=16,
        fontweight="bold",
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
        x = np.arange(len(scenarios))
        width = 0.35

        # Model 1 values
        model1_vals = [
            get_metric_value(data, "train", model1, s, metric) for s in scenarios
        ]
        model1_vals = [v if v is not None else 0 for v in model1_vals]

        # Model 2 values
        model2_vals = [
            get_metric_value(data, "train", model2, s, metric) for s in scenarios
        ]
        model2_vals = [v if v is not None else 0 for v in model2_vals]

        ax.bar(
            x - width / 2,
            model1_vals,
            width,
            label=model_labels_dict[model1],
            color=model_colors_dict[model1],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            model2_vals,
            width,
            label=model_labels_dict[model2],
            color=model_colors_dict[model2],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Calculate CV (Coefficient of Variation)
        cv_values = []
        for v1, v2 in zip(model1_vals, model2_vals):
            if v1 > 0 and v2 > 0:
                mean = (v1 + v2) / 2
                std = np.std([v1, v2])
                cv = (std / mean) * 100
                cv_values.append(cv)

        if cv_values:
            mean_cv = np.mean(cv_values)
            ax.text(
                0.98,
                0.98,
                f"Avg CV: {mean_cv:.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=9,
            )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", " ").title() for s in scenarios],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_ylabel("Value", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Hide unused subplots if any
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / "seed_robustness_scenarios"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_cross_scenario_stability(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #7: Horizontal bar chart ranking scenarios by difficulty"""
    logger.info("    7. Scenario difficulty ranking")

    # Placeholder implementation
    pass
