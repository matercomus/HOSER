"""
Application-focused plots.

Plots:
- #12: Application Use Case Radar Charts (3 radars)
- #13: Improvement Percentage Heatmap
"""

import logging
from pathlib import Path
from typing import Any, Dict

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


def plot_all(data: Dict, output_dir: Path, dpi: int = 300, loader=None, config=None):
    """Generate all application plots

    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        loader: Optional ScenarioDataLoader for config-based filtering
        config: Optional plot-specific configuration
    """
    logger.info("  🎯 Application plots...")

    plot_application_radars(data, output_dir, dpi, loader=loader, config=config)

    # Generate both individual and grid heatmaps
    plot_improvement_heatmaps_individual(
        data, output_dir, dpi, loader=loader, config=config
    )
    plot_improvement_heatmap_grid(data, output_dir, dpi, loader=loader, config=config)


def plot_application_radars(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #12: 3 radar charts for different applications

    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        loader: Optional ScenarioDataLoader for config-based filtering
        config: Optional plot-specific configuration
    """
    logger.info("    12. Application use case radars")

    # Define metrics for each application
    applications = {
        "Route Planning": {
            "metrics": ["Distance_JSD", "Radius_JSD", "Hausdorff_km", "DTW_km"],
            "labels": [
                "Distance\nQuality",
                "Spatial\nComplexity",
                "Route\nDeviation",
                "Path\nSimilarity",
            ],
            "description": "Emphasis on spatial accuracy and route quality",
        },
        "Traffic Simulation": {
            "metrics": ["Distance_JSD", "Duration_JSD", "DTW_km", "EDR"],
            "labels": [
                "Distance\nRealism",
                "Duration\nRealism",
                "Temporal\nAlignment",
                "Sequence\nSimilarity",
            ],
            "description": "Focus on realistic distributions and temporal patterns",
        },
        "Urban Planning": {
            "metrics": ["Distance_JSD", "Radius_JSD", "Duration_JSD", "Hausdorff_km"],
            "labels": [
                "Trip Length\nPatterns",
                "Spatial\nSpread",
                "Time\nPatterns",
                "Coverage\nArea",
            ],
            "description": "Aggregate patterns and spatial coverage",
        },
    }

    # Dynamically detect all models and get their colors/labels
    vanilla_models, distilled_models = classify_models(data, "train")
    models = sorted(vanilla_models + distilled_models)
    model_colors = get_model_colors(data, "train")
    model_labels = get_model_labels(data, "train")

    if not models:
        logger.warning("    No models found for radar charts")
        return

    # Use first available scenario as representative
    scenarios = get_available_scenarios(data, "train")
    scenario = scenarios[0] if scenarios else "off_peak"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection="polar"))
    fig.suptitle(
        "Application-Specific Performance Profiles", fontsize=16, fontweight="bold"
    )

    for idx, (app_name, app_config) in enumerate(applications.items()):
        ax = axes[idx]
        metrics = app_config["metrics"]
        labels = app_config["labels"]

        # Number of variables
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each model
        for model in models:
            values = []
            for metric in metrics:
                val = get_metric_value(data, "train", model, scenario, metric)
                if val is not None:
                    values.append(val)
                else:
                    values.append(0)

            # Normalize values (invert so higher is better)
            # For JSD, Hausdorff, DTW, EDR: lower is better, so invert
            max_vals = [0.3, 0.3, 2.0, 50.0]  # Approximate max reasonable values
            norm_values = []
            for v, max_v in zip[tuple[Any, float]](values, max_vals):
                # Invert and normalize
                norm = 1 - min(v / max_v, 1.0)
                norm_values.append(norm)

            norm_values += norm_values[:1]  # Complete the circle

            ax.plot(
                angles,
                norm_values,
                "o-",
                linewidth=2,
                label=model_labels[model],
                color=model_colors[model],
            )
            ax.fill(angles, norm_values, alpha=0.15, color=model_colors[model])

        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.set_title(
            f"{app_name}\n{app_config['description']}",
            fontsize=11,
            fontweight="bold",
            pad=20,
        )
        ax.grid(True, linestyle="--", alpha=0.7)

        if idx == 2:
            ax.legend(loc="upper left", bbox_to_anchor=(1.2, 1.0), framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / "application_use_case_radar"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_improvement_heatmaps_individual(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #13a: Individual distance heatmaps showing raw metric values from real data

    Creates one heatmap per distilled model showing distance from real data across
    scenarios and metrics. Uses per-metric color normalization where each metric column
    has its own color scale (green=best, red=worst) while displaying actual raw values.

    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        loader: Optional ScenarioDataLoader for config-based filtering
        config: Optional plot-specific configuration (colormap, percentile_range)
    """
    logger.info("    13a. Individual distance heatmaps (distance from real data)")

    # Detect distilled models (vanilla not needed for distance-from-real-data viz)
    _, distilled_models = classify_models(data, "train")

    if not distilled_models:
        logger.warning("    ⚠️  No distilled models found, skipping individual heatmaps")
        return

    logger.info(f"    Generating {len(distilled_models)} distance heatmap(s)")

    # DYNAMIC: Extract scenarios and metrics from actual data
    scenarios = get_available_scenarios(data, "train")
    metrics = get_available_metrics(data, "train")

    # Apply metric filtering if loader provided
    if loader:
        metrics = loader.get_filtered_metrics(metrics)

    metric_labels = get_metric_display_labels(metrics)

    # Validate we have data to plot
    if not scenarios:
        logger.warning(
            "    ⚠️  No scenarios found in data, skipping individual heatmaps"
        )
        return

    if not metrics:
        logger.warning("    ⚠️  No metrics found in data, skipping individual heatmaps")
        return

    logger.info(
        f"    Using {len(scenarios)} scenarios and {len(metrics)} metrics from data"
    )

    # Generate one heatmap for each distilled model
    for distilled_model in distilled_models:
        # Build distance matrix showing raw metric values (distance from real data)
        distance_matrix = np.zeros((len(scenarios), len(metrics)))

        for i, scenario in enumerate(scenarios):
            for j, metric in enumerate(metrics):
                value = get_metric_value(
                    data,
                    "train",
                    distilled_model,
                    scenario,
                    metric,
                )
                if value is not None:
                    distance_matrix[i, j] = value
                else:
                    distance_matrix[i, j] = np.nan  # Use NaN for missing values

        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate bounds using percentiles to handle outliers
        # Use 5th and 95th percentiles for robust range estimation
        valid_values = distance_matrix[~np.isnan(distance_matrix)]
        if len(valid_values) > 0:
            vmin = np.percentile(valid_values, 5)
            vmax = np.percentile(valid_values, 95)
            # Ensure vmin and vmax are different
            if vmax - vmin < 0.0001:
                vmax = vmin + 0.01
        else:
            vmin, vmax = 0, 1

        # Use colormap from config or default to RdYlGn_r (green=low/good, red=high/bad)
        cmap = config.get("colormap", "RdYlGn_r") if config else "RdYlGn_r"

        # Normalize per metric (column-wise) for color mapping
        # But keep original values for annotations
        normalized_matrix = np.zeros_like(distance_matrix)
        for j in range(distance_matrix.shape[1]):  # For each metric
            col_data = distance_matrix[:, j]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.any():
                col_min = np.nanmin(col_data)
                col_max = np.nanmax(col_data)
                if col_max - col_min > 0.0001:
                    # Normalize to 0-1 range for this metric
                    normalized_matrix[:, j] = (col_data - col_min) / (col_max - col_min)
                else:
                    normalized_matrix[:, j] = 0.5  # Neutral color if all same
            else:
                normalized_matrix[:, j] = np.nan

        # Create heatmap with normalized colors but original value annotations
        sns.heatmap(
            normalized_matrix,
            annot=distance_matrix,  # Show original values
            fmt=".3f",
            cmap=cmap,
            xticklabels=metric_labels,
            yticklabels=[s.replace("_", " ").title() for s in scenarios],
            ax=ax,
            cbar_kws={"label": "Normalized per Metric\n(Green=Best, Red=Worst)"},
            vmin=0,
            vmax=1,
        )

        # Create title with model name
        distilled_label = distilled_model.replace("_", " ").title()

        ax.set_title(
            f"Model Distance from Real Data: {distilled_label}\n"
            f"Across Scenarios and Metrics",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Metric", fontsize=11, fontweight="bold")
        ax.set_ylabel("Scenario", fontsize=11, fontweight="bold")

        plt.tight_layout()

        # Save with descriptive filename
        output_path = output_dir / f"distance_heatmap_{distilled_model}"
        plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
        plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"      ✓ {distilled_model}")


def plot_improvement_heatmap_grid(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #13b: Grid of heatmaps showing raw distance metrics for all models

    Creates a grid with all models (vanilla + distilled) side-by-side for easy comparison.
    Uses per-metric color normalization applied consistently across all models, so colors
    are comparable across subplots for each metric. Raw values are shown as annotations.

    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        loader: Optional ScenarioDataLoader for config-based filtering
        config: Optional plot-specific configuration (colormap, percentile_range)
    """
    logger.info("    13b. Distance heatmap grid (distance from real data)")

    # Detect all models
    vanilla_models, distilled_models = classify_models(data, "train")
    all_models = vanilla_models + distilled_models

    if not all_models:
        logger.warning("    ⚠️  No models found, skipping grid heatmap")
        return

    n_models = len(all_models)

    logger.info(f"    Creating grid of quality heatmaps for {n_models} model(s)")

    # DYNAMIC: Extract scenarios and metrics from actual data
    scenarios = get_available_scenarios(data, "train")
    metrics = get_available_metrics(data, "train")

    # Apply metric filtering if loader provided
    if loader:
        metrics = loader.get_filtered_metrics(metrics)

    metric_labels = get_metric_display_labels(metrics)
    scenario_labels = [s.replace("_", " ").title() for s in scenarios]

    # Validate we have data to plot
    if not scenarios:
        logger.warning("    ⚠️  No scenarios found in data, skipping grid heatmap")
        return

    if not metrics:
        logger.warning("    ⚠️  No metrics found in data, skipping grid heatmap")
        return

    logger.info(
        f"    Using {len(scenarios)} scenarios and {len(metrics)} metrics from data"
    )

    # Calculate figure size: scale with number of models
    # Arrange in a grid layout (e.g., 2x2 for 4 models, 3x2 for 5-6 models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    if n_models <= 2:
        fig_width = 8 * n_cols
        fig_height = 6 * n_rows
        annot_fontsize = 8
    else:
        fig_width = 6 * n_cols
        fig_height = 5 * n_rows
        annot_fontsize = 7

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False
    )

    # Overall title
    fig.suptitle(
        "Model Quality Across Scenarios (Distance from Real Data)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Store all distance values for shared colorbar
    all_distances = []

    # First pass: calculate all distance scores
    distance_data = {}
    for idx, model in enumerate(all_models):
        distance_matrix = np.zeros((len(scenarios), len(metrics)))

        for s_idx, scenario in enumerate(scenarios):
            for m_idx, metric in enumerate(metrics):
                value = get_metric_value(
                    data,
                    "train",
                    model,
                    scenario,
                    metric,
                )
                if value is not None:
                    distance_matrix[s_idx, m_idx] = value
                    all_distances.append(value)
                else:
                    distance_matrix[s_idx, m_idx] = np.nan

        distance_data[idx] = distance_matrix

    # Determine colorbar range using percentiles to exclude outliers
    if all_distances:
        # Use 5th and 95th percentiles for robust range estimation
        vmin = np.percentile(all_distances, 5)
        vmax = np.percentile(all_distances, 95)

        # Ensure vmin and vmax are different
        if vmax - vmin < 0.0001:
            vmax = vmin + 0.01
    else:
        vmin, vmax = 0, 1

    # Use colormap from config or default to RdYlGn_r (green=low/good, red=high/bad)
    cmap = config.get("colormap", "RdYlGn_r") if config else "RdYlGn_r"

    # Calculate per-metric min/max across all models for consistent normalization
    n_metrics = len(metrics)
    metric_ranges = {}
    for j in range(n_metrics):
        all_values_for_metric = []
        for idx in range(len(all_models)):
            col_data = distance_data[idx][:, j]
            all_values_for_metric.extend(col_data[~np.isnan(col_data)])

        if all_values_for_metric:
            metric_ranges[j] = (
                np.min(all_values_for_metric),
                np.max(all_values_for_metric),
            )
        else:
            metric_ranges[j] = (0, 1)

    # Second pass: create heatmaps with per-metric normalization
    for idx, model in enumerate(all_models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        distance_matrix = distance_data[idx]

        # Normalize per metric (column-wise) for color mapping
        normalized_matrix = np.zeros_like(distance_matrix)
        for j in range(n_metrics):
            col_min, col_max = metric_ranges[j]
            if col_max - col_min > 0.0001:
                normalized_matrix[:, j] = (distance_matrix[:, j] - col_min) / (
                    col_max - col_min
                )
            else:
                normalized_matrix[:, j] = 0.5  # Neutral if no variation

        # Create heatmap
        # Only show colorbar on rightmost column
        show_cbar = col == n_cols - 1

        sns.heatmap(
            normalized_matrix,
            annot=distance_matrix,  # Show original values
            fmt=".3f",
            cmap=cmap,
            xticklabels=metric_labels,
            yticklabels=scenario_labels if col == 0 else [],
            ax=ax,
            cbar=show_cbar,
            cbar_kws={"label": "Normalized per Metric\n(Green=Best, Red=Worst)"}
            if show_cbar
            else {},
            vmin=0,
            vmax=1,
            annot_kws={"fontsize": annot_fontsize},
        )

        # Subplot title with model name
        model_label = model.replace("_", " ").title()

        title_fontsize = 10 if n_models > 2 else 11
        ax.set_title(
            f"Distance: {model_label}",
            fontsize=title_fontsize,
            fontweight="bold",
            pad=8,
        )

        # Labels
        if row == n_rows - 1:  # Bottom row
            ax.set_xlabel("Metric", fontsize=9, fontweight="bold")
        else:
            ax.set_xlabel("")

        if col == 0:  # Leftmost column
            ax.set_ylabel("Scenario", fontsize=9, fontweight="bold")
        else:
            ax.set_ylabel("")

        # Adjust tick label sizes
        ax.tick_params(axis="both", labelsize=8)

    # Hide any unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_path = output_dir / "distance_heatmap_grid"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"      ✓ Grid saved with {n_models} model(s)")


def plot_improvement_heatmap(
    data: Dict, output_dir: Path, dpi: int, loader=None, config=None
):
    """Plot #13: Heatmap showing raw distance metrics (DEPRECATED - kept for compatibility)

    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        loader: Optional ScenarioDataLoader for config-based filtering
        config: Optional plot-specific configuration
    """
    logger.info("    13. Distance heatmap (legacy single model)")

    # Use dynamic extraction even for legacy function
    scenarios = get_available_scenarios(data, "train")
    metrics = get_available_metrics(data, "train")

    # Apply metric filtering if loader provided
    if loader:
        metrics = loader.get_filtered_metrics(metrics)

    metric_labels = get_metric_display_labels(metrics)

    if not scenarios or not metrics:
        logger.warning("    ⚠️  No scenarios/metrics found, skipping legacy heatmap")
        return

    # Default to first distilled model or fallback to any model
    vanilla_models, distilled_models = classify_models(data, "train")
    if distilled_models:
        model = distilled_models[0]
    elif vanilla_models:
        model = vanilla_models[0]
    else:
        logger.warning("    ⚠️  No models found, skipping legacy heatmap")
        return

    # Build distance matrix showing raw metric values (distance from real data)
    distance_matrix = np.zeros((len(scenarios), len(metrics)))

    for i, scenario in enumerate(scenarios):
        for j, metric in enumerate(metrics):
            value = get_metric_value(
                data,
                "train",
                model,
                scenario,
                metric,
            )
            if value is not None:
                distance_matrix[i, j] = value
            else:
                distance_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate bounds using percentiles to handle outliers
    # Use 5th and 95th percentiles for robust range estimation
    valid_values = distance_matrix[~np.isnan(distance_matrix)]
    if len(valid_values) > 0:
        vmin = np.percentile(valid_values, 5)
        vmax = np.percentile(valid_values, 95)
        # Ensure vmin and vmax are different
        if vmax - vmin < 0.0001:
            vmax = vmin + 0.01
    else:
        vmin, vmax = 0, 1

    # Use colormap from config or default to RdYlGn_r (green=low/good, red=high/bad)
    cmap = config.get("colormap", "RdYlGn_r") if config else "RdYlGn_r"

    # Normalize per metric (column-wise) for color mapping
    normalized_matrix = np.zeros_like(distance_matrix)
    for j in range(distance_matrix.shape[1]):  # For each metric
        col_data = distance_matrix[:, j]
        valid_mask = ~np.isnan(col_data)
        if valid_mask.any():
            col_min = np.nanmin(col_data)
            col_max = np.nanmax(col_data)
            if col_max - col_min > 0.0001:
                normalized_matrix[:, j] = (col_data - col_min) / (col_max - col_min)
            else:
                normalized_matrix[:, j] = 0.5  # Neutral color if all same
        else:
            normalized_matrix[:, j] = np.nan

    # Create heatmap with normalized colors but original value annotations
    sns.heatmap(
        normalized_matrix,
        annot=distance_matrix,  # Show original values
        fmt=".3f",
        cmap=cmap,
        xticklabels=metric_labels,
        yticklabels=[s.replace("_", " ").title() for s in scenarios],
        ax=ax,
        cbar_kws={"label": "Normalized per Metric\n(Green=Best, Red=Worst)"},
        vmin=0,
        vmax=1,
    )

    model_label = model.replace("_", " ").title()
    ax.set_title(
        f"Model Distance from Real Data: {model_label}\nAcross Scenarios and Metrics",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Metric", fontsize=11, fontweight="bold")
    ax.set_ylabel("Scenario", fontsize=11, fontweight="bold")

    plt.tight_layout()

    output_path = output_dir / "distance_heatmap"
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
