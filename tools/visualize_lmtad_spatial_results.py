#!/usr/bin/env python3
"""
Visualize LM-TAD Perplexity Evaluation Results

This script generates publication-quality visualizations from aggregated
LM-TAD perplexity evaluation results, focusing on model comparison through
perplexity distributions, rankings, and segment-level analysis.

Usage:
    uv run python tools/visualize_lmtad_spatial_results.py \
        --input analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \
        --output-dir figures/lmtad_perplexity/porto_hoser \
        --dataset porto_hoser
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D

# Import model detection utility
from tools.model_detection import get_model_color, get_display_name  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
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

# Color scheme for perplexity-based visualization
PERPLEXITY_COLORS = {
    "perplexity": "#3498db",  # Blue
    "distribution": "#9b59b6",  # Purple
    "model_comparison": "#2ecc71",  # Green
    "real": "#34495e",  # Dark grey
}


def load_aggregated_results(json_path: Path) -> Dict:
    """Load aggregated results from JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)


# Backward compatibility wrappers for old function names
def plot_spatial_abnormality_rates_comparison(
    results: Dict, output_dir: Path, dataset: str
):
    """
    Backward compatibility wrapper for plot_perplexity_distribution_comparison.

    This function maintains backward compatibility with the old API.
    It calls the new perplexity-focused distribution comparison plot.
    """
    logger.warning(
        "plot_spatial_abnormality_rates_comparison is deprecated. "
        "Use plot_perplexity_distribution_comparison instead."
    )

    # Backwards compatibility: support old 'statistical_tests' structure
    statistical_tests = results.get("statistical_analysis", {}).get(
        "statistical_tests", []
    )
    if statistical_tests:
        # Real rate if available
        real_rate = (
            results.get("summary_statistics", {})
            .get(dataset, {})
            .get("real_spatial_abnormality_rate")
            if dataset
            else None
        )

        # Validate inputs
        for test in statistical_tests:
            gen_rate = test.get("generated_rate")
            ci_lower = test.get("ci_lower")
            ci_upper = test.get("ci_upper")
            # NaN check
            if gen_rate is None or (isinstance(gen_rate, float) and np.isnan(gen_rate)):
                raise AssertionError("Rate cannot be NaN")
            # Non-negative check
            if gen_rate < 0:
                raise AssertionError("Rate must be non-negative")
            # CI sanity check
            if ci_lower is not None and ci_upper is not None:
                if ci_lower > ci_upper:
                    raise AssertionError("ci_lower cannot be > ci_upper")

        # Create simple bar chart with CI error bars and real rate line
        fig, ax = plt.subplots(figsize=(10, 6))
        models = [t.get("model", f"model_{i}") for i, t in enumerate(statistical_tests)]
        rates = [float(t.get("generated_rate", 0)) for t in statistical_tests]
        ci_lowers = [
            float(t.get("ci_lower", r)) for t, r in zip(statistical_tests, rates)
        ]
        ci_uppers = [
            float(t.get("ci_upper", r)) for t, r in zip(statistical_tests, rates)
        ]
        error_lower = [r - lower for r, lower in zip(rates, ci_lowers)]
        error_upper = [upper - r for upper, r in zip(ci_uppers, rates)]
        errors = np.array([error_lower, error_upper])

        ax.bar(models, rates, color=[get_model_color(m) for m in models], alpha=0.8)
        ax.errorbar(models, rates, yerr=errors, fmt="none", ecolor="black", capsize=5)
        if real_rate is not None:
            ax.axhline(y=real_rate, color=get_model_color("real"), linestyle="--")

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Spatial Abnormality Rate (%)", fontsize=12)
        ax.set_title(f"Spatial Abnormality Rates Comparison: {dataset}")
        plt.tight_layout()
        output_file = output_dir / f"spatial_abnormality_rates_{dataset}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Saved to {output_file}")
        return

    # Fallback: use perplexity-focused plot if no statistical_tests present
    return plot_perplexity_distribution_comparison(results, output_dir, dataset)


def plot_route_switch_vs_detour_breakdown(
    results: Dict, output_dir: Path, dataset: str
):
    """
    Backward compatibility placeholder for plot_route_switch_vs_detour_breakdown.

    This function is deprecated as route_switch/detour visualizations have been
    replaced with perplexity-focused visualizations. This function now generates
    a placeholder plot explaining the change.
    """
    logger.warning(
        "plot_route_switch_vs_detour_breakdown is deprecated. "
        "Perplexity-focused visualizations have replaced route_switch/detour analysis."
    )

    # Create a simple info plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5,
        0.5,
        "Route Switch/Detour Visualizations Removed\n\n"
        "Perplexity-focused visualizations now provide\n"
        "more detailed model comparisons.\n\n"
        "Use plot_perplexity_distribution_comparison()\n"
        "or plot_model_rankings_by_perplexity()",
        ha="center",
        va="center",
        fontsize=14,
        transform=ax.transAxes,
    )
    ax.set_title(
        f"Visualization Update Notice: {dataset}", fontsize=16, fontweight="bold"
    )
    ax.axis("off")

    output_file = output_dir / f"route_switch_vs_detour_{dataset}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved notice to {output_file}")


def plot_statistical_significance_spatial(
    results: Dict, output_dir: Path, dataset: str
):
    """
    Backward compatibility wrapper for plot_statistical_significance_perplexity.

    This function maintains backward compatibility with the old API.
    It calls the new perplexity-focused statistical significance plot.
    """
    logger.warning(
        "plot_statistical_significance_spatial is deprecated. "
        "Use plot_statistical_significance_perplexity instead."
    )

    # Backwards compatibility: accept 'statistical_tests' from old format
    statistical_tests = results.get("statistical_analysis", {}).get(
        "statistical_tests", []
    )
    if statistical_tests:
        # Extract rates and significance flags
        models = [t.get("model", f"model_{i}") for i, t in enumerate(statistical_tests)]
        rates = [t.get("generated_rate") for t in statistical_tests]

        # Validate rates: ensure no NaN
        for r in rates:
            if r is None or (isinstance(r, float) and np.isnan(r)):
                raise AssertionError("Rates cannot contain NaN")

        # Build a simple significance plot (bars colored by significance)
        sig_flags = [bool(t.get("significant", False)) for t in statistical_tests]
        colors = ["red" if s else "green" for s in sig_flags]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        ax.bar(x, [float(r) for r in rates], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [get_display_name(m) for m in models], rotation=45, ha="right"
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Spatial Abnormality Rate (%)")
        ax.set_title(
            f"Statistical Significance of Spatial Abnormality Rates: {dataset}"
        )
        plt.tight_layout()
        output_file = output_dir / f"statistical_significance_spatial_{dataset}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Saved to {output_file}")
        return

    return plot_statistical_significance_perplexity(results, output_dir, dataset)


def plot_model_rankings_spatial(results: Dict, output_dir: Path, dataset: str):
    """
    Backward compatibility wrapper for plot_model_rankings_by_perplexity.

    This function maintains backward compatibility with the old API.
    It calls the new perplexity-focused model rankings plot.
    """
    logger.warning(
        "plot_model_rankings_spatial is deprecated. "
        "Use plot_model_rankings_by_perplexity instead."
    )
    return plot_model_rankings_by_perplexity(results, output_dir, dataset)


def plot_perplexity_distribution_comparison(
    results: Dict, output_dir: Path, dataset: str
):
    """
    Plot perplexity distribution comparison across models using violin plots.

    Creates a violin plot showing the distribution of perplexity scores across
    different models, with box plots overlaid for quartile information.
    """
    generated = results.get("generated_data", {}).get(dataset, {})

    if not generated:
        logger.warning(f"No generated data for {dataset}")
        return

    # Collect perplexity data from all models
    all_perplexities = {}
    model_stats = {}

    for model_name, model_data in generated.items():
        log_perp_stats = model_data.get("log_perplexity_stats", {})
        if log_perp_stats and "mean" in log_perp_stats:
            all_perplexities[model_name] = model_data
            model_stats[model_name] = log_perp_stats

    if not all_perplexities:
        logger.warning("No perplexity statistics available")
        return

    # Create synthetic distribution data based on stats
    # For aggregated data, we only have summary statistics, so we'll use
    # normal distribution approximation for visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top plot: Mean and standard deviation comparison
    models = list(model_stats.keys())
    means = [model_stats[m].get("mean", 0) for m in models]
    stds = [model_stats[m].get("std", 0) for m in models]
    medians = [
        model_stats[m].get("median", np.mean([model_stats[m].get("mean", 0)]))
        for m in models
    ]

    colors = [get_model_color(m) for m in models]

    bars = ax1.bar(
        models,
        means,
        yerr=stds,
        color=colors,
        alpha=0.8,
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.1,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Mean Log Perplexity ± Std", fontsize=12)
    ax1.set_title(
        f"Perplexity Statistics Comparison: {dataset}",
        fontsize=14,
        fontweight="bold",
    )
    # Use numeric tick positions for models
    x = np.arange(len(models))
    ax1.set_xticks(x)
    ax1.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Bottom plot: Summary statistics (median, IQR approximation)
    x = np.arange(len(models))
    # Quartile approximations (not used directly; kept for future enhancement)

    # Box plots (using statistical approximations)
    box_parts = ax2.boxplot(
        [np.random.normal(med, std, 1000) for med, std in zip(medians, stds)],
        positions=x,
        widths=0.6,
        patch_artist=True,
        tick_labels=[get_display_name(m) for m in models],
    )

    for patch, color in zip(box_parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Log Perplexity Distribution", fontsize=12)
    ax2.set_title(
        f"Perplexity Distribution Approximation: {dataset}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / f"perplexity_distribution_comparison_{dataset}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_per_od_pair_perplexity_comparison(
    results: Dict,
    output_dir: Path,
    dataset: str,
    vmin_pct: float = 5.0,
    vmax_pct: float = 95.0,
    clip_outliers: bool = True,
    cmap: mpl.colors.Colormap | None = None,
    inf_policy: str = "clip",  # options: 'clip', 'mask'
    annotate_inf: bool = True,
    return_fig: bool = False,
    norm: str | None = None,  # 'linear', 'log', 'twoslope'
    norm_center: float | None = None,
):
    """
    Plot per-OD-pair perplexity comparison across models.

    Creates a heatmap showing perplexity scores for different OD pairs across models.

    Parameters:
    - results (Dict): Aggregated results dict.
    - output_dir (Path): Output directory where images are saved.
    - dataset (str): Dataset key inside results.
    - vmin_pct, vmax_pct (float): Percentiles to use for vmin/vmax to avoid outliers skew.
    - clip_outliers (bool): If True, clip values outside vmin/vmax.
    - cmap (mpl.colors.Colormap or None): Matplotlib colormap to use. Defaults to RdYlGn_r (green->yellow->red).
    - inf_policy (str): How to handle `inf` values produced by evaluation. Options:
        * 'clip' (default): Replace `inf` with `vmax` so they show as maximum (abnormal) color.
        * 'mask': Replace `inf` with NaN and visualize as white mask.
    - annotate_inf (bool): If True, annotate inf cells with `inf` text and red 'x' marker; disabled when clipped and annotate_inf False.
    - return_fig (bool): If True, return the matplotlib Figure object for tests/inspection. Default False.
    """
    od_data = results.get("od_pair_perplexities", {}).get(dataset, {})

    if not od_data:
        logger.warning(f"No OD pair perplexity data for {dataset}")
        return

    # Get list of OD pairs and models
    od_pairs = list(od_data.keys())
    if not od_pairs:
        logger.warning("No OD pairs found in data")
        return

    # Collect all models that have data for these OD pairs
    all_models = set()
    for od_pair in od_pairs:
        for model_name in od_data[od_pair].keys():
            all_models.add(model_name)

    models = sorted(list(all_models))

    if not models:
        logger.warning("No models found in OD pair data")
        return

    # Build matrix of mean perplexities
    perplexity_matrix = []
    valid_od_pairs = []

    for od_pair in od_pairs:
        row = []
        has_data = False
        for model in models:
            model_data = od_data[od_pair].get(model, {})
            mean_perp = model_data.get("mean_log_perplexity")
            if mean_perp is not None:
                row.append(mean_perp)
                has_data = True
            else:
                row.append(np.nan)
        if has_data:
            perplexity_matrix.append(row)
            valid_od_pairs.append(od_pair)

    if not perplexity_matrix:
        logger.warning("No valid perplexity data found")
        return

    perplexity_matrix = np.array(perplexity_matrix)

    # Limit to top 20 OD pairs for readability
    if len(valid_od_pairs) > 20:
        # Sort by average perplexity across all models
        avg_perplexities = np.nanmean(perplexity_matrix, axis=1)
        top_indices = np.argsort(avg_perplexities)[-20:]
        perplexity_matrix = perplexity_matrix[top_indices]
        valid_od_pairs = [valid_od_pairs[i] for i in top_indices]

    # Create heatmap
    fig, ax = plt.subplots(
        figsize=(max(10, len(models) * 1.5), max(8, len(valid_od_pairs) * 0.4))
    )

    # Convert to numpy array and mask invalid values (NaN/inf)
    matrix = np.array(perplexity_matrix, dtype=float)
    mask = ~np.isfinite(matrix)
    if not np.any(~mask):
        logger.warning("No finite perplexity values found for OD pairs")
        return
    mtx_masked = np.ma.masked_invalid(matrix)

    # Compute robust bounds (percentiles) to avoid color skew from outliers
    finite_vals = matrix[~mask]
    try:
        vmin = float(np.nanpercentile(finite_vals, vmin_pct))
        vmax = float(np.nanpercentile(finite_vals, vmax_pct))
    except Exception:
        # Fall back to min/max
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))

    if vmin == vmax:
        # Avoid identical vmin/vmax which breaks coloring
        vmin = vmin - 1e-6
        vmax = vmax + 1e-6

    # Choose default colormap appropriate for abnormality measures (green -> yellow -> red)
    if cmap is None:
        cmap = mpl.cm.RdYlGn_r
    else:
        cmap = mpl.cm.get_cmap(cmap)
    cmap = cmap.copy()
    cmap.set_bad("white")  # ensure NaN/invalid shown as white

    # Optionally create normalization for the colormap
    if norm == "log":
        # LogNorm requires all positive values
        try:
            norm_obj = mpl.colors.LogNorm(vmin=max(1e-6, vmin), vmax=max(vmax, 1e-6))
        except Exception:
            norm_obj = None
    elif norm == "twoslope":
        center = (
            norm_center
            if norm_center is not None
            else float(np.nanpercentile(finite_vals, 50))
        )
        try:
            norm_obj = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        except Exception:
            norm_obj = None
    else:
        norm_obj = None

    # Optionally clip colors to the chosen percentiles
    if clip_outliers:
        # Norm with clip ensures values outside vmin/vmax are clipped in colormap
        norm_clip = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    else:
        norm_clip = None

    # Prepare matrix to plot: handle Inf policy and outlier clipping
    matrix_to_plot = np.array(perplexity_matrix, dtype=float)
    inf_mask = np.isinf(matrix_to_plot)
    nan_mask = np.isnan(matrix_to_plot)

    # Count and report Inf/NaN
    n_inf = int(np.sum(inf_mask))
    n_nan = int(np.sum(nan_mask))
    if n_inf > 0 or n_nan > 0:
        logger.info(
            f"  ⚠️ Found {n_inf} inf and {n_nan} NaN values in OD matrix for {dataset}"
        )

    if n_inf > 0:
        for i, od in enumerate(valid_od_pairs):
            row_inf = inf_mask[i]
            if np.any(row_inf):
                inf_models = [models[j] for j in range(len(models)) if row_inf[j]]
                logger.info(
                    f"  ⚠️ OD pair {od} has inf perplexity for models: {inf_models}"
                )

    if n_nan > 0:
        for i, od in enumerate(valid_od_pairs):
            row_nan = nan_mask[i]
            if np.any(row_nan):
                nan_models = [models[j] for j in range(len(models)) if row_nan[j]]
                logger.info(
                    f"  ⚠️ OD pair {od} has NaN perplexity for models: {nan_models}"
                )

    if inf_policy == "clip":
        if np.isfinite(vmax):
            matrix_to_plot[inf_mask] = vmax
        else:
            # If vmax not finite (degenerate), just set to large finite value
            matrix_to_plot[inf_mask] = np.nanmax(
                matrix_to_plot[np.isfinite(matrix_to_plot)]
            )
    elif inf_policy == "mask":
        matrix_to_plot[inf_mask] = np.nan

    # Optionally clip outliers in the matrix itself
    # Count values that will be clipped for diagnostics
    n_high_clip = int(np.sum(matrix > vmax))
    n_low_clip = int(np.sum(matrix < vmin))
    if (n_high_clip > 0) or (n_low_clip > 0):
        logger.info(
            f"  ⚠️ Clipping {n_high_clip} values above vmax and {n_low_clip} values below vmin for {dataset}"
        )
    if clip_outliers:
        matrix_to_plot = np.clip(matrix_to_plot, vmin, vmax)

    mtx_masked = np.ma.masked_invalid(matrix_to_plot)

    # If user specified LogNorm/Twoslope, prefer those, otherwise use clip-normalize or no normalization
    norm_to_use = norm_obj if norm_obj is not None else norm_clip
    im = ax.imshow(mtx_masked, cmap=cmap, aspect="auto", norm=norm_to_use)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(valid_od_pairs)))
    ax.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax.set_yticklabels(valid_od_pairs)

    # Add text annotations and overlay markers for NaN/Inf
    for i in range(len(valid_od_pairs)):
        for j in range(len(models)):
            value = matrix_to_plot[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=("white" if value > (vmin + vmax) / 2 else "black"),
                    fontsize=8,
                )
            elif np.isinf(value):
                # Show 'inf' explicitly when evaluation returned infinite logs
                if annotate_inf:
                    ax.text(
                        j, i, "inf", ha="center", va="center", color="black", fontsize=8
                    )
                    # Add a red marker for visibility
                    ax.plot(j, i, marker="x", color="red", markersize=6)
            elif np.isnan(value):
                # Show 'N/A' for missing values
                ax.text(
                    j, i, "N/A", ha="center", va="center", color="black", fontsize=8
                )
            else:
                # NaN / missing - show as blank
                pass

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, extend="both")
    cbar.set_label("Mean Log Perplexity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Log histogram and percentiles for diagnostics
    try:
        pct_vals = np.nanpercentile(finite_vals, [vmin_pct, 25, 50, 75, vmax_pct])
        logger.info(
            f"Per-OD heatmap percentiles ({vmin_pct},{vmax_pct}): {pct_vals.tolist()}"
        )
        # Log clipping counts
        clipped_low = np.sum(finite_vals < vmin)
        clipped_high = np.sum(finite_vals > vmax)
        logger.info(
            f"Per-OD heatmap clipping: low={clipped_low}, high={clipped_high}, total={len(finite_vals)}"
        )
    except Exception:
        pass

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("OD Pair (Origin → Destination)", fontsize=12)
    ax.set_title(
        f"Per-OD-Pair Perplexity Comparison: {dataset}",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend entry for Inf marker if present and annotated
    if annotate_inf and n_inf > 0:
        inf_handle = Line2D(
            [0], [0], marker="x", color="red", linestyle="", markersize=6
        )
        # Keep existing legends from other axes if present; append our new handle
        existing_handles, existing_labels = ax.get_legend_handles_labels()
        handles = existing_handles + [inf_handle]
        labels = existing_labels + ["inf"]
        ax.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    output_file = output_dir / f"per_od_pair_perplexity_{dataset}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")

    logger.info(f"  ✓ Saved to {output_file}")

    if return_fig:
        return fig

    plt.close(fig)


def plot_model_rankings_by_perplexity(results: Dict, output_dir: Path, dataset: str):
    """
    Plot model rankings by mean perplexity (lower is better).

    Models are sorted by mean log perplexity in ascending order.
    """
    generated = results.get("generated_data", {}).get(dataset, {})

    if not generated:
        logger.warning(f"No generated data for {dataset}")
        return

    # Collect model statistics
    model_data = {}
    for model_name, data in generated.items():
        log_perp_stats = data.get("log_perplexity_stats", {})
        if log_perp_stats and "mean" in log_perp_stats:
            model_data[model_name] = {
                "mean": log_perp_stats["mean"],
                "std": log_perp_stats.get("std", 0),
                "median": log_perp_stats.get("median", log_perp_stats["mean"]),
                "count": log_perp_stats.get("count", 0),
            }

    if not model_data:
        logger.warning("No perplexity statistics available for ranking")
        return

    # Sort by mean perplexity (ascending - lower is better)
    sorted_models = sorted(model_data.items(), key=lambda x: x[1]["mean"])

    logger.info(
        f"  Ranking {len(sorted_models)} models by mean perplexity for {dataset}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(sorted_models) * 0.5)))

    # Left plot: Mean perplexity ranking
    models = [m[0] for m in sorted_models]
    means = [m[1]["mean"] for m in sorted_models]
    stds = [m[1]["std"] for m in sorted_models]

    colors = [get_model_color(m) for m in models]

    y_pos = np.arange(len(models))

    bars = ax1.barh(
        y_pos,
        means,
        xerr=stds,
        color=colors,
        alpha=0.8,
        error_kw={"elinewidth": 2, "capthick": 2, "capsize": 4},
    )

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        width = bar.get_width()
        ax1.text(
            width + std + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f}±{std:.2f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([get_display_name(m) for m in models])
    ax1.set_xlabel("Mean Log Perplexity ± Std", fontsize=12)
    ax1.set_ylabel("Model (ranked by perplexity)", fontsize=12)
    # Title for model ranking subplot
    ax1.set_title(
        "Model Rankings by Mean Log Perplexity",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, axis="x")

    # Right plot: Median vs Mean comparison
    medians = [m[1]["median"] for m in sorted_models]

    x = np.arange(len(models))
    width = 0.35

    ax2.bar(x - width / 2, means, width, label="Mean", color=colors, alpha=0.8)
    ax2.bar(
        x + width / 2,
        medians,
        width,
        label="Median",
        color=colors,
        alpha=0.5,
        hatch="//",
    )

    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Log Perplexity", fontsize=12)
    ax2.set_title(
        f"Mean vs Median Perplexity: {dataset}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / f"model_rankings_perplexity_{dataset}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_statistical_significance_perplexity(
    results: Dict, output_dir: Path, dataset: str
):
    """
    Plot statistical comparison of perplexity across models with confidence intervals.

    Shows perplexity differences with statistical significance testing.
    """
    # For this version, we'll use the summary statistics
    # Statistical tests might be available in future versions
    generated = results.get("generated_data", {}).get(dataset, {})

    if not generated:
        logger.warning(f"No generated data for {dataset}")
        return

    # Collect perplexity statistics
    model_stats = {}
    for model_name, data in generated.items():
        log_perp_stats = data.get("log_perplexity_stats", {})
        if log_perp_stats and "mean" in log_perp_stats:
            model_stats[model_name] = log_perp_stats

    if not model_stats:
        logger.warning("No perplexity statistics for statistical significance analysis")
        return

    # Sort by model name for consistency
    sorted_models = sorted(model_stats.keys())

    models = []
    means = []
    stds = []
    counts = []
    ci_lowers = []
    ci_uppers = []

    # Calculate 95% confidence intervals
    for model in sorted_models:
        stats = model_stats[model]
        mean = stats["mean"]
        std = stats.get("std", 0)
        count = stats.get("count", 100)  # Default if not available

        # 95% CI: mean ± 1.96 * std / sqrt(n)
        ci_margin = 1.96 * std / np.sqrt(count) if count > 0 else 0

        models.append(model)
        means.append(mean)
        stds.append(std)
        counts.append(count)
        ci_lowers.append(mean - ci_margin)
        ci_uppers.append(mean + ci_margin)

    # Input validation: no NaNs, no negative means, and ci_lower <= ci_upper
    for i, (mean, ci_lower, ci_upper) in enumerate(zip(means, ci_lowers, ci_uppers)):
        if not np.isfinite(mean) or np.isnan(mean):
            raise AssertionError("Mean perplexity must be a finite number")
        if mean < 0:
            raise AssertionError("Mean perplexity must be non-negative")
        if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
            raise AssertionError("CI bounds must be finite")
        if ci_lower > ci_upper:
            raise AssertionError("ci_lower cannot be > ci_upper")

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    colors = [get_model_color(m) for m in models]

    # Error bars (ensure non-negative values)
    error_lowers = np.maximum(0, np.array(means) - np.array(ci_lowers))
    error_uppers = np.maximum(0, np.array(ci_uppers) - np.array(means))
    errors = np.array([error_lowers, error_uppers])

    bars = ax.barh(
        x,
        means,
        xerr=errors,
        color=colors,
        alpha=0.8,
        error_kw={"elinewidth": 2, "capsize": 4, "capthick": 2},
    )

    # Add value labels
    for i, (bar, mean, ci_lower, ci_upper) in enumerate(
        zip(bars, means, ci_lowers, ci_uppers)
    ):
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Mean Log Perplexity with 95% CI", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Perplexity Comparison with Confidence Intervals: {dataset}\n(Error bars show 95% confidence intervals)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks(x)
    ax.set_yticklabels([get_display_name(m) for m in models])
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_file = output_dir / f"statistical_significance_perplexity_{dataset}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_segment_level_perplexity_aggregate(
    results: Dict, output_dir: Path, dataset: str
):
    """
    Plot aggregate segment-level perplexity statistics.

    Shows aggregate statistics for segment-level perplexity scores across models.
    """
    segment_data = results.get("segment_perplexity_stats", {}).get(dataset, {})

    if not segment_data:
        logger.warning(f"No segment-level perplexity data for {dataset}")
        return

    # Collect data for each model
    models = []
    mean_perps = []
    std_perps = []
    counts = []
    percentiles = {}

    for model_name, stats in segment_data.items():
        if "mean" in stats:
            models.append(model_name)
            mean_perps.append(stats["mean"])
            std_perps.append(stats.get("std", 0))
            counts.append(stats.get("count", 0))

            # Collect percentile data if available
            for p in [25, 50, 75, 90, 95]:
                key = f"p{p}"
                if key not in percentiles:
                    percentiles[key] = []
                percentiles[key].append(stats.get(key, 0))

    if not models:
        logger.warning("No valid segment perplexity data found")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean and standard deviation
    colors = [get_model_color(m) for m in models]
    bars = ax1.bar(
        models,
        mean_perps,
        yerr=std_perps,
        color=colors,
        alpha=0.8,
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    for bar, mean in zip(bars, mean_perps):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Mean Log Perplexity", fontsize=12)
    ax1.set_title("Segment-Level Mean Perplexity", fontsize=12, fontweight="bold")
    # Use numeric tick positions for models
    x = np.arange(len(models))
    ax1.set_xticks(x)
    ax1.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Distribution by percentiles
    if percentiles:
        percentile_labels = sorted(percentiles.keys())
        x = np.arange(len(models))
        width = 0.15
        colors_percentiles = plt.cm.viridis(np.linspace(0, 1, len(percentile_labels)))

        for i, (p, vals) in enumerate(percentiles.items()):
            ax2.bar(
                x + i * width,
                vals,
                width,
                label=p,
                color=colors_percentiles[i],
                alpha=0.8,
            )

        ax2.set_xlabel("Model", fontsize=12)
        ax2.set_ylabel("Percentile Value", fontsize=12)
        ax2.set_title("Segment Perplexity Percentiles", fontsize=12, fontweight="bold")
        ax2.set_xticks(x + width * len(percentile_labels) / 2)
        ax2.set_xticklabels(
            [get_display_name(m) for m in models], rotation=45, ha="right"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Count of segments
    bars = ax3.bar(models, counts, color=colors, alpha=0.8)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_ylabel("Number of Segments", fontsize=12)
    ax3.set_title("Total Segments Evaluated", fontsize=12, fontweight="bold")
    # Use numeric tick positions for models
    x = np.arange(len(models))
    ax3.set_xticks(x)
    ax3.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Coefficient of variation (std/mean)
    cv = [std / mean if mean > 0 else 0 for std, mean in zip(std_perps, mean_perps)]
    bars = ax4.bar(models, cv, color=colors, alpha=0.8)

    for bar, val in zip(bars, cv):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(cv) * 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax4.set_xlabel("Model", fontsize=12)
    ax4.set_ylabel("Coefficient of Variation", fontsize=12)
    ax4.set_title("Perplexity Variability (Std/Mean)", fontsize=12, fontweight="bold")
    # Use numeric tick positions for models
    x = np.arange(len(models))
    ax4.set_xticks(x)
    ax4.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Segment-Level Perplexity Statistics: {dataset}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    output_file = output_dir / f"segment_level_perplexity_{dataset}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_comprehensive_perplexity_summary(
    results: Dict, output_dir: Path, dataset: str
):
    """
    Plot a comprehensive summary of all perplexity metrics.

    Creates a multi-panel figure summarizing all perplexity-related statistics.
    """
    generated = results.get("generated_data", {}).get(dataset, {})

    if not generated:
        logger.warning(f"No generated data for {dataset}")
        return

    # Collect all available statistics
    models = []
    perplexity_stats = []
    # Placeholder for future per-OD pair stats usage

    for model_name, data in generated.items():
        log_perp_stats = data.get("log_perplexity_stats", {})
        if log_perp_stats:
            models.append(model_name)
            perplexity_stats.append(log_perp_stats)

    if not models:
        logger.warning("No perplexity statistics available")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean perplexity comparison
    means = [stats.get("mean", 0) for stats in perplexity_stats]
    stds = [stats.get("std", 0) for stats in perplexity_stats]
    colors = [get_model_color(m) for m in models]

    ax1.bar(models, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Mean Log Perplexity", fontsize=12)
    ax1.set_title("Mean Perplexity Comparison", fontsize=12, fontweight="bold")
    # Use numeric tick positions for models
    ax1.set_xticks(np.arange(len(models)))
    ax1.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Median vs Mean
    medians = [stats.get("median", stats.get("mean", 0)) for stats in perplexity_stats]

    x = np.arange(len(models))
    width = 0.35

    ax2.bar(x - width / 2, means, width, label="Mean", color=colors, alpha=0.8)
    ax2.bar(
        x + width / 2,
        medians,
        width,
        label="Median",
        color=colors,
        alpha=0.5,
        hatch="//",
    )

    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Perplexity", fontsize=12)
    ax2.set_title("Mean vs Median", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Range (max - min)
    ranges = [
        stats.get("max", stats.get("mean", 0)) - stats.get("min", stats.get("mean", 0))
        for stats in perplexity_stats
    ]

    x = np.arange(len(models))
    ax3.bar(x, ranges, color=colors, alpha=0.8)
    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_ylabel("Perplexity Range (Max - Min)", fontsize=12)
    ax3.set_title("Perplexity Range", fontsize=12, fontweight="bold")
    ax3.set_xticks(np.arange(len(models)))
    ax3.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Best and worst performers
    best_models = []
    best_perps = []
    worst_models = []
    worst_perps = []

    # Sort by mean perplexity
    sorted_by_perp = sorted(zip(models, means), key=lambda x: x[1])

    # Top 3 best (or all if less than 3)
    for model, perp in sorted_by_perp[:3]:
        best_models.append(get_display_name(model))
        best_perps.append(perp)

    # Top 3 worst (or all if less than 3)
    for model, perp in sorted_by_perp[-3:]:
        worst_models.append(get_display_name(model))
        worst_perps.append(perp)

    # Adjust number of positions based on available data
    n_positions = max(len(best_perps), len(worst_perps), 1)
    x_pos = np.arange(n_positions)

    # Pad arrays if needed to ensure they have the same length
    while len(best_perps) < n_positions:
        best_perps.append(0)
    while len(worst_perps) < n_positions:
        worst_perps.append(0)

    ax4.bar(
        x_pos - 0.2,
        best_perps[:n_positions],
        0.4,
        label="Best",
        color="#2ecc71",
        alpha=0.8,
    )
    ax4.bar(
        x_pos + 0.2,
        worst_perps[:n_positions],
        0.4,
        label="Worst",
        color="#e74c3c",
        alpha=0.8,
    )

    ax4.set_xlabel("Rank", fontsize=12)
    ax4.set_ylabel("Mean Log Perplexity", fontsize=12)
    ax4.set_title("Best vs Worst Models", fontsize=12, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(
        [
            f"{i + 1}{'st' if i == 0 else 'nd' if i == 1 else 'rd'}"
            for i in range(n_positions)
        ]
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Comprehensive Perplexity Summary: {dataset}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    output_file = output_dir / f"comprehensive_perplexity_summary_{dataset}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for LM-TAD perplexity evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations
  uv run python tools/visualize_lmtad_spatial_results.py \\
    --input analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \\
    --output-dir figures/lmtad_perplexity/porto_hoser \\
    --dataset porto_hoser
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to aggregated results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., porto_hoser, Beijing)",
    )
    parser.add_argument(
        "--viz-cmap",
        type=str,
        default=None,
        help="Matplotlib colormap name for OD heatmap (e.g. RdYlGn_r, YlOrRd, viridis).",
    )
    parser.add_argument(
        "--viz-inf-policy",
        choices=["clip", "mask"],
        default="clip",
        help="How to handle inf values in heatmap: 'clip' (default) or 'mask' (white/NaN).",
    )
    parser.add_argument(
        "--viz-no-annotate-inf",
        dest="viz_annotate_inf",
        action="store_false",
        help="Disable annotation of inf cells (default: annotate)",
    )
    parser.add_argument(
        "--viz-clip-outliers",
        action="store_true",
        help="Clip color values to percentile bounds (vmin_pct/vmax_pct).",
    )
    parser.add_argument(
        "--viz-vmin-pct",
        type=float,
        default=5.0,
        help="Vmin percentile for color clipping (default 5.0)",
    )
    parser.add_argument(
        "--viz-vmax-pct",
        type=float,
        default=95.0,
        help="Vmax percentile for color clipping (default 95.0)",
    )
    parser.add_argument(
        "--viz-norm",
        choices=["linear", "log", "twoslope"],
        default="linear",
        help="Color normalization: linear (default), log, or twoslope (requires center)",
    )
    parser.add_argument(
        "--viz-norm-center",
        type=float,
        default=None,
        help="Center value for TwoSlopeNorm (default median)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load results
    results = load_aggregated_results(args.input)

    # Generate all plots
    logger.info(f"📊 Generating perplexity visualizations for {args.dataset}...")

    plot_perplexity_distribution_comparison(results, args.output_dir, args.dataset)
    plot_per_od_pair_perplexity_comparison(
        results,
        args.output_dir,
        args.dataset,
        vmin_pct=args.viz_vmin_pct,
        vmax_pct=args.viz_vmax_pct,
        clip_outliers=args.viz_clip_outliers,
        cmap=args.viz_cmap,
        inf_policy=args.viz_inf_policy,
        annotate_inf=args.viz_annotate_inf,
        norm=(args.viz_norm if args.viz_norm != "linear" else None),
        norm_center=args.viz_norm_center,
    )
    plot_model_rankings_by_perplexity(results, args.output_dir, args.dataset)
    plot_statistical_significance_perplexity(results, args.output_dir, args.dataset)
    plot_segment_level_perplexity_aggregate(results, args.output_dir, args.dataset)
    plot_comprehensive_perplexity_summary(results, args.output_dir, args.dataset)

    logger.info(f"✅ All visualizations saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
