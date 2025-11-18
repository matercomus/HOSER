#!/usr/bin/env python3
"""
Visualize LM-TAD Spatial Abnormality Evaluation Results

This script generates publication-quality visualizations from aggregated
LM-TAD spatial abnormality evaluation results.

Usage:
    uv run python tools/visualize_lmtad_spatial_results.py \\
        --input analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \\
        --output-dir figures/lmtad_spatial_abnormality/porto_hoser \\
        --dataset porto_hoser
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

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

# Color scheme
SPATIAL_COLORS = {
    "route_switch": "#e74c3c",  # Red
    "detour": "#f39c12",  # Orange
    "non_outlier": "#2ecc71",  # Green
    "real": "#34495e",  # Dark grey
}


def load_aggregated_results(json_path: Path) -> Dict:
    """Load aggregated results from JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_spatial_abnormality_rates_comparison(
    results: Dict, output_dir: Path, dataset: str
):
    """Plot real vs generated spatial abnormality rates for each model with confidence intervals"""
    summary = results.get("summary_statistics", {}).get(dataset, {})
    if not summary:
        logger.warning(f"No summary statistics for {dataset}")
        return

    real_rate = summary.get("real_spatial_abnormality_rate", 0)

    # Get model rates and CIs from statistical tests
    tests = (
        results.get("statistical_analysis", {}).get("statistical_tests", [])
        if isinstance(
            results.get("statistical_analysis", {}).get("statistical_tests", []), list
        )
        else []
    )

    model_data = {}
    for test in tests:
        model = test.get("model")
        if not model:
            continue

        rate = test.get("generated_rate", 0)
        ci_lower = test.get("ci_lower", rate)
        ci_upper = test.get("ci_upper", rate)

        # Validate data types and ranges
        assert isinstance(rate, (int, float)), (
            f"Rate must be numeric for {model}, got {type(rate)}"
        )
        assert isinstance(ci_lower, (int, float)), (
            f"ci_lower must be numeric for {model}, got {type(ci_lower)}"
        )
        assert isinstance(ci_upper, (int, float)), (
            f"ci_upper must be numeric for {model}, got {type(ci_upper)}"
        )
        assert not np.isnan(rate), f"Rate cannot be NaN for {model}"
        assert not np.isnan(ci_lower), f"ci_lower cannot be NaN for {model}"
        assert not np.isnan(ci_upper), f"ci_upper cannot be NaN for {model}"
        assert rate >= 0, f"Rate must be non-negative for {model}, got {rate}"
        assert ci_lower >= 0, (
            f"ci_lower must be non-negative for {model}, got {ci_lower}"
        )
        assert ci_upper >= 0, (
            f"ci_upper must be non-negative for {model}, got {ci_upper}"
        )
        assert ci_lower <= ci_upper, (
            f"Invalid CI bounds for {model}: ci_lower ({ci_lower:.4f}) > ci_upper ({ci_upper:.4f})"
        )

        # Warn if CI bounds are invalid relative to rate (will be clamped to 0 in visualization)
        if ci_lower > rate:
            logger.warning(
                f"Invalid CI bounds for {model}: ci_lower ({ci_lower:.4f}) > rate ({rate:.4f})"
            )
        if ci_upper < rate:
            logger.warning(
                f"Invalid CI bounds for {model}: ci_upper ({ci_upper:.4f}) < rate ({rate:.4f})"
            )

        model_data[model] = {
            "rate": rate,
            "error_lower": rate - ci_lower,
            "error_upper": ci_upper - rate,
            "effect_size": test.get("effect_size", "unknown"),
            "cohens_h": test.get("cohens_h", 0),
        }

    if not model_data:
        logger.warning(f"No model data found for {dataset}")
        return

    # Sort by rate
    sorted_models = sorted(model_data.items(), key=lambda x: x[1]["rate"])

    logger.info(f"  Plotting {len(sorted_models)} models for {dataset}")

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_models) * 0.5)))
    models = [m[0] for m in sorted_models]
    rates = [m[1]["rate"] for m in sorted_models]
    error_lowers = [m[1]["error_lower"] for m in sorted_models]
    error_uppers = [m[1]["error_upper"] for m in sorted_models]

    # Create error bars (ensure non-negative values)
    errors = np.array(
        [
            np.maximum(0, error_lowers),  # Clamp to non-negative
            np.maximum(0, error_uppers),  # Clamp to non-negative
        ]
    )

    # Get colors using model detection utility
    colors = [get_model_color(m) for m in models]
    bars = ax.barh(
        models,
        rates,
        xerr=errors,
        color=colors,
        alpha=0.8,
        error_kw={"elinewidth": 2, "capsize": 4, "capthick": 2, "alpha": 0.7},
    )

    # Add value labels
    for bar, rate in zip(bars, rates):
        width = bar.get_width()
        ax.text(
            width + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.2f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Add real rate line
    ax.axvline(
        real_rate,
        color=SPATIAL_COLORS["real"],
        linestyle="--",
        linewidth=2,
        label=f"Real Data ({real_rate:.2f}%)",
    )

    ax.set_xlabel("Spatial Abnormality Rate (%) with 95% CI", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        f"Spatial Abnormality Rates: {dataset} - Real vs Generated\n(Error bars show 95% confidence intervals)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_file = output_dir / f"spatial_abnormality_rates_{dataset}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_route_switch_vs_detour_breakdown(
    results: Dict, output_dir: Path, dataset: str
):
    """Plot route switch vs detour breakdown as stacked bar chart"""
    summary = results.get("summary_statistics", {}).get(dataset, {})
    if not summary:
        logger.warning(f"No summary statistics for {dataset}")
        return

    real_route_switch = summary.get("real_route_switch_rate", 0)
    real_detour = summary.get("real_detour_rate", 0)

    # Get generated data
    generated = results.get("generated_data", {}).get(dataset, {})

    if not generated:
        logger.warning(f"No generated data for {dataset}")
        return

    models = []
    route_switch_rates = []
    detour_rates = []

    for model_name, model_data in generated.items():
        models.append(model_name)
        route_switch_rates.append(model_data.get("route_switch_rate", 0))
        detour_rates.append(model_data.get("detour_rate", 0))

    # Sort by total spatial abnormality rate
    sorted_indices = sorted(
        range(len(models)),
        key=lambda i: route_switch_rates[i] + detour_rates[i],
    )
    models = [models[i] for i in sorted_indices]
    route_switch_rates = [route_switch_rates[i] for i in sorted_indices]
    detour_rates = [detour_rates[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.6

    # Stacked bars
    ax.bar(
        x,
        route_switch_rates,
        width,
        label="Route Switch",
        color=SPATIAL_COLORS["route_switch"],
        alpha=0.8,
    )
    ax.bar(
        x,
        detour_rates,
        width,
        bottom=route_switch_rates,
        label="Detour",
        color=SPATIAL_COLORS["detour"],
        alpha=0.8,
    )

    # Add real data lines
    ax.axhline(
        real_route_switch,
        color=SPATIAL_COLORS["route_switch"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Real Route Switch ({real_route_switch:.2f}%)",
    )
    ax.axhline(
        real_route_switch + real_detour,
        color=SPATIAL_COLORS["detour"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Real Total ({real_route_switch + real_detour:.2f}%)",
    )

    # Add value labels
    for i, (rs, det) in enumerate(zip(route_switch_rates, detour_rates)):
        total = rs + det
        if total > 0:
            ax.text(
                i,
                total + 0.1,
                f"{total:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spatial Abnormality Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Route Switch vs Detour Breakdown: {dataset}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / f"route_switch_vs_detour_{dataset}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_model_rankings_spatial(results: Dict, output_dir: Path, dataset: str):
    """Plot model rankings by spatial abnormality reproduction rate"""
    tests = (
        results.get("statistical_analysis", {}).get("statistical_tests", [])
        if isinstance(
            results.get("statistical_analysis", {}).get("statistical_tests", []), list
        )
        else []
    )

    if not tests:
        logger.warning(f"No statistical tests for {dataset}")
        return

    # Sort by absolute deviation from real rate
    summary = results.get("summary_statistics", {}).get(dataset, {})
    real_rate = summary.get("real_spatial_abnormality_rate", 0)

    sorted_tests = sorted(
        tests,
        key=lambda t: abs(t.get("generated_rate", 0) - real_rate),
    )

    models = [t.get("model") for t in sorted_tests]
    rates = [t.get("generated_rate", 0) for t in sorted_tests]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(models))
    colors = [get_model_color(m) for m in models]

    bars = ax.barh(x, rates, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        width = bar.get_width()
        ax.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.2f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Add real rate line
    ax.axvline(
        real_rate,
        color=SPATIAL_COLORS["real"],
        linestyle="--",
        linewidth=2,
        label=f"Real Data ({real_rate:.2f}%)",
    )

    ax.set_xlabel("Spatial Abnormality Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model (ranked by deviation)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Model Rankings by Spatial Abnormality Reproduction: {dataset}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks(x)
    ax.set_yticklabels([get_display_name(m) for m in models])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_file = output_dir / f"model_rankings_spatial_{dataset}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_statistical_significance_spatial(
    results: Dict, output_dir: Path, dataset: str
):
    """Plot statistical significance with confidence intervals"""
    tests = (
        results.get("statistical_analysis", {}).get("statistical_tests", [])
        if isinstance(
            results.get("statistical_analysis", {}).get("statistical_tests", []), list
        )
        else []
    )

    if not tests:
        logger.warning(f"No statistical tests for {dataset}")
        return

    summary = results.get("summary_statistics", {}).get(dataset, {})
    real_rate = summary.get("real_spatial_abnormality_rate", 0)

    # Sort by model name for consistency
    sorted_tests = sorted(tests, key=lambda t: t.get("model", ""))

    models = [t.get("model") for t in sorted_tests]
    rates = [t.get("generated_rate", 0) for t in sorted_tests]
    ci_lowers = [t.get("ci_lower", rate) for t, rate in zip(sorted_tests, rates)]
    ci_uppers = [t.get("ci_upper", rate) for t, rate in zip(sorted_tests, rates)]
    significant = [t.get("significant", False) for t in sorted_tests]

    # Validate data before plotting
    assert len(models) == len(rates) == len(ci_lowers) == len(ci_uppers), (
        f"Mismatched array lengths: models={len(models)}, rates={len(rates)}, "
        f"ci_lowers={len(ci_lowers)}, ci_uppers={len(ci_uppers)}"
    )
    rates_arr = np.array(rates)
    ci_lowers_arr = np.array(ci_lowers)
    ci_uppers_arr = np.array(ci_uppers)
    assert np.all(~np.isnan(rates_arr)), "Rates cannot contain NaN values"
    assert np.all(~np.isnan(ci_lowers_arr)), "ci_lowers cannot contain NaN values"
    assert np.all(~np.isnan(ci_uppers_arr)), "ci_uppers cannot contain NaN values"
    assert np.all(rates_arr >= 0), "Rates must be non-negative"
    assert np.all(ci_lowers_arr >= 0), "ci_lowers must be non-negative"
    assert np.all(ci_uppers_arr >= 0), "ci_uppers must be non-negative"
    assert np.all(ci_lowers_arr <= ci_uppers_arr), (
        "ci_lower values must be <= ci_upper values for all models"
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    colors = [
        "#e74c3c" if sig else "#95a5a6" for sig in significant
    ]  # Red if significant, grey if not

    # Error bars (ensure non-negative values)
    error_lowers = np.maximum(0, rates_arr - ci_lowers_arr)
    error_uppers = np.maximum(0, ci_uppers_arr - rates_arr)
    errors = np.array([error_lowers, error_uppers])

    # Final validation: ensure error bars are non-negative
    assert np.all(error_lowers >= 0), "Error lower bounds must be non-negative"
    assert np.all(error_uppers >= 0), "Error upper bounds must be non-negative"

    bars = ax.barh(
        x,
        rates,
        xerr=errors,
        color=colors,
        alpha=0.8,
        error_kw={"elinewidth": 2, "capsize": 4, "capthick": 2},
    )

    # Add real rate line
    ax.axvline(
        real_rate,
        color=SPATIAL_COLORS["real"],
        linestyle="--",
        linewidth=2,
        label=f"Real Data ({real_rate:.2f}%)",
    )

    # Add significance markers
    for i, (bar, sig) in enumerate(zip(bars, significant)):
        if sig:
            ax.text(
                bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                "*",
                ha="left",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#e74c3c",
            )

    ax.set_xlabel(
        "Spatial Abnormality Rate (%) with 95% CI", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Statistical Significance: Spatial Abnormality Rates - {dataset}\n(* = p < 0.05 after FDR correction)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks(x)
    ax.set_yticklabels([get_display_name(m) for m in models])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_file = output_dir / f"statistical_significance_spatial_{dataset}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def plot_perplexity_distribution_spatial(results: Dict, output_dir: Path, dataset: str):
    """Plot perplexity distribution comparison (spatial abnormal vs normal)"""
    generated = results.get("generated_data", {}).get(dataset, {})

    if not generated:
        logger.warning(f"No generated data for {dataset}")
        return

    # We need individual perplexity values, but they may not be in aggregated results
    # For now, create a summary plot based on available statistics
    models = []
    spatial_means = []
    spatial_stds = []

    for model_name, model_data in generated.items():
        log_perp_stats = model_data.get("log_perplexity_stats", {})
        if log_perp_stats:
            models.append(model_name)
            spatial_means.append(log_perp_stats.get("mean", 0))
            spatial_stds.append(log_perp_stats.get("std", 0))

    if not models:
        logger.warning("No perplexity statistics available")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    colors = [get_model_color(m) for m in models]

    ax.bar(
        x,
        spatial_means,
        yerr=spatial_stds,
        color=colors,
        alpha=0.8,
        error_kw={"elinewidth": 2, "capsize": 4, "capthick": 2},
    )

    # Add source statistics lines (defaults from Porto evaluation)
    ax.axhline(
        7.03,
        color=SPATIAL_COLORS["route_switch"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Source Route Switch Mean (~7.03)",
    )
    ax.axhline(
        8.41,
        color=SPATIAL_COLORS["detour"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Source Detour Mean (~8.41)",
    )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Log Perplexity", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Log Perplexity Distribution (Spatial Abnormal Trajectories): {dataset}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([get_display_name(m) for m in models], rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / f"perplexity_distribution_spatial_{dataset}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Saved to {output_file}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for LM-TAD spatial abnormality evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations
  uv run python tools/visualize_lmtad_spatial_results.py \\
    --input analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \\
    --output-dir figures/lmtad_spatial_abnormality/porto_hoser \\
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

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load results
    results = load_aggregated_results(args.input)

    # Generate all plots
    logger.info(f"📊 Generating visualizations for {args.dataset}...")

    plot_spatial_abnormality_rates_comparison(results, args.output_dir, args.dataset)
    plot_route_switch_vs_detour_breakdown(results, args.output_dir, args.dataset)
    plot_model_rankings_spatial(results, args.output_dir, args.dataset)
    plot_statistical_significance_spatial(results, args.output_dir, args.dataset)
    plot_perplexity_distribution_spatial(results, args.output_dir, args.dataset)

    logger.info(f"✅ All visualizations saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
