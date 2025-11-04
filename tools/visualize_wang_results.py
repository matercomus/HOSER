#!/usr/bin/env python3
"""
Wang Statistical Abnormality Detection - Visualization Generator

This script generates publication-quality visualizations from the aggregated
Wang abnormality detection results.

Usage:
    uv run python tools/visualize_wang_results.py
"""

import json
import logging
from pathlib import Path
from typing import Dict

try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
if HAS_MATPLOTLIB:
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
COLORS = {
    "real": "#34495e",  # Dark gray
    "distilled": "#2ecc71",  # Green
    "distilled_seed44": "#27ae60",  # Dark green
    "vanilla": "#e74c3c",  # Red
    "vanilla_seed43": "#c0392b",  # Dark red
    "vanilla_seed44": "#a93226",  # Darker red
    "distill_phase1": "#3498db",  # Blue
    "distill_phase1_seed43": "#2980b9",  # Dark blue
    "distill_phase1_seed44": "#1f618d",  # Darker blue
    "distill_phase2": "#9b59b6",  # Purple
    "distill_phase2_seed43": "#8e44ad",  # Dark purple
    "distill_phase2_seed44": "#7d3c98",  # Darker purple
}

PATTERN_COLORS = {
    "Abp1_normal": "#2ecc71",  # Green
    "Abp2_temporal_delay": "#f39c12",  # Orange
    "Abp3_route_deviation": "#e74c3c",  # Red
    "Abp4_both_deviations": "#8e44ad",  # Purple
}


def load_aggregated_results(json_path: Path) -> Dict:
    """Load aggregated results from JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_abnormality_rates_comparison(results: Dict, output_dir: Path, dataset: str):
    """Plot real vs generated abnormality rates for each model"""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return

    summary = results.get("summary_statistics", {}).get(dataset, {})
    if not summary:
        logger.warning(f"No summary statistics for {dataset}")
        return

    real_rate = summary.get("mean_real_rate", 0)

    # Get model names from comparisons
    model_rates = {}
    for comp in results.get("comparisons", []):
        if comp.get("dataset") == dataset:
            model = comp.get("model")
            if model is None:  # Skip real data entries
                continue
            rate = comp.get("generated_rate", 0)
            split = comp.get("od_source", "unknown")
            key = f"{model}_{split}"
            model_rates[key] = rate

    if not model_rates:
        logger.warning(f"No model rates found for {dataset}")
        return

    # Sort by rate
    sorted_models = sorted(model_rates.items(), key=lambda x: x[1])

    logger.info(f"  Plotting {len(sorted_models)} models for {dataset}")

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_models) * 0.5)))
    models = [m[0] for m in sorted_models]
    rates = [m[1] for m in sorted_models]

    # Plot bars
    colors = [COLORS.get(m.split("_")[0], "#95a5a6") for m in models]
    bars = ax.barh(models, rates, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.2f}%",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Add real rate line
    ax.axvline(
        real_rate,
        color=COLORS["real"],
        linestyle="--",
        linewidth=2,
        label=f"Real Data ({real_rate:.2f}%)",
    )

    ax.set_xlabel("Abnormality Rate (%)", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        f"Abnormality Rates: {dataset} - Real vs Generated",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"abnormality_rates_{dataset.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        output_dir / f"abnormality_rates_{dataset.lower()}.svg", bbox_inches="tight"
    )
    plt.close()

    logger.info(f"✅ Saved: abnormality_rates_{dataset.lower()}.png")


def plot_pattern_distribution(results: Dict, output_dir: Path):
    """Plot pattern distribution (Abp1-4) for each dataset"""
    if not HAS_MATPLOTLIB:
        return

    pattern_data = results.get("statistical_analysis", {}).get(
        "pattern_distributions", {}
    )

    if not pattern_data:
        logger.warning("No pattern distribution data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = ["Beijing", "porto_hoser", "BJUT_Beijing"]
    dataset_labels = ["Beijing", "Porto", "BJUT Beijing"]

    for ax, dataset, label in zip(axes, datasets, dataset_labels):
        data = pattern_data.get(dataset, {})
        if not data:
            ax.text(0.5, 0.5, f"No data for {label}", ha="center", va="center")
            continue

        percentages = data.get("pattern_percentages", {})
        patterns = [
            "Abp1_normal",
            "Abp2_temporal_delay",
            "Abp3_route_deviation",
            "Abp4_both_deviations",
        ]
        pattern_labels = [
            "Normal",
            "Temporal Delay",
            "Route Deviation",
            "Both Deviations",
        ]
        values = [percentages.get(p, 0) for p in patterns]
        colors = [PATTERN_COLORS.get(p, "#95a5a6") for p in patterns]

        bars = ax.bar(pattern_labels, values, color=colors)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_title(f"{label}\nPattern Distribution", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.1 if values else 100)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 1:  # Only label if >1%
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "pattern_distributions.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pattern_distributions.svg", bbox_inches="tight")
    plt.close()

    logger.info("✅ Saved: pattern_distributions.png")


def plot_model_rankings(results: Dict, output_dir: Path):
    """Plot model rankings by realism (deviation from real rate)"""
    if not HAS_MATPLOTLIB:
        return

    rankings = results.get("statistical_analysis", {}).get("model_rankings", {})

    if not rankings:
        logger.warning("No model rankings found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = ["Beijing", "porto_hoser", "BJUT_Beijing"]
    dataset_labels = ["Beijing", "Porto", "BJUT Beijing"]

    for ax, dataset, label in zip(axes, datasets, dataset_labels):
        dataset_rankings = rankings.get(dataset, [])
        if not dataset_rankings:
            ax.text(0.5, 0.5, f"No rankings for {label}", ha="center", va="center")
            continue

        # Get top 6 models
        top_models = dataset_rankings[:6]
        models = [f"{m['model']}\n({m['od_source']})" for m in top_models]
        real_rates = [m["real_rate"] for m in top_models]
        gen_rates = [m["generated_rate"] for m in top_models]

        x = np.arange(len(models))
        width = 0.35

        ax.bar(
            x - width / 2,
            real_rates,
            width,
            label="Real",
            color=COLORS["real"],
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            gen_rates,
            width,
            label="Generated",
            color=COLORS.get("distilled", "#3498db"),
            alpha=0.8,
        )

        ax.set_ylabel("Abnormality Rate (%)", fontsize=12)
        ax.set_title(f"{label}\nTop Models by Realism", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_rankings.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "model_rankings.svg", bbox_inches="tight")
    plt.close()

    logger.info("✅ Saved: model_rankings.png")


def plot_cross_dataset_comparison(results: Dict, output_dir: Path):
    """Plot cross-dataset comparison (Beijing → BJUT)"""
    if not HAS_MATPLOTLIB:
        return

    cross_data = results.get("statistical_analysis", {}).get(
        "cross_dataset_comparison", {}
    )
    beijing_bjut = cross_data.get("beijing_to_bjut", {})

    if not beijing_bjut:
        logger.warning("No cross-dataset comparison data found")
        return

    models = list(beijing_bjut.keys())
    beijing_rates = [beijing_bjut[m]["beijing_rate"] for m in models]
    bjut_rates = [beijing_bjut[m]["bjut_rate"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        beijing_rates,
        width,
        label="Beijing (Same-Network)",
        color=COLORS.get("distilled", "#3498db"),
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        bjut_rates,
        width,
        label="BJUT (Cross-Network)",
        color=COLORS["vanilla"],
        alpha=0.8,
    )

    ax.set_ylabel("Abnormality Rate (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Cross-Dataset Transfer: Beijing → BJUT Beijing", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(
        output_dir / "cross_dataset_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "cross_dataset_comparison.svg", bbox_inches="tight")
    plt.close()

    logger.info("✅ Saved: cross_dataset_comparison.png")


def plot_statistical_significance(results: Dict, output_dir: Path):
    """Plot statistical significance test results (p-values)"""
    if not HAS_MATPLOTLIB:
        return

    tests = results.get("statistical_analysis", {}).get("statistical_tests", {})

    if not tests:
        logger.warning("No statistical test data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = ["Beijing", "porto_hoser", "BJUT_Beijing"]
    dataset_labels = ["Beijing", "Porto", "BJUT Beijing"]

    for ax, dataset, label in zip(axes, datasets, dataset_labels):
        dataset_tests = tests.get(dataset, [])
        if not dataset_tests:
            ax.text(0.5, 0.5, f"No tests for {label}", ha="center", va="center")
            continue

        # Get p-values (log scale)
        models = [f"{t['model']}\n({t['od_source']})" for t in dataset_tests]
        p_values = [max(t["p_value"], 1e-300) for t in dataset_tests]  # Avoid log(0)
        log_p_values = [-np.log10(p) for p in p_values]

        ax.barh(
            models, log_p_values, color=COLORS.get("distilled", "#3498db"), alpha=0.8
        )
        ax.axvline(
            -np.log10(0.001), color="red", linestyle="--", linewidth=2, label="p=0.001"
        )

        ax.set_xlabel("-log10(p-value)", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        ax.set_title(
            f"{label}\nStatistical Significance", fontsize=13, fontweight="bold"
        )
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "statistical_significance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "statistical_significance.svg", bbox_inches="tight")
    plt.close()

    logger.info("✅ Saved: statistical_significance.png")


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent
    results_file = project_root / "wang_results_aggregated.json"
    output_dir = project_root / "figures" / "wang_abnormality"

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.info("Run: uv run python tools/analyze_wang_results.py")
        return

    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available. Install with: uv add matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 Loading results from {results_file}")
    results = load_aggregated_results(results_file)

    logger.info("📊 Generating visualizations...")

    # Generate all plots
    for dataset in ["Beijing", "porto_hoser", "BJUT_Beijing"]:
        plot_abnormality_rates_comparison(results, output_dir, dataset)

    plot_pattern_distribution(results, output_dir)
    plot_model_rankings(results, output_dir)
    plot_cross_dataset_comparison(results, output_dir)
    plot_statistical_significance(results, output_dir)

    logger.info(f"✅ All visualizations saved to {output_dir}")
    logger.info("   Files:")
    logger.info("   - abnormality_rates_*.png/svg (per dataset)")
    logger.info("   - pattern_distributions.png/svg")
    logger.info("   - model_rankings.png/svg")
    logger.info("   - cross_dataset_comparison.png/svg")
    logger.info("   - statistical_significance.png/svg")


if __name__ == "__main__":
    main()
