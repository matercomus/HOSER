#!/usr/bin/env python3
"""
LM-TAD Evaluation Results Visualization Script
Creates comprehensive visualizations for teacher-student model comparison
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime
import re
from typing import Dict
import warnings
import sys

# Add tools directory to path for model detection
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from tools.model_detection import extract_model_name, get_display_name, get_model_color  # noqa: E402

warnings.filterwarnings("ignore")

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Configuration
BASE_DIR = Path("/home/matt/Dev/HOSER")
EVAL_DIR = (
    BASE_DIR
    / "hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732"
    / "eval_lmtad_simple"
    / "porto_hoser"
)
OUTPUT_DIR = EVAL_DIR / "figures"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def monitor_for_results(max_wait_hours=3, check_interval_seconds=30) -> bool:
    """
    Monitor for evaluation results to become available
    Returns True when results are found
    """
    print("Monitoring for evaluation results...")
    print(f"Looking for: {EVAL_DIR}/evaluation_results.json")
    print(f"Check interval: {check_interval_seconds}s")
    print(f"Max wait time: {max_wait_hours} hours\n")

    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600

    while (time.time() - start_time) < max_wait_seconds:
        if (EVAL_DIR / "evaluation_results.json").exists():
            print(
                f"\n✓ Results found! Elapsed time: {(time.time() - start_time) / 60:.1f} minutes"
            )
            return True

        elapsed_min = (time.time() - start_time) / 60
        print(f"  Waiting... ({elapsed_min:.1f} min elapsed)", end="\r")
        time.sleep(check_interval_seconds)

    print(f"\n✗ Timeout after {max_wait_hours} hours")
    return False


def parse_results_json(json_path: Path) -> Dict:
    """Parse the main evaluation results JSON file"""
    print(f"Parsing {json_path}")
    with open(json_path, "r") as f:
        results = json.load(f)
    return results


def organize_results_by_model(results: Dict) -> Dict:
    """
    Organize results dictionary by model name and split.

    Converts keys like '2025-11-07_00-13-07_distill_phase1_train' into
    organized structure by model type and split.

    Returns:
        Dictionary with structure: {model_name: {split: data}}
    """
    organized = {}

    for key, data in results.items():
        # Extract model name from key using centralized utility
        model_name = extract_model_name(key)

        # Extract split from key (train/test)
        if "_train" in key:
            split = "train"
        elif "_test" in key:
            split = "test"
        else:
            split = "unknown"

        # Initialize model entry if needed
        if model_name not in organized:
            organized[model_name] = {}

        # Store data by split
        organized[model_name][split] = data

    return organized


def parse_csv_files(csv_dir: Path) -> pd.DataFrame:
    """
    Parse all CSV files following the naming pattern:
    YYYY-MM-DD_HH-MM-SS_{model}_{seed}_{split}.csv
    or YYYY-MM-DD_HH-MM-SS_{model}_{split}.csv
    """
    print(f"Parsing CSV files from {csv_dir}")

    all_data = []
    csv_files = list(csv_dir.glob("*.csv"))

    if not csv_files:
        print("  No CSV files found")
        return pd.DataFrame()

    # Parse filenames to extract model, seed, and split information
    filename_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_([a-z_]+)(?:_seed(\d+))?_([a-z]+)\.csv"
    )

    for csv_file in csv_files:
        match = filename_pattern.match(csv_file.name)
        if match:
            timestamp, model, seed_str, split = match.groups()
            seed = int(seed_str) if seed_str else 42

            df = pd.read_csv(csv_file)
            df["timestamp"] = timestamp
            df["model"] = model
            df["seed"] = seed
            df["split"] = split

            all_data.append(df)
            print(f"  Found: {model} (seed={seed}, split={split})")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Loaded {len(combined_df)} records from {len(csv_files)} files")
        return combined_df
    else:
        print("  No data loaded")
        return pd.DataFrame()


def create_model_comparison_plot(results: Dict, output_dir: Path):
    """Create bar chart comparing mean perplexity across all individual models"""
    print("\n1. Creating model comparison plot...")

    # Organize results by model
    organized = organize_results_by_model(results)

    # Get all individual models (including seed variants) in logical order
    all_model_keys = [
        "vanilla",
        "vanilla_seed43",
        "vanilla_seed44",
        "distill_phase1",
        "distill_phase1_seed43",
        "distill_phase1_seed44",
        "distill_phase2",
        "distill_phase2_seed43",
        "distill_phase2_seed44",
    ]

    fig, ax1 = plt.subplots(figsize=(18, 8))

    # Extract mean perplexity data for each individual model
    models = []
    model_labels = []
    train_perplexity = []
    test_perplexity = []
    train_std = []
    test_std = []

    for model_key in all_model_keys:
        # Check if this model exists in organized results
        if model_key not in organized:
            continue

        # Get train and test data for this specific model
        train_data = organized[model_key].get("train", {})
        test_data = organized[model_key].get("test", {})

        if isinstance(train_data, dict) or isinstance(test_data, dict):
            models.append(model_key)
            model_labels.append(get_display_name(model_key))

            if isinstance(train_data, dict):
                train_perplexity.append(train_data.get("mean_perplexity", 0))
                train_std.append(train_data.get("std_perplexity", 0))
            else:
                train_perplexity.append(0)
                train_std.append(0)

            if isinstance(test_data, dict):
                test_perplexity.append(test_data.get("mean_perplexity", 0))
                test_std.append(test_data.get("std_perplexity", 0))
            else:
                test_perplexity.append(0)
                test_std.append(0)

    if not models:
        print("  ⚠ No model comparison data found")
        return

    x = np.arange(len(models))
    width = 0.35

    # Plot perplexity bars
    bars1 = ax1.bar(
        x - width / 2,
        train_perplexity,
        width,
        label="Train",
        yerr=train_std,
        capsize=5,
        color="skyblue",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        test_perplexity,
        width,
        label="Test",
        yerr=test_std,
        capsize=5,
        color="lightcoral",
        alpha=0.8,
    )

    ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Model Comparison: Mean Perplexity (All Variants)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=45, ha="right")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(train_std + test_std) * 0.05,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout(pad=2.0)

    # Save in both formats
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "model_comparison.svg", bbox_inches="tight")
    print(f"  ✓ Saved to {output_dir}/model_comparison.png and .svg")
    plt.close()


def create_seed_stability_plot(results: Dict, output_dir: Path):
    """Create box plots showing perplexity distribution across seeds for all models"""
    print("\n2. Creating seed stability analysis...")

    # Organize results by model
    organized = organize_results_by_model(results)

    # Extract perplexity values from results
    perplexity_data = []

    for model_name, splits in organized.items():
        for split, data in splits.items():
            if isinstance(data, dict) and "perplexity_values" in data:
                for perplexity in data["perplexity_values"]:
                    perplexity_data.append(
                        {"model": model_name, "split": split, "perplexity": perplexity}
                    )

    if not perplexity_data:
        print("  ⚠ No perplexity data in results")
        return

    df = pd.DataFrame(perplexity_data)

    # Create a single plot with all models
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get all unique models and sort them logically
    all_models = sorted(
        df["model"].unique(),
        key=lambda x: (
            "vanilla" in x,
            "distill_phase1" in x,
            "distill_phase2" in x,
            "seed42" in x,
            "seed43" in x,
            "seed44" in x,
            x,
        ),
    )

    # Create box plot for all models
    sns.boxplot(
        data=df,
        x="model",
        y="perplexity",
        ax=ax,
        palette=[get_model_color(m) for m in all_models],
        order=all_models,
    )

    ax.set_title(
        "Seed Stability Analysis: Perplexity Distribution Across All Models",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=10)

    # Add mean markers
    for i, model in enumerate(all_models):
        model_data = df[df["model"] == model]["perplexity"]
        if len(model_data) > 0:
            mean_val = model_data.mean()
            ax.plot(
                i,
                mean_val,
                marker="D",
                color="red",
                markersize=8,
                markeredgecolor="darkred",
                markeredgewidth=1,
                zorder=10,
            )

    plt.tight_layout(pad=2.0)

    plt.savefig(output_dir / "seed_stability.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "seed_stability.svg", bbox_inches="tight")
    print(f"  ✓ Saved to {output_dir}/seed_stability.png and .svg")
    plt.close()


def create_perplexity_distribution_plot(results: Dict, output_dir: Path):
    """Create KDE curves for perplexity distributions"""
    print("\n3. Creating perplexity distribution comparison...")

    # Organize results by model
    organized = organize_results_by_model(results)

    # Extract perplexity values from results
    perplexity_data = []

    for model_name, splits in organized.items():
        for split, data in splits.items():
            if isinstance(data, dict) and "perplexity_values" in data:
                for perplexity in data["perplexity_values"]:
                    perplexity_data.append(
                        {"model": model_name, "split": split, "perplexity": perplexity}
                    )

    if not perplexity_data:
        print("  ⚠ No perplexity data in results")
        return

    df = pd.DataFrame(perplexity_data)

    # Import scipy for KDE
    from scipy import stats

    # Create a single plot combining all models (differences are negligible)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Combine train and test data since differences are negligible
    all_perplexities = df["perplexity"].values
    min_perp = all_perplexities.min()
    max_perp = all_perplexities.max()
    # Create range with some padding
    padding = (max_perp - min_perp) * 0.1
    x_range = np.linspace(min_perp - padding, max_perp + padding, 300)

    # Use different line styles for seed variants
    linestyles = {
        "vanilla": "-",
        "vanilla_seed43": "--",
        "vanilla_seed44": "-.",
        "distill_phase1": "-",
        "distill_phase1_seed43": "--",
        "distill_phase1_seed44": "-.",
        "distill_phase2": "-",
        "distill_phase2_seed43": "--",
        "distill_phase2_seed44": "-.",
    }

    # All model keys
    all_model_keys = [
        "vanilla",
        "vanilla_seed43",
        "vanilla_seed44",
        "distill_phase1",
        "distill_phase1_seed43",
        "distill_phase1_seed44",
        "distill_phase2",
        "distill_phase2_seed43",
        "distill_phase2_seed44",
    ]

    for model_key in all_model_keys:
        # Get data for this model (combine train and test)
        model_data = df[df["model"] == model_key]["perplexity"].values

        if len(model_data) > 0:
            color = get_model_color(model_key)
            label = get_display_name(model_key)
            linestyle = linestyles.get(model_key, "-")

            # Calculate KDE
            kde = stats.gaussian_kde(model_data)
            ax.plot(
                x_range,
                kde(x_range),
                color=color,
                linewidth=2.5,
                label=label,
                linestyle=linestyle,
                alpha=0.85,
            )

    ax.set_title(
        "Perplexity Distribution Comparison (All Models)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Perplexity", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.95, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout(pad=2.0)

    plt.savefig(
        output_dir / "perplexity_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "perplexity_distributions.svg", bbox_inches="tight")
    print(f"  ✓ Saved to {output_dir}/perplexity_distributions.png and .svg")
    plt.close()


def create_outlier_rate_comparison(
    results: Dict, csv_df: pd.DataFrame, output_dir: Path
):
    """Create bar chart comparing outlier rates"""
    print("\n4. Creating outlier rate comparison...")

    # Organize results by model
    organized = organize_results_by_model(results)

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # From results.json - show all 9 individual models
    all_model_keys = [
        "vanilla",
        "vanilla_seed43",
        "vanilla_seed44",
        "distill_phase1",
        "distill_phase1_seed43",
        "distill_phase1_seed44",
        "distill_phase2",
        "distill_phase2_seed43",
        "distill_phase2_seed44",
    ]

    models = []
    model_labels = []
    outlier_rates = []

    for model_key in all_model_keys:
        if model_key not in organized:
            continue

        # Aggregate outlier rates across splits for this model
        rates = []
        for split, data in organized[model_key].items():
            if isinstance(data, dict):
                rates.append(data.get("outlier_rate", 0))

        if rates:
            models.append(model_key)
            model_labels.append(get_display_name(model_key))
            outlier_rates.append(np.mean(rates))

    if models:
        x = np.arange(len(models))
        colors = [get_model_color(m) for m in models]
        bars = ax1.bar(x, outlier_rates, color=colors, alpha=0.8)
        ax1.set_title(
            "Outlier Rates by Model (All Variants)", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Outlier Rate", fontweight="bold")
        ax1.set_xlabel("Model", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, rotation=45, ha="right")
        ax1.set_ylim(0, max(outlier_rates) * 1.2 if outlier_rates else 0.1)

        # Add percentage labels
        for bar, rate in zip(bars, outlier_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout(pad=2.0)

    plt.savefig(output_dir / "outlier_rates.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "outlier_rates.svg", bbox_inches="tight")
    print(f"  ✓ Saved to {output_dir}/outlier_rates.png and .svg")
    plt.close()


def create_distillation_progression_plot(results: Dict, output_dir: Path):
    """Create comparison plot showing all model variants (vanilla, phase1, phase2) as separate models"""
    print("\n5. Creating model comparison analysis...")

    # Organize results by model
    organized = organize_results_by_model(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Treat all as separate models, not a progression
    model_keys = ["vanilla", "distill_phase1", "distill_phase2"]
    model_labels = ["Vanilla", "HParam Tuning Phase 1", "HParam Tuning Phase 2"]

    # Train and test perplexity
    train_values = []
    test_values = []
    train_stds = []
    test_stds = []

    for model_key in model_keys:
        # Find all variants of this model
        model_variants = [
            m for m in organized.keys() if m.startswith(model_key) or m == model_key
        ]

        if not model_variants:
            train_values.append(0)
            train_stds.append(0)
            test_values.append(0)
            test_stds.append(0)
            continue

        # Aggregate train data
        train_vals = []
        train_std_vals = []
        test_vals = []
        test_std_vals = []

        for variant in model_variants:
            if "train" in organized[variant]:
                data = organized[variant]["train"]
                if isinstance(data, dict):
                    train_vals.append(data.get("mean_perplexity", 0))
                    train_std_vals.append(data.get("std_perplexity", 0))

            if "test" in organized[variant]:
                data = organized[variant]["test"]
                if isinstance(data, dict):
                    test_vals.append(data.get("mean_perplexity", 0))
                    test_std_vals.append(data.get("std_perplexity", 0))

        train_values.append(np.mean(train_vals) if train_vals else 0)
        train_stds.append(np.mean(train_std_vals) if train_std_vals else 0)
        test_values.append(np.mean(test_vals) if test_vals else 0)
        test_stds.append(np.mean(test_std_vals) if test_std_vals else 0)

    if train_values and test_values:
        x = np.arange(len(model_labels))
        width = 0.35

        # Left plot: Bar chart with error bars (clearer than line plot)
        bars1 = ax1.bar(
            x - width / 2,
            train_values,
            width,
            label="Train",
            yerr=train_stds,
            capsize=5,
            alpha=0.8,
            color="skyblue",
            edgecolor="navy",
            linewidth=1.5,
        )
        bars2 = ax1.bar(
            x + width / 2,
            test_values,
            width,
            label="Test",
            yerr=test_stds,
            capsize=5,
            alpha=0.8,
            color="lightcoral",
            edgecolor="darkred",
            linewidth=1.5,
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, rotation=0, ha="center")
        ax1.set_xlabel("Model", fontweight="bold", fontsize=12)
        ax1.set_ylabel("Perplexity", fontweight="bold", fontsize=12)
        ax1.set_title(
            "Model Comparison: Train vs Test Perplexity", fontsize=14, fontweight="bold"
        )
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(train_stds + test_stds) * 0.05,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    # Right plot: Side-by-side comparison (same as left but different view)
    if train_values and len(train_values) >= 3:
        x = np.arange(len(model_labels))
        width = 0.35

        bars1 = ax2.bar(
            x - width / 2,
            train_values,
            width,
            label="Train",
            alpha=0.8,
            color="skyblue",
            edgecolor="navy",
            linewidth=1.5,
        )
        bars2 = ax2.bar(
            x + width / 2,
            test_values,
            width,
            label="Test",
            alpha=0.8,
            color="lightcoral",
            edgecolor="darkred",
            linewidth=1.5,
        )

        ax2.set_xticks(x)
        ax2.set_xticklabels(model_labels, rotation=0, ha="center")
        ax2.set_xlabel("Model", fontweight="bold", fontsize=12)
        ax2.set_ylabel("Perplexity", fontweight="bold", fontsize=12)
        ax2.set_title("Perplexity by Model", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()

    plt.savefig(
        output_dir / "distillation_progression.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "distillation_progression.svg", bbox_inches="tight")
    print(f"  ✓ Saved to {output_dir}/distillation_progression.png and .svg")
    plt.close()


def create_summary_report(results: Dict, csv_df: pd.DataFrame, output_dir: Path):
    """Create a text summary report"""
    print("\n6. Creating summary report...")

    report_path = output_dir / "visualization_summary.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LM-TAD Evaluation Results - Visualization Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results Directory: {EVAL_DIR}\n\n")

        f.write("-" * 80 + "\n")
        f.write("DATA SOURCES\n")
        f.write("-" * 80 + "\n")

        if (EVAL_DIR / "evaluation_results.json").exists():
            f.write(f"✓ Main results: {EVAL_DIR / 'evaluation_results.json'}\n")

        if not csv_df.empty:
            f.write(
                f"✓ CSV data files: {len(csv_df['model'].unique())} unique models\n"
            )
            f.write(f"  - Total records: {len(csv_df)}\n")
            # Filter out NaN values and convert to strings
            models = [str(m) for m in csv_df["model"].unique() if pd.notna(m)]
            splits = [str(s) for s in csv_df["split"].unique() if pd.notna(s)]
            f.write(f"  - Models: {', '.join(models)}\n")
            f.write(f"  - Splits: {', '.join(splits)}\n")
        else:
            f.write("✗ CSV data not available\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("VISUALIZATIONS CREATED\n")
        f.write("-" * 80 + "\n")

        plots = [
            "model_comparison.png/svg - Mean perplexity comparison (all variants)",
            "seed_stability.png/svg - Perplexity distribution across seeds",
            "perplexity_distributions.png/svg - Overlaid histograms",
            "outlier_rates.png/svg - Outlier rate comparison",
        ]

        for plot in plots:
            f.write(f"✓ {plot}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("KEY METRICS\n")
        f.write("-" * 80 + "\n")

        for model_key in ["vanilla", "distill_phase1", "distill_phase2"]:
            if model_key in results:
                data = results[model_key]
                if isinstance(data, dict):
                    f.write(f"\n{model_key.upper().replace('_', ' ')}:\n")
                    f.write(
                        f"  - Mean Perplexity: {data.get('mean_perplexity', 'N/A')}\n"
                    )
                    f.write(
                        f"  - Std Perplexity: {data.get('std_perplexity', 'N/A')}\n"
                    )
                    f.write(f"  - Outlier Rate: {data.get('outlier_rate', 'N/A')}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

    print(f"  ✓ Saved summary to {report_path}")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("LM-TAD Evaluation Results Visualization")
    print("=" * 80 + "\n")

    # Check if results are available
    if not monitor_for_results(max_wait_hours=3, check_interval_seconds=30):
        print("\n⚠ Results not available yet. Exiting.")
        return

    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80 + "\n")

    # Load results
    results = {}
    csv_df = pd.DataFrame()

    if (EVAL_DIR / "evaluation_results.json").exists():
        results = parse_results_json(EVAL_DIR / "evaluation_results.json")
        print(f"\n✓ Loaded {len(results)} metrics from results.json")
    else:
        print("\n⚠ evaluation_results.json not found")

    if (EVAL_DIR / "evaluation_summary.csv").exists():
        csv_df = pd.read_csv(EVAL_DIR / "evaluation_summary.csv")
        print(f"✓ Loaded summary CSV with {len(csv_df)} rows")
    else:
        # Try to load all CSV files
        csv_df = parse_csv_files(EVAL_DIR)

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    # Create all plots
    create_model_comparison_plot(results, OUTPUT_DIR)
    create_seed_stability_plot(results, OUTPUT_DIR)
    create_perplexity_distribution_plot(results, OUTPUT_DIR)
    create_outlier_rate_comparison(results, csv_df, OUTPUT_DIR)
    create_summary_report(results, csv_df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("=" * 80 + "\n")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Total figures created: 4")
    print("Formats: PNG (high-res) + SVG (vector)\n")


if __name__ == "__main__":
    main()
