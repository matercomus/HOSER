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
import glob
import re
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
BASE_DIR = Path("/home/matt/Dev/HOSER")
EVAL_DIR = BASE_DIR / "hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732" / "eval_lmtad_simple" / "porto_hoser"
OUTPUT_DIR = EVAL_DIR / "figures"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    'vanilla': '#E74C3C',
    'vanilla_seed43': '#F1948A',
    'vanilla_seed44': '#F5B7B1',
    'distill_phase1': '#3498DB',
    'distill_phase1_seed43': '#85C1E9',
    'distill_phase1_seed44': '#AED6F1',
    'distill_phase2': '#2ECC71',
    'distill_phase2_seed43': '#7DCEA0',
    'distill_phase2_seed44': '#A9DFBF',
    'real': '#95A5A6'
}


def monitor_for_results(max_wait_hours=3, check_interval_seconds=30) -> bool:
    """
    Monitor for evaluation results to become available
    Returns True when results are found
    """
    print(f"Monitoring for evaluation results...")
    print(f"Looking for: {EVAL_DIR}/evaluation_results.json")
    print(f"Check interval: {check_interval_seconds}s")
    print(f"Max wait time: {max_wait_hours} hours\n")

    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600

    while (time.time() - start_time) < max_wait_seconds:
        if (EVAL_DIR / "evaluation_results.json").exists():
            print(f"\n✓ Results found! Elapsed time: {(time.time() - start_time)/60:.1f} minutes")
            return True

        elapsed_min = (time.time() - start_time) / 60
        print(f"  Waiting... ({elapsed_min:.1f} min elapsed)", end='\r')
        time.sleep(check_interval_seconds)

    print(f"\n✗ Timeout after {max_wait_hours} hours")
    return False


def parse_results_json(json_path: Path) -> Dict:
    """Parse the main evaluation results JSON file"""
    print(f"Parsing {json_path}")
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


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
        r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_([a-z_]+)(?:_seed(\d+))?_([a-z]+)\.csv'
    )

    for csv_file in csv_files:
        match = filename_pattern.match(csv_file.name)
        if match:
            timestamp, model, seed_str, split = match.groups()
            seed = int(seed_str) if seed_str else 42

            df = pd.read_csv(csv_file)
            df['timestamp'] = timestamp
            df['model'] = model
            df['seed'] = seed
            df['split'] = split

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
    """Create bar chart comparing mean perplexity across models"""
    print("\n1. Creating model comparison plot...")

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Extract mean perplexity data
    models = []
    train_perplexity = []
    test_perplexity = []
    train_std = []
    test_std = []
    outlier_rates = []

    # Check for the structure in results
    for model_key in ['vanilla', 'distill_phase1', 'distill_phase2']:
        # Handle different possible data structures
        train_data = results.get(f'{model_key}_train', results.get(f'{model_key}', {}))
        test_data = results.get(f'{model_key}_test', results.get(f'{model_key}', {}))

        if isinstance(train_data, dict):
            models.append(model_key)
            train_perplexity.append(train_data.get('mean_perplexity', 0))
            train_std.append(train_data.get('std_perplexity', 0))
            outlier_rates.append(train_data.get('outlier_rate', 0))
        elif isinstance(train_data, (int, float)):
            models.append(model_key)
            train_perplexity.append(train_data)
            train_std.append(0)
            outlier_rates.append(0)

        if isinstance(test_data, dict):
            test_perplexity.append(test_data.get('mean_perplexity', 0))
            test_std.append(test_data.get('std_perplexity', 0))
        elif isinstance(test_data, (int, float)):
            test_perplexity.append(test_data)
            test_std.append(0)

    if not models:
        print("  ⚠ No model comparison data found")
        return

    x = np.arange(len(models))
    width = 0.35

    # Plot perplexity bars
    bars1 = ax1.bar(x - width/2, train_perplexity, width, label='Train',
                    yerr=train_std, capsize=5, color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_perplexity, width, label='Test',
                    yerr=test_std, capsize=5, color='lightcoral', alpha=0.8)

    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison: Mean Perplexity', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='upper left')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # Add outlier rates on secondary y-axis if available
    if outlier_rates and any(rate > 0 for rate in outlier_rates):
        ax2 = ax1.twinx()
        line = ax2.plot(x, outlier_rates, color='red', marker='o', linewidth=2,
                       markersize=8, label='Outlier Rate')
        ax2.set_ylabel('Outlier Rate', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')

        for i, rate in enumerate(outlier_rates):
            ax2.text(i, rate + max(outlier_rates) * 0.02, f'{rate:.1%}',
                    ha='center', va='bottom', fontsize=10, color='red')

    plt.tight_layout()

    # Save in both formats
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "model_comparison.svg", bbox_inches='tight')
    print(f"  ✓ Saved to {output_dir}/model_comparison.png and .svg")
    plt.close()


def create_seed_stability_plot(csv_df: pd.DataFrame, output_dir: Path):
    """Create box plots showing perplexity distribution across seeds"""
    print("\n2. Creating seed stability analysis...")

    if csv_df.empty or 'perplexity' not in csv_df.columns:
        print("  ⚠ No perplexity data in CSV files")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    model_types = ['vanilla', 'distill_phase1', 'distill_phase2']
    splits = ['train', 'test']

    for i, model_type in enumerate(model_types):
        for j, split in enumerate(splits):
            ax = axes[i * 2 + j]

            # Filter data for this model and split
            subset = csv_df[
                (csv_df['model'].str.contains(model_type)) &
                (csv_df['split'] == split)
            ]

            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{model_type} - {split}', fontweight='bold')
                continue

            # Create box plot
            sns.boxplot(data=subset, x='model', y='perplexity', ax=ax,
                       palette=[MODEL_COLORS.get(m, '#999999') for m in subset['model'].unique()])

            ax.set_title(f'{model_type} - {split}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model Variant', fontweight='bold')
            ax.set_ylabel('Perplexity', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)

            # Add mean markers
            for k, model in enumerate(subset['model'].unique()):
                model_data = subset[subset['model'] == model]['perplexity']
                ax.plot(k, model_data.mean(), marker='D', color='red',
                       markersize=8, markeredgecolor='darkred', markeredgewidth=1)

    plt.suptitle('Seed Stability Analysis: Perplexity Distribution', fontsize=18,
                fontweight='bold', y=0.98)
    plt.tight_layout()

    plt.savefig(output_dir / "seed_stability.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "seed_stability.svg", bbox_inches='tight')
    print(f"  ✓ Saved to {output_dir}/seed_stability.png and .svg")
    plt.close()


def create_perplexity_distribution_plot(csv_df: pd.DataFrame, output_dir: Path):
    """Create overlaid histograms for perplexity distributions"""
    print("\n3. Creating perplexity distribution comparison...")

    if csv_df.empty or 'perplexity' not in csv_df.columns:
        print("  ⚠ No perplexity data in CSV files")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, split in enumerate(['train', 'test']):
        ax = axes[i]

        split_data = csv_df[csv_df['split'] == split]

        if split_data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{split.title()} Split', fontweight='bold')
            continue

        # Plot distributions for each model
        for model in split_data['model'].unique():
            model_data = split_data[split_data['model'] == model]['perplexity']
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                alpha = 0.7 if 'seed' not in model else 0.4
                label = model.replace('_', ' ').title()

                ax.hist(model_data, bins=50, alpha=alpha, color=color,
                       label=label, density=True)

        ax.set_title(f'{split.title()} Split - Perplexity Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Perplexity', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_dir / "perplexity_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "perplexity_distributions.svg", bbox_inches='tight')
    print(f"  ✓ Saved to {output_dir}/perplexity_distributions.png and .svg")
    plt.close()


def create_outlier_rate_comparison(results: Dict, csv_df: pd.DataFrame, output_dir: Path):
    """Create bar chart comparing outlier rates"""
    print("\n4. Creating outlier rate comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # From results.json
    models = []
    outlier_rates = []

    for model_key in ['vanilla', 'distill_phase1', 'distill_phase2']:
        if model_key in results:
            data = results[model_key]
            if isinstance(data, dict):
                models.append(model_key.replace('_', ' ').title())
                outlier_rates.append(data.get('outlier_rate', 0))

    if models:
        bars = ax1.bar(models, outlier_rates, color=['#E74C3C', '#3498DB', '#2ECC71'],
                      alpha=0.8)
        ax1.set_title('Outlier Rates by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Outlier Rate', fontweight='bold')
        ax1.set_ylim(0, max(outlier_rates) * 1.2 if outlier_rates else 0.1)

        # Add percentage labels
        for bar, rate in zip(bars, outlier_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1%}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    # From CSV data if available
    if not csv_df.empty and 'is_outlier' in csv_df.columns:
        outlier_by_model = csv_df.groupby(['model', 'split'])['is_outlier'].mean().reset_index()

        # Train data
        train_outliers = outlier_by_model[outlier_by_model['split'] == 'train']
        if not train_outliers.empty:
            ax2.bar(train_outliers['model'], train_outliers['is_outlier'],
                   color='skyblue', alpha=0.8, label='Train')

        # Test data
        test_outliers = outlier_by_model[outlier_by_model['split'] == 'test']
        if not test_outliers.empty:
            ax2.bar(test_outliers['model'], test_outliers['is_outlier'],
                   color='lightcoral', alpha=0.8, label='Test')

        ax2.set_title('Outlier Rates by Model (CSV Data)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Outlier Rate', fontweight='bold')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No outlier data in CSV', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)

    plt.tight_layout()

    plt.savefig(output_dir / "outlier_rates.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "outlier_rates.svg", bbox_inches='tight')
    print(f"  ✓ Saved to {output_dir}/outlier_rates.png and .svg")
    plt.close()


def create_distillation_progression_plot(results: Dict, output_dir: Path):
    """Create line plot showing distillation progression"""
    print("\n5. Creating distillation progression analysis...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    stages = ['Vanilla', 'Phase 1', 'Phase 2']
    stage_keys = ['vanilla', 'distill_phase1', 'distill_phase2']

    # Train perplexity
    train_values = []
    test_values = []
    train_stds = []
    test_stds = []

    for stage_key in stage_keys:
        if stage_key in results:
            data = results[stage_key]
            if isinstance(data, dict):
                train_values.append(data.get('mean_perplexity', 0))
                train_stds.append(data.get('std_perplexity', 0))
                test_values.append(data.get('mean_perplexity_test', data.get('mean_perplexity', 0)))
                test_stds.append(data.get('std_perplexity_test', data.get('std_perplexity', 0)))

    if train_values and test_values:
        x = range(len(stages))

        # Plot with error bars
        ax1.errorbar(x, train_values, yerr=train_stds, marker='o', linewidth=2,
                    markersize=8, label='Train', capsize=5)
        ax1.errorbar(x, test_values, yerr=test_stds, marker='s', linewidth=2,
                    markersize=8, label='Test', capsize=5)

        ax1.set_xticks(x)
        ax1.set_xticklabels(stages)
        ax1.set_xlabel('Distillation Stage', fontweight='bold')
        ax1.set_ylabel('Perplexity', fontweight='bold')
        ax1.set_title('Distillation Progression', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add improvement percentages
        for i in range(1, len(train_values)):
            train_improvement = (train_values[0] - train_values[i]) / train_values[0] * 100
            test_improvement = (test_values[0] - test_values[i]) / test_values[0] * 100

            ax1.annotate(f'{train_improvement:.1f}%', (i, train_values[i]),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=10, color='blue')
            ax1.annotate(f'{test_improvement:.1f}%', (i, test_values[i]),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=10, color='red')

    # Comparison bar chart
    if train_values and len(train_values) >= 3:
        x = np.arange(len(stages))
        width = 0.35

        bars1 = ax2.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
        bars2 = ax2.bar(x + width/2, test_values, width, label='Test', alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(stages)
        ax2.set_xlabel('Distillation Stage', fontweight='bold')
        ax2.set_ylabel('Perplexity', fontweight='bold')
        ax2.set_title('Perplexity by Stage', fontsize=14, fontweight='bold')
        ax2.legend()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    plt.savefig(output_dir / "distillation_progression.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "distillation_progression.svg", bbox_inches='tight')
    print(f"  ✓ Saved to {output_dir}/distillation_progression.png and .svg")
    plt.close()


def create_summary_report(results: Dict, csv_df: pd.DataFrame, output_dir: Path):
    """Create a text summary report"""
    print("\n6. Creating summary report...")

    report_path = output_dir / "visualization_summary.txt"

    with open(report_path, 'w') as f:
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
            f.write(f"✓ CSV data files: {len(csv_df['model'].unique())} unique models\n")
            f.write(f"  - Total records: {len(csv_df)}\n")
            f.write(f"  - Models: {', '.join(csv_df['model'].unique())}\n")
            f.write(f"  - Splits: {', '.join(csv_df['split'].unique())}\n")
        else:
            f.write("✗ CSV data not available\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("VISUALIZATIONS CREATED\n")
        f.write("-" * 80 + "\n")

        plots = [
            "model_comparison.png/svg - Mean perplexity comparison",
            "seed_stability.png/svg - Perplexity distribution across seeds",
            "perplexity_distributions.png/svg - Overlaid histograms",
            "outlier_rates.png/svg - Outlier rate comparison",
            "distillation_progression.png/svg - Stage-wise improvement"
        ]

        for plot in plots:
            f.write(f"✓ {plot}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("KEY METRICS\n")
        f.write("-" * 80 + "\n")

        for model_key in ['vanilla', 'distill_phase1', 'distill_phase2']:
            if model_key in results:
                data = results[model_key]
                if isinstance(data, dict):
                    f.write(f"\n{model_key.upper().replace('_', ' ')}:\n")
                    f.write(f"  - Mean Perplexity: {data.get('mean_perplexity', 'N/A')}\n")
                    f.write(f"  - Std Perplexity: {data.get('std_perplexity', 'N/A')}\n")
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
    create_seed_stability_plot(csv_df, OUTPUT_DIR)
    create_perplexity_distribution_plot(csv_df, OUTPUT_DIR)
    create_outlier_rate_comparison(results, csv_df, OUTPUT_DIR)
    create_distillation_progression_plot(results, OUTPUT_DIR)
    create_summary_report(results, csv_df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("=" * 80 + "\n")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total figures created: 5")
    print(f"Formats: PNG (high-res) + SVG (vector)\n")


if __name__ == "__main__":
    main()