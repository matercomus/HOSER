#!/usr/bin/env python3
"""
Cross-Seed Statistical Analysis for HOSER Evaluation Results

Computes mean ± std, confidence intervals, and variance metrics across
multiple random seeds for trajectory generation evaluation.

Usage:
    uv run python scripts/analysis/cross_seed_analysis.py \
        --eval_dirs hoser-distill-optuna-6/eval hoser-distill-optuna-porto-eval-*/eval \
        --output_dir docs/results \
        --confidence 0.95
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute cross-seed statistics for HOSER evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval_dirs",
        nargs="+",
        required=True,
        help="Evaluation directories containing timestamped results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("docs/results"),
        help="Output directory for analysis reports",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )
    parser.add_argument(
        "--min_seeds",
        type=int,
        default=2,
        help="Minimum number of seeds required for analysis (default: 2)",
    )
    return parser.parse_args()


def load_all_results(eval_dirs: List[str]) -> List[Dict[str, Any]]:
    """Load all evaluation results from multiple directories."""
    all_results = []

    for eval_dir_pattern in eval_dirs:
        eval_paths = list(Path().glob(eval_dir_pattern))

        for eval_dir in eval_paths:
            if not eval_dir.is_dir():
                continue

            # Find all results.json files in timestamped subdirectories
            results_files = sorted(eval_dir.glob("*/results.json"))

            print(f"📂 Loading {len(results_files)} results from {eval_dir}")

            for results_file in results_files:
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                        all_results.append(data)
                except Exception as e:
                    print(f"⚠️  Failed to load {results_file}: {e}")

    print(f"✅ Loaded {len(all_results)} total evaluation results")
    return all_results


def group_results_by_model_and_od(
    results: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """
    Group results by (dataset, model_type, od_source).

    Returns:
        Dict mapping (dataset, model_type, od_source) -> list of result dicts
    """
    grouped = defaultdict(list)

    for result in results:
        metadata = result.get("metadata", {})

        # Infer dataset from generated_file path or use metadata
        gen_file = metadata.get("generated_file", "")
        if "beijing" in gen_file.lower() or "Beijing" in metadata.get(
            "road_network_file", ""
        ):
            dataset = "Beijing"
        elif "porto" in gen_file.lower() or "porto" in metadata.get(
            "road_network_file", ""
        ):
            dataset = "Porto"
        else:
            dataset = "Unknown"

        model_type = metadata.get("model_type", "unknown")
        od_source = metadata.get("od_source", "unknown")

        # Normalize model_type (remove seed suffixes for grouping)
        if "_seed" in model_type:
            # Extract base model type (e.g., "distill_phase2_seed43" -> "distill_phase2")
            base_model = "_".join(model_type.split("_")[:-1])
        else:
            base_model = model_type

        key = (dataset, base_model, od_source)
        grouped[key].append(result)

    return grouped


def compute_statistics(
    values: List[float], confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a list of values.

    Returns:
        Dict with mean, std, sem, ci_lower, ci_upper, cv, min, max, median
    """
    if not values or len(values) < 1:
        return {}

    values_array = np.array(values)
    n = len(values_array)

    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1) if n > 1 else 0.0
    sem = stats.sem(values_array) if n > 1 else 0.0

    # Confidence interval (t-distribution for small samples)
    if n > 1:
        ci = stats.t.interval(
            confidence,
            df=n - 1,
            loc=mean,
            scale=sem,
        )
        ci_lower, ci_upper = ci
    else:
        ci_lower = ci_upper = mean

    # Coefficient of variation (only for ratio scale metrics)
    cv = (std / mean * 100) if mean != 0 else 0.0

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cv": cv,
        "min": np.min(values_array),
        "max": np.max(values_array),
        "median": np.median(values_array),
        "n": n,
    }


def analyze_cross_seed_variance(
    grouped_results: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    min_seeds: int,
) -> Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]]:
    """
    Analyze cross-seed variance for each (dataset, model, od_source) group.

    Returns:
        Dict mapping group key -> metric name -> statistics dict
    """
    analysis = {}

    # Metrics to analyze (exclude metadata fields)
    metric_keys = [
        "Distance_JSD",
        "Distance_real_mean",
        "Distance_gen_mean",
        "Duration_JSD",
        "Duration_real_mean",
        "Duration_gen_mean",
        "Radius_JSD",
        "Radius_real_mean",
        "Radius_gen_mean",
        "Hausdorff_km",
        "DTW_km",
        "EDR",
        "matched_od_pairs",
        "total_generated_od_pairs",
    ]

    for group_key, results in grouped_results.items():
        dataset, model_type, od_source = group_key

        if len(results) < min_seeds:
            print(
                f"⚠️  Skipping {dataset}/{model_type}/{od_source}: "
                f"only {len(results)} seed(s), need {min_seeds}"
            )
            continue

        print(f"📊 Analyzing {dataset}/{model_type}/{od_source} ({len(results)} seeds)")

        # Extract seeds for this group
        seeds = [r.get("metadata", {}).get("seed", "unknown") for r in results]
        print(f"   Seeds: {sorted(set(seeds))}")

        # Compute statistics for each metric
        group_analysis = {}

        for metric in metric_keys:
            # Collect values across seeds
            values = []
            for result in results:
                if metric in result and isinstance(result[metric], (int, float)):
                    values.append(float(result[metric]))

            if values:
                stats_dict = compute_statistics(values)
                group_analysis[metric] = stats_dict

        analysis[group_key] = group_analysis

    return analysis


def generate_markdown_report(
    analysis: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]],
    output_dir: Path,
    confidence: float,
) -> None:
    """Generate markdown report with cross-seed statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "CROSS_SEED_ANALYSIS.md"

    with open(report_file, "w") as f:
        f.write("# Cross-Seed Statistical Analysis\n\n")
        f.write("## Overview\n\n")
        f.write(
            "This document presents cross-seed statistical analysis of HOSER evaluation results. "
            "All metrics are reported as **mean ± std** with confidence intervals.\n\n"
        )
        f.write(f"**Confidence Level**: {confidence * 100:.0f}%\n\n")
        f.write("---\n\n")

        # Group by dataset
        by_dataset = defaultdict(dict)
        for (dataset, model_type, od_source), metrics in analysis.items():
            if dataset not in by_dataset:
                by_dataset[dataset] = {}
            if model_type not in by_dataset[dataset]:
                by_dataset[dataset][model_type] = {}
            by_dataset[dataset][model_type][od_source] = metrics

        # Generate sections for each dataset
        for dataset in sorted(by_dataset.keys()):
            f.write(f"## {dataset} Dataset\n\n")

            models = by_dataset[dataset]

            for model_type in sorted(models.keys()):
                f.write(f"### {model_type.replace('_', ' ').title()}\n\n")

                od_sources = models[model_type]

                for od_source in sorted(od_sources.keys()):
                    metrics = od_sources[od_source]

                    f.write(f"#### {od_source.upper()} OD\n\n")

                    # Create table
                    f.write("| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |\n")
                    f.write("|--------|------|---------|--------|-----|--------|\n")

                    # Main metrics first
                    priority_metrics = [
                        "Distance_JSD",
                        "Duration_JSD",
                        "Radius_JSD",
                        "Hausdorff_km",
                        "DTW_km",
                        "EDR",
                    ]

                    for metric in priority_metrics:
                        if metric in metrics:
                            stats = metrics[metric]
                            f.write(
                                f"| {metric} | "
                                f"{stats['mean']:.4f} | "
                                f"{stats['std']:.4f} | "
                                f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] | "
                                f"{stats['cv']:.2f}% | "
                                f"{stats['n']} |\n"
                            )

                    # OD match metrics
                    if (
                        "matched_od_pairs" in metrics
                        and "total_generated_od_pairs" in metrics
                    ):
                        matched = metrics["matched_od_pairs"]
                        total = metrics["total_generated_od_pairs"]
                        match_rate = matched["mean"] / total["mean"] * 100
                        f.write(
                            f"| OD Match Rate | "
                            f"{match_rate:.2f}% | "
                            f"- | "
                            f"- | "
                            f"- | "
                            f"{matched['n']} |\n"
                        )

                    f.write("\n")

                    # High variance metrics warning
                    high_cv_metrics = [
                        m for m, s in metrics.items() if s.get("cv", 0) > 10
                    ]
                    if high_cv_metrics:
                        f.write(
                            f"⚠️  **High Variance Metrics** (CV > 10%): "
                            f"{', '.join(high_cv_metrics)}\n\n"
                        )

        f.write("---\n\n")
        f.write("## Statistical Notes\n\n")
        f.write("- **Mean ± Std**: Arithmetic mean with standard deviation\n")
        f.write("- **95% CI**: 95% confidence interval using t-distribution\n")
        f.write("- **CV%**: Coefficient of variation (std/mean × 100)\n")
        f.write("- **N Seeds**: Number of random seeds in analysis\n")
        f.write("\n")
        f.write("**Interpretation**:\n")
        f.write("- Low CV (<5%): Stable, seed-independent results\n")
        f.write("- Medium CV (5-10%): Moderate seed sensitivity\n")
        f.write("- High CV (>10%): High seed sensitivity, interpret with caution\n")

    print(f"✅ Report saved: {report_file}")


def main():
    """Main entry point."""
    args = parse_args()

    print("🔬 HOSER Cross-Seed Statistical Analysis")
    print("=" * 60)

    # Load all results
    all_results = load_all_results(args.eval_dirs)

    if not all_results:
        print("❌ No evaluation results found")
        sys.exit(1)

    # Group by model and OD source
    grouped = group_results_by_model_and_od(all_results)
    print(f"\n📦 Grouped into {len(grouped)} (dataset, model, OD) combinations")

    # Analyze cross-seed variance
    analysis = analyze_cross_seed_variance(grouped, args.min_seeds)

    if not analysis:
        print("❌ No groups with sufficient seeds for analysis")
        sys.exit(1)

    # Generate report
    generate_markdown_report(analysis, args.output_dir, args.confidence)

    print("\n✅ Cross-seed analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
