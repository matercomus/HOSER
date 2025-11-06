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
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy import stats

# Import paired statistical tests (optional - only needed for paired analysis)
try:
    from tools.paired_statistical_tests import compare_models_paired

    HAS_PAIRED_TESTS = True
except ImportError:
    HAS_PAIRED_TESTS = False


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
    values: List[float], confidence: float = 0.95, compute_cv: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a list of values.

    Args:
        values: List of numeric values to analyze
        confidence: Confidence level for intervals (default 0.95)
        compute_cv: Whether to compute coefficient of variation (default True).
                   Set to False for interval scale metrics (e.g., bounded 0-1 metrics)

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
    # CV assumes true zero and ratio relationships - inappropriate for interval scales
    if compute_cv:
        cv = (std / mean * 100) if mean != 0 else 0.0
    else:
        cv = None  # Explicitly None for interval scale metrics

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
    # Classified by measurement scale for appropriate statistical treatment
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
        "Hausdorff_km",  # Raw distance (ratio scale - CV appropriate)
        "Hausdorff_norm",  # Normalized by trajectory length (ratio scale - CV appropriate)
        "DTW_km",  # Raw distance (ratio scale - CV appropriate)
        "DTW_norm",  # Normalized by trajectory length (ratio scale - CV appropriate)
        "EDR",  # Edit Distance on Real sequence (interval scale, 0-1 bounded - CV problematic)
        "matched_od_pairs",
        "total_generated_od_pairs",
    ]

    # Interval scale metrics (CV not appropriate - bounded or without true zero)
    interval_scale_metrics = {
        "EDR",  # 0-1 bounded scale
    }

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
                # Determine if CV should be computed based on metric scale
                compute_cv = metric not in interval_scale_metrics
                stats_dict = compute_statistics(values, compute_cv=compute_cv)
                group_analysis[metric] = stats_dict

        analysis[group_key] = group_analysis

    return analysis


def load_trajectory_metrics_for_group(
    eval_dirs: List[str], dataset: str, model_type: str, od_source: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Load trajectory-level metrics for a specific (dataset, model, od_source) group.

    Args:
        eval_dirs: List of evaluation directory patterns
        dataset: Dataset name (e.g., 'Beijing', 'Porto')
        model_type: Model type (e.g., 'vanilla', 'distilled_phase1')
        od_source: OD source ('train' or 'test')

    Returns:
        List of trajectory metrics dictionaries, one per seed, or None if not found
    """
    trajectory_metrics_list = []

    for eval_dir_pattern in eval_dirs:
        eval_paths = list(Path().glob(eval_dir_pattern))

        for eval_dir in eval_paths:
            if not eval_dir.is_dir():
                continue

            # Find all trajectory_metrics.json files in timestamped subdirectories
            metrics_files = sorted(eval_dir.glob("*/trajectory_metrics.json"))

            for metrics_file in metrics_files:
                try:
                    with open(metrics_file) as f:
                        data = json.load(f)

                    # Check if this matches our criteria
                    metadata = data.get("metadata", {})
                    file_od_source = metadata.get("od_source", "")

                    # Check dataset and OD source match
                    gen_file = metadata.get("generated_file", "")
                    matches_dataset = (
                        dataset.lower() in gen_file.lower()
                        or dataset in metadata.get("real_data_file", "")
                    )
                    matches_od = file_od_source == od_source

                    if matches_dataset and matches_od:
                        # Infer model type from filename
                        if (
                            model_type in gen_file
                            or model_type.replace("_", "-") in gen_file
                        ):
                            trajectory_metrics_list.append(data)

                except Exception as e:
                    print(f"⚠️  Failed to load {metrics_file}: {e}")

    return trajectory_metrics_list if trajectory_metrics_list else None


def run_paired_analysis_across_seeds(
    grouped_results: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    eval_dirs: List[str],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Perform paired statistical tests between models across seeds.

    For each dataset and OD source, compare models pairwise on the same trajectories.
    Results are aggregated across seeds to show consistency.

    Args:
        grouped_results: Grouped evaluation results by (dataset, model, od_source)
        eval_dirs: Evaluation directory patterns
        alpha: Significance level for tests

    Returns:
        Dict with paired comparison results organized by comparison key
    """
    if not HAS_PAIRED_TESTS:
        print(
            "⚠️  Paired statistical tests not available (scipy or paired_statistical_tests missing)"
        )
        return {}

    print("\n🔬 Running paired statistical analysis between models...")

    paired_results = {}

    # Group by dataset and OD source to find models to compare
    by_dataset_od = defaultdict(list)
    for (dataset, model_type, od_source), results in grouped_results.items():
        key = (dataset, od_source)
        by_dataset_od[key].append((model_type, results))

    # For each dataset/OD combination, compare models pairwise
    for (dataset, od_source), models_and_results in by_dataset_od.items():
        if len(models_and_results) < 2:
            continue  # Need at least 2 models to compare

        # Try all pairwise comparisons
        for i in range(len(models_and_results)):
            for j in range(i + 1, len(models_and_results)):
                model1_type, results1 = models_and_results[i]
                model2_type, results2 = models_and_results[j]

                comparison_key = f"{dataset}_{model1_type}_vs_{model2_type}_{od_source}"

                print(
                    f"  Comparing {model1_type} vs {model2_type} on {dataset} {od_source} OD..."
                )

                # Load trajectory-level metrics for both models
                traj_metrics1 = load_trajectory_metrics_for_group(
                    eval_dirs, dataset, model1_type, od_source
                )
                traj_metrics2 = load_trajectory_metrics_for_group(
                    eval_dirs, dataset, model2_type, od_source
                )

                if not traj_metrics1 or not traj_metrics2:
                    print(
                        "    ⚠️  Trajectory metrics not available, skipping paired analysis"
                    )
                    continue

                if len(traj_metrics1) != len(traj_metrics2):
                    print(
                        f"    ⚠️  Different number of seeds ({len(traj_metrics1)} vs {len(traj_metrics2)}), skipping"
                    )
                    continue

                # Perform paired tests for key metrics
                metrics_to_test = ["hausdorff_norm", "dtw_norm", "edr"]
                comparison_results = {}

                for metric_name in metrics_to_test:
                    # Aggregate values across seeds (matched by OD pairs within each seed)
                    all_values1 = []
                    all_values2 = []

                    for seed_metrics1, seed_metrics2 in zip(
                        traj_metrics1, traj_metrics2
                    ):
                        trajs1 = seed_metrics1.get("trajectory_metrics", [])
                        trajs2 = seed_metrics2.get("trajectory_metrics", [])

                        # Match by OD pair
                        od_index1 = {}
                        for traj in trajs1:
                            od_pair = tuple(traj["od_pair"])
                            if od_pair not in od_index1:
                                od_index1[od_pair] = []
                            od_index1[od_pair].append(traj[metric_name])

                        for traj in trajs2:
                            od_pair = tuple(traj["od_pair"])
                            if od_pair in od_index1 and od_index1[od_pair]:
                                val1 = od_index1[od_pair].pop(0)
                                val2 = traj[metric_name]
                                all_values1.append(val1)
                                all_values2.append(val2)

                    if len(all_values1) < 10:
                        print(
                            f"      ⚠️  Too few matched pairs for {metric_name} ({len(all_values1)}), skipping"
                        )
                        continue

                    # Perform paired test
                    try:
                        test_result = compare_models_paired(
                            model1_values=all_values1,
                            model2_values=all_values2,
                            model1_name=model1_type,
                            model2_name=model2_type,
                            metric_name=metric_name,
                            alpha=alpha,
                            check_assumptions=True,
                        )

                        comparison_results[metric_name] = {
                            "test_name": test_result.test_name,
                            "n_pairs": test_result.n_pairs,
                            "model1_mean": test_result.model1_mean,
                            "model2_mean": test_result.model2_mean,
                            "mean_difference": test_result.mean_difference,
                            "p_value": test_result.p_value,
                            "significant": test_result.significant,
                            "cohens_d": test_result.cohens_d,
                        }

                        sig_marker = "✓" if test_result.significant else "✗"
                        print(
                            f"      {sig_marker} {metric_name}: p={test_result.p_value:.4f}, "
                            f"d={test_result.cohens_d:.3f}"
                        )

                    except Exception as e:
                        print(f"      ⚠️  Paired test failed for {metric_name}: {e}")

                if comparison_results:
                    paired_results[comparison_key] = {
                        "dataset": dataset,
                        "model1": model1_type,
                        "model2": model2_type,
                        "od_source": od_source,
                        "metrics": comparison_results,
                    }

    return paired_results


def generate_markdown_report(
    analysis: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]],
    output_dir: Path,
    confidence: float,
    paired_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Generate markdown report with cross-seed statistics and optional paired tests."""
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

        if paired_results:
            f.write(
                "**Paired Statistical Tests**: Model comparisons using paired tests on matched "
                "trajectories are included in a separate section below.\n\n"
            )

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
                        "Hausdorff_norm",
                        "DTW_km",
                        "DTW_norm",
                        "EDR",
                    ]

                    for metric in priority_metrics:
                        if metric in metrics:
                            stats = metrics[metric]
                            # Handle CV for interval scale metrics (None)
                            cv_str = (
                                f"{stats['cv']:.2f}%"
                                if stats["cv"] is not None
                                else "N/A*"
                            )
                            f.write(
                                f"| {metric} | "
                                f"{stats['mean']:.4f} | "
                                f"{stats['std']:.4f} | "
                                f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] | "
                                f"{cv_str} | "
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

                    # High variance metrics warning (only for metrics with CV computed)
                    high_cv_metrics = [
                        m
                        for m, s in metrics.items()
                        if s.get("cv") is not None and s.get("cv", 0) > 10
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
        f.write(
            "- **CV%**: Coefficient of variation (std/mean × 100) for ratio scale metrics\n"
        )
        f.write("- **N Seeds**: Number of random seeds in analysis\n")
        f.write("\n")
        f.write("**CV% Interpretation**:\n")
        f.write("- Low CV (<5%): Stable, seed-independent results\n")
        f.write("- Medium CV (5-10%): Moderate seed sensitivity\n")
        f.write("- High CV (>10%): High seed sensitivity, interpret with caution\n")
        f.write("\n")
        f.write(
            "**Note**: CV% marked as 'N/A*' for interval scale metrics (e.g., EDR) where "
            "coefficient of variation is not appropriate. EDR is bounded 0-1, making CV% "
            "potentially misleading. For these metrics, use standard deviation and "
            "confidence intervals instead.\n"
        )
        f.write("\n")
        f.write("**Metric Scale Classification**:\n")
        f.write(
            "- **Ratio Scale** (CV appropriate): JSD metrics, Hausdorff (raw & normalized), "
            "DTW (raw & normalized), real/gen means, OD match counts\n"
        )
        f.write("- **Interval Scale** (CV not appropriate): EDR (0-1 bounded)\n")

        # Add paired statistical tests section if available
        if paired_results:
            f.write("\n---\n\n")
            f.write("## Paired Statistical Tests\n\n")
            f.write(
                "This section presents pairwise model comparisons using paired statistical tests "
                "on matched trajectories (same OD pairs). Tests are performed on trajectory-level "
                "metrics and aggregated across all available seeds.\n\n"
            )

            # Group paired results by dataset
            paired_by_dataset = defaultdict(list)
            for comp_key, comp_data in paired_results.items():
                dataset = comp_data["dataset"]
                paired_by_dataset[dataset].append((comp_key, comp_data))

            for dataset in sorted(paired_by_dataset.keys()):
                f.write(f"### {dataset} Dataset\n\n")

                for comp_key, comp_data in paired_by_dataset[dataset]:
                    model1 = comp_data["model1"]
                    model2 = comp_data["model2"]
                    od_source = comp_data["od_source"]
                    metrics = comp_data["metrics"]

                    f.write(f"#### {model1} vs {model2} ({od_source.upper()} OD)\n\n")

                    if not metrics:
                        f.write("*No paired test results available*\n\n")
                        continue

                    # Create table
                    f.write(
                        "| Metric | Test | P-value | Significant | Cohen's d | Effect Size | N Pairs |\n"
                    )
                    f.write(
                        "|--------|------|---------|-------------|-----------|-------------|--------|\n"
                    )

                    for metric_name, metric_results in metrics.items():
                        sig_marker = "✓" if metric_results["significant"] else "✗"

                        # Interpret effect size
                        d = abs(metric_results["cohens_d"])
                        if d < 0.2:
                            effect = "Negligible"
                        elif d < 0.5:
                            effect = "Small"
                        elif d < 0.8:
                            effect = "Medium"
                        else:
                            effect = "Large"

                        f.write(
                            f"| {metric_name} | "
                            f"{metric_results['test_name']} | "
                            f"{metric_results['p_value']:.4f} | "
                            f"{sig_marker} | "
                            f"{metric_results['cohens_d']:.3f} | "
                            f"{effect} | "
                            f"{metric_results['n_pairs']} |\n"
                        )

                    f.write("\n")
                    f.write(
                        f"**Mean Values**: {model1} = {metrics[list(metrics.keys())[0]]['model1_mean']:.4f}, "
                    )
                    f.write(
                        f"{model2} = {metrics[list(metrics.keys())[0]]['model2_mean']:.4f}\n\n"
                    )

            f.write("\n**Interpretation Guide**:\n\n")
            f.write("- **P-value < 0.05**: Statistically significant difference\n")
            f.write("- **Cohen's d Effect Size**:\n")
            f.write("  - |d| < 0.2: Negligible effect\n")
            f.write("  - 0.2 ≤ |d| < 0.5: Small effect\n")
            f.write("  - 0.5 ≤ |d| < 0.8: Medium effect\n")
            f.write("  - |d| ≥ 0.8: Large effect\n\n")
            f.write(
                "**Note**: Paired tests compare models on the same trajectories (matched by OD pairs), "
                "providing more statistical power than unpaired comparisons.\n"
            )

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

    # Run paired statistical analysis if trajectory metrics are available
    paired_results = run_paired_analysis_across_seeds(
        grouped, args.eval_dirs, alpha=1 - args.confidence
    )

    # Generate report (with or without paired results)
    generate_markdown_report(analysis, args.output_dir, args.confidence, paired_results)

    print("\n✅ Cross-seed analysis complete!")
    if paired_results:
        print(f"   Including {len(paired_results)} paired model comparisons")
    print("=" * 60)


if __name__ == "__main__":
    main()
