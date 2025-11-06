#!/usr/bin/env python3
"""
Paired Statistical Tests - Demonstration and Validation

This script demonstrates the use of paired statistical tests for comparing
trajectory generation models. It includes both synthetic examples and
shows how to apply these tests to real evaluation data.

Usage:
    uv run python examples/paired_tests_demo.py
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.paired_statistical_tests import (  # noqa: E402
    paired_ttest,
    wilcoxon_signed_rank,
    compare_models_paired,
    format_paired_test_result,
    compute_cohens_d_paired,
)


def demo_basic_paired_test():
    """Demonstrate basic paired t-test with synthetic data"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Paired T-Test")
    print("=" * 70)

    # Simulate: 100 OD pairs, both models generate trajectories
    # Distilled model consistently has lower Hausdorff distance
    np.random.seed(42)
    n_pairs = 100

    # Vanilla model: higher Hausdorff (mean=0.5, std=0.2)
    vanilla_hausdorff = np.random.normal(0.5, 0.2, n_pairs)

    # Distilled model: lower Hausdorff by 0.15 on average
    # Key: same random state shifted, creating correlation
    distilled_hausdorff = vanilla_hausdorff - 0.15 + np.random.normal(0, 0.05, n_pairs)

    print(f"\nVanilla mean: {vanilla_hausdorff.mean():.4f}")
    print(f"Distilled mean: {distilled_hausdorff.mean():.4f}")
    print(f"Mean difference: {(vanilla_hausdorff - distilled_hausdorff).mean():.4f}")

    # Perform paired t-test
    t_stat, p_val, mean_diff, significant = paired_ttest(
        vanilla_hausdorff, distilled_hausdorff
    )

    print("\nPaired t-test results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Significant (α=0.05): {'Yes ✓' if significant else 'No ✗'}")

    # Compute effect size
    cohens_d = compute_cohens_d_paired(vanilla_hausdorff, distilled_hausdorff)
    print(f"  Cohen's d: {cohens_d:.4f}")


def demo_wilcoxon_test():
    """Demonstrate Wilcoxon signed-rank test with non-normal data"""
    print("\n" + "=" * 70)
    print("DEMO 2: Wilcoxon Signed-Rank Test (Non-Normal Data)")
    print("=" * 70)

    # Simulate: 50 OD pairs with skewed distribution
    np.random.seed(123)
    n_pairs = 50

    # Vanilla model: exponential distribution (right-skewed)
    vanilla_edr = np.random.exponential(0.4, n_pairs)

    # Distilled model: consistently lower by ~0.1
    distilled_edr = vanilla_edr * 0.7 + np.random.normal(0, 0.02, n_pairs)

    print(f"\nVanilla median: {np.median(vanilla_edr):.4f}")
    print(f"Distilled median: {np.median(distilled_edr):.4f}")

    # Perform Wilcoxon test
    w_stat, p_val, median_diff, significant = wilcoxon_signed_rank(
        vanilla_edr, distilled_edr
    )

    print("\nWilcoxon signed-rank test results:")
    print(f"  W-statistic: {w_stat:.4f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Median difference: {median_diff:.4f}")
    print(f"  Significant (α=0.05): {'Yes ✓' if significant else 'No ✗'}")


def demo_comprehensive_comparison():
    """Demonstrate comprehensive paired comparison with automatic test selection"""
    print("\n" + "=" * 70)
    print("DEMO 3: Comprehensive Paired Comparison (Automatic Test Selection)")
    print("=" * 70)

    # Simulate: 200 OD pairs, DTW distance comparison
    np.random.seed(456)
    n_pairs = 200

    # Vanilla model: DTW values
    vanilla_dtw = np.random.gamma(2, 0.15, n_pairs)

    # Distilled model: 25% improvement
    distilled_dtw = vanilla_dtw * 0.75 + np.random.normal(0, 0.03, n_pairs)

    # Use comprehensive comparison
    result = compare_models_paired(
        model1_values=vanilla_dtw,
        model2_values=distilled_dtw,
        model1_name="vanilla",
        model2_name="distilled",
        metric_name="DTW_norm",
        alpha=0.05,
        check_assumptions=True,
    )

    # Print formatted results
    print(format_paired_test_result(result))

    # Interpretation
    if result.significant:
        if abs(result.cohens_d) >= 0.8:
            interpretation = "LARGE practical effect"
        elif abs(result.cohens_d) >= 0.5:
            interpretation = "MEDIUM practical effect"
        elif abs(result.cohens_d) >= 0.2:
            interpretation = "SMALL practical effect"
        else:
            interpretation = "negligible practical effect"

        print(f"✅ **Conclusion**: Distilled model shows {interpretation}")
        print(
            f"   Mean improvement: {(1 - result.model2_mean / result.model1_mean) * 100:.1f}%"
        )
    else:
        print("❌ **Conclusion**: No significant difference between models")


def demo_multiple_metrics():
    """Demonstrate comparing multiple metrics with multiple testing correction"""
    print("\n" + "=" * 70)
    print("DEMO 4: Multiple Metrics Comparison (With FDR Correction)")
    print("=" * 70)

    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        print("\nNote: statsmodels not available, skipping FDR correction example")
        return

    # Simulate: 150 OD pairs, 3 different metrics
    np.random.seed(789)
    n_pairs = 150

    metrics = {}

    # Metric 1: Hausdorff (large effect)
    vanilla_hd = np.random.normal(0.45, 0.15, n_pairs)
    distilled_hd = vanilla_hd * 0.65 + np.random.normal(0, 0.05, n_pairs)
    metrics["Hausdorff_norm"] = (vanilla_hd, distilled_hd)

    # Metric 2: DTW (medium effect)
    vanilla_dtw = np.random.normal(0.35, 0.12, n_pairs)
    distilled_dtw = vanilla_dtw * 0.80 + np.random.normal(0, 0.04, n_pairs)
    metrics["DTW_norm"] = (vanilla_dtw, distilled_dtw)

    # Metric 3: EDR (small/no effect)
    vanilla_edr = np.random.normal(0.52, 0.18, n_pairs)
    distilled_edr = vanilla_edr * 0.95 + np.random.normal(0, 0.08, n_pairs)
    metrics["EDR"] = (vanilla_edr, distilled_edr)

    # Perform paired tests for all metrics
    results = {}
    p_values = []

    for metric_name, (vanilla_vals, distilled_vals) in metrics.items():
        print(f"\nAnalyzing {metric_name}...")

        result = compare_models_paired(
            model1_values=vanilla_vals,
            model2_values=distilled_vals,
            model1_name="vanilla",
            model2_name="distilled",
            metric_name=metric_name,
            alpha=0.05,
            check_assumptions=True,
        )

        results[metric_name] = result
        p_values.append(result.p_value)

    # Apply FDR correction
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Multiple Metrics Comparison")
    print("=" * 70)
    header = f"{'Metric':<20} {'p_raw':<12} {'p_corrected':<12} {'Significant':<12} {'Cohen d':<10}"
    print(header)
    print("-" * 70)

    for i, (metric_name, result) in enumerate(results.items()):
        sig_str = "✓ Yes" if reject[i] else "✗ No"
        print(
            f"{metric_name:<20} {result.p_value:<12.6f} {p_corrected[i]:<12.6f} "
            f"{sig_str:<12} {result.cohens_d:>7.3f}"
        )

    print("\n✅ FDR correction applied successfully")
    print(f"   Significant after correction: {sum(reject)}/{len(reject)} metrics")


def demo_power_comparison():
    """Demonstrate why paired tests have greater power than unpaired tests"""
    print("\n" + "=" * 70)
    print("DEMO 5: Paired vs Unpaired Tests - Statistical Power")
    print("=" * 70)

    try:
        from scipy import stats
    except ImportError:
        print("\nNote: scipy not available, cannot run power comparison")
        return

    # Simulate: 30 OD pairs with HIGH between-pair variability
    np.random.seed(999)
    n_pairs = 30

    # Base difficulty for each OD pair (high variability)
    base_difficulty = np.random.uniform(0.1, 0.9, n_pairs)

    # Vanilla and distilled both affected by base difficulty (creates correlation)
    vanilla = base_difficulty + np.random.normal(0, 0.1, n_pairs)
    distilled = (
        base_difficulty - 0.15 + np.random.normal(0, 0.1, n_pairs)
    )  # 0.15 better

    print(f"\nSimulated data (n={n_pairs} OD pairs):")
    print(
        f"  Base difficulty range: [{base_difficulty.min():.2f}, {base_difficulty.max():.2f}]"
    )
    print("  True mean difference: -0.15 (distilled better)")
    print(f"  Actual mean difference: {(vanilla - distilled).mean():.4f}")

    # Unpaired t-test (WRONG - ignores pairing)
    t_unpaired, p_unpaired = stats.ttest_ind(vanilla, distilled)

    # Paired t-test (CORRECT - accounts for pairing)
    t_paired, p_paired = stats.ttest_rel(vanilla, distilled)

    print("\nUNPAIRED t-test (incorrect approach):")
    print(f"  t-statistic: {t_unpaired:.4f}")
    print(f"  p-value: {p_unpaired:.4f}")
    print(f"  Significant (α=0.05): {'Yes ✓' if p_unpaired < 0.05 else 'No ✗'}")

    print("\nPAIRED t-test (correct approach):")
    print(f"  t-statistic: {t_paired:.4f}")
    print(f"  p-value: {p_paired:.4f}")
    print(f"  Significant (α=0.05): {'Yes ✓' if p_paired < 0.05 else 'No ✗'}")

    print("\n🔍 Power Difference:")
    print(f"   Paired test is {abs(t_paired / t_unpaired):.2f}x more powerful!")
    print("   Paired test accounts for within-pair correlation")
    print("   Removes between-pair variability (different OD difficulty)")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("PAIRED STATISTICAL TESTS - DEMONSTRATION & VALIDATION")
    print("=" * 70)
    print("\nThis script demonstrates the correct use of paired statistical tests")
    print("for comparing trajectory generation models on the same OD pairs.")

    # Run demonstrations
    demo_basic_paired_test()
    demo_wilcoxon_test()
    demo_comprehensive_comparison()
    demo_multiple_metrics()
    demo_power_comparison()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✅ All demonstrations completed successfully!")
    print("\nKey Takeaways:")
    print("  1. Use PAIRED tests when comparing same OD pairs across models")
    print("  2. Paired tests have GREATER statistical power")
    print("  3. Check normality assumptions (use Wilcoxon if violated)")
    print("  4. Report effect sizes (Cohen's d) alongside p-values")
    print("  5. Apply multiple testing correction for multiple metrics")
    print("\nFor more details, see: docs/PAIRED_STATISTICAL_TESTS_GUIDE.md")
    print("For implementation, see: tools/paired_statistical_tests.py")


if __name__ == "__main__":
    main()
