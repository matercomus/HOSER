#!/usr/bin/env python3
"""
Standalone tool for paired statistical comparison of models.

This script loads trajectory-level metrics from evaluation directories and performs
paired statistical tests to compare model performance on the same OD pairs.

Usage:
    uv run python tools/compare_models_paired_analysis.py \
        --eval-dirs eval/model1/eval/2025-01-01_12-00-00 eval/model2/eval/2025-01-01_12-00-00 \
        --model-names vanilla distilled \
        --output paired_comparison_results.json

Author: HOSER Project
Date: 2025-11-06
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import paired statistical tests module
try:
    from tools.paired_statistical_tests import compare_models_paired
except ImportError:
    # Try relative import if running from tools directory
    try:
        from paired_statistical_tests import compare_models_paired
    except ImportError:
        logger.error("Could not import paired_statistical_tests module")
        sys.exit(1)


def load_trajectory_metrics(eval_dir: Path) -> Dict[str, Any]:
    """
    Load trajectory-level metrics from an evaluation directory.

    Args:
        eval_dir: Path to evaluation directory containing trajectory_metrics.json

    Returns:
        Dictionary with metadata and trajectory metrics
    """
    trajectory_metrics_file = eval_dir / "trajectory_metrics.json"

    if not trajectory_metrics_file.exists():
        raise FileNotFoundError(
            f"trajectory_metrics.json not found in {eval_dir}. "
            "Make sure evaluation was run with trajectory-level metrics enabled."
        )

    with open(trajectory_metrics_file, "r") as f:
        data = json.load(f)

    logger.info(
        f"Loaded {len(data.get('trajectory_metrics', []))} trajectory metrics from {eval_dir.name}"
    )

    return data


def match_trajectory_pairs(
    metrics1: List[Dict], metrics2: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match trajectory metrics by OD pair across two models.

    Args:
        metrics1: List of trajectory metrics from model 1
        metrics2: List of trajectory metrics from model 2

    Returns:
        Tuple of (matched_metrics1, matched_metrics2) with same length and aligned OD pairs
    """
    # Build OD pair index for model 1
    od_index1 = {}
    for metric in metrics1:
        od_pair = tuple(metric["od_pair"])
        if od_pair not in od_index1:
            od_index1[od_pair] = []
        od_index1[od_pair].append(metric)

    # Match with model 2
    matched1 = []
    matched2 = []

    for metric2 in metrics2:
        od_pair = tuple(metric2["od_pair"])
        if od_pair in od_index1 and od_index1[od_pair]:
            # Match available - pair them up
            metric1 = od_index1[od_pair].pop(0)
            matched1.append(metric1)
            matched2.append(metric2)

    logger.info(f"Matched {len(matched1)} trajectory pairs across models")

    if len(matched1) == 0:
        logger.warning("No matching OD pairs found between models!")

    return matched1, matched2


def extract_metric_values(matched_metrics: List[Dict], metric_name: str) -> List[float]:
    """
    Extract values for a specific metric from matched trajectory metrics.

    Args:
        matched_metrics: List of matched trajectory metric dictionaries
        metric_name: Name of metric to extract (e.g., 'hausdorff_norm', 'dtw_norm', 'edr')

    Returns:
        List of metric values
    """
    values = [m[metric_name] for m in matched_metrics]
    return values


def perform_paired_comparison(
    matched_metrics1: List[Dict],
    matched_metrics2: List[Dict],
    model1_name: str,
    model2_name: str,
    metrics_to_compare: List[str],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform paired statistical tests for all metrics.

    Args:
        matched_metrics1: Matched trajectory metrics from model 1
        matched_metrics2: Matched trajectory metrics from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        metrics_to_compare: List of metric names to compare
        alpha: Significance level

    Returns:
        Dictionary with comparison results for each metric
    """
    results = {}

    for metric_name in metrics_to_compare:
        logger.info(f"Comparing {metric_name}...")

        # Extract metric values
        values1 = extract_metric_values(matched_metrics1, metric_name)
        values2 = extract_metric_values(matched_metrics2, metric_name)

        # Perform paired comparison
        try:
            test_result = compare_models_paired(
                model1_values=values1,
                model2_values=values2,
                model1_name=model1_name,
                model2_name=model2_name,
                metric_name=metric_name,
                alpha=alpha,
                check_assumptions=True,
            )

            # Convert result to dictionary
            results[metric_name] = {
                "test_name": test_result.test_name,
                "n_pairs": test_result.n_pairs,
                "model1_mean": test_result.model1_mean,
                "model2_mean": test_result.model2_mean,
                "mean_difference": test_result.mean_difference,
                "std_difference": test_result.std_difference,
                "test_statistic": test_result.test_statistic,
                "p_value": test_result.p_value,
                "significant": test_result.significant,
                "cohens_d": test_result.cohens_d,
                "assumptions_met": test_result.assumptions_met,
                "warnings": test_result.warnings,
            }

            # Print summary
            sig_marker = "✓" if test_result.significant else "✗"
            logger.info(
                f"  {sig_marker} {metric_name}: p={test_result.p_value:.4f}, "
                f"Cohen's d={test_result.cohens_d:.3f}, "
                f"{model1_name} mean={test_result.model1_mean:.4f}, "
                f"{model2_name} mean={test_result.model2_mean:.4f}"
            )

        except Exception as e:
            logger.error(f"Failed to compare {metric_name}: {e}")
            results[metric_name] = {"error": str(e)}

    return results


def generate_markdown_summary(
    comparison_results: Dict[str, Any], output_file: Path
) -> None:
    """
    Generate a markdown summary of paired comparison results.

    Args:
        comparison_results: Dictionary with comparison results
        output_file: Path to save markdown summary
    """
    md_file = output_file.with_suffix(".md")

    with open(md_file, "w") as f:
        f.write("# Paired Statistical Comparison\n\n")
        f.write("## Models\n\n")
        f.write(f"- Model 1: {comparison_results['model1_name']}\n")
        f.write(f"- Model 2: {comparison_results['model2_name']}\n\n")

        f.write("## Summary\n\n")
        f.write(
            f"- Number of matched trajectory pairs: {comparison_results['n_matched_pairs']}\n"
        )
        f.write(f"- Significance level (α): {comparison_results['alpha']}\n\n")

        f.write("## Metric Comparisons\n\n")

        for metric_name, metric_results in comparison_results["metrics"].items():
            if "error" in metric_results:
                f.write(f"### {metric_name}\n\n")
                f.write(f"**Error**: {metric_results['error']}\n\n")
                continue

            sig_marker = (
                "✓ Significant"
                if metric_results["significant"]
                else "✗ Not Significant"
            )
            f.write(f"### {metric_name} ({sig_marker})\n\n")

            f.write(f"- **Test**: {metric_results['test_name']}\n")
            f.write(f"- **P-value**: {metric_results['p_value']:.4f}\n")
            f.write(f"- **Cohen's d**: {metric_results['cohens_d']:.3f}\n")
            f.write(
                f"- **{comparison_results['model1_name']} mean**: {metric_results['model1_mean']:.4f}\n"
            )
            f.write(
                f"- **{comparison_results['model2_name']} mean**: {metric_results['model2_mean']:.4f}\n"
            )
            f.write(f"- **Mean difference**: {metric_results['mean_difference']:.4f}\n")
            f.write(f"- **Std difference**: {metric_results['std_difference']:.4f}\n\n")

            if metric_results.get("warnings"):
                f.write("**Warnings**:\n")
                for warning in metric_results["warnings"]:
                    f.write(f"- {warning}\n")
                f.write("\n")

    logger.info(f"Markdown summary saved to {md_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform paired statistical comparison of models using trajectory-level metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  uv run python tools/compare_models_paired_analysis.py \\
    --eval-dirs eval/vanilla/eval/2025-01-01_12-00-00 eval/distilled/eval/2025-01-01_12-00-00 \\
    --model-names vanilla distilled \\
    --output paired_comparison.json

  # Compare with custom metrics
  uv run python tools/compare_models_paired_analysis.py \\
    --eval-dirs dir1 dir2 \\
    --model-names model1 model2 \\
    --metrics hausdorff_norm dtw_norm \\
    --output results.json
        """,
    )

    parser.add_argument(
        "--eval-dirs",
        type=str,
        nargs=2,
        required=True,
        help="Paths to two evaluation directories containing trajectory_metrics.json",
    )

    parser.add_argument(
        "--model-names",
        type=str,
        nargs=2,
        required=True,
        help="Names of the two models (for reporting)",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["hausdorff_km", "hausdorff_norm", "dtw_km", "dtw_norm", "edr"],
        help="Metrics to compare (default: hausdorff_km, hausdorff_norm, dtw_km, dtw_norm, edr)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for JSON results",
    )

    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip generating markdown summary",
    )

    args = parser.parse_args()

    # Validate inputs
    eval_dir1 = Path(args.eval_dirs[0])
    eval_dir2 = Path(args.eval_dirs[1])

    if not eval_dir1.exists():
        logger.error(f"Evaluation directory not found: {eval_dir1}")
        sys.exit(1)

    if not eval_dir2.exists():
        logger.error(f"Evaluation directory not found: {eval_dir2}")
        sys.exit(1)

    model1_name = args.model_names[0]
    model2_name = args.model_names[1]

    logger.info(f"Comparing {model1_name} vs {model2_name}")

    # Load trajectory metrics
    try:
        data1 = load_trajectory_metrics(eval_dir1)
        data2 = load_trajectory_metrics(eval_dir2)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Match trajectory pairs by OD pair
    matched_metrics1, matched_metrics2 = match_trajectory_pairs(
        data1["trajectory_metrics"], data2["trajectory_metrics"]
    )

    if len(matched_metrics1) == 0:
        logger.error("No matching trajectory pairs found. Cannot perform paired tests.")
        sys.exit(1)

    # Perform paired comparison
    comparison_results = perform_paired_comparison(
        matched_metrics1=matched_metrics1,
        matched_metrics2=matched_metrics2,
        model1_name=model1_name,
        model2_name=model2_name,
        metrics_to_compare=args.metrics,
        alpha=args.alpha,
    )

    # Prepare output
    output_data = {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "model1_eval_dir": str(eval_dir1),
        "model2_eval_dir": str(eval_dir2),
        "n_matched_pairs": len(matched_metrics1),
        "alpha": args.alpha,
        "metrics": comparison_results,
        "metadata": {
            "model1_metadata": data1.get("metadata", {}),
            "model2_metadata": data2.get("metadata", {}),
        },
    }

    # Save JSON results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Generate markdown summary
    if not args.no_markdown:
        generate_markdown_summary(output_data, output_file)

    # Print summary
    print("\n" + "=" * 70)
    print(f"Paired Statistical Comparison: {model1_name} vs {model2_name}")
    print("=" * 70)
    print(f"Matched trajectory pairs: {len(matched_metrics1)}")
    print(f"Significance level (α): {args.alpha}")
    print()

    significant_count = sum(
        1 for m in comparison_results.values() if m.get("significant", False)
    )
    print(f"Significant differences: {significant_count}/{len(args.metrics)} metrics")
    print()

    for metric_name, result in comparison_results.items():
        if "error" in result:
            print(f"  ✗ {metric_name}: ERROR - {result['error']}")
        else:
            sig = "✓" if result["significant"] else "✗"
            print(
                f"  {sig} {metric_name}: p={result['p_value']:.4f}, "
                f"d={result['cohens_d']:.3f}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()
