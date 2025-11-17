#!/usr/bin/env python3
"""
Analyze and Aggregate LM-TAD Spatial Abnormality Evaluation Results

This script aggregates spatial abnormality evaluation results from multiple models
and performs statistical comparisons with real data rates.

Usage:
    uv run python tools/analyze_lmtad_spatial_results.py \\
        --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \\
        --dataset porto_hoser \\
        --output analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Import statistical functions from analyze_wang_results
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.analyze_wang_results import (  # noqa: E402
    compute_cohens_h,
    compute_proportion_ci,
    interpret_cohens_h,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import sys  # noqa: E402


@dataclass
class SpatialEvaluationMetrics:
    """Metrics for a single spatial abnormality evaluation result"""

    dataset: str
    model: Optional[str]  # None for real data
    is_real: bool
    total_trajectories: int
    spatial_abnormal_count: int
    spatial_abnormality_rate: float
    route_switch_count: int
    route_switch_rate: float
    detour_count: int
    detour_rate: float
    log_perplexity_stats: Optional[Dict[str, float]] = None
    source_statistics: Optional[Dict[str, float]] = None


@dataclass
class StatisticalComparison:
    """Statistical comparison between real and generated"""

    dataset: str
    model: str
    real_rate: float
    generated_rate: float
    difference: float  # generated - real
    relative_difference_pct: float  # (generated - real) / real * 100
    trajectory_count_real: int
    trajectory_count_generated: int
    p_value: float
    cohens_h: float
    effect_size: str
    ci_lower: float
    ci_upper: float


def load_evaluation_result(result_file: Path) -> Optional[Dict]:
    """Load evaluation result from JSON file

    Args:
        result_file: Path to evaluation result JSON file

    Returns:
        Dictionary with evaluation results or None if failed
    """
    try:
        with open(result_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {result_file}: {e}")
        return None


def load_source_real_rates(source_eval_dir: Path) -> Dict:
    """Load real spatial abnormality rates from source evaluation

    Args:
        source_eval_dir: Path to LM-TAD source evaluation directory

    Returns:
        Dictionary with real rates
    """
    # Read from EVALUATION_ANALYSIS.md or compute from TSV
    tsv_files = list(source_eval_dir.glob("ckpt_best_outliers_*.tsv"))
    if not tsv_files:
        logger.warning("No TSV files found, using default rates")
        return {
            "spatial_abnormality_rate": 6.54,
            "route_switch_rate": 3.27,
            "detour_rate": 3.27,
        }

    import pandas as pd

    tsv_file = tsv_files[0]
    df = pd.read_csv(tsv_file, sep="\t")

    total = len(df)
    route_switch_count = len(df[df["outlier"] == "route switch"])
    detour_count = len(df[df["outlier"] == "detour"])
    spatial_abnormal_count = route_switch_count + detour_count

    return {
        "spatial_abnormality_rate": (spatial_abnormal_count / total * 100)
        if total > 0
        else 0,
        "route_switch_rate": (route_switch_count / total * 100) if total > 0 else 0,
        "detour_rate": (detour_count / total * 100) if total > 0 else 0,
        "total_trajectories": total,
        "route_switch_count": route_switch_count,
        "detour_count": detour_count,
    }


def compute_statistical_test(
    real_count: int,
    real_total: int,
    gen_count: int,
    gen_total: int,
) -> Tuple[float, float]:
    """Perform chi-square test for proportions

    Args:
        real_count: Number of spatial abnormal in real data
        real_total: Total trajectories in real data
        gen_count: Number of spatial abnormal in generated data
        gen_total: Total trajectories in generated data

    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    # Create contingency table
    # [real_abnormal, real_normal]
    # [gen_abnormal, gen_normal]
    real_normal = real_total - real_count
    gen_normal = gen_total - gen_count

    contingency_table = np.array([[real_count, real_normal], [gen_count, gen_normal]])

    # Chi-square test
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]

    return float(chi2), float(p_value)


def aggregate_lmtad_spatial_results(
    eval_dir: Path, dataset: str, source_eval_dir: Path
) -> Dict:
    """Aggregate spatial abnormality evaluation results

    Args:
        eval_dir: Evaluation directory
        dataset: Dataset name
        source_eval_dir: Path to LM-TAD source evaluation directory

    Returns:
        Dictionary with aggregated results and statistical comparisons
    """
    logger.info(f"📊 Aggregating LM-TAD spatial results for {dataset}")

    # Load real rates from source evaluation
    real_rates = load_source_real_rates(source_eval_dir)
    logger.info(
        f"✅ Real spatial abnormality rate: {real_rates['spatial_abnormality_rate']:.2f}%"
    )
    logger.info(f"   Route switch: {real_rates['route_switch_rate']:.2f}%")
    logger.info(f"   Detour: {real_rates['detour_rate']:.2f}%")

    # Find all evaluation result files
    eval_results_dir = eval_dir / "eval_lmtad_spatial" / dataset
    if not eval_results_dir.exists():
        logger.warning(f"Evaluation results directory not found: {eval_results_dir}")
        return {}

    result_files = list(eval_results_dir.glob("*_spatial_evaluation.json"))
    logger.info(f"  Found {len(result_files)} evaluation result files")

    # Load all results
    generated_results = []
    for result_file in result_files:
        result = load_evaluation_result(result_file)
        if result:
            generated_results.append(result)

    if not generated_results:
        logger.warning("No evaluation results found")
        return {}

    # Build aggregated structure
    summary_stats = {
        dataset: {
            "real_spatial_abnormality_rate": real_rates["spatial_abnormality_rate"],
            "real_route_switch_rate": real_rates["route_switch_rate"],
            "real_detour_rate": real_rates["detour_rate"],
            "real_total_trajectories": real_rates.get("total_trajectories", 0),
            "generated_spatial_rates": [],
        }
    }

    real_data = {
        dataset: {
            "dataset": dataset,
            "model": None,
            "is_real": True,
            "total_trajectories": real_rates.get("total_trajectories", 0),
            "spatial_abnormal_count": real_rates.get("route_switch_count", 0)
            + real_rates.get("detour_count", 0),
            "spatial_abnormality_rate": real_rates["spatial_abnormality_rate"],
            "route_switch_count": real_rates.get("route_switch_count", 0),
            "route_switch_rate": real_rates["route_switch_rate"],
            "detour_count": real_rates.get("detour_count", 0),
            "detour_rate": real_rates["detour_rate"],
        }
    }

    generated_data = {dataset: {}}

    # Process generated results
    for result in generated_results:
        model_name = result.get("model", "unknown")
        generated_data[dataset][model_name] = {
            "dataset": dataset,
            "model": model_name,
            "is_real": False,
            "total_trajectories": result.get("total_trajectories", 0),
            "spatial_abnormal_count": result.get("spatial_abnormal_count", 0),
            "spatial_abnormality_rate": result.get("spatial_abnormality_rate", 0),
            "route_switch_count": result.get("by_type", {})
            .get("route_switch", {})
            .get("count", 0),
            "route_switch_rate": result.get("by_type", {})
            .get("route_switch", {})
            .get("rate", 0),
            "detour_count": result.get("by_type", {}).get("detour", {}).get("count", 0),
            "detour_rate": result.get("by_type", {}).get("detour", {}).get("rate", 0),
            "log_perplexity_stats": result.get("log_perplexity_stats", {}),
        }

        summary_stats[dataset]["generated_spatial_rates"].append(
            result.get("spatial_abnormality_rate", 0)
        )

    # Perform statistical comparisons
    statistical_tests = []
    real_total = real_rates.get("total_trajectories", 0)
    real_spatial_count = real_rates.get("route_switch_count", 0) + real_rates.get(
        "detour_count", 0
    )

    for model_name, gen_result in generated_data[dataset].items():
        gen_total = gen_result["total_trajectories"]
        gen_spatial_count = gen_result["spatial_abnormal_count"]
        gen_rate = gen_result["spatial_abnormality_rate"]
        real_rate = real_rates["spatial_abnormality_rate"]

        # Statistical test
        chi2, p_value = compute_statistical_test(
            real_count=real_spatial_count,
            real_total=real_total,
            gen_count=gen_spatial_count,
            gen_total=gen_total,
        )

        # Effect size
        cohens_h = compute_cohens_h(real_rate, gen_rate)
        effect_size = interpret_cohens_h(cohens_h)

        # Confidence interval
        ci_lower, ci_upper = compute_proportion_ci(gen_spatial_count, gen_total)

        statistical_tests.append(
            {
                "dataset": dataset,
                "model": model_name,
                "real_rate": real_rate,
                "generated_rate": gen_rate,
                "difference": gen_rate - real_rate,
                "relative_difference_pct": (
                    ((gen_rate - real_rate) / real_rate * 100) if real_rate > 0 else 0
                ),
                "trajectory_count_real": real_total,
                "trajectory_count_generated": gen_total,
                "chi2_statistic": chi2,
                "p_value": p_value,
                "cohens_h": cohens_h,
                "effect_size": effect_size,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    # Apply FDR correction
    if statistical_tests:
        p_values = [test["p_value"] for test in statistical_tests]
        _, p_values_corrected, _, _ = multipletests(
            p_values, alpha=0.05, method="fdr_bh"
        )

        for i, test in enumerate(statistical_tests):
            test["p_value_corrected"] = float(p_values_corrected[i])
            test["significant"] = p_values_corrected[i] < 0.05

    # Build final result structure
    result = {
        "summary_statistics": summary_stats,
        "real_data": real_data,
        "generated_data": generated_data,
        "statistical_analysis": {
            "statistical_tests": statistical_tests,
            "correction_method": "FDR (Benjamini-Hochberg)",
            "alpha": 0.05,
        },
    }

    logger.info(f"✅ Aggregated results for {len(generated_results)} models")
    logger.info(f"   Performed {len(statistical_tests)} statistical comparisons")

    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Aggregate LM-TAD spatial abnormality evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate results
  uv run python tools/analyze_lmtad_spatial_results.py \\
    --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \\
    --dataset porto_hoser \\
    --source-eval-dir /home/matt/Dev/LMTAD/.../eval \\
    --output analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json
        """,
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., porto_hoser, Beijing)",
    )
    parser.add_argument(
        "--source-eval-dir",
        type=Path,
        required=True,
        help="Path to LM-TAD source evaluation directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.eval_dir.exists():
        logger.error(f"Evaluation directory not found: {args.eval_dir}")
        return 1

    if not args.source_eval_dir.exists():
        logger.error(f"Source eval directory not found: {args.source_eval_dir}")
        return 1

    # Aggregate results
    try:
        result = aggregate_lmtad_spatial_results(
            eval_dir=args.eval_dir,
            dataset=args.dataset,
            source_eval_dir=args.source_eval_dir,
        )

        if not result:
            logger.warning("No results to aggregate")
            return 1

        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"✅ Aggregated results saved to {args.output}")
        return 0

    except Exception as e:
        logger.error(f"❌ Aggregation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
