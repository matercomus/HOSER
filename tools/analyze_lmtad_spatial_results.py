#!/usr/bin/env python3
"""
Analyze and Aggregate LM-TAD Perplexity-Based Evaluation Results

This script aggregates perplexity-based evaluation results from multiple models
and performs statistical comparisons on log-perplexity distributions.

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
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Import statistical functions from analyze_wang_results
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import sys  # noqa: E402


def ensure_json_serializable(obj):
    """Recursively convert object to JSON-serializable types

    Handles numpy types, booleans, and nested structures.
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, str):
        return obj
    elif obj is None:
        return None
    else:
        # Last resort: convert to string
        return str(obj)


# Backward compatibility wrappers
def aggregate_lmtad_spatial_results(
    eval_dir: Path,
    dataset: str,
    source_eval_dir: Path,
) -> Optional[Dict]:
    """
    Backward compatibility wrapper for aggregate_lmtad_perplexity_results.

    This function maintains backward compatibility with the old API by converting
    the new perplexity-focused results to the old spatial abnormality format.
    """
    logger.warning(
        "aggregate_lmtad_spatial_results is deprecated. "
        "Use aggregate_lmtad_perplexity_results instead."
    )
    result = aggregate_lmtad_perplexity_results(
        eval_dir=eval_dir,
        dataset=dataset,
        source_eval_dir=source_eval_dir,
    )

    if not result:
        return None

    # Convert to old format for backward compatibility
    # The old format used "real_data" instead of "generated_data" for real data
    # and had "statistical_tests" instead of "distribution_tests"
    old_format = result.copy()

    # Add deprecated "statistical_tests" key
    if "statistical_analysis" in old_format:
        distribution_tests = old_format["statistical_analysis"].get(
            "distribution_tests", []
        )
        # Convert distribution tests to old format
        old_format["statistical_analysis"]["statistical_tests"] = [
            {
                "model": test.get("model", ""),
                "generated_rate": test.get("mean_perplexity", 0)
                * 10,  # Convert perplexity to percentage
                "ci_lower": test.get("ci_lower", 0) * 10,
                "ci_upper": test.get("ci_upper", 0) * 10,
                "effect_size": test.get("effect_size", "unknown"),
                "cohens_h": test.get("cohens_h", 0),
                "significant": test.get("significant", False),
            }
            for test in distribution_tests
        ]

    return old_format


def load_source_real_rates(source_eval_dir: Path) -> Optional[Dict]:
    """
    Backward compatibility wrapper for load_source_perplexity_rates.

    This function maintains backward compatibility with the old API.
    Returns the old format with spatial_abnormality_rate.
    """
    logger.warning(
        "load_source_real_rates is deprecated. "
        "Use load_source_perplexity_rates instead."
    )
    result = load_source_perplexity_rates(source_eval_dir)

    if not result:
        return None

    # Convert to old format
    old_format = result.copy()
    # Add the old spatial_abnormality_rate key
    if "spatial_abnormality_rate" not in old_format:
        # Convert from perplexity stats if available
        if "log_perplexity_stats" in old_format:
            mean_perp = old_format["log_perplexity_stats"].get("mean", 0)
            old_format["spatial_abnormality_rate"] = max(
                0, min(100, (mean_perp - 5) * 20)
            )
        else:
            old_format["spatial_abnormality_rate"] = 0

    return old_format


def compute_statistical_test(
    results_1: Dict, results_2: Dict, test_type: str = "ks", **kwargs
) -> Optional[Dict]:
    """
    Backward compatibility wrapper for compare_perplexity_distributions.

    This function maintains backward compatibility with the old API.
    """
    logger.warning(
        "compute_statistical_test is deprecated. "
        "Use compare_perplexity_distributions instead."
    )

    # Handle old parameter names
    if "real_count" in kwargs:
        kwargs["count_1"] = kwargs.pop("real_count")
    if "generated_count" in kwargs:
        kwargs["count_2"] = kwargs.pop("generated_count")

    result = compare_perplexity_distributions(
        results_1=results_1, results_2=results_2, test_type=test_type, **kwargs
    )

    if not result:
        return None

    # Convert to old format
    old_format = result.copy()
    # Ensure old parameter names are present
    if "statistic" in old_format and "chi2" not in old_format:
        old_format["chi2"] = old_format["statistic"]
    if "p_value" not in old_format and "pvalue" in old_format:
        old_format["p_value"] = old_format["pvalue"]

    return old_format


@dataclass
class PerplexityEvaluationMetrics:
    """Metrics for a single perplexity-based evaluation result"""

    dataset: str
    model: Optional[str]  # None for real data
    is_real: bool
    total_trajectories: int
    log_perplexity_stats: Dict[str, float]
    segment_log_perplexities: Optional[List[List[float]]] = None
    od_pair_data: Optional[Dict[str, Dict[str, Union[float, List[float]]]]] = None


@dataclass
class PerplexityStatisticalComparison:
    """Statistical comparison of perplexity distributions between models"""

    dataset: str
    model1: str
    model2: str
    mean_perplexity_1: float
    mean_perplexity_2: float
    std_perplexity_1: float
    std_perplexity_2: float
    ks_statistic: float
    ks_p_value: float
    mannwhitney_u_statistic: float
    mannwhitney_u_p_value: float
    trajectory_count_1: int
    trajectory_count_2: int
    significant_ks: bool
    significant_mw: bool


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


def build_od_pair_data(evaluation_results: List[Dict]) -> Dict[str, Dict]:
    """Build OD pair comparison structure from evaluation results

    Args:
        evaluation_results: List of evaluation result dictionaries
        Each result should contain: model, dataset, and trajectories_with_perplexity

    Returns:
        Dictionary with structure: od_pair -> {model: {log_perplexity, ...}}
    """
    od_pair_data = {}

    for result in evaluation_results:
        model_name = result.get("model", "unknown")
        trajectories = result.get("trajectories_with_perplexity", [])

        for trajectory in trajectories:
            # Get first and last road ID for OD pair
            road_sequence = trajectory.get("road_sequence", [])
            if not road_sequence or len(road_sequence) < 2:
                continue

            # Create OD pair key as "first_road-last_road"
            od_key = f"{road_sequence[0]}-{road_sequence[-1]}"

            # Initialize OD pair if not exists
            if od_key not in od_pair_data:
                od_pair_data[od_key] = {}

            # Add data for this model
            log_perplexity = trajectory.get("log_perplexity")

            if log_perplexity is not None:
                # Get segment log perplexities if available
                segment_log_perplexities = trajectory.get(
                    "segment_log_perplexities", []
                )

                od_pair_data[od_key][model_name] = {
                    "log_perplexity": log_perplexity,
                    "segment_log_perplexities": segment_log_perplexities,
                    "trajectory": trajectory,
                }

    return od_pair_data


def compute_per_od_pair_statistics(od_pair_data: Dict, models: List[str]) -> Dict:
    """Compute statistics per OD pair for multiple models

    Args:
        od_pair_data: Dictionary mapping OD pairs to model data
        models: List of model names to analyze

    Returns:
        Dictionary with per-OD-pair statistics
    """
    per_od_statistics = {}

    for od_key, model_data in od_pair_data.items():
        # Get models that have data for this OD pair
        available_models = [m for m in models if m in model_data]

        if len(available_models) < 2:
            # Need at least 2 models for comparison
            continue

        # Collect log perplexities for each model
        od_perplexities = {}
        for model_name in available_models:
            if model_name in model_data:
                log_perplexity = model_data[model_name].get("log_perplexity")
                if log_perplexity is not None:
                    od_perplexities[model_name] = log_perplexity

        # Compute statistics if we have data from at least 2 models
        if len(od_perplexities) >= 2:
            od_stats = {
                "od_key": od_key,
                "models": available_models,
                "perplexities": od_perplexities,
                "mean_log_perplexity": np.mean(list(od_perplexities.values())),
                "std_log_perplexity": np.std(list(od_perplexities.values())),
                "min_log_perplexity": min(od_perplexities.values()),
                "max_log_perplexity": max(od_perplexities.values()),
            }
            per_od_statistics[od_key] = od_stats

    return per_od_statistics


def load_source_perplexity_rates(source_eval_dir: Path) -> Optional[Dict]:
    """Load perplexity data from source evaluation (real data)

    Args:
        source_eval_dir: Path to LM-TAD source evaluation directory

    Returns:
        Dictionary with perplexity data or None
    """
    # For now, real data perplexity rates are not available
    # This is a placeholder for when the data structure is implemented
    logger.info("Real data perplexity data not yet implemented for source evaluation")
    return None


def compare_perplexity_distributions(
    perplexities_1: np.ndarray,
    perplexities_2: np.ndarray,
    model_name_1: str,
    model_name_2: str,
) -> Dict[str, Union[float, bool, int]]:
    """Compare perplexity distributions between two models

    Args:
        perplexities_1: Array of log perplexities for model 1
        perplexities_2: Array of log perplexities for model 2
        model_name_1: Name of first model
        model_name_2: Name of second model

    Returns:
        Dictionary with statistical test results
    """
    # Validate inputs
    if len(perplexities_1) == 0 or len(perplexities_2) == 0:
        logger.warning(
            f"Cannot perform comparison: empty perplexity arrays for {model_name_1} and {model_name_2}"
        )
        return {
            "ks_statistic": float("nan"),
            "ks_p_value": float("nan"),
            "mannwhitney_u_statistic": float("nan"),
            "mannwhitney_u_p_value": float("nan"),
            "significant_ks": False,
            "significant_mw": False,
        }

    # Kolmogorov-Smirnov test
    try:
        ks_statistic, ks_p_value = stats.ks_2samp(perplexities_1, perplexities_2)
    except Exception as e:
        logger.warning(f"Kolmogorov-Smirnov test failed: {e}")
        ks_statistic, ks_p_value = float("nan"), float("nan")

    # Mann-Whitney U test
    try:
        mw_statistic, mw_p_value = stats.mannwhitneyu(
            perplexities_1, perplexities_2, alternative="two-sided"
        )
    except Exception as e:
        logger.warning(f"Mann-Whitney U test failed: {e}")
        mw_statistic, mw_p_value = float("nan"), float("nan")

    return {
        "ks_statistic": float(ks_statistic)
        if not np.isnan(ks_statistic)
        else float("nan"),
        "ks_p_value": float(ks_p_value) if not np.isnan(ks_p_value) else float("nan"),
        "mannwhitney_u_statistic": float(mw_statistic)
        if not np.isnan(mw_statistic)
        else float("nan"),
        "mannwhitney_u_p_value": float(mw_p_value)
        if not np.isnan(mw_p_value)
        else float("nan"),
        "significant_ks": bool(ks_p_value < 0.05)
        if not np.isnan(ks_p_value)
        else False,
        "significant_mw": bool(mw_p_value < 0.05)
        if not np.isnan(mw_p_value)
        else False,
    }


def paired_perplexity_test(
    od_pair_data: Dict, models: List[str], min_pairs: int = 5
) -> List[Dict]:
    """Perform paired t-test on log-perplexity per OD pair

    Args:
        od_pair_data: Dictionary mapping OD pairs to model data
        models: List of model names to compare
        min_pairs: Minimum number of shared OD pairs for comparison

    Returns:
        List of paired test results
    """
    results = []

    # Compare each pair of models
    for i, model_1 in enumerate(models):
        for j, model_2 in enumerate(models[i + 1 :], start=i + 1):
            # Find shared OD pairs
            shared_od_pairs = []
            perplexities_1 = []
            perplexities_2 = []

            for od_key, model_data in od_pair_data.items():
                if model_1 in model_data and model_2 in model_data:
                    perp_1 = model_data[model_1].get("log_perplexity")
                    perp_2 = model_data[model_2].get("log_perplexity")

                    if perp_1 is not None and perp_2 is not None:
                        shared_od_pairs.append(od_key)
                        perplexities_1.append(perp_1)
                        perplexities_2.append(perp_2)

            # Need minimum number of paired observations
            if len(shared_od_pairs) < min_pairs:
                logger.warning(
                    f"Insufficient paired OD pairs for {model_1} vs {model_2}: "
                    f"{len(shared_od_pairs)} < {min_pairs}"
                )
                continue

            # Perform paired t-test
            try:
                t_statistic, p_value = stats.ttest_rel(perplexities_1, perplexities_2)
            except Exception as e:
                logger.warning(f"Paired t-test failed for {model_1} vs {model_2}: {e}")
                t_statistic, p_value = float("nan"), float("nan")

            # Compute effect size (Cohen's d)
            diff = np.array(perplexities_1) - np.array(perplexities_2)
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

            results.append(
                {
                    "model_1": model_1,
                    "model_2": model_2,
                    "shared_od_pairs": len(shared_od_pairs),
                    "mean_diff": np.mean(diff),
                    "std_diff": np.std(diff),
                    "cohens_d": float(cohens_d),
                    "t_statistic": float(t_statistic)
                    if not np.isnan(t_statistic)
                    else float("nan"),
                    "p_value": float(p_value)
                    if not np.isnan(p_value)
                    else float("nan"),
                    "significant": bool(p_value < 0.05)
                    if not np.isnan(p_value)
                    else False,
                }
            )

    return results


def aggregate_lmtad_perplexity_results(
    eval_dir: Path, dataset: str, source_eval_dir: Path
) -> Dict:
    """Aggregate perplexity-based evaluation results

    Args:
        eval_dir: Evaluation directory
        dataset: Dataset name
        source_eval_dir: Path to LM-TAD source evaluation directory

    Returns:
        Dictionary with aggregated results and perplexity-based statistical comparisons
    """
    logger.info(f"📊 Aggregating LM-TAD perplexity results for {dataset}")

    # Load perplexity data from source evaluation (real data - not yet available)
    real_perplexity = load_source_perplexity_rates(source_eval_dir)
    if real_perplexity:
        logger.info("✅ Real perplexity data loaded")
    else:
        logger.info("⚠️  Real perplexity data not available")

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

    # Build model data structure
    generated_data = {dataset: {}}
    all_perplexities = {}
    model_names = []

    # Process generated results
    for result in generated_results:
        model_name = result.get("model", "unknown")
        model_names.append(model_name)

        # Extract perplexity statistics
        log_perplexity_stats = result.get("log_perplexity_stats", {})

        # Collect trajectory-level perplexity data if available
        trajectories_with_perplexity = result.get("trajectories_with_perplexity", [])

        # Store model data
        generated_data[dataset][model_name] = {
            "dataset": dataset,
            "model": model_name,
            "is_real": False,
            "total_trajectories": result.get("total_trajectories", 0),
            "log_perplexity_stats": log_perplexity_stats,
            "trajectories_with_perplexity": trajectories_with_perplexity,
        }

        # Collect perplexity values for distribution comparison
        if "mean" in log_perplexity_stats:
            # Use mean perplexity as a representative value
            all_perplexities[model_name] = log_perplexity_stats["mean"]
        elif trajectories_with_perplexity:
            # Collect all log perplexities from trajectories
            perplexities = [
                t.get("log_perplexity")
                for t in trajectories_with_perplexity
                if t.get("log_perplexity") is not None
            ]
            if perplexities:
                all_perplexities[model_name] = perplexities

    # Build OD pair comparison structure
    od_pair_data = build_od_pair_data(generated_results)

    # Compute per-OD-pair statistics if we have multiple models
    per_od_statistics = {}
    if len(model_names) >= 2:
        per_od_statistics = compute_per_od_pair_statistics(od_pair_data, model_names)
        logger.info(
            f"✅ Computed statistics for {len(per_od_statistics)} OD pairs with multiple models"
        )

    # Perform statistical comparisons between models
    perplexity_comparisons = []
    distribution_tests = []

    # Collect all perplexity distributions for comparison
    if all_perplexities:
        model_list = list(all_perplexities.keys())
        for i, model_1 in enumerate(model_list):
            for j, model_2 in enumerate(model_list[i + 1 :], start=i + 1):
                perp_1 = all_perplexities[model_1]
                perp_2 = all_perplexities[model_2]

                # Handle different data types
                if isinstance(perp_1, (list, np.ndarray)) and isinstance(
                    perp_2, (list, np.ndarray)
                ):
                    # Distribution comparison
                    perp_1_array = np.array(perp_1)
                    perp_2_array = np.array(perp_2)

                    if len(perp_1_array) > 0 and len(perp_2_array) > 0:
                        test_result = compare_perplexity_distributions(
                            perp_1_array, perp_2_array, model_1, model_2
                        )

                        distribution_tests.append(
                            {
                                "dataset": dataset,
                                "model_1": model_1,
                                "model_2": model_2,
                                "mean_perplexity_1": float(np.mean(perp_1_array)),
                                "mean_perplexity_2": float(np.mean(perp_2_array)),
                                "std_perplexity_1": float(np.std(perp_1_array)),
                                "std_perplexity_2": float(np.std(perp_2_array)),
                                "trajectory_count_1": len(perp_1_array),
                                "trajectory_count_2": len(perp_2_array),
                                **test_result,
                            }
                        )
                else:
                    # Simple mean comparison
                    mean_diff = perp_1 - perp_2
                    perplexity_comparisons.append(
                        {
                            "dataset": dataset,
                            "model_1": model_1,
                            "model_2": model_2,
                            "mean_perplexity_1": float(perp_1),
                            "mean_perplexity_2": float(perp_2),
                            "mean_diff": float(mean_diff),
                            "abs_diff": float(abs(mean_diff)),
                        }
                    )

    # Perform paired t-tests on OD pairs
    paired_tests = []
    if len(model_names) >= 2 and od_pair_data:
        paired_tests = paired_perplexity_test(od_pair_data, model_names)
        logger.info(f"✅ Performed {len(paired_tests)} paired tests on OD pairs")

    # Build summary statistics
    summary_stats = {
        dataset: {
            "total_models": len(model_names),
            "model_names": model_names,
            "total_od_pairs": len(od_pair_data),
            "compared_od_pairs": len(per_od_statistics),
            "per_model_perplexity": {
                model: generated_data[dataset][model]["log_perplexity_stats"]
                for model in model_names
            },
        }
    }

    # Apply FDR correction for distribution tests
    if distribution_tests:
        p_values_ks = [test["ks_p_value"] for test in distribution_tests]
        p_values_mw = [test["mannwhitney_u_p_value"] for test in distribution_tests]

        # Filter out NaN values for FDR correction
        valid_indices_ks = [i for i, p in enumerate(p_values_ks) if not np.isnan(p)]
        valid_indices_mw = [i for i, p in enumerate(p_values_mw) if not np.isnan(p)]

        if valid_indices_ks:
            valid_p_values_ks = [p_values_ks[i] for i in valid_indices_ks]
            _, p_values_corrected_ks, _, _ = multipletests(
                valid_p_values_ks, alpha=0.05, method="fdr_bh"
            )
            for i, idx in enumerate(valid_indices_ks):
                distribution_tests[idx]["ks_p_value_corrected"] = float(
                    p_values_corrected_ks[i]
                )
                distribution_tests[idx]["ks_significant_corrected"] = bool(
                    p_values_corrected_ks[i] < 0.05
                )

        if valid_indices_mw:
            valid_p_values_mw = [p_values_mw[i] for i in valid_indices_mw]
            _, p_values_corrected_mw, _, _ = multipletests(
                valid_p_values_mw, alpha=0.05, method="fdr_bh"
            )
            for i, idx in enumerate(valid_indices_mw):
                distribution_tests[idx]["mw_p_value_corrected"] = float(
                    p_values_corrected_mw[i]
                )
                distribution_tests[idx]["mw_significant_corrected"] = bool(
                    p_values_corrected_mw[i] < 0.05
                )

        # Apply FDR correction for paired t-tests
        if paired_tests:
            paired_p_values = [test["p_value"] for test in paired_tests]
            valid_indices_paired = [
                i for i, p in enumerate(paired_p_values) if not np.isnan(p)
            ]
            if valid_indices_paired:
                valid_paired_p_values = [
                    paired_p_values[i] for i in valid_indices_paired
                ]
                _, p_values_corrected_paired, _, _ = multipletests(
                    valid_paired_p_values, alpha=0.05, method="fdr_bh"
                )
                for i, idx in enumerate(valid_indices_paired):
                    paired_tests[idx]["p_value_corrected"] = float(
                        p_values_corrected_paired[i]
                    )
                    paired_tests[idx]["significant_corrected"] = bool(
                        p_values_corrected_paired[i] < 0.05
                    )

    # Build final result structure
    result = {
        "summary_statistics": summary_stats,
        "generated_data": generated_data,
        "od_pair_data": od_pair_data,
        "per_od_pair_statistics": per_od_statistics,
        "statistical_analysis": {
            "perplexity_comparisons": perplexity_comparisons,
            "distribution_tests": distribution_tests,
            "paired_tests": paired_tests,
            "correction_method": "FDR (Benjamini-Hochberg)",
            "alpha": 0.05,
        },
    }

    logger.info(f"✅ Aggregated results for {len(model_names)} models")
    logger.info(f"   Performed {len(distribution_tests)} distribution comparisons")
    logger.info(f"   Performed {len(paired_tests)} paired tests")
    logger.info(f"   Built data for {len(od_pair_data)} OD pairs")

    # Ensure all values are JSON serializable
    return ensure_json_serializable(result)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Aggregate LM-TAD perplexity-based evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate perplexity-based results
  uv run python tools/analyze_lmtad_spatial_results.py \\
    --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \\
    --dataset porto_hoser \\
    --source-eval-dir /home/matt/Dev/LMTAD/.../eval \\
    --output analysis_abnormal/porto_hoser/lmtad_perplexity_results_aggregated.json
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

    # Aggregate perplexity-based results
    try:
        result = aggregate_lmtad_perplexity_results(
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
