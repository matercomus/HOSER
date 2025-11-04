#!/usr/bin/env python3
"""
Wang Statistical Abnormality Detection - Results Analysis and Aggregation

This script collects and aggregates all Wang statistical abnormality detection
results from evaluation directories, performing statistical analysis and preparing
data for research report generation.

Usage:
    uv run python tools/analyze_wang_results.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available - statistical tests will be limited")


@dataclass
class DetectionMetrics:
    """Metrics for a single detection result"""

    dataset: str
    od_source: str
    model: Optional[str]  # None for real data
    is_real: bool
    total_trajectories: int
    abnormal_count: int
    abnormal_rate: float
    pattern_counts: Dict[str, int]  # Abp1-4 counts
    abnormal_by_category: Dict[str, int]
    baseline_usage: Optional[Dict[str, int]] = None  # od_specific, global, none
    wang_metadata: Optional[Dict[str, Any]] = None


@dataclass
class StatisticalComparison:
    """Statistical comparison between real and generated"""

    dataset: str
    od_source: str
    model: str
    real_rate: float
    generated_rate: float
    difference: float  # generated - real
    relative_difference_pct: float  # (generated - real) / real * 100
    trajectory_count_real: int
    trajectory_count_generated: int


class WangResultsCollector:
    """Collect and aggregate Wang statistical detection results"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[DetectionMetrics] = []
        self.comparisons: List[StatisticalComparison] = []

    def collect_from_eval_dir(self, eval_dir: Path) -> None:
        """Collect results from an evaluation directory

        Args:
            eval_dir: Path to evaluation directory (e.g., hoser-distill-optuna-6)
        """
        abnormal_dir = eval_dir / "abnormal"
        if not abnormal_dir.exists():
            logger.warning(f"Abnormal directory not found: {abnormal_dir}")
            return

        logger.info(f"📂 Collecting results from {eval_dir.name}")

        # Find all comparison_report.json files
        comparison_files = list(abnormal_dir.rglob("comparison_report.json"))
        logger.info(f"  Found {len(comparison_files)} comparison reports")

        for comp_file in comparison_files:
            self._process_comparison_report(comp_file, eval_dir)

        # Also load individual detection_results.json for pattern_counts and baseline_usage
        detection_files = list(abnormal_dir.rglob("detection_results.json"))
        logger.info(f"  Found {len(detection_files)} detection results files")

        for det_file in detection_files:
            self._process_detection_results(det_file)

    def _process_comparison_report(self, comp_file: Path, eval_dir: Path) -> None:
        """Process a comparison_report.json file

        Args:
            comp_file: Path to comparison_report.json
            eval_dir: Evaluation directory (for context)
        """
        try:
            with open(comp_file, "r") as f:
                data = json.load(f)

            dataset = data.get("dataset", "unknown")
            od_source = data.get("od_source", "unknown")

            # Process real data
            real_data = data.get("real_data", {})
            if real_data:
                self.results.append(
                    DetectionMetrics(
                        dataset=dataset,
                        od_source=od_source,
                        model=None,
                        is_real=True,
                        total_trajectories=real_data.get("total_trajectories", 0),
                        abnormal_count=real_data.get("abnormal_count", 0),
                        abnormal_rate=real_data.get("abnormal_rate", 0.0),
                        pattern_counts={},  # Will be filled from detection_results.json
                        abnormal_by_category=real_data.get("abnormal_by_category", {}),
                    )
                )

            # Process generated data
            generated_data = data.get("generated_data", {})
            for model_name, model_results in generated_data.items():
                self.results.append(
                    DetectionMetrics(
                        dataset=dataset,
                        od_source=od_source,
                        model=model_name,
                        is_real=False,
                        total_trajectories=model_results.get("total_trajectories", 0),
                        abnormal_count=model_results.get("abnormal_count", 0),
                        abnormal_rate=model_results.get("abnormal_rate", 0.0),
                        pattern_counts={},  # Will be filled from detection_results.json
                        abnormal_by_category=model_results.get(
                            "abnormal_by_category", {}
                        ),
                    )
                )

                # Create comparison if we have real data
                if real_data:
                    self.comparisons.append(
                        StatisticalComparison(
                            dataset=dataset,
                            od_source=od_source,
                            model=model_name,
                            real_rate=real_data.get("abnormal_rate", 0.0),
                            generated_rate=model_results.get("abnormal_rate", 0.0),
                            difference=model_results.get("abnormal_rate", 0.0)
                            - real_data.get("abnormal_rate", 0.0),
                            relative_difference_pct=(
                                (
                                    model_results.get("abnormal_rate", 0.0)
                                    - real_data.get("abnormal_rate", 0.0)
                                )
                                / real_data.get("abnormal_rate", 1.0)
                                * 100
                            )
                            if real_data.get("abnormal_rate", 0) > 0
                            else 0.0,
                            trajectory_count_real=real_data.get(
                                "total_trajectories", 0
                            ),
                            trajectory_count_generated=model_results.get(
                                "total_trajectories", 0
                            ),
                        )
                    )

        except Exception as e:
            logger.error(f"Error processing {comp_file}: {e}")

    def _process_detection_results(self, det_file: Path) -> None:
        """Process a detection_results.json file to extract pattern_counts and metadata

        Args:
            det_file: Path to detection_results.json
        """
        try:
            with open(det_file, "r") as f:
                data = json.load(f)

            dataset = data.get("dataset", "unknown")
            pattern_counts = data.get("pattern_counts", {})
            wang_metadata = data.get("wang_metadata", {})
            baseline_usage = (
                wang_metadata.get("baseline_usage") if wang_metadata else None
            )

            # Reconstruct pattern_counts from abnormal_indices if not present
            if not pattern_counts:
                abnormal_indices = data.get("abnormal_indices", {})
                total = data.get("total_trajectories", 0)

                # Map Wang categories to patterns
                abp2_count = len(abnormal_indices.get("wang_temporal_delay", []))
                abp3_count = len(abnormal_indices.get("wang_route_deviation", []))
                abp4_count = len(abnormal_indices.get("wang_both_deviations", []))
                abp1_count = total - abp2_count - abp3_count - abp4_count

                pattern_counts = {
                    "Abp1_normal": max(0, abp1_count),
                    "Abp2_temporal_delay": abp2_count,
                    "Abp3_route_deviation": abp3_count,
                    "Abp4_both_deviations": abp4_count,
                }

            # Try to identify dataset, od_source, and model from path
            # Path structure: abnormal/{dataset}/{od_source}/{real_data|generated}/{model}/detection_results.json
            parts = det_file.parts
            dataset_idx = None
            od_source_idx = None
            model_idx = None

            for i, part in enumerate(parts):
                if part == "abnormal" and i + 1 < len(parts):
                    dataset_idx = i + 1
                elif part in ["train", "test"]:
                    od_source_idx = i
                elif part in ["real_data", "generated"]:
                    if part == "generated" and i + 1 < len(parts):
                        model_idx = i + 1

            # Update matching results
            dataset_name = parts[dataset_idx] if dataset_idx else dataset
            od_source = parts[od_source_idx] if od_source_idx else "unknown"
            is_real = (
                parts[od_source_idx + 1] == "real_data" if od_source_idx else False
            )
            model_name = parts[model_idx] if model_idx and not is_real else None

            # Find matching result and update it
            for result in self.results:
                if (
                    result.dataset == dataset_name
                    and result.od_source == od_source
                    and result.is_real == is_real
                    and result.model == model_name
                ):
                    result.pattern_counts = pattern_counts
                    result.wang_metadata = wang_metadata
                    if baseline_usage:
                        result.baseline_usage = baseline_usage
                    break

        except Exception as e:
            logger.warning(f"Error processing {det_file}: {e}")

    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate all collected results into structured format

        Returns:
            Dictionary with aggregated results organized by dataset, OD source, and model
        """
        logger.info("📊 Aggregating results...")

        # Organize by dataset
        # by_dataset = defaultdict(lambda: defaultdict(dict))  # Not used, keeping for future use

        # Group real data
        real_data_by_dataset = defaultdict(dict)
        for result in self.results:
            if result.is_real:
                real_data_by_dataset[result.dataset][result.od_source] = result

        # Group generated data
        generated_by_dataset = defaultdict(lambda: defaultdict(dict))
        for result in self.results:
            if not result.is_real:
                generated_by_dataset[result.dataset][result.od_source][result.model] = (
                    result
                )

        # Calculate summary statistics
        summary_stats = {}
        for dataset in set(r.dataset for r in self.results):
            dataset_results = [r for r in self.results if r.dataset == dataset]
            real_results = [r for r in dataset_results if r.is_real]
            generated_results = [r for r in dataset_results if not r.is_real]

            summary_stats[dataset] = {
                "total_evaluations": len(dataset_results),
                "real_data_evaluations": len(real_results),
                "generated_evaluations": len(generated_results),
                "models_evaluated": len(
                    set(r.model for r in generated_results if r.model)
                ),
                "real_abnormality_rates": [r.abnormal_rate for r in real_results],
                "generated_abnormality_rates": [
                    r.abnormal_rate for r in generated_results
                ],
                "mean_real_rate": statistics.mean(
                    [r.abnormal_rate for r in real_results]
                )
                if real_results
                else 0.0,
                "mean_generated_rate": statistics.mean(
                    [r.abnormal_rate for r in generated_results]
                )
                if generated_results
                else 0.0,
            }

        return {
            "summary_statistics": summary_stats,
            "real_data": {
                dataset: {
                    od_source: asdict(result)
                    for od_source, result in real_data_by_dataset[dataset].items()
                }
                for dataset in real_data_by_dataset
            },
            "generated_data": {
                dataset: {
                    od_source: {
                        model: asdict(result)
                        for model, result in generated_by_dataset[dataset][
                            od_source
                        ].items()
                    }
                    for od_source in generated_by_dataset[dataset]
                }
                for dataset in generated_by_dataset
            },
            "comparisons": [asdict(comp) for comp in self.comparisons],
        }

    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis on collected results

        Returns:
            Dictionary with statistical analysis results
        """
        logger.info("📊 Performing statistical analysis...")

        analysis = {
            "model_rankings": self._rank_models_by_realism(),
            "pattern_distributions": self._analyze_pattern_distributions(),
            "baseline_usage_analysis": self._analyze_baseline_usage(),
            "cross_dataset_comparison": self._analyze_cross_dataset(),
            "statistical_tests": self._perform_significance_tests(),
        }

        return analysis

    def _rank_models_by_realism(self) -> Dict[str, List[Dict[str, Any]]]:
        """Rank models by how close their abnormality rates are to real data

        Returns:
            Dictionary mapping dataset to list of models ranked by realism
        """
        rankings = {}

        for dataset in set(c.dataset for c in self.comparisons):
            dataset_comparisons = [c for c in self.comparisons if c.dataset == dataset]

            # Calculate absolute deviation from real rate
            model_scores = []
            for comp in dataset_comparisons:
                absolute_deviation = abs(comp.difference)
                model_scores.append(
                    {
                        "model": comp.model,
                        "od_source": comp.od_source,
                        "real_rate": comp.real_rate,
                        "generated_rate": comp.generated_rate,
                        "absolute_deviation": absolute_deviation,
                        "relative_deviation_pct": comp.relative_difference_pct,
                    }
                )

            # Sort by absolute deviation (smaller is better)
            model_scores.sort(key=lambda x: x["absolute_deviation"])

            rankings[dataset] = model_scores

        return rankings

    def _analyze_pattern_distributions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze pattern distributions (Abp1-4) across datasets

        Returns:
            Dictionary with pattern distribution analysis
        """
        pattern_analysis = {}

        for dataset in set(r.dataset for r in self.results):
            dataset_results = [r for r in self.results if r.dataset == dataset]

            # Aggregate pattern counts
            total_patterns = {
                "Abp1_normal": 0,
                "Abp2_temporal_delay": 0,
                "Abp3_route_deviation": 0,
                "Abp4_both_deviations": 0,
            }

            total_trajectories = 0

            for result in dataset_results:
                if result.pattern_counts:
                    for pattern, count in result.pattern_counts.items():
                        total_patterns[pattern] = total_patterns.get(pattern, 0) + count
                    total_trajectories += result.total_trajectories

            # Calculate percentages
            pattern_percentages = {}
            if total_trajectories > 0:
                for pattern, count in total_patterns.items():
                    pattern_percentages[pattern] = (count / total_trajectories) * 100

            pattern_analysis[dataset] = {
                "total_trajectories": total_trajectories,
                "pattern_counts": total_patterns,
                "pattern_percentages": pattern_percentages,
            }

        return pattern_analysis

    def _analyze_baseline_usage(self) -> Dict[str, Dict[str, Any]]:
        """Analyze baseline usage statistics (OD-specific vs global)

        Returns:
            Dictionary with baseline usage analysis
        """
        baseline_analysis = {}

        for dataset in set(r.dataset for r in self.results):
            dataset_results = [
                r for r in self.results if r.dataset == dataset and r.baseline_usage
            ]

            if not dataset_results:
                continue

            total_od_specific = 0
            total_global = 0
            total_none = 0

            for result in dataset_results:
                if result.baseline_usage:
                    total_od_specific += result.baseline_usage.get("od_specific", 0)
                    total_global += result.baseline_usage.get("global", 0)
                    total_none += result.baseline_usage.get("none", 0)

            total = total_od_specific + total_global + total_none

            baseline_analysis[dataset] = {
                "od_specific": total_od_specific,
                "global": total_global,
                "none": total_none,
                "total": total,
                "od_specific_pct": (total_od_specific / total * 100)
                if total > 0
                else 0,
                "global_pct": (total_global / total * 100) if total > 0 else 0,
            }

        return baseline_analysis

    def _analyze_cross_dataset(self) -> Dict[str, Any]:
        """Analyze cross-dataset evaluation results

        Returns:
            Dictionary with cross-dataset analysis
        """
        cross_dataset_analysis = {}

        # Find BJUT_Beijing results (cross-dataset evaluation)
        bjut_results = [r for r in self.results if r.dataset == "BJUT_Beijing"]
        beijing_results = [r for r in self.results if r.dataset == "Beijing"]

        if bjut_results and beijing_results:
            # Compare same models on Beijing vs BJUT_Beijing
            model_comparisons = {}

            for bjut_result in bjut_results:
                if not bjut_result.is_real and bjut_result.model:
                    # Find corresponding Beijing result
                    beijing_match = next(
                        (
                            r
                            for r in beijing_results
                            if r.model == bjut_result.model
                            and r.od_source == bjut_result.od_source
                            and not r.is_real
                        ),
                        None,
                    )

                    if beijing_match:
                        model_comparisons[bjut_result.model] = {
                            "beijing_rate": beijing_match.abnormal_rate,
                            "bjut_rate": bjut_result.abnormal_rate,
                            "difference": bjut_result.abnormal_rate
                            - beijing_match.abnormal_rate,
                            "relative_change_pct": (
                                (
                                    bjut_result.abnormal_rate
                                    - beijing_match.abnormal_rate
                                )
                                / beijing_match.abnormal_rate
                                * 100
                            )
                            if beijing_match.abnormal_rate > 0
                            else 0,
                        }

            cross_dataset_analysis["beijing_to_bjut"] = model_comparisons

        return cross_dataset_analysis

    def _perform_significance_tests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Perform statistical significance tests comparing real vs generated

        Returns:
            Dictionary with statistical test results
        """
        test_results = {}

        if not HAS_SCIPY:
            logger.warning("scipy not available - skipping statistical tests")
            return test_results

        for dataset in set(c.dataset for c in self.comparisons):
            dataset_comparisons = [c for c in self.comparisons if c.dataset == dataset]
            dataset_tests = []

            for comp in dataset_comparisons:
                # Chi-square test for proportions
                # Real: comp.trajectory_count_real trajectories, comp.real_rate abnormal
                # Generated: comp.trajectory_count_generated trajectories, comp.generated_rate abnormal

                real_abnormal = int(comp.trajectory_count_real * comp.real_rate / 100)
                real_normal = comp.trajectory_count_real - real_abnormal

                gen_abnormal = int(
                    comp.trajectory_count_generated * comp.generated_rate / 100
                )
                gen_normal = comp.trajectory_count_generated - gen_abnormal

                # Contingency table
                contingency = [[real_abnormal, real_normal], [gen_abnormal, gen_normal]]

                try:
                    chi2, p_value = stats.chi2_contingency(contingency)[:2]

                    dataset_tests.append(
                        {
                            "model": comp.model,
                            "od_source": comp.od_source,
                            "real_rate": comp.real_rate,
                            "generated_rate": comp.generated_rate,
                            "chi2": float(chi2),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "interpretation": "Significantly different"
                            if p_value < 0.05
                            else "Not significantly different",
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Error performing chi-square test for {comp.model}: {e}"
                    )
                    dataset_tests.append(
                        {
                            "model": comp.model,
                            "od_source": comp.od_source,
                            "error": str(e),
                        }
                    )

            test_results[dataset] = dataset_tests

        return test_results

    def _ensure_json_serializable(self, obj):
        """Recursively convert object to JSON-serializable types"""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_json_serializable(item) for item in obj]
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
            return str(obj)

    def save_aggregated_results(self, output_file: Path) -> None:
        """Save aggregated results to JSON file

        Args:
            output_file: Path to output JSON file
        """
        aggregated = self.aggregate_results()
        statistical_analysis = self.perform_statistical_analysis()

        # Combine results
        full_results = {
            **aggregated,
            "statistical_analysis": statistical_analysis,
        }

        # Ensure JSON serializable
        full_results = self._ensure_json_serializable(full_results)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2)

        logger.info(
            f"💾 Saved aggregated results with statistical analysis to {output_file}"
        )


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent

    collector = WangResultsCollector(project_root)

    # Collect from Beijing eval directory
    beijing_eval = project_root / "hoser-distill-optuna-6"
    if beijing_eval.exists():
        collector.collect_from_eval_dir(beijing_eval)
    else:
        logger.warning(f"Beijing eval directory not found: {beijing_eval}")

    # Collect from Porto eval directory
    porto_eval_pattern = "hoser-distill-optuna-porto-eval-*"
    porto_eval_dirs = list(project_root.glob(porto_eval_pattern))
    if porto_eval_dirs:
        for porto_eval in porto_eval_dirs:
            collector.collect_from_eval_dir(porto_eval)
    else:
        logger.warning(f"Porto eval directory not found: {porto_eval_pattern}")

    logger.info(f"✅ Collected {len(collector.results)} result entries")
    logger.info(f"✅ Collected {len(collector.comparisons)} comparisons")

    # Save aggregated results
    output_file = project_root / "wang_results_aggregated.json"
    collector.save_aggregated_results(output_file)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 COLLECTION SUMMARY")
    print("=" * 70)
    aggregated = collector.aggregate_results()
    summary = aggregated["summary_statistics"]

    for dataset, stats in summary.items():
        print(f"\n{dataset}:")
        print(f"  Real data evaluations: {stats['real_data_evaluations']}")
        print(f"  Generated evaluations: {stats['generated_evaluations']}")
        print(f"  Models evaluated: {stats['models_evaluated']}")
        print(f"  Mean real abnormality rate: {stats['mean_real_rate']:.2f}%")
        print(f"  Mean generated abnormality rate: {stats['mean_generated_rate']:.2f}%")

    # Print statistical analysis summary
    print("\n" + "=" * 70)
    print("📊 STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)
    statistical_analysis = collector.perform_statistical_analysis()

    # Model rankings
    print("\n🏆 Model Rankings (by realism, closest to real data):")
    for dataset, rankings in statistical_analysis["model_rankings"].items():
        print(f"\n  {dataset}:")
        for i, model_info in enumerate(rankings[:5], 1):  # Top 5
            print(
                f"    {i}. {model_info['model']} ({model_info['od_source']}): "
                f"deviation = {model_info['absolute_deviation']:.2f}% "
                f"(real: {model_info['real_rate']:.2f}%, "
                f"generated: {model_info['generated_rate']:.2f}%)"
            )

    print("\n" + "=" * 70)
    print(f"💾 Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
