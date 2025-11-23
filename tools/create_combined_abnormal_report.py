#!/usr/bin/env python3
"""
Create Combined Abnormal Trajectory Analysis Report

This script combines Wang temporal and LM-TAD spatial abnormality evaluation results
into a comprehensive analysis report.

Usage:
    uv run python tools/create_combined_abnormal_report.py \\
        --wang-results analysis_abnormal/porto_hoser/wang_results_aggregated.json \\
        --lmtad-spatial-results analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \\
        --output analysis_abnormal/porto_hoser/COMBINED_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md \\
        --dataset porto_hoser
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_wang_results(wang_results_file: Path) -> Optional[Dict]:
    """Load Wang temporal abnormality results from JSON file"""
    if not wang_results_file.exists():
        logger.warning(f"Wang results file not found: {wang_results_file}")
        return None

    logger.info(f"📂 Loading Wang results from {wang_results_file}")
    with open(wang_results_file, "r") as f:
        return json.load(f)


def load_lmtad_spatial_results(lmtad_spatial_file: Path) -> Optional[Dict]:
    """Load LM-TAD spatial abnormality results from JSON file"""
    if not lmtad_spatial_file.exists():
        logger.warning(f"LM-TAD spatial results file not found: {lmtad_spatial_file}")
        return None

    logger.info(f"📂 Loading LM-TAD spatial results from {lmtad_spatial_file}")
    with open(lmtad_spatial_file, "r") as f:
        return json.load(f)


def create_combined_report(
    wang_results: Optional[Dict],
    lmtad_spatial_results: Optional[Dict],
    output_file: Path,
    dataset: str,
) -> None:
    """Create combined abnormal trajectory analysis report

    Args:
        wang_results: Wang temporal abnormality results
        lmtad_spatial_results: LM-TAD spatial abnormality results
        output_file: Output markdown file path
        output_file: Output markdown file path
        dataset: Dataset name
    """
    logger.info(f"📝 Creating combined report for {dataset}...")

    report_lines = []
    report_lines.append("# Comprehensive Abnormal Trajectory Analysis Report")
    report_lines.append(f"## {dataset.upper()} Dataset Evaluation")
    report_lines.append("")
    report_lines.append(f"**Date**: {datetime.now().strftime('%B %d, %Y')}")
    report_lines.append(f"**Dataset**: {dataset}")
    report_lines.append(
        "**Analysis Type**: Combined temporal (Wang) and spatial (LM-TAD) abnormality evaluation"
    )
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append(
        "This comprehensive analysis evaluates trajectory generation models on both "
        "**temporal abnormalities** (detected by Wang statistical method) and "
        "**spatial abnormalities** (detected by LM-TAD teacher model)."
    )
    report_lines.append("")

    # Temporal Abnormality Summary
    if wang_results:
        wang_summary = wang_results.get("summary_statistics", {}).get(dataset, {})
        if wang_summary:
            real_temporal_rate = wang_summary.get("mean_real_rate", 0)
            generated_temporal_rates = wang_summary.get("generated_rates", [])
            mean_generated_temporal = (
                sum(generated_temporal_rates) / len(generated_temporal_rates)
                if generated_temporal_rates
                else 0
            )

            report_lines.append("### Temporal Abnormality (Wang Method)")
            report_lines.append("")
            report_lines.append(f"- **Real Data Rate**: {real_temporal_rate:.2f}%")
            report_lines.append(
                f"- **Mean Generated Rate**: {mean_generated_temporal:.2f}%"
            )
            report_lines.append(
                f"- **Gap**: {real_temporal_rate - mean_generated_temporal:.2f} percentage points"
            )
            report_lines.append("")

    # Spatial Abnormality Summary
    if lmtad_spatial_results:
        spatial_summary = lmtad_spatial_results.get("summary_statistics", {}).get(
            dataset, {}
        )
        if spatial_summary:
            real_spatial_rate = spatial_summary.get("real_spatial_abnormality_rate", 0)
            generated_spatial_rates = spatial_summary.get("generated_spatial_rates", [])
            mean_generated_spatial = (
                sum(generated_spatial_rates) / len(generated_spatial_rates)
                if generated_spatial_rates
                else 0
            )

            report_lines.append("### Spatial Abnormality (LM-TAD Method)")
            report_lines.append("")
            report_lines.append(f"- **Real Data Rate**: {real_spatial_rate:.2f}%")
            report_lines.append(
                f"- **Mean Generated Rate**: {mean_generated_spatial:.2f}%"
            )
            report_lines.append(
                f"- **Gap**: {real_spatial_rate - mean_generated_spatial:.2f} percentage points"
            )
            report_lines.append("")
            report_lines.append(
                f"- **Route Switch Rate**: {spatial_summary.get('real_route_switch_rate', 0):.2f}%"
            )
            report_lines.append(
                f"- **Detour Rate**: {spatial_summary.get('real_detour_rate', 0):.2f}%"
            )
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Temporal Abnormality Analysis Section
    if wang_results:
        report_lines.append("## Temporal Abnormality Analysis (Wang Method)")
        report_lines.append("")
        report_lines.append(
            "The Wang statistical method identifies temporal abnormalities by comparing "
            "trajectories against statistical baselines (global and OD-specific)."
        )
        report_lines.append("")

        wang_summary = wang_results.get("summary_statistics", {}).get(dataset, {})
        if wang_summary:
            report_lines.append("### Real Data Statistics")
            report_lines.append("")
            report_lines.append(
                f"- **Test Split Rate**: {wang_summary.get('test_real_rate', 0):.2f}%"
            )
            report_lines.append(
                f"- **Train Split Rate**: {wang_summary.get('train_real_rate', 0):.2f}%"
            )
            report_lines.append("")

        # Model comparisons
        wang_tests = (
            wang_results.get("statistical_analysis", {})
            .get("statistical_tests", {})
            .get(dataset, [])
        )

        if wang_tests:
            report_lines.append("### Model Performance on Temporal Abnormalities")
            report_lines.append("")
            report_lines.append(
                "| Model | Generated Rate | Real Rate | Difference | Effect Size |"
            )
            report_lines.append(
                "|-------|---------------|-----------|------------|-------------|"
            )

            for test in sorted(wang_tests, key=lambda x: abs(x.get("difference", 0))):
                model = test.get("model", "unknown")
                gen_rate = test.get("generated_rate", 0)
                real_rate = test.get("real_rate", 0)
                diff = test.get("difference", 0)
                effect = test.get("effect_size", "unknown")

                report_lines.append(
                    f"| {model} | {gen_rate:.2f}% | {real_rate:.2f}% | {diff:+.2f}pp | {effect} |"
                )

            report_lines.append("")

        # Visualizations
        report_lines.append("### Visualizations")
        report_lines.append("")
        report_lines.append(
            "See `figures/wang_abnormality/{dataset}/` for detailed visualizations."
        )
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Spatial Abnormality Analysis Section
    if lmtad_spatial_results:
        report_lines.append("## Spatial Abnormality Analysis (LM-TAD Method)")
        report_lines.append("")
        report_lines.append(
            "The LM-TAD teacher model identifies spatial abnormalities (route switches and detours) "
            "by evaluating trajectory perplexity. High perplexity indicates spatial deviations from "
            "expected routes."
        )
        report_lines.append("")

        spatial_summary = lmtad_spatial_results.get("summary_statistics", {}).get(
            dataset, {}
        )
        if spatial_summary:
            report_lines.append("### Real Data Statistics")
            report_lines.append("")
            report_lines.append(
                f"- **Total Spatial Abnormal Rate**: {spatial_summary.get('real_spatial_abnormality_rate', 0):.2f}%"
            )
            report_lines.append(
                f"- **Route Switch Rate**: {spatial_summary.get('real_route_switch_rate', 0):.2f}%"
            )
            report_lines.append(
                f"- **Detour Rate**: {spatial_summary.get('real_detour_rate', 0):.2f}%"
            )
            report_lines.append("")

        # Model comparisons
        spatial_tests = (
            lmtad_spatial_results.get("statistical_analysis", {}).get(
                "statistical_tests", []
            )
            if isinstance(
                lmtad_spatial_results.get("statistical_analysis", {}).get(
                    "statistical_tests", []
                ),
                list,
            )
            else []
        )

        if spatial_tests:
            report_lines.append("### Model Performance on Spatial Abnormalities")
            report_lines.append("")
            report_lines.append(
                "| Model | Generated Rate | Real Rate | Difference | Effect Size | Significant |"
            )
            report_lines.append(
                "|-------|---------------|-----------|------------|-------------|-------------|"
            )

            for test in sorted(
                spatial_tests, key=lambda x: abs(x.get("difference", 0))
            ):
                model = test.get("model", "unknown")
                gen_rate = test.get("generated_rate", 0)
                real_rate = test.get("real_rate", 0)
                diff = test.get("difference", 0)
                effect = test.get("effect_size", "unknown")
                significant = "Yes" if test.get("significant", False) else "No"

                report_lines.append(
                    f"| {model} | {gen_rate:.2f}% | {real_rate:.2f}% | {diff:+.2f}pp | {effect} | {significant} |"
                )

            report_lines.append("")

            # Route switch vs detour breakdown
            generated_data = lmtad_spatial_results.get("generated_data", {}).get(
                dataset, {}
            )
            if generated_data:
                report_lines.append("### Route Switch vs Detour Breakdown")
                report_lines.append("")
                report_lines.append(
                    "| Model | Route Switch Rate | Detour Rate | Total Spatial Rate |"
                )
                report_lines.append(
                    "|-------|------------------|-------------|-------------------|"
                )

                for model_name, model_data in generated_data.items():
                    rs_rate = model_data.get("route_switch_rate", 0)
                    det_rate = model_data.get("detour_rate", 0)
                    total = model_data.get("spatial_abnormality_rate", 0)

                    report_lines.append(
                        f"| {model_name} | {rs_rate:.2f}% | {det_rate:.2f}% | {total:.2f}% |"
                    )

                report_lines.append("")

        # Visualizations
        report_lines.append("### Visualizations")
        report_lines.append("")
        report_lines.append(
            f"See `figures/lmtad_spatial_abnormality/{dataset}/` for detailed visualizations."
        )
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Combined Model Rankings
    report_lines.append("## Combined Model Rankings")
    report_lines.append("")
    report_lines.append(
        "Models are ranked by their ability to reproduce both temporal and spatial abnormalities."
    )
    report_lines.append("")

    # Create combined rankings if both results available
    if wang_results and lmtad_spatial_results:
        wang_tests = (
            wang_results.get("statistical_analysis", {})
            .get("statistical_tests", {})
            .get(dataset, [])
        )
        spatial_tests = (
            lmtad_spatial_results.get("statistical_analysis", {}).get(
                "statistical_tests", []
            )
            if isinstance(
                lmtad_spatial_results.get("statistical_analysis", {}).get(
                    "statistical_tests", []
                ),
                list,
            )
            else []
        )

        if wang_tests and spatial_tests:
            # Create model scores (lower deviation = better)
            model_scores = {}
            for test in wang_tests:
                model = test.get("model", "unknown")
                deviation = abs(test.get("difference", 0))
                model_scores[model] = model_scores.get(model, 0) + deviation

            for test in spatial_tests:
                model = test.get("model", "unknown")
                deviation = abs(test.get("difference", 0))
                model_scores[model] = model_scores.get(model, 0) + deviation

            # Sort by combined score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])

            report_lines.append("| Rank | Model | Combined Deviation Score |")
            report_lines.append("|------|-------|-------------------------|")
            for rank, (model, score) in enumerate(sorted_models, 1):
                report_lines.append(f"| {rank} | {model} | {score:.2f} |")
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Key Insights
    report_lines.append("## Key Insights")
    report_lines.append("")
    report_lines.append(
        "1. **Temporal vs Spatial Abnormalities**: Models may perform differently on temporal vs spatial abnormalities."
    )
    report_lines.append(
        "2. **Model Consistency**: Check if models that perform well on temporal abnormalities also perform well on spatial abnormalities."
    )
    report_lines.append(
        "3. **Generation Gaps**: Both temporal and spatial abnormality rates are typically lower in generated trajectories compared to real data."
    )
    report_lines.append(
        "4. **Statistical Significance**: Most model comparisons show statistically significant differences from real data."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Technical Details
    report_lines.append("## Technical Details")
    report_lines.append("")
    report_lines.append("### Wang Temporal Detection")
    report_lines.append("- Method: Statistical baseline comparison")
    report_lines.append(
        "- Detects: Temporal delays, route deviations, combined patterns"
    )
    report_lines.append(
        "- Configuration: See `config/abnormal_detection_statistical.yaml`"
    )
    report_lines.append("")
    report_lines.append("### LM-TAD Spatial Detection")
    report_lines.append(
        "- Method: Perplexity-based analysis (no automatic label inference)"
    )
    report_lines.append(
        "- Detects: Per-trajectory and per-segment perplexity patterns (used as indicators of abnormality)"
    )
    report_lines.append(
        "- Notes: Perplexity is an indicator only — the pipeline does not infer spatial-abnormality types from perplexity. Source labels (route_switch/detour) are retained as metadata when present."
    )
    report_lines.append("")

    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"✅ Combined report saved to {output_file}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Create combined abnormal trajectory analysis report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined report
  uv run python tools/create_combined_abnormal_report.py \\
    --wang-results analysis_abnormal/porto_hoser/wang_results_aggregated.json \\
    --lmtad-spatial-results analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \\
    --output analysis_abnormal/porto_hoser/COMBINED_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md \\
    --dataset porto_hoser

  # Create report with only one result type
  uv run python tools/create_combined_abnormal_report.py \\
    --lmtad-spatial-results analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \\
    --output analysis_abnormal/porto_hoser/LM_TAD_SPATIAL_REPORT.md \\
    --dataset porto_hoser
        """,
    )

    parser.add_argument(
        "--wang-results",
        type=Path,
        help="Path to Wang results JSON file (optional)",
    )
    parser.add_argument(
        "--lmtad-spatial-results",
        type=Path,
        help="Path to LM-TAD spatial results JSON file (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output markdown file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., porto_hoser, Beijing)",
    )

    args = parser.parse_args()

    # Validate that at least one result file is provided
    if not args.wang_results and not args.lmtad_spatial_results:
        logger.error(
            "At least one result file (--wang-results or --lmtad-spatial-results) must be provided"
        )
        return 1

    # Load results
    wang_results = None
    if args.wang_results:
        wang_results = load_wang_results(args.wang_results)

    lmtad_spatial_results = None
    if args.lmtad_spatial_results:
        lmtad_spatial_results = load_lmtad_spatial_results(args.lmtad_spatial_results)

    # Create report
    try:
        create_combined_report(
            wang_results=wang_results,
            lmtad_spatial_results=lmtad_spatial_results,
            output_file=args.output,
            dataset=args.dataset,
        )
        return 0

    except Exception as e:
        logger.error(f"❌ Report creation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
