#!/usr/bin/env python3
"""
Aggregate evaluation metrics (overall and per-scenario) from HOSER evaluation results.

Usage:
    uv run python scripts/analysis/aggregate_eval_scenarios.py \
        --root /path/to/eval_bundle \
        --dataset porto_hoser \
        --out /path/to/output
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate HOSER evaluation metrics and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of evaluation bundle",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="porto_hoser",
        help="Dataset identifier (default: porto_hoser)",
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        help="Override path to eval runs (default: <root>/eval)",
    )
    parser.add_argument(
        "--scenarios_dir",
        type=Path,
        help="Override path to scenarios (default: <root>/scenarios)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="auto",
        help="Comma-separated model list or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--od_sources",
        type=str,
        default="train,test",
        help="Comma-separated OD sources (default: train,test)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: <root>/analysis)",
    )
    parser.add_argument(
        "--md",
        type=bool,
        default=True,
        help="Generate Markdown fragments (default: True)",
    )
    parser.add_argument(
        "--fig_prefix",
        type=str,
        default="figures/",
        help="Prefix for figure paths in markdown (default: figures/)",
    )
    return parser.parse_args()


def infer_model_type(metadata: dict[str, Any]) -> str:
    """Infer model type from metadata, handling both Porto and Beijing formats."""
    # Porto format: has explicit model_type
    if "model_type" in metadata:
        return metadata["model_type"]

    # Beijing format: infer from generated_file path
    if "generated_file" in metadata:
        gen_file = metadata["generated_file"]
        # Extract from path like "gene/Beijing/seed42/2025-10-08_18-23-41.csv"
        # or "gene/Beijing/seed42/2025-10-08_18-23-41_distill.csv"
        # Check if filename contains model type indicator
        if "distill" in gen_file.lower():
            # Check which seed from parent directory
            if "seed43" in gen_file:
                return "distill_seed43"
            elif "seed44" in gen_file:
                return "distill_seed44"
            else:
                return "distill"
        elif "vanilla" in gen_file.lower():
            if "seed43" in gen_file:
                return "vanilla_seed43"
            elif "seed44" in gen_file:
                return "vanilla_seed44"
            else:
                return "vanilla"

    return "unknown"


def infer_seed(metadata: dict[str, Any]) -> int:
    """Infer seed from metadata."""
    if "seed" in metadata:
        return metadata["seed"]

    # Infer from generated_file path
    if "generated_file" in metadata:
        gen_file = metadata["generated_file"]
        if "seed42" in gen_file or "/seed42/" in gen_file:
            return 42
        elif "seed43" in gen_file or "/seed43/" in gen_file:
            return 43
        elif "seed44" in gen_file or "/seed44/" in gen_file:
            return 44

    return 42  # default


def load_eval_results(eval_dir: Path) -> list[dict[str, Any]]:
    """Load all evaluation results from eval/<timestamp>/results.json files."""
    results = []
    results_files = sorted(eval_dir.glob("*/results.json"))

    # For Beijing format (no explicit model_type), infer from sequence
    # Pattern: vanilla_train, vanilla_test, distilled_train, distilled_test, distilled_seed44_train, distilled_seed44_test
    beijing_model_sequence = [
        "vanilla",
        "vanilla",
        "distilled",
        "distilled",
        "distilled_seed44",
        "distilled_seed44",
    ]

    for idx, results_file in enumerate(results_files):
        try:
            with open(results_file) as f:
                data = json.load(f)
                # Normalize metadata for both Porto and Beijing formats
                if "metadata" in data:
                    if "model_type" not in data["metadata"]:
                        # Try to infer from filename first
                        inferred = infer_model_type(data["metadata"])
                        if inferred == "unknown" and idx < len(beijing_model_sequence):
                            # Fall back to Beijing sequence pattern
                            inferred = beijing_model_sequence[idx]
                        data["metadata"]["model_type"] = inferred
                    if "seed" not in data["metadata"]:
                        data["metadata"]["seed"] = infer_seed(data["metadata"])
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {results_file}: {e}", file=sys.stderr)
    return results


def load_scenario_metrics(
    scenarios_dir: Path, od_source: str, model: str
) -> dict[str, Any] | None:
    """Load scenario metrics for a given OD source and model."""
    scenario_file = scenarios_dir / od_source / model / "scenario_metrics.json"
    if not scenario_file.exists():
        return None
    try:
        with open(scenario_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {scenario_file}: {e}", file=sys.stderr)
        return None


def map_model_to_group(model: str) -> str:
    """Map model name to group (distilled or vanilla)."""
    if model.startswith("distill"):  # Handles both "distill" and "distilled"
        return "distilled"
    elif model.startswith("vanilla"):
        return "vanilla"
    else:
        return "unknown"


def compute_cv(values: list[float]) -> float:
    """Compute coefficient of variation (CV%) from a list of values."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance**0.5
    return 100.0 * std / mean


def aggregate_by_group(
    results: list[dict[str, Any]], od_source: str
) -> dict[str, dict[str, Any]]:
    """Aggregate results by model group (distilled/vanilla) for a given OD source."""
    # Group results by model type
    grouped = defaultdict(list)
    for result in results:
        if result["metadata"]["od_source"] != od_source:
            continue
        model = result["metadata"]["model_type"]
        group = map_model_to_group(model)
        grouped[group].append(result)

    # Compute aggregates
    aggregates = {}
    for group, group_results in grouped.items():
        metrics = {
            "match_rate": [],
            "Distance_JSD": [],
            "Radius_JSD": [],
            "Duration_JSD": [],
            "Distance_gen_mean": [],
            "Hausdorff_km": [],
            "DTW_km": [],
            "EDR": [],
        }

        for res in group_results:
            match_rate = (
                res["matched_od_pairs"] / res["total_generated_od_pairs"]
                if res["total_generated_od_pairs"] > 0
                else 0.0
            )
            metrics["match_rate"].append(match_rate * 100)
            for key in [
                "Distance_JSD",
                "Radius_JSD",
                "Duration_JSD",
                "Distance_gen_mean",
                "Hausdorff_km",
                "DTW_km",
                "EDR",
            ]:
                metrics[key].append(res[key])

        # Compute mean, min, max, std, CV%
        agg = {}
        for metric, values in metrics.items():
            agg[f"{metric}_mean"] = sum(values) / len(values) if values else 0.0
            agg[f"{metric}_min"] = min(values) if values else 0.0
            agg[f"{metric}_max"] = max(values) if values else 0.0
            agg[f"{metric}_std"] = (
                (sum((x - agg[f"{metric}_mean"]) ** 2 for x in values) / len(values))
                ** 0.5
                if values
                else 0.0
            )
            agg[f"{metric}_cv"] = compute_cv(values)

        aggregates[group] = agg

    return aggregates


def aggregate_scenarios_by_group(
    scenarios_dir: Path, od_source: str, models: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate scenario metrics by model group for a given OD source."""
    # Load all scenario metrics
    scenario_data = {}
    for model in models:
        data = load_scenario_metrics(scenarios_dir, od_source, model)
        if data and "individual_scenarios" in data:
            scenario_data[model] = data["individual_scenarios"]

    # Group by model type
    grouped_scenarios = defaultdict(lambda: defaultdict(list))
    for model, scenarios in scenario_data.items():
        group = map_model_to_group(model)
        for scenario_name, scenario_info in scenarios.items():
            # Extract metrics from nested structure
            if "metrics" in scenario_info:
                scenario_metrics = scenario_info["metrics"]
                # Add match_rate calculation
                if (
                    "matched_od_pairs" in scenario_metrics
                    and "total_generated_od_pairs" in scenario_metrics
                ):
                    scenario_metrics["match_rate"] = (
                        scenario_metrics["matched_od_pairs"]
                        / scenario_metrics["total_generated_od_pairs"]
                        if scenario_metrics["total_generated_od_pairs"] > 0
                        else 0.0
                    )
                grouped_scenarios[scenario_name][group].append(scenario_metrics)

    # Compute aggregates per scenario
    scenario_aggregates = {}
    for scenario_name, group_data in grouped_scenarios.items():
        scenario_agg = {}
        for group, metrics_list in group_data.items():
            metrics = {
                "match_rate": [],
                "Distance_JSD": [],
                "Radius_JSD": [],
                "Duration_JSD": [],
                "Distance_gen_mean": [],
                "Hausdorff_km": [],
                "DTW_km": [],
                "EDR": [],
            }

            for m in metrics_list:
                if "match_rate" in m:
                    metrics["match_rate"].append(m["match_rate"] * 100)
                for key in [
                    "Distance_JSD",
                    "Radius_JSD",
                    "Duration_JSD",
                    "Distance_gen_mean",
                    "Hausdorff_km",
                    "DTW_km",
                    "EDR",
                ]:
                    if key in m:
                        metrics[key].append(m[key])

            # Compute aggregates
            agg = {}
            for metric, values in metrics.items():
                if values:
                    agg[f"{metric}_mean"] = sum(values) / len(values)
                    agg[f"{metric}_cv"] = compute_cv(values)
                else:
                    agg[f"{metric}_mean"] = None
                    agg[f"{metric}_cv"] = None

            scenario_agg[group] = agg

        scenario_aggregates[scenario_name] = scenario_agg

    return scenario_aggregates


def compute_deltas(
    scenario_aggregates: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Compute deltas (distilled - vanilla) for scenario metrics."""
    deltas = {}
    for scenario_name, groups in scenario_aggregates.items():
        if "distilled" in groups and "vanilla" in groups:
            delta = {}
            dist = groups["distilled"]
            van = groups["vanilla"]
            for metric in [
                "match_rate",
                "Distance_JSD",
                "Radius_JSD",
                "Duration_JSD",
                "Distance_gen_mean",
                "DTW_km",
                "EDR",
            ]:
                dist_val = dist.get(f"{metric}_mean")
                van_val = van.get(f"{metric}_mean")
                if dist_val is not None and van_val is not None:
                    delta[f"delta_{metric}"] = dist_val - van_val
                else:
                    delta[f"delta_{metric}"] = None
            deltas[scenario_name] = delta
    return deltas


def get_top_scenarios(
    deltas: dict[str, dict[str, float]], metric: str, n: int = 5
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Get top N scenarios with largest positive and negative deltas for a metric."""
    delta_key = f"delta_{metric}"
    valid_scenarios = [
        (s, d[delta_key]) for s, d in deltas.items() if d.get(delta_key) is not None
    ]
    valid_scenarios.sort(key=lambda x: x[1], reverse=True)

    top_positive = valid_scenarios[:n]
    top_negative = valid_scenarios[-n:][::-1]

    return top_positive, top_negative


def save_results_overview(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Save 12-row results overview as CSV."""
    rows = []
    for res in results:
        match_rate = (
            res["matched_od_pairs"] / res["total_generated_od_pairs"] * 100
            if res["total_generated_od_pairs"] > 0
            else 0.0
        )
        rows.append(
            {
                "model": res["metadata"]["model_type"],
                "od_source": res["metadata"]["od_source"],
                "matched_od": res["matched_od_pairs"],
                "total_generated": res["total_generated_od_pairs"],
                "match_rate": match_rate,
                "Distance_JSD": res["Distance_JSD"],
                "Radius_JSD": res["Radius_JSD"],
                "Duration_JSD": res["Duration_JSD"],
                "Distance_gen_mean": res["Distance_gen_mean"],
                "Hausdorff_km": res["Hausdorff_km"],
                "DTW_km": res["DTW_km"],
                "EDR": res["EDR"],
            }
        )

    df = pl.DataFrame(rows)
    df = df.sort(["od_source", "model"])
    df.write_csv(out_dir / "results_overview.csv")
    print(f"✓ Saved results_overview.csv ({len(rows)} rows)")


def save_aggregates(
    aggregates: dict[str, dict[str, Any]], out_dir: Path, od_source: str
) -> None:
    """Save aggregated metrics as CSV."""
    rows = []
    for group, metrics in aggregates.items():
        row = {"group": group, "od_source": od_source}
        row.update(metrics)
        rows.append(row)

    df = pl.DataFrame(rows)
    out_file = out_dir / f"aggregates_{od_source}.csv"
    df.write_csv(out_file)
    print(f"✓ Saved aggregates_{od_source}.csv ({len(rows)} groups)")


def save_scenario_aggregates(
    scenario_aggregates: dict[str, dict[str, dict[str, Any]]],
    deltas: dict[str, dict[str, float]],
    out_dir: Path,
    od_source: str,
) -> None:
    """Save per-scenario aggregates as CSV."""
    rows = []
    for scenario_name in sorted(scenario_aggregates.keys()):
        groups = scenario_aggregates[scenario_name]
        delta = deltas.get(scenario_name, {})

        row = {"scenario": scenario_name, "od_source": od_source}

        # Distilled metrics
        if "distilled" in groups:
            for metric in [
                "match_rate",
                "Distance_JSD",
                "Radius_JSD",
                "Duration_JSD",
                "Distance_gen_mean",
                "DTW_km",
                "EDR",
            ]:
                row[f"distilled_{metric}"] = groups["distilled"].get(f"{metric}_mean")
                row[f"distilled_{metric}_cv"] = groups["distilled"].get(f"{metric}_cv")

        # Vanilla metrics
        if "vanilla" in groups:
            for metric in [
                "match_rate",
                "Distance_JSD",
                "Radius_JSD",
                "Duration_JSD",
                "Distance_gen_mean",
                "DTW_km",
                "EDR",
            ]:
                row[f"vanilla_{metric}"] = groups["vanilla"].get(f"{metric}_mean")
                row[f"vanilla_{metric}_cv"] = groups["vanilla"].get(f"{metric}_cv")

        # Deltas
        for key, value in delta.items():
            row[key] = value

        rows.append(row)

    df = pl.DataFrame(rows)
    out_file = out_dir / f"scenarios_{od_source}.csv"
    df.write_csv(out_file)
    print(f"✓ Saved scenarios_{od_source}.csv ({len(rows)} scenarios)")


def save_top_scenarios(
    deltas: dict[str, dict[str, float]], out_dir: Path, od_source: str
) -> None:
    """Save top scenarios with largest deltas."""
    metrics = ["Distance_JSD", "match_rate", "Radius_JSD"]
    rows = []

    for metric in metrics:
        top_pos, top_neg = get_top_scenarios(deltas, metric, n=5)

        for scenario, delta_val in top_pos:
            rows.append(
                {
                    "od_source": od_source,
                    "metric": metric,
                    "direction": "distilled_better",
                    "scenario": scenario,
                    "delta": delta_val,
                }
            )

        for scenario, delta_val in top_neg:
            rows.append(
                {
                    "od_source": od_source,
                    "metric": metric,
                    "direction": "vanilla_better",
                    "scenario": scenario,
                    "delta": delta_val,
                }
            )

    df = pl.DataFrame(rows)
    out_file = out_dir / f"top_scenarios_{od_source}.csv"
    df.write_csv(out_file)
    print(f"✓ Saved top_scenarios_{od_source}.csv ({len(rows)} entries)")


def generate_markdown_results_table(
    results: list[dict[str, Any]],
    aggregates_train: dict[str, dict[str, Any]],
    aggregates_test: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    """Generate Markdown fragment for results table."""
    md = []

    # Real data baseline
    real_train = next(
        (r for r in results if r["metadata"]["od_source"] == "train"), None
    )
    real_test = next((r for r in results if r["metadata"]["od_source"] == "test"), None)

    if real_train and real_test:
        md.append("**Real Data Baseline:**\n")
        md.append(
            "| OD Source | Distance (km) | Duration (hours) | Radius of Gyration |"
        )
        md.append(
            "|-----------|---------------|------------------|-------------------|"
        )
        md.append(
            f"| Train | {real_train['Distance_real_mean']:.3f} | "
            f"{real_train['Duration_real_mean']:.3f} | {real_train['Radius_real_mean']:.3f} |"
        )
        md.append(
            f"| Test | {real_test['Distance_real_mean']:.3f} | "
            f"{real_test['Duration_real_mean']:.3f} | {real_test['Radius_real_mean']:.3f} |"
        )
        md.append("")

    # Aggregated comparison
    md.append("**Aggregated Comparison (by Model Type):**\n")
    md.append("**Train Set:**\n")
    md.append(
        "| Group | Match Rate | Distance JSD | Radius JSD | Distance (km) | DTW (km) | EDR |"
    )
    md.append(
        "|-------|------------|--------------|------------|---------------|----------|-----|"
    )

    for group in ["distilled", "vanilla"]:
        if group in aggregates_train:
            agg = aggregates_train[group]
            md.append(
                f"| **{group.capitalize()}** | "
                f"{agg['match_rate_mean']:.1f}% ± {agg['match_rate_std']:.1f}% | "
                f"{agg['Distance_JSD_mean']:.4f} ± {agg['Distance_JSD_std']:.4f} | "
                f"{agg['Radius_JSD_mean']:.4f} ± {agg['Radius_JSD_std']:.4f} | "
                f"{agg['Distance_gen_mean_mean']:.3f} ± {agg['Distance_gen_mean_std']:.3f} | "
                f"{agg['DTW_km_mean']:.2f} ± {agg['DTW_km_std']:.2f} | "
                f"{agg['EDR_mean']:.3f} ± {agg['EDR_std']:.3f} |"
            )

    md.append("\n**Test Set:**\n")
    md.append(
        "| Group | Match Rate | Distance JSD | Radius JSD | Distance (km) | DTW (km) | EDR |"
    )
    md.append(
        "|-------|------------|--------------|------------|---------------|----------|-----|"
    )

    for group in ["distilled", "vanilla"]:
        if group in aggregates_test:
            agg = aggregates_test[group]
            md.append(
                f"| **{group.capitalize()}** | "
                f"{agg['match_rate_mean']:.1f}% ± {agg['match_rate_std']:.1f}% | "
                f"{agg['Distance_JSD_mean']:.4f} ± {agg['Distance_JSD_std']:.4f} | "
                f"{agg['Radius_JSD_mean']:.4f} ± {agg['Radius_JSD_std']:.4f} | "
                f"{agg['Distance_gen_mean_mean']:.3f} ± {agg['Distance_gen_mean_std']:.3f} | "
                f"{agg['DTW_km_mean']:.2f} ± {agg['DTW_km_std']:.2f} | "
                f"{agg['EDR_mean']:.3f} ± {agg['EDR_std']:.3f} |"
            )

    with open(out_dir / "md" / "results_table.md", "w") as f:
        f.write("\n".join(md))
    print("✓ Saved md/results_table.md")


def generate_markdown_scenario_analysis(
    scenario_aggregates_train: dict[str, dict[str, dict[str, Any]]],
    scenario_aggregates_test: dict[str, dict[str, dict[str, Any]]],
    deltas_train: dict[str, dict[str, float]],
    deltas_test: dict[str, dict[str, float]],
    out_dir: Path,
) -> None:
    """Generate Markdown fragment for scenario analysis."""
    md = []

    md.append("### 4.5.2 Per-Scenario Performance Comparison\n")

    # Train scenarios (top 10 by coverage or importance)
    md.append("**Train Set Scenarios:**\n")
    md.append(
        "| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | "
        "Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |"
    )
    md.append(
        "|----------|------------------|----------------|----------|"
        "--------------------|------------------|------------|"
    )

    for scenario in sorted(list(scenario_aggregates_train.keys())[:15]):
        groups = scenario_aggregates_train[scenario]
        delta = deltas_train.get(scenario, {})

        dist_match = groups.get("distilled", {}).get("match_rate_mean")
        van_match = groups.get("vanilla", {}).get("match_rate_mean")
        delta_match = delta.get("delta_match_rate")

        dist_jsd = groups.get("distilled", {}).get("Distance_JSD_mean")
        van_jsd = groups.get("vanilla", {}).get("Distance_JSD_mean")
        delta_jsd = delta.get("delta_Distance_JSD")

        # Skip if missing critical data
        if (
            dist_match is None
            or van_match is None
            or dist_jsd is None
            or van_jsd is None
        ):
            continue

        delta_match_str = f"{delta_match:+.1f}%" if delta_match is not None else "N/A"
        delta_jsd_str = f"{delta_jsd:+.4f}" if delta_jsd is not None else "N/A"

        md.append(
            f"| `{scenario}` | "
            f"{dist_match:.1f}% | "
            f"{van_match:.1f}% | "
            f"{delta_match_str} | "
            f"{dist_jsd:.4f} | "
            f"{van_jsd:.4f} | "
            f"{delta_jsd_str} |"
        )

    md.append("\n**Test Set Scenarios:**\n")
    md.append(
        "| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | "
        "Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |"
    )
    md.append(
        "|----------|------------------|----------------|----------|"
        "--------------------|------------------|------------|"
    )

    for scenario in sorted(list(scenario_aggregates_test.keys())[:15]):
        groups = scenario_aggregates_test[scenario]
        delta = deltas_test.get(scenario, {})

        dist_match = groups.get("distilled", {}).get("match_rate_mean")
        van_match = groups.get("vanilla", {}).get("match_rate_mean")
        delta_match = delta.get("delta_match_rate")

        dist_jsd = groups.get("distilled", {}).get("Distance_JSD_mean")
        van_jsd = groups.get("vanilla", {}).get("Distance_JSD_mean")
        delta_jsd = delta.get("delta_Distance_JSD")

        # Skip if missing critical data
        if (
            dist_match is None
            or van_match is None
            or dist_jsd is None
            or van_jsd is None
        ):
            continue

        delta_match_str = f"{delta_match:+.1f}%" if delta_match is not None else "N/A"
        delta_jsd_str = f"{delta_jsd:+.4f}" if delta_jsd is not None else "N/A"

        md.append(
            f"| `{scenario}` | "
            f"{dist_match:.1f}% | "
            f"{van_match:.1f}% | "
            f"{delta_match_str} | "
            f"{dist_jsd:.4f} | "
            f"{van_jsd:.4f} | "
            f"{delta_jsd_str} |"
        )

    # Top scenarios
    md.append("\n### 4.5.3 Notable Scenarios\n")
    md.append("**Top-5 Scenarios Where Distilled Outperforms (Distance JSD, Test):**\n")

    top_pos, _ = get_top_scenarios(deltas_test, "Distance_JSD", n=5)
    for i, (scenario, delta) in enumerate(top_pos, 1):
        md.append(f"{i}. `{scenario}`: Δ = {delta:+.4f} (distilled better)")

    md.append("\n**Top-5 Scenarios Where Vanilla Outperforms (Distance JSD, Test):**\n")
    _, top_neg = get_top_scenarios(deltas_test, "Distance_JSD", n=5)
    for i, (scenario, delta) in enumerate(top_neg, 1):
        md.append(f"{i}. `{scenario}`: Δ = {delta:+.4f} (vanilla better)")

    with open(out_dir / "md" / "scenario_analysis.md", "w") as f:
        f.write("\n".join(md))
    print("✓ Saved md/scenario_analysis.md")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup paths
    root = args.root
    eval_dir = args.eval_dir or root / "eval"
    scenarios_dir = args.scenarios_dir or root / "scenarios"
    out_dir = args.out or root / "analysis"

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.md:
        (out_dir / "md").mkdir(exist_ok=True)

    print(f"Aggregating evaluation from: {root}")
    print(f"  Eval dir: {eval_dir}")
    print(f"  Scenarios dir: {scenarios_dir}")
    print(f"  Output dir: {out_dir}\n")

    # Load eval results
    results = load_eval_results(eval_dir)
    if not results:
        print("Error: No evaluation results found!", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} evaluation runs")

    # Detect models
    models = sorted({r["metadata"]["model_type"] for r in results})
    print(f"Models: {', '.join(models)}\n")

    # Save results overview
    save_results_overview(results, out_dir)

    # Process each OD source
    od_sources = args.od_sources.split(",")

    all_aggregates = {}
    all_scenario_aggregates = {}
    all_deltas = {}

    for od_source in od_sources:
        print(f"\nProcessing {od_source} set...")

        # Overall aggregates
        aggregates = aggregate_by_group(results, od_source)
        all_aggregates[od_source] = aggregates
        save_aggregates(aggregates, out_dir, od_source)

        # Scenario aggregates
        scenario_aggregates = aggregate_scenarios_by_group(
            scenarios_dir, od_source, models
        )
        all_scenario_aggregates[od_source] = scenario_aggregates
        print(f"  Found {len(scenario_aggregates)} scenarios")

        # Compute deltas
        deltas = compute_deltas(scenario_aggregates)
        all_deltas[od_source] = deltas

        # Save scenario data
        save_scenario_aggregates(scenario_aggregates, deltas, out_dir, od_source)
        save_top_scenarios(deltas, out_dir, od_source)

    # Save consolidated JSON
    consolidated = {
        "aggregates": all_aggregates,
        "scenario_aggregates": all_scenario_aggregates,
        "deltas": all_deltas,
    }
    with open(out_dir / "aggregates.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    print("\n✓ Saved aggregates.json")

    # Generate Markdown fragments
    if args.md:
        print("\nGenerating Markdown fragments...")
        generate_markdown_results_table(
            results,
            all_aggregates.get("train", {}),
            all_aggregates.get("test", {}),
            out_dir,
        )
        generate_markdown_scenario_analysis(
            all_scenario_aggregates.get("train", {}),
            all_scenario_aggregates.get("test", {}),
            all_deltas.get("train", {}),
            all_deltas.get("test", {}),
            out_dir,
        )

    print("\n✅ Aggregation complete!")
    print(f"\nOutputs in: {out_dir}")
    print("  - CSV files: results_overview.csv, aggregates_*.csv, scenarios_*.csv")
    print("  - JSON: aggregates.json")
    if args.md:
        print("  - Markdown: md/results_table.md, md/scenario_analysis.md")


if __name__ == "__main__":
    main()
