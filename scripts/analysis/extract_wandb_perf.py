#!/usr/bin/env python3
"""
Extract performance metrics from Weights & Biases generation runs.

This tool searches for trajectory generation runs in WandB and extracts
performance metrics (timing, throughput, etc.) that were logged during
generation.

Examples:
    # Search for Porto Phase 1 generation runs
    uv run python scripts/analysis/extract_wandb_perf.py \
      --project hoser-generation \
      --tags porto_hoser generation

    # Search with date filter
    uv run python scripts/analysis/extract_wandb_perf.py \
      --project hoser-generation \
      --created_after 2025-10-26 \
      --created_before 2025-10-28

    # Extract specific run
    uv run python scripts/analysis/extract_wandb_perf.py \
      --run_id abc123xyz \
      --output porto_phase1_perf.json
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import wandb


def extract_perf_metrics_from_run(
    run_id: str,
    entity: str = "matercomus",
    project: str = "hoser-generation",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Extract performance metrics from a single WandB run.

    Args:
        run_id: WandB run ID
        entity: WandB entity/username
        project: WandB project name
        verbose: Print detailed information

    Returns:
        Dictionary of performance metrics if found
    """
    if verbose:
        print(f"🔍 Fetching run: {entity}/{project}/{run_id}")

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        if verbose:
            print("\n📊 Run info:")
            print(f"   Name: {run.name}")
            print(f"   State: {run.state}")
            print(f"   Created: {run.created_at}")
            print(f"   Tags: {', '.join(run.tags)}")

        # Extract performance metrics from summary
        summary = run.summary
        perf_metrics = {}

        # Check for perf/* keys (logged by gene.py)
        perf_keys = [
            "perf/total_time_mean",
            "perf/total_time_std",
            "perf/throughput",
            "perf/forward_time_mean",
            "perf/forward_count_mean",
        ]

        found_metrics = False
        for key in perf_keys:
            if key in summary:
                perf_metrics[key.replace("perf/", "")] = summary[key]
                found_metrics = True

        # Also check config for context
        config = run.config
        if config:
            perf_metrics["config"] = {
                "dataset": config.get("dataset"),
                "seed": config.get("seed"),
                "num_gene": config.get("num_gene"),
                "search_method": config.get("search_method"),
                "beam_width": config.get("beam_width"),
                "cuda": config.get("cuda"),
                "od_source": config.get("od_source"),
            }

        # Add metadata
        perf_metrics["metadata"] = {
            "run_id": run.id,
            "run_name": run.name,
            "created_at": str(run.created_at),
            "state": run.state,
            "tags": run.tags,
        }

        if verbose:
            if found_metrics:
                print("\n✅ Performance metrics found:")
                for key, value in perf_metrics.items():
                    if key not in ["config", "metadata"]:
                        print(f"   {key}: {value}")
            else:
                print(
                    "\n⚠️  No performance metrics found (may be older run without perf logging)"
                )

        return perf_metrics if found_metrics else None

    except Exception as e:
        print(f"❌ Error extracting metrics: {e}")
        return None


def search_generation_runs(
    entity: str = "matercomus",
    project: str = "hoser-generation",
    tags: Optional[List[str]] = None,
    dataset: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Search for trajectory generation runs in WandB.

    Args:
        entity: WandB entity/username
        project: WandB project name
        tags: Filter by tags (e.g., ['generation', 'porto_hoser'])
        dataset: Filter by dataset name
        created_after: Filter by creation date (YYYY-MM-DD)
        created_before: Filter by creation date (YYYY-MM-DD)
        limit: Maximum number of results

    Returns:
        List of runs with extracted performance metrics
    """
    print(f"🔍 Searching generation runs in {entity}/{project}")

    filters = []
    if tags:
        filters.append(f"   - Tags: {', '.join(tags)}")
    if dataset:
        filters.append(f"   - Dataset: {dataset}")
    if created_after:
        filters.append(f"   - Created after: {created_after}")
    if created_before:
        filters.append(f"   - Created before: {created_before}")

    if filters:
        print("Filters:")
        for f in filters:
            print(f)

    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        matching_runs = []
        for run in runs:
            # Apply filters
            if tags:
                if not all(tag in run.tags for tag in tags):
                    continue

            if dataset:
                if run.config.get("dataset") != dataset:
                    continue

            if created_after:
                created_date = datetime.fromisoformat(str(run.created_at).split("T")[0])
                filter_date = datetime.fromisoformat(created_after)
                if created_date < filter_date:
                    continue

            if created_before:
                created_date = datetime.fromisoformat(str(run.created_at).split("T")[0])
                filter_date = datetime.fromisoformat(created_before)
                if created_date > filter_date:
                    continue

            # Extract summary info
            summary = run.summary
            perf_data = {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created": str(run.created_at),
                "tags": run.tags,
                "dataset": run.config.get("dataset"),
                "seed": run.config.get("seed"),
                "num_gene": run.config.get("num_gene"),
                "od_source": run.config.get("od_source"),
                "search_method": run.config.get("search_method"),
                # Performance metrics (if available)
                "total_time_mean": summary.get("perf/total_time_mean"),
                "throughput": summary.get("perf/throughput"),
                "forward_time_mean": summary.get("perf/forward_time_mean"),
            }

            matching_runs.append(perf_data)

            if len(matching_runs) >= limit:
                break

        print(f"\n📊 Found {len(matching_runs)} matching runs:")
        print(
            f"{'ID':<12} {'Dataset':<15} {'OD':<6} {'Seed':<6} {'#Traj':<8} {'Throughput':<12} {'State':<10}"
        )
        print("-" * 90)

        for run_info in matching_runs:
            throughput = run_info.get("throughput")
            throughput_str = f"{throughput:.2f} t/s" if throughput else "N/A"
            print(
                f"{run_info['id']:<12} {run_info['dataset'] or 'N/A':<15} "
                f"{run_info['od_source'] or 'N/A':<6} {run_info['seed'] or 'N/A':<6} "
                f"{run_info['num_gene'] or 'N/A':<8} {throughput_str:<12} {run_info['state']:<10}"
            )

        return matching_runs

    except Exception as e:
        print(f"❌ Error searching runs: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract performance metrics from WandB trajectory generation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common arguments
    parser.add_argument(
        "--entity",
        type=str,
        default="matercomus",
        help="WandB entity/username (default: matercomus)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="hoser-generation",
        help="WandB project name (default: hoser-generation)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    # Extract-specific arguments
    parser.add_argument(
        "--run_id", type=str, help="WandB run ID to extract metrics from"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file for extracted metrics",
    )

    # Search-specific arguments
    parser.add_argument(
        "--search", action="store_true", help="Search for runs instead of extracting"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Filter by tags (e.g., porto_hoser generation)",
    )
    parser.add_argument("--dataset", type=str, help="Filter by dataset name")
    parser.add_argument(
        "--created_after",
        type=str,
        help="Filter by creation date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--created_before",
        type=str,
        help="Filter by creation date YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of search results (default: 50)",
    )
    parser.add_argument(
        "--extract_all",
        action="store_true",
        help="Extract detailed metrics from all matching runs",
    )

    args = parser.parse_args()

    if args.search or (args.tags or args.dataset or args.created_after):
        # Search mode
        matching_runs = search_generation_runs(
            entity=args.entity,
            project=args.project,
            tags=args.tags,
            dataset=args.dataset,
            created_after=args.created_after,
            created_before=args.created_before,
            limit=args.limit,
        )

        if matching_runs and args.extract_all:
            print("\n⬇️  Extracting detailed metrics from all matching runs...")
            detailed_metrics = []
            for run_info in matching_runs:
                metrics = extract_perf_metrics_from_run(
                    run_id=run_info["id"],
                    entity=args.entity,
                    project=args.project,
                    verbose=False,
                )
                if metrics:
                    detailed_metrics.append(metrics)

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(detailed_metrics, f, indent=2)
                print(
                    f"\n✅ Saved {len(detailed_metrics)} metric sets to: {args.output}"
                )
            else:
                print("\n📊 Extracted metrics summary:")
                print(json.dumps(detailed_metrics, indent=2))

        elif matching_runs:
            print("\n💡 To extract detailed metrics, add --extract_all")
            print("💡 To extract from specific run: --run_id <ID>")

    elif args.run_id:
        # Extract mode
        metrics = extract_perf_metrics_from_run(
            run_id=args.run_id,
            entity=args.entity,
            project=args.project,
            verbose=not args.quiet,
        )

        if metrics:
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✅ Metrics saved to: {args.output}")
            else:
                print("\n📊 Extracted metrics:")
                print(json.dumps(metrics, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        parser.print_help()
        print("\n❌ Error: Must specify --run_id, --search, or search filters")
        sys.exit(1)


if __name__ == "__main__":
    main()
