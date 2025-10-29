#!/usr/bin/env python3
"""
Generate all scenario analysis plots.

Usage:
    uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6
"""

import argparse
import logging
from pathlib import Path

from scenario_plots.data_loader import ScenarioDataLoader
from scenario_plots import metrics_plots
from scenario_plots import temporal_spatial_plots
from scenario_plots import robustness_plots
from scenario_plots import analysis_plots
from scenario_plots import application_plots

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Plot configuration: module, function, display name, count
PLOT_REGISTRY = {
    "metrics": {
        "module": metrics_plots,
        "function": "plot_all",
        "name": "Metrics plots",
        "count": 4,
        "ids": "#1, #2, #3, #8",
    },
    "temporal": {
        "module": temporal_spatial_plots,
        "function": "plot_all",
        "name": "Temporal/Spatial plots",
        "count": 2,
        "ids": "#4, #5",
    },
    "robustness": {
        "module": robustness_plots,
        "function": "plot_all",
        "name": "Robustness plots",
        "count": 2,
        "ids": "#6, #7",
    },
    "analysis": {
        "module": analysis_plots,
        "function": "plot_all",
        "name": "Analysis plots",
        "count": 3,
        "ids": "#9, #10, #11",
    },
    "application": {
        "module": application_plots,
        "function": "plot_all",
        "name": "Application plots",
        "count": 3,
        "ids": "#12, #13a, #13b",
        "description": "radar + heatmaps",
    },
    # Granular application sub-options
    "heatmaps": {
        "module": application_plots,
        "functions": [
            "plot_improvement_heatmaps_individual",
            "plot_improvement_heatmap_grid",
        ],
        "name": "Improvement heatmaps",
        "count": 2,
        "ids": "#13a, #13b",
        "parent": "application",
    },
    "radar": {
        "module": application_plots,
        "function": "plot_application_radars",
        "name": "Application radar charts",
        "count": 1,
        "ids": "#12",
        "parent": "application",
    },
}


def generate_plots(data, output_dir, dpi, plot_groups, loader=None):
    """Generate selected plot groups dynamically

    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        plot_groups: List of plot group names to generate
        loader: Optional ScenarioDataLoader for config-based filtering
    """

    generate_all = "all" in plot_groups
    generated = []

    for plot_key, plot_config in PLOT_REGISTRY.items():
        # Skip sub-options if parent is being generated
        if "parent" in plot_config and plot_config["parent"] in plot_groups:
            continue

        # Check if this plot group should be generated
        if not (generate_all or plot_key in plot_groups):
            continue

        logger.info(
            f"  {plot_config['name']} ({plot_config['count']} plots: {plot_config['ids']})..."
        )

        try:
            module = plot_config["module"]

            # Handle multiple functions (e.g., heatmaps)
            if "functions" in plot_config:
                for func_name in plot_config["functions"]:
                    func = getattr(module, func_name)
                    func(data, output_dir, dpi, loader=loader)
            else:
                # Single function
                func_name = plot_config["function"]
                func = getattr(module, func_name)
                func(data, output_dir, dpi, loader=loader)

            generated.append(plot_key)

        except Exception as e:
            logger.error(f"    ❌ Error generating {plot_key}: {e}")

    return generated


def main():
    # Build available options dynamically from registry
    available_options = ", ".join(["all"] + list(PLOT_REGISTRY.keys()))

    parser = argparse.ArgumentParser(
        description="Generate scenario analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Generate all plots
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6
  
  # Generate only application plots (radar + heatmaps)
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots application
  
  # Generate only heatmaps (skip radar charts)
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots heatmaps
  
  # Generate multiple specific plot groups
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots metrics,heatmaps
  
  # Custom DPI for high-resolution output
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots heatmaps --dpi 600

Available plot groups: {available_options}
        """,
    )
    parser.add_argument(
        "--eval-dir", required=True, help="Evaluation directory containing scenarios/"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution (default: 300)"
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help=f"Comma-separated list of plot groups to generate (default: all). "
        f"Available: {available_options}",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    output_dir = eval_dir / "figures" / "scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"📂 Loading scenario data from {eval_dir}")
    loader = ScenarioDataLoader(eval_dir)
    data = loader.load_all()

    # Parse plot selection
    plot_groups = [p.strip().lower() for p in args.plots.split(",")]

    # Validate plot groups
    invalid = [p for p in plot_groups if p not in ["all"] + list(PLOT_REGISTRY.keys())]
    if invalid:
        logger.error(f"❌ Invalid plot group(s): {', '.join(invalid)}")
        logger.info(f"Available options: {available_options}")
        return 1

    logger.info("\n📊 Generating scenario visualizations...")

    # Generate plots dynamically
    generated = generate_plots(data, output_dir, args.dpi, plot_groups, loader=loader)

    if generated:
        logger.info(
            f"\n✅ Generated {len(generated)} plot group(s): {', '.join(generated)}"
        )
        logger.info(f"📁 Saved to {output_dir}")
    else:
        logger.warning("\n⚠️  No plots were generated")

    return 0


if __name__ == "__main__":
    exit(main())
