#!/usr/bin/env python3
"""
Generate all scenario analysis plots.

Usage:
    uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6
    uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots heatmaps_only
    uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots application.improvement_heatmaps
    uv run python create_scenario_plots.py --list-plots
"""

import argparse
import importlib
import logging
from pathlib import Path

from scenario_plots.config_loader import PlotConfigLoader
from scenario_plots.data_loader import ScenarioDataLoader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Plot configuration is now loaded from config/scenario_plots.yaml
# This provides a config-driven, extensible plotting system


def generate_plots(
    data, output_dir, dpi, plot_selections, plot_config_loader, data_loader=None
):
    """Generate selected plots based on configuration

    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        plot_selections: List of plot groups or IDs to generate
        plot_config_loader: PlotConfigLoader instance
        data_loader: Optional ScenarioDataLoader for config-based filtering

    Returns:
        List of generated plot IDs
    """
    # Resolve all selections to individual plots
    plots_to_generate = []
    for selection in plot_selections:
        plots = plot_config_loader.get_enabled_plots(selection)
        plots_to_generate.extend(plots)

    # Remove duplicates while preserving order
    seen = set()
    unique_plots = []
    for plot in plots_to_generate:
        if plot.id not in seen:
            seen.add(plot.id)
            unique_plots.append(plot)

    generated = []

    for plot in unique_plots:
        logger.info(f"  📊 {plot.description} ({plot.id})")

        try:
            # Dynamically import module
            module = importlib.import_module(f"scenario_plots.{plot.module_name}")

            # Get dataset-specific overrides if available
            plot_config = plot.config.copy()
            if data_loader:
                overrides = data_loader.get_plot_override(plot.id)
                plot_config.update(overrides)

            # Call each function for this plot
            for func_name in plot.functions:
                func = getattr(module, func_name)
                # Pass loader and config to function
                func(data, output_dir, dpi, loader=data_loader, config=plot_config)

            generated.append(plot.id)

        except Exception as e:
            logger.error(f"    ❌ Error generating {plot.id}: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    return generated


def print_available_plots(config_loader: PlotConfigLoader):
    """Print all available plots and groups"""
    print("\n" + "=" * 70)
    print("Available Plot Groups")
    print("=" * 70)

    groups = config_loader.list_available_groups()
    for name, desc in sorted(groups.items()):
        print(f"  {name:20} - {desc}")

    print("\n" + "=" * 70)
    print("Available Individual Plots")
    print("=" * 70)

    plots = config_loader.list_available_plots()
    for plot_id, desc in sorted(plots.items()):
        print(f"  {plot_id:40} - {desc}")

    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)
    print("  # Run all plots")
    print("  --plots all")
    print("")
    print("  # Run a plot group")
    print("  --plots heatmaps_only")
    print("")
    print("  # Run a single plot")
    print("  --plots application.improvement_heatmaps")
    print("")
    print("  # Run multiple selections")
    print("  --plots metrics.scenario_heatmap,application.radar_charts")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate scenario analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available plots and groups
  uv run python create_scenario_plots.py --list-plots
  
  # Generate all plots
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6
  
  # Generate only application plots
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots application
  
  # Generate only heatmaps
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots heatmaps_only
  
  # Generate a single specific plot (fast for development)
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \\
    --plots application.improvement_heatmaps
  
  # Generate multiple specific plots
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \\
    --plots metrics.scenario_heatmap,application.radar_charts
  
  # Custom DPI for high-resolution output
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \\
    --plots heatmaps_only --dpi 600
        """,
    )
    parser.add_argument(
        "--eval-dir",
        help="Evaluation directory containing scenarios/",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution (default: 300)"
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help="Comma-separated list of plot groups or individual plots (default: all). "
        "Use --list-plots to see all options.",
    )
    parser.add_argument(
        "--list-plots",
        action="store_true",
        help="List all available plots and groups, then exit",
    )

    args = parser.parse_args()

    # Load plot configuration
    config_dir = Path(__file__).parent / "config"
    plot_config = PlotConfigLoader(config_dir)

    # List mode - show available plots and exit
    if args.list_plots:
        print_available_plots(plot_config)
        return 0

    # Validate required arguments for generation mode
    if not args.eval_dir:
        logger.error("❌ --eval-dir is required (or use --list-plots to see options)")
        return 1

    eval_dir = Path(args.eval_dir).resolve()
    output_dir = eval_dir / "figures" / "scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"📂 Loading scenario data from {eval_dir}")
    data_loader = ScenarioDataLoader(eval_dir)
    data = data_loader.load_all()

    # Parse plot selection
    plot_selections = [p.strip() for p in args.plots.split(",")]

    logger.info("\n📊 Generating scenario visualizations...")

    # Generate plots dynamically
    generated = generate_plots(
        data, output_dir, args.dpi, plot_selections, plot_config, data_loader
    )

    if generated:
        logger.info(f"\n✅ Generated {len(generated)} plot(s):")
        for plot_id in generated:
            logger.info(f"  ✓ {plot_id}")
        logger.info(f"📁 Saved to {output_dir}")
    else:
        logger.warning("\n⚠️  No plots were generated")

    return 0


if __name__ == "__main__":
    exit(main())
