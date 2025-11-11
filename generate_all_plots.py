#!/usr/bin/env python3
"""
Unified plotting orchestrator for HOSER evaluation results.

This script imports and calls existing plotting modules directly:
1. Aggregate analysis CSVs
2. Distribution plots (distance, radius, duration)
3. Trajectory visualizations (optional, slow)
4. Analysis figures (heatmaps, comparisons)
5. Scenario plots (if available)

Usage:
    # Run all (skip trajectories by default)
    uv run python generate_all_plots.py --eval-dir hoser-distill-optuna-6

    # Include trajectory visualizations (~30-60 min)
    uv run python generate_all_plots.py --eval-dir eval_dir --include-trajectories

    # Skip specific categories
    uv run python generate_all_plots.py --eval-dir eval_dir --skip-distributions
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class PlottingOrchestrator:
    """Orchestrate all plotting tasks by calling existing modules directly"""

    def __init__(
        self,
        eval_dir: Path,
        dataset: Optional[str] = None,
        skip_trajectories: bool = True,
        skip_distributions: bool = False,
        skip_analysis: bool = False,
        skip_scenarios: bool = False,
    ):
        self.eval_dir = Path(eval_dir).resolve()
        self.dataset = dataset or self._auto_detect_dataset()
        self.skip_trajectories = skip_trajectories
        self.skip_distributions = skip_distributions
        self.skip_analysis = skip_analysis
        self.skip_scenarios = skip_scenarios

        # Verify eval directory exists
        if not self.eval_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {self.eval_dir}")

        # Track completion
        self.completed_steps = []
        self.failed_steps = []
        self.skipped_steps = []

    def _auto_detect_dataset(self) -> str:
        """Auto-detect dataset from evaluation.yaml"""
        config_path = self.eval_dir / "config" / "evaluation.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("dataset", "Beijing")
            except Exception as e:
                logger.warning(f"Failed to load evaluation.yaml: {e}")
        return "Beijing"

    def _run_step(self, name: str, func: Callable) -> bool:
        """Run a plotting step with error handling and timing"""
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Step: {name}")
        logger.info(f"{'=' * 70}")

        try:
            start = time.time()
            func()
            elapsed = time.time() - start
            logger.info(f"✅ {name} completed in {elapsed:.1f}s")
            self.completed_steps.append(name)
            return True
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            logger.debug("Full traceback:", exc_info=True)
            self.failed_steps.append(name)
            return False

    def _run_aggregate_analysis(self):
        """Step 1: Aggregate analysis CSVs"""
        import sys
        from scripts.analysis import aggregate_eval_scenarios

        # Mock sys.argv to call main() function
        old_argv = sys.argv
        try:
            sys.argv = ["aggregate_eval_scenarios.py", "--root", str(self.eval_dir)]
            aggregate_eval_scenarios.main()
        finally:
            sys.argv = old_argv

    def _run_distribution_plots(self):
        """Step 2: Distribution plots"""
        from create_distribution_plots import DistributionPlotter

        plotter = DistributionPlotter(eval_dir=str(self.eval_dir), dataset=self.dataset)
        plotter.generate_all_plots()

    def _run_trajectory_plots(self):
        """Step 3: Trajectory visualizations"""
        from visualize_trajectories import TrajectoryVisualizer, VisualizationConfig

        config = VisualizationConfig(
            eval_dir=self.eval_dir,
            dataset=self.dataset,
            # Skip per-model plots (separate and overlaid)
            generate_separate=False,
            generate_overlaid=False,
            # Generate cross-model comparisons
            generate_cross_model=True,
            # Generate scenario-based cross-model comparisons (important!)
            generate_scenario_cross_model=True,
        )
        visualizer = TrajectoryVisualizer(config)
        visualizer.run()

    def _run_analysis_figures(self):
        """Step 4: Analysis figures"""
        from create_analysis_figures import EvaluationVisualizer

        visualizer = EvaluationVisualizer(
            eval_dir=str(self.eval_dir), dataset=self.dataset
        )
        visualizer.create_all_figures()

    def _run_scenario_plots(self):
        """Step 5: Scenario plots (if available)"""
        scenarios_dir = self.eval_dir / "scenarios"
        if not scenarios_dir.exists() or not any(scenarios_dir.iterdir()):
            logger.info("ℹ️  No scenario analysis found, skipping scenario plots")
            return

        from create_scenario_plots import generate_plots
        from scenario_plots.config_loader import PlotConfigLoader
        from scenario_plots.data_loader import ScenarioDataLoader

        config_dir = Path(__file__).parent / "config"
        plot_config = PlotConfigLoader(config_dir)
        data_loader = ScenarioDataLoader(self.eval_dir)
        data = data_loader.load_all()

        output_dir = self.eval_dir / "figures" / "scenarios"
        output_dir.mkdir(parents=True, exist_ok=True)

        generate_plots(data, output_dir, 300, ["all"], plot_config, data_loader)

    def run_all(self) -> int:
        """Run all enabled plotting tasks"""
        logger.info("\n" + "=" * 70)
        logger.info("HOSER Evaluation: Unified Plotting Pipeline")
        logger.info("=" * 70)
        logger.info(f"Evaluation Directory: {self.eval_dir}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Include Trajectories: {not self.skip_trajectories}")
        logger.info("=" * 70)

        total_start = time.time()

        # Step 1: Aggregate Analysis
        self._run_step("Aggregate Analysis CSVs", self._run_aggregate_analysis)

        # Step 2: Distribution Plots
        if self.skip_distributions:
            logger.info(f"\n{'=' * 70}")
            logger.info("Step: Distribution Plots (SKIPPED)")
            logger.info("=" * 70)
            self.skipped_steps.append("Distribution Plots")
        else:
            self._run_step("Distribution Plots", self._run_distribution_plots)

        # Step 3: Trajectory Visualizations (optional)
        if self.skip_trajectories:
            logger.info(f"\n{'=' * 70}")
            logger.info("Step: Trajectory Visualizations (SKIPPED)")
            logger.info("=" * 70)
            logger.info("ℹ️  Use --include-trajectories to generate trajectory plots")
            self.skipped_steps.append("Trajectory Visualizations")
        else:
            self._run_step("Trajectory Visualizations", self._run_trajectory_plots)

        # Step 4: Analysis Figures
        if self.skip_analysis:
            logger.info(f"\n{'=' * 70}")
            logger.info("Step: Analysis Figures (SKIPPED)")
            logger.info("=" * 70)
            self.skipped_steps.append("Analysis Figures")
        else:
            self._run_step("Analysis Figures", self._run_analysis_figures)

        # Step 5: Scenario Plots
        if self.skip_scenarios:
            logger.info(f"\n{'=' * 70}")
            logger.info("Step: Scenario Plots (SKIPPED)")
            logger.info("=" * 70)
            self.skipped_steps.append("Scenario Plots")
        else:
            self._run_step("Scenario Plots", self._run_scenario_plots)

        # Summary
        total_elapsed = time.time() - total_start
        self._print_summary(total_elapsed)

        return 1 if self.failed_steps else 0

    def _print_summary(self, total_elapsed: float):
        """Print execution summary"""
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline Summary")
        logger.info("=" * 70)
        logger.info(f"Total Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
        logger.info(
            f"Completed: {len(self.completed_steps)} | "
            f"Failed: {len(self.failed_steps)} | "
            f"Skipped: {len(self.skipped_steps)}"
        )

        if self.completed_steps:
            logger.info("\n✅ Completed Steps:")
            for step in self.completed_steps:
                logger.info(f"  ✓ {step}")

        if self.skipped_steps:
            logger.info("\nℹ️  Skipped Steps:")
            for step in self.skipped_steps:
                logger.info(f"  - {step}")

        if self.failed_steps:
            logger.info("\n❌ Failed Steps:")
            for step in self.failed_steps:
                logger.info(f"  ✗ {step}")
            return

        logger.info("\n" + "=" * 70)
        logger.info("✅ All plotting tasks completed successfully!")
        logger.info("=" * 70)
        logger.info(f"\nFigures saved to: {self.eval_dir / 'figures'}")

        # Show directory structure
        figures_dir = self.eval_dir / "figures"
        if figures_dir.exists():
            logger.info("\nGenerated directories:")
            for subdir in sorted(figures_dir.iterdir()):
                if subdir.is_dir():
                    file_count = len(list(subdir.rglob("*.*")))
                    logger.info(f"  • {subdir.name}/ ({file_count} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all evaluation plots by calling existing modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all plots (skip trajectories by default)
  uv run python generate_all_plots.py --eval-dir hoser-distill-optuna-6
  
  # Include trajectory visualizations (~30-60 min)
  uv run python generate_all_plots.py --eval-dir eval_dir --include-trajectories
  
  # Skip specific categories
  uv run python generate_all_plots.py --eval-dir eval_dir --skip-distributions
  uv run python generate_all_plots.py --eval-dir eval_dir --skip-analysis --skip-scenarios
  
  # Specify dataset explicitly (auto-detected from evaluation.yaml by default)
  uv run python generate_all_plots.py --eval-dir eval_xyz --dataset porto

Output Structure:
  eval_directory/
  ├── analysis/           # Aggregate CSVs
  └── figures/
      ├── distributions/  # Distance, radius, duration plots
      ├── trajectories/   # Trajectory visualizations (if enabled)
      ├── analysis/       # Heatmaps, comparisons, radar charts
      └── scenarios/      # Scenario-specific plots (if available)
        """,
    )

    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Evaluation directory containing results",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name (auto-detected from evaluation.yaml if not provided)",
    )
    parser.add_argument(
        "--include-trajectories",
        action="store_true",
        help="Include trajectory visualizations (~30-60 min, skipped by default)",
    )
    parser.add_argument(
        "--skip-distributions",
        action="store_true",
        help="Skip distribution plots",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis figures",
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="Skip scenario plots",
    )

    args = parser.parse_args()

    try:
        orchestrator = PlottingOrchestrator(
            eval_dir=args.eval_dir,
            dataset=args.dataset,
            skip_trajectories=not args.include_trajectories,
            skip_distributions=args.skip_distributions,
            skip_analysis=args.skip_analysis,
            skip_scenarios=args.skip_scenarios,
        )
        return orchestrator.run_all()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
