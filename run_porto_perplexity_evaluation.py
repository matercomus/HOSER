#!/usr/bin/env python3
"""
Run Porto Perplexity Evaluation with New Implementation

This script efficiently runs the new per-road-segment perplexity evaluation
on Porto dataset, reusing existing files and only computing what's needed.

Usage:
    python run_porto_perplexity_evaluation.py

Features:
- Reuses existing trajectory files
- Only evaluates models that haven't been processed
- Leverages existing LMTAD teacher model
- Comprehensive output with cross-model comparison
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_lmtad_spatial_abnormal import evaluate_spatial_abnormal_trajectories
from tools.analyze_lmtad_spatial_results import aggregate_lmtad_perplexity_results
from tools.visualize_lmtad_spatial_results import (
    plot_perplexity_distribution_comparison,
    plot_per_od_pair_perplexity_comparison,
    plot_model_rankings_by_perplexity,
    plot_comprehensive_perplexity_summary
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PortoPerplexityEvaluator:
    """Comprehensive evaluator for Porto dataset using new perplexity approach."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.base_dir = self.project_root / "hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732"
        self.dataset = "porto_hoser"

        # LMTAD teacher checkpoint (adjust path as needed)
        self.lmtad_checkpoint = Path("/home/matt/Dev/LMTAD/code/results/porto_hoser/ckpt_best.pt")

        # Models to evaluate
        self.models = [
            "vanilla_25epoch_seed42",
            "distill_phase1_seed42",
            "distill_phase2_seed42"
        ]

        # Output directories
        self.eval_dir = self.base_dir / "eval_lmtad_spatial" / self.dataset
        self.analysis_dir = self.base_dir / "analysis_abnormal" / self.dataset
        self.figures_dir = self.base_dir / "figures" / "lmtad_spatial_abnormality"

        # Trajectory file patterns
        self.trajectory_pattern = "vanilla_spatial_abnormal.csv"

    def check_existing_results(self) -> Dict[str, bool]:
        """Check which models already have evaluation results."""
        existing_results = {}

        for model in self.models:
            result_file = self.eval_dir / f"{model}_lmtad_spatial_evaluation.json"
            existing_results[model] = result_file.exists()

        return existing_results

    def get_trajectory_file(self, model: str) -> Optional[Path]:
        """Get the trajectory file path for a model."""
        # For Porto, we use the vanilla spatial abnormal trajectories
        # You can extend this to use model-specific trajectories if needed
        traj_file = self.base_dir / "gene_abnormal_lmtad_spatial" / self.dataset / "seed42" / self.trajectory_pattern

        if traj_file.exists():
            return traj_file
        else:
            logger.warning(f"Trajectory file not found for {model}: {traj_file}")
            return None

    def evaluate_models(self) -> List[Path]:
        """Evaluate all models that don't have existing results."""
        logger.info("=" * 80)
        logger.info("🚀 PORTO PERPLEXITY EVALUATION")
        logger.info("=" * 80)

        # Check existing results
        existing_results = self.check_existing_results()
        logger.info(f"📊 Existing results: {existing_results}")

        # Create output directory
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        evaluated_files = []

        for model in self.models:
            if existing_results[model]:
                logger.info(f"✅ Skipping {model} (already evaluated)")
                evaluated_files.append(self.eval_dir / f"{model}_lmtad_spatial_evaluation.json")
                continue

            trajectory_file = self.get_trajectory_file(model)
            if not trajectory_file:
                logger.warning(f"❌ Skipping {model} (no trajectory file)")
                continue

            logger.info(f"🔍 Evaluating {model}...")

            try:
                # Run evaluation
                result_file = evaluate_spatial_abnormal_trajectories(
                    trajectory_file=str(trajectory_file),
                    lmtad_checkpoint=str(self.lmtad_checkpoint),
                    dataset=self.dataset,
                    output_dir=str(self.eval_dir),
                    model_name=model
                )

                evaluated_files.append(Path(result_file))
                logger.info(f"✅ {model} evaluation complete: {result_file}")

            except Exception as e:
                logger.error(f"❌ Failed to evaluate {model}: {e}")
                continue

        logger.info(f"📊 Total evaluated files: {len(evaluated_files)}")
        return evaluated_files

    def aggregate_results(self, result_files: List[Path]) -> Path:
        """Aggregate evaluation results across models."""
        logger.info("📈 Aggregating results across models...")

        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate results
        output_file = self.analysis_dir / "lmtad_perplexity_results_aggregated.json"

        try:
            aggregate_lmtad_perplexity_results(
                eval_dir=str(self.eval_dir),
                dataset=self.dataset,
                output=str(output_file)
            )

            logger.info(f"✅ Aggregation complete: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"❌ Aggregation failed: {e}")
            raise

    def generate_cross_model_comparison(self, aggregated_file: Path) -> Path:
        """Generate cross-model OD comparison."""
        logger.info("🔄 Building cross-model OD comparison...")

        from tools.evaluate_lmtad_spatial_abnormal import _build_cross_model_od_comparison

        # Load aggregated results
        with open(aggregated_file) as f:
            aggregated_results = json.load(f)

        # Extract evaluation results for each model
        evaluation_results = []
        for model_name, model_data in aggregated_results.items():
            if isinstance(model_data, dict) and 'trajectories' in model_data:
                evaluation_results.append({
                    "model": model_name,
                    "trajectories": model_data['trajectories']
                })

        if not evaluation_results:
            logger.warning("No evaluation results found for cross-model comparison")
            return aggregated_file

        # Build cross-model comparison
        comparison_file = self.analysis_dir / "cross_model_od_comparison.json"

        try:
            _build_cross_model_od_comparison(
                evaluation_results=evaluation_results,
                output_path=str(comparison_file)
            )

            logger.info(f"✅ Cross-model comparison complete: {comparison_file}")
            return comparison_file

        except Exception as e:
            logger.error(f"❌ Cross-model comparison failed: {e}")
            return aggregated_file

    def create_visualizations(self, aggregated_file: Path, comparison_file: Path) -> List[Path]:
        """Create comprehensive visualizations."""
        logger.info("📊 Creating visualizations...")

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        try:
            # Load data
            with open(aggregated_file) as f:
                aggregated_results = json.load(f)

            with open(comparison_file) as f:
                comparison_results = json.load(f)

            # Create visualizations
            viz_files = [
                ("perplexity_distributions.png", lambda: plot_perplexity_distribution_comparison(
                    aggregated_results, str(self.figures_dir / "perplexity_distributions.png")
                )),
                ("od_pair_comparison.png", lambda: plot_per_od_pair_perplexity_comparison(
                    comparison_results, str(self.figures_dir / "od_pair_comparison.png")
                )),
                ("model_rankings.png", lambda: plot_model_rankings_by_perplexity(
                    comparison_results, str(self.figures_dir / "model_rankings.png")
                )),
                ("comprehensive_summary.png", lambda: plot_comprehensive_perplexity_summary(
                    aggregated_results, comparison_results, str(self.figures_dir / "comprehensive_summary.png")
                ))
            ]

            for filename, plot_func in viz_files:
                try:
                    plot_func()
                    created_files.append(self.figures_dir / filename)
                    logger.info(f"✅ Created: {filename}")
                except Exception as e:
                    logger.error(f"❌ Failed to create {filename}: {e}")

        except Exception as e:
            logger.error(f"❌ Visualization creation failed: {e}")

        return created_files

    def print_summary(self, result_files: List[Path], aggregated_file: Path,
                      comparison_file: Path, viz_files: List[Path]):
        """Print comprehensive summary of the evaluation."""
        logger.info("=" * 80)
        logger.info("📋 PORTO PERPLEXITY EVALUATION SUMMARY")
        logger.info("=" * 80)

        logger.info(f"📊 Individual model results: {len(result_files)} files")
        for file in result_files:
            logger.info(f"   • {file.name}")

        logger.info(f"📈 Aggregated results: {aggregated_file.name}")
        logger.info(f"🔄 Cross-model comparison: {comparison_file.name}")

        logger.info(f"📊 Visualizations created: {len(viz_files)} files")
        for file in viz_files:
            logger.info(f"   • {file.name}")

        # Load and display key metrics
        try:
            with open(aggregated_file) as f:
                aggregated_results = json.load(f)

            with open(comparison_file) as f:
                comparison_results = json.load(f)

            logger.info("\n🎯 KEY METRICS:")
            for model_name, data in aggregated_results.items():
                if isinstance(data, dict) and 'log_perplexity_stats' in data:
                    stats = data['log_perplexity_stats']
                    logger.info(f"   {model_name}:")
                    logger.info(f"     Mean perplexity: {stats.get('mean', 'N/A'):.3f}")
                    logger.info(f"     Std perplexity: {stats.get('std', 'N/A'):.3f}")
                    logger.info(f"     Failed trajectories: {data.get('failed_trajectory_count', 0)}")

            # Cross-model insights
            if 'od_summary' in comparison_results:
                od_summary = comparison_results['od_summary']
                if 'model_performance_ranking' in od_summary:
                    rankings = od_summary['model_performance_ranking']
                    logger.info("\n🏆 MODEL RANKINGS (by # of OD pairs where best):")
                    for model, count in sorted(rankings.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"   {model}: {count} OD pairs")

        except Exception as e:
            logger.warning(f"Could not load summary statistics: {e}")

        logger.info("=" * 80)
        logger.info("🎉 Evaluation complete! Check the figures directory for visualizations.")
        logger.info("=" * 80)

    def run_full_evaluation(self):
        """Run the complete Porto perplexity evaluation."""
        try:
            # Step 1: Evaluate models
            result_files = self.evaluate_models()

            if not result_files:
                logger.warning("No models were evaluated. Exiting.")
                return

            # Step 2: Aggregate results
            aggregated_file = self.aggregate_results(result_files)

            # Step 3: Create cross-model comparison
            comparison_file = self.generate_cross_model_comparison(aggregated_file)

            # Step 4: Create visualizations
            viz_files = self.create_visualizations(aggregated_file, comparison_file)

            # Step 5: Print summary
            self.print_summary(result_files, aggregated_file, comparison_file, viz_files)

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise


def main():
    """Main entry point."""
    evaluator = PortoPerplexityEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()