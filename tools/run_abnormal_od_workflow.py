#!/usr/bin/env python3
"""
Abnormal OD Workflow Orchestrator

This module provides a reusable programmatic interface for executing the complete
abnormal OD workflow (Phases 0-5) on any evaluation directory and dataset.

The workflow includes:
- Phase 0: Wang statistical detection (if needed)
- Phase 3: Extract abnormal OD pairs
- Phase 4: Generate trajectories for abnormal ODs
- Phase 5: Evaluate models on abnormal ODs
- Analysis: Aggregate and visualize results

Usage (Programmatic):
    from pathlib import Path
    from tools.run_abnormal_od_workflow import run_abnormal_od_workflow
    
    analysis_dir = run_abnormal_od_workflow(
        eval_dir=Path("hoser-distill-optuna-porto-eval-xyz"),
        dataset="porto_hoser",
        real_data_dir=Path("data/porto_hoser"),
        num_trajectories=50,
        max_pairs_per_category=20,
        seed=42,
        skip_detection=True
    )

Usage (CLI):
    uv run python tools/run_abnormal_od_workflow.py \\
        --eval-dir hoser-distill-optuna-porto-eval-xyz \\
        --dataset porto_hoser \\
        --real-data-dir data/porto_hoser \\
        --skip-detection \\
        --num-traj 50 \\
        --max-pairs 20 \\
        --seed 42

Documentation:
    See docs/ABNORMAL_OD_WORKFLOW_GUIDE.md for comprehensive guide
    See tools/TOOLS_README.md for programmatic interface documentation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import programmatic interfaces
from tools.analyze_abnormal import run_abnormal_analysis
from tools.config_loader import load_evaluation_config
from tools.extract_abnormal_od_pairs import extract_and_save_abnormal_od_pairs
from tools.generate_abnormal_od import generate_abnormal_od_trajectories
from tools.evaluate_abnormal_od import evaluate_abnormal_od
from tools.analyze_wang_results import analyze_wang_results
from tools.visualize_wang_results import generate_wang_visualizations
from tools.translate_od_pairs import (
    load_road_mapping,
    translate_od_pairs,
    filter_od_pairs_by_quality,
    save_translated_od_pairs,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set publication-quality plot defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "font.family": "sans-serif",
    }
)


class AbnormalODWorkflowRunner:
    """Orchestrates the complete abnormal OD workflow"""

    def __init__(
        self,
        eval_dir: Path,
        dataset: str,
        real_data_dir: Optional[Path] = None,
        num_trajectories: int = 50,
        max_pairs_per_category: Optional[int] = 20,
        seed: int = 42,
        skip_detection: bool = False,
        detection_config: Optional[Path] = None,
    ):
        """
        Initialize workflow runner.

        Args:
            eval_dir: Evaluation directory containing models and detection results
            dataset: Dataset name (e.g., 'porto_hoser', 'BJUT_Beijing')
            real_data_dir: Directory containing train.csv and test.csv (optional, will use config if not provided)
            num_trajectories: Number of trajectories to generate per OD pair
            max_pairs_per_category: Max OD pairs per abnormality category (None = all)
            seed: Random seed for generation
            skip_detection: Skip Phase 0 if detection already exists
            detection_config: Path to detection config (required if skip_detection=False)
        """
        self.eval_dir = Path(eval_dir).resolve()
        self.dataset = dataset
        self.num_trajectories = num_trajectories
        self.max_pairs_per_category = max_pairs_per_category
        self.skip_detection = skip_detection
        self.detection_config = detection_config

        # Load config using shared config loader
        self.config = load_evaluation_config(eval_dir=self.eval_dir)

        # Override dataset if provided (takes precedence over config)
        if dataset:
            self.config.dataset = dataset

        # Override seed from parameter or use config
        self.seed = seed if seed != 42 else self.config.seed

        # Determine real_data_dir from config if not provided
        if real_data_dir is None:
            self.real_data_dir = self.config.get_data_dir()
        else:
            self.real_data_dir = Path(real_data_dir)
            if not self.real_data_dir.is_absolute():
                # Relative paths are relative to current working directory (project root)
                # not the eval directory, since user provides paths from project root
                original_cwd = Path.cwd()
                self.real_data_dir = (original_cwd / self.real_data_dir).resolve()
            else:
                # Already absolute, just resolve
                self.real_data_dir = self.real_data_dir.resolve()

        # Derived paths
        self.train_csv = self.real_data_dir / "train.csv"
        self.test_csv = self.real_data_dir / "test.csv"
        self.abnormal_dir = self.eval_dir / "abnormal" / self.dataset
        self.models_dir = self.eval_dir / "models"

        # Output paths
        self.od_pairs_file = self.eval_dir / f"abnormal_od_pairs_{self.dataset}.json"
        self.gene_dir = (
            self.eval_dir / "gene_abnormal" / self.dataset / f"seed{self.seed}"
        )
        self.eval_output_dir = self.eval_dir / "eval_abnormal" / self.dataset
        self.analysis_dir = self.eval_dir / "analysis_abnormal" / self.dataset
        self.figures_dir = self.eval_dir / "figures" / "abnormal_od" / self.dataset

        # Validation
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate required inputs exist"""
        assert self.eval_dir.exists(), (
            f"Evaluation directory not found: {self.eval_dir}"
        )
        assert self.real_data_dir.exists(), (
            f"Real data directory not found: {self.real_data_dir}"
        )
        assert self.train_csv.exists(), f"Train data not found: {self.train_csv}"
        assert self.test_csv.exists(), f"Test data not found: {self.test_csv}"

        if not self.skip_detection:
            assert self.detection_config, (
                "detection_config is required when skip_detection=False"
            )
            assert Path(self.detection_config).exists(), (
                f"Detection config not found: {self.detection_config}"
            )

    def detect_abnormalities(self) -> bool:
        """
        Phase 0: Run Wang statistical detection on real data.

        Returns:
            True if detection ran successfully, False if skipped
        """
        if self.skip_detection:
            logger.info("⏭️  Phase 0: Skipping detection (skip_detection=True)")

            # Check if detection results exist
            train_detection = (
                self.abnormal_dir / "train" / "real_data" / "detection_results.json"
            )
            test_detection = (
                self.abnormal_dir / "test" / "real_data" / "detection_results.json"
            )

            assert train_detection.exists() and test_detection.exists(), (
                "Detection results not found. Set skip_detection=False to run detection."
            )

            return False

        logger.info("=" * 80)
        logger.info("🔍 Phase 0: Running Wang Detection")
        logger.info("=" * 80)

        # Run detection on train split
        train_output = self.abnormal_dir / "train" / "real_data"
        logger.info("Running detection on train split...")
        self._run_detection(self.train_csv, train_output)

        # Run detection on test split
        test_output = self.abnormal_dir / "test" / "real_data"
        logger.info("Running detection on test split...")
        self._run_detection(self.test_csv, test_output)

        logger.info("✅ Phase 0 complete: Detection results generated")
        return True

    def _run_detection(self, real_file: Path, output_dir: Path):
        """Run detection using programmatic interface"""
        logger.info(f"  Processing: {real_file}")

        # Call programmatic interface
        run_abnormal_analysis(
            real_file=real_file,
            dataset=self.dataset,
            config_path=self.detection_config,
            output_dir=output_dir,
        )

    def extract_abnormal_od_pairs(self) -> Path:
        """
        Phase 3: Extract abnormal OD pairs from detection results.

        Returns:
            Path to generated OD pairs JSON file
        """
        logger.info("=" * 80)
        logger.info("📊 Phase 3: Extracting Abnormal OD Pairs")
        logger.info("=" * 80)

        train_detection = (
            self.abnormal_dir / "train" / "real_data" / "detection_results.json"
        )
        test_detection = (
            self.abnormal_dir / "test" / "real_data" / "detection_results.json"
        )

        logger.info(f"Extracting OD pairs to {self.od_pairs_file}")

        # Call programmatic interface
        extract_and_save_abnormal_od_pairs(
            detection_results_files=[train_detection, test_detection],
            real_data_files=[self.train_csv, self.test_csv],
            dataset_name=self.dataset,
            output_file=self.od_pairs_file,
        )

        # Load and log summary
        with open(self.od_pairs_file, "r") as f:
            od_data = json.load(f)

        total_pairs = sum(
            len(pairs) for pairs in od_data["od_pairs_by_category"].values()
        )
        logger.info(f"✅ Phase 3 complete: {total_pairs} abnormal OD pairs extracted")
        logger.info(f"   Categories: {list(od_data['od_pairs_by_category'].keys())}")

        return self.od_pairs_file

    def translate_od_pairs(self) -> Path:
        """Translate and filter OD pairs using road network mapping

        Returns:
            Path to translated and filtered OD pairs JSON file
        """
        # Determine source and target datasets for translation
        source_dataset = getattr(self.config, "source_dataset", self.dataset)
        target_dataset = getattr(self.config, "target_dataset", self.config.dataset)

        # Assert translation is needed (always required)
        assert source_dataset and target_dataset, (
            "Source and target datasets must be configured for translation"
        )
        assert source_dataset != target_dataset, (
            f"Source dataset ({source_dataset}) and target dataset ({target_dataset}) "
            f"must be different for translation to be required"
        )

        logger.info("=" * 80)
        logger.info("🗺️  Translation: Translating OD Pairs")
        logger.info("=" * 80)
        logger.info(f"Source dataset: {source_dataset}")
        logger.info(f"Target dataset: {target_dataset}")
        logger.info(f"Max distance threshold: {self.config.translation_max_distance}m")

        # Load original OD pairs
        logger.info(f"Loading OD pairs from {self.od_pairs_file}")
        with open(self.od_pairs_file, "r") as f:
            original_data = json.load(f)

        # Collect all OD pairs from all categories
        all_od_pairs = []
        category_od_pairs = {}
        for category, pairs in original_data.get("od_pairs_by_category", {}).items():
            category_od_pairs[category] = pairs
            all_od_pairs.extend(pairs)

        if not all_od_pairs:
            logger.warning("No OD pairs found to translate")
            return self.od_pairs_file

        logger.info(
            f"Found {len(all_od_pairs)} total OD pairs across {len(category_od_pairs)} categories"
        )

        # Load mapping file
        mapping_file = self.config.get_translation_mapping_file()
        assert mapping_file and mapping_file.exists(), (
            f"Translation mapping file not found: {mapping_file}. "
            f"Translation is required for cross-dataset evaluation."
        )

        logger.info(f"Loading mapping from {mapping_file}")
        mapping = load_road_mapping(mapping_file)

        # Translate OD pairs
        logger.info("Translating OD pairs...")
        translated_pairs = translate_od_pairs(all_od_pairs, mapping)

        # Filter by quality
        logger.info("Filtering by quality...")
        max_distance = getattr(self.config, "translation_max_distance", 20.0)
        filtered_pairs, stats = filter_od_pairs_by_quality(
            translated_pairs, mapping, max_distance
        )

        # Redistribute filtered pairs to categories
        # This is a simplified approach - in practice, we'd track which pairs belong to which category
        translated_by_category = {}
        total_translated = 0

        for category, original_pairs in category_od_pairs.items():
            num_pairs_in_category = len(original_pairs)
            if total_translated + num_pairs_in_category <= len(filtered_pairs):
                # Take pairs for this category
                category_start = total_translated
                category_end = category_start + num_pairs_in_category
                category_translated = filtered_pairs[category_start:category_end]
                translated_by_category[category] = category_translated
                total_translated = category_end
            else:
                # Take remaining pairs
                category_translated = filtered_pairs[total_translated:]
                if category_translated:
                    translated_by_category[category] = category_translated
                total_translated = len(filtered_pairs)
                break

        # Save translated pairs
        translated_file = (
            self.od_pairs_file.parent / f"{self.od_pairs_file.stem}_translated.json"
        )
        save_translated_od_pairs(
            original_data=original_data,
            translated_pairs_by_category=translated_by_category,
            translation_stats=stats,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            output_file=translated_file,
        )

        # Update od_pairs_file to point to translated file
        self.od_pairs_file = translated_file

        logger.info(f"✅ Translation complete: {translated_file}")
        return translated_file

    def generate_trajectories(self) -> Path:
        """
        Phase 4: Generate trajectories for abnormal OD pairs.

        Returns:
            Path to generation output directory
        """
        logger.info("=" * 80)
        logger.info("🚗 Phase 4: Generating Trajectories for Abnormal ODs")
        logger.info("=" * 80)

        logger.info(f"Generating {self.num_trajectories} trajectories per OD pair")
        logger.info(f"Output directory: {self.gene_dir}")

        # Call programmatic interface
        generate_abnormal_od_trajectories(
            od_pairs_file=self.od_pairs_file,
            model_dir=self.models_dir,
            output_dir=self.gene_dir,
            dataset=self.dataset,
            num_traj_per_od=self.num_trajectories,
            max_pairs_per_category=self.max_pairs_per_category,
            seed=self.seed,
            cuda_device=0,
            beam_search=self.config.beam_search,  # Use A* search by default
            beam_width=self.config.beam_width,
        )

        # Count generated files
        generated_files = list(self.gene_dir.glob("*_abnormal_od.csv"))
        logger.info(
            f"✅ Phase 4 complete: Generated files for {len(generated_files)} models"
        )

        return self.gene_dir

    def evaluate_trajectories(self) -> Path:
        """
        Phase 5: Evaluate models on abnormal OD trajectories.

        Returns:
            Path to evaluation output directory
        """
        logger.info("=" * 80)
        logger.info("📈 Phase 5: Evaluating Models on Abnormal ODs")
        logger.info("=" * 80)

        logger.info(f"Evaluating against real data: {self.train_csv}")
        logger.info(f"Output directory: {self.eval_output_dir}")

        # Call programmatic interface
        evaluate_abnormal_od(
            generated_dir=self.gene_dir,
            real_abnormal_file=self.train_csv,
            abnormal_od_pairs_file=self.od_pairs_file,
            output_dir=self.eval_output_dir,
            dataset=self.dataset,
        )

        logger.info("✅ Phase 5 complete: Evaluation results saved")

        return self.eval_output_dir

    def run_analysis_and_visualization(self):
        """
        Analyze and visualize abnormal OD workflow results using programmatic interfaces.
        """
        logger.info("=" * 80)
        logger.info("📊 Analysis: Aggregating and Visualizing Results")
        logger.info("=" * 80)

        # Create analysis directory
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Run Wang results analysis using programmatic interface
        logger.info("Analyzing Wang detection results...")
        wang_results_file = self.analysis_dir / "wang_results_aggregated.json"

        analyze_wang_results(eval_dirs=[self.eval_dir], output_file=wang_results_file)

        logger.info(f"✅ Wang results aggregated: {wang_results_file}")

        # Generate Wang visualizations using programmatic interface
        logger.info("Generating Wang visualizations...")
        wang_figures_dir = self.eval_dir / "figures" / "wang_abnormality" / self.dataset

        generate_wang_visualizations(
            results_file=wang_results_file, output_dir=wang_figures_dir
        )

        logger.info(f"✅ Wang visualizations generated: {wang_figures_dir}")

        # Generate abnormal OD evaluation plots
        logger.info("Generating abnormal OD evaluation plots...")
        self._generate_abnormal_od_plots()

        logger.info(f"✅ Abnormal OD plots generated: {self.figures_dir}")

        # Create summary report
        self._create_summary_report()

    def _generate_abnormal_od_plots(self):
        """Generate plots for abnormal OD evaluation results"""
        # Create figures directory
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Load comparison report
        comparison_file = self.eval_output_dir / "comparison_report.json"
        if not comparison_file.exists():
            logger.warning(f"Comparison report not found: {comparison_file}")
            return

        with open(comparison_file, "r") as f:
            comparison_data = json.load(f)

        model_results = comparison_data.get("model_results", {})
        if not model_results:
            logger.warning("No model results found in comparison report")
            return

        # Generate various plots
        self._plot_abnormality_reproduction_rates(model_results)
        self._plot_similarity_metrics(model_results)
        self._plot_abnormality_by_category(model_results)
        self._plot_metrics_comparison_heatmap(model_results)

        logger.info(f"   Generated 4 plot types in {self.figures_dir}")

    def _plot_abnormality_reproduction_rates(self, model_results: Dict[str, Any]):
        """Plot abnormality reproduction rates across models"""
        models = []
        rates = []
        counts = []

        for model_name, results in sorted(model_results.items()):
            total_abnormal = sum(
                cat_data["count"]
                for cat_data in results["abnormality_detection"].values()
            )
            total_traj = results["total_trajectories"]
            rate = (total_abnormal / total_traj * 100) if total_traj > 0 else 0

            models.append(model_name)
            rates.append(rate)
            counts.append(total_abnormal)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Bar colors - blue for distilled, red for vanilla
        colors = ["#3498db" if "distill" in m.lower() else "#e74c3c" for m in models]

        bars = ax.barh(models, rates, color=colors, alpha=0.8)

        # Add value labels
        for i, (bar, rate, count, total) in enumerate(
            zip(
                bars,
                rates,
                counts,
                [model_results[m]["total_trajectories"] for m in models],
            )
        ):
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}% ({count}/{total})",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlabel("Abnormality Reproduction Rate (%)", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        ax.set_title(
            f"Abnormality Reproduction Rates - {self.dataset}\n"
            f"(Generated trajectories reproducing abnormal patterns)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#3498db", alpha=0.8, label="Distilled Models"),
            Patch(facecolor="#e74c3c", alpha=0.8, label="Vanilla Models"),
        ]
        ax.legend(handles=legend_elements, loc="best")

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "abnormality_reproduction_rates.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "abnormality_reproduction_rates.svg", bbox_inches="tight"
        )
        plt.close()

    def _plot_similarity_metrics(self, model_results: Dict[str, Any]):
        """Plot similarity metrics comparison across models"""
        models = list(sorted(model_results.keys()))

        edr_scores = []
        dtw_scores = []
        hausdorff_scores = []

        for model_name in models:
            metrics = model_results[model_name]["similarity_metrics"]
            edr_scores.append(metrics.get("edr", 0))
            dtw_scores.append(metrics.get("dtw", 0))
            hausdorff_scores.append(metrics.get("hausdorff", 0))

        # Create grouped bar plot
        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))

        bars1 = ax.bar(
            x - width, edr_scores, width, label="EDR", color="#2ecc71", alpha=0.8
        )
        bars2 = ax.bar(x, dtw_scores, width, label="DTW", color="#3498db", alpha=0.8)
        bars3 = ax.bar(
            x + width,
            hausdorff_scores,
            width,
            label="Hausdorff",
            color="#e74c3c",
            alpha=0.8,
        )

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

        ax.set_ylabel("Similarity Score", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_title(
            f"Trajectory Similarity Metrics - {self.dataset}\n"
            f"(Lower is better - distance from real abnormal trajectories)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "similarity_metrics_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "similarity_metrics_comparison.svg", bbox_inches="tight"
        )
        plt.close()

    def _plot_abnormality_by_category(self, model_results: Dict[str, Any]):
        """Plot abnormality detection by category across models"""
        # Collect all categories
        all_categories = set()
        for results in model_results.values():
            all_categories.update(results["abnormality_detection"].keys())

        categories = sorted(all_categories)
        models = list(sorted(model_results.keys()))

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 6))

        # Prepare data
        category_data = {cat: [] for cat in categories}
        for model_name in models:
            abnormal_by_cat = model_results[model_name]["abnormality_detection"]
            for cat in categories:
                count = abnormal_by_cat.get(cat, {}).get("count", 0)
                category_data[cat].append(count)

        # Color scheme for categories
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

        # Create stacked bars
        bottom = np.zeros(len(models))
        for i, cat in enumerate(categories):
            ax.bar(
                models,
                category_data[cat],
                bottom=bottom,
                label=cat,
                color=colors[i],
                alpha=0.8,
            )
            bottom += np.array(category_data[cat])

        ax.set_ylabel("Abnormal Trajectory Count", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_title(
            f"Abnormality Detection by Category - {self.dataset}\n"
            f"(Distribution of abnormal patterns in generated trajectories)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "abnormality_by_category.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.figures_dir / "abnormality_by_category.svg", bbox_inches="tight"
        )
        plt.close()

    def _plot_metrics_comparison_heatmap(self, model_results: Dict[str, Any]):
        """Create heatmap comparing all metrics across models"""
        models = list(sorted(model_results.keys()))

        # Collect metrics
        metrics_data = []
        metric_names = ["EDR", "DTW", "Hausdorff", "Abnormality Rate (%)"]

        for model_name in models:
            results = model_results[model_name]
            metrics = results["similarity_metrics"]

            total_abnormal = sum(
                cat_data["count"]
                for cat_data in results["abnormality_detection"].values()
            )
            total_traj = results["total_trajectories"]
            abnormal_rate = (total_abnormal / total_traj * 100) if total_traj > 0 else 0

            metrics_data.append(
                [
                    metrics.get("edr", 0),
                    metrics.get("dtw", 0),
                    metrics.get("hausdorff", 0),
                    abnormal_rate,
                ]
            )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(models) * 0.5 + 2))

        # Normalize each column separately for better visualization
        metrics_array = np.array(metrics_data)
        normalized_data = np.zeros_like(metrics_array)
        for i in range(metrics_array.shape[1]):
            col = metrics_array[:, i]
            if col.max() > 0:
                normalized_data[:, i] = col / col.max()

        im = ax.imshow(normalized_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(metric_names)
        ax.set_yticklabels(models)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values in cells
        for i in range(len(models)):
            for j in range(len(metric_names)):
                ax.text(
                    j,
                    i,
                    f"{metrics_array[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

        ax.set_title(
            f"Metrics Comparison Heatmap - {self.dataset}\n"
            f"(Normalized by column, darker = worse)",
            fontsize=14,
            fontweight="bold",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized Score (0=best, 1=worst)", rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "metrics_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(self.figures_dir / "metrics_heatmap.svg", bbox_inches="tight")
        plt.close()

    def _create_summary_report(self):
        """Create a summary report of the workflow execution"""
        summary_file = self.analysis_dir / "workflow_summary.json"

        summary = {
            "workflow": "Abnormal OD Analysis",
            "dataset": self.dataset,
            "eval_dir": str(self.eval_dir),
            "configuration": {
                "num_trajectories": self.num_trajectories,
                "max_pairs_per_category": self.max_pairs_per_category,
                "seed": self.seed,
            },
            "outputs": {
                "od_pairs": str(self.od_pairs_file),
                "generated_trajectories": str(self.gene_dir),
                "evaluation_results": str(self.eval_output_dir),
                "analysis": str(self.analysis_dir),
                "figures": str(self.figures_dir),
            },
            "files": {
                "wang_results": str(self.analysis_dir / "wang_results_aggregated.json"),
                "wang_visualizations": str(
                    self.eval_dir / "figures" / "wang_abnormality" / self.dataset
                ),
                "abnormal_od_plots": str(self.figures_dir),
            },
            "plots_generated": [
                "abnormality_reproduction_rates.png/svg",
                "similarity_metrics_comparison.png/svg",
                "abnormality_by_category.png/svg",
                "metrics_heatmap.png/svg",
            ],
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📄 Workflow summary: {summary_file}")

    def run_complete_workflow(self):
        """
        Execute the complete abnormal OD workflow (Phases 0-5 + Analysis).

        This is the main entry point for running the entire workflow.
        """
        # Change to eval directory (like python_pipeline.py does)
        original_cwd = os.getcwd()
        try:
            os.chdir(self.eval_dir)
            logger.info(f"Working directory: {self.eval_dir}")
        except Exception as e:
            logger.error(f"Failed to change to eval directory: {e}")
            raise

        logger.info("=" * 80)
        logger.info(f"🚀 Starting Abnormal OD Workflow for {self.dataset}")
        logger.info(f"   Evaluation Directory: {self.eval_dir}")
        logger.info(f"   Dataset: {self.dataset}")
        logger.info(f"   Data Directory: {self.real_data_dir}")
        logger.info("=" * 80)

        try:
            # Phase 0: Detection (if needed)
            self.detect_abnormalities()

            # Phase 3: Extract OD pairs
            self.extract_abnormal_od_pairs()

            # Phase 4: Generate trajectories
            self.generate_trajectories()

            # Phase 5: Evaluate
            self.evaluate_trajectories()

            # Analysis and visualization
            self.run_analysis_and_visualization()

            logger.info("=" * 80)
            logger.info("✅ Complete workflow finished successfully!")
            logger.info(f"📁 Results directory: {self.eval_output_dir}")
            logger.info(f"📊 Analysis directory: {self.analysis_dir}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            raise
        finally:
            # Restore original working directory
            os.chdir(original_cwd)


def run_abnormal_od_workflow(
    eval_dir: Path,
    dataset: str,
    real_data_dir: Optional[Path] = None,
    num_trajectories: int = 50,
    max_pairs_per_category: Optional[int] = 20,
    seed: int = 42,
    skip_detection: bool = False,
    detection_config: Optional[Path] = None,
) -> Path:
    """
    Run the complete abnormal OD workflow (programmatic interface).

    Args:
        eval_dir: Evaluation directory containing models and detection results
        dataset: Dataset name (e.g., 'porto_hoser', 'BJUT_Beijing')
        real_data_dir: Directory containing train.csv and test.csv (optional, will use config if not provided)
        num_trajectories: Number of trajectories to generate per OD pair
        max_pairs_per_category: Max OD pairs per abnormality category (None = all)
        seed: Random seed for generation (will be overridden by config if present)
        skip_detection: Skip Phase 0 if detection already exists
        detection_config: Path to detection config (required if skip_detection=False)

    Returns:
        Path to analysis output directory

    Example:
        >>> from pathlib import Path
        >>> from tools.run_abnormal_od_workflow import run_abnormal_od_workflow
        >>>
        >>> analysis_dir = run_abnormal_od_workflow(
        ...     eval_dir=Path("hoser-distill-optuna-porto-eval-xyz"),
        ...     dataset="porto_hoser",
        ...     num_trajectories=50,
        ...     max_pairs_per_category=20,
        ...     seed=42,
        ...     skip_detection=True  # Detection already done
        ... )
        >>> print(f"Analysis complete: {analysis_dir}")
    """
    runner = AbnormalODWorkflowRunner(
        eval_dir=eval_dir,
        dataset=dataset,
        real_data_dir=real_data_dir,
        num_trajectories=num_trajectories,
        max_pairs_per_category=max_pairs_per_category,
        seed=seed,
        skip_detection=skip_detection,
        detection_config=detection_config,
    )

    runner.run_complete_workflow()

    return runner.analysis_dir


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run complete abnormal OD workflow on evaluation directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow with detection
  uv run python tools/run_abnormal_od_workflow.py \\
    --eval-dir hoser-distill-optuna-porto-eval-xyz \\
    --dataset porto_hoser \\
    --real-data-dir data/porto_hoser \\
    --detection-config config/abnormal_detection_statistical.yaml \\
    --num-traj 50 \\
    --max-pairs 20 \\
    --seed 42

  # Run workflow skipping detection (already done)
  uv run python tools/run_abnormal_od_workflow.py \\
    --eval-dir hoser-distill-optuna-porto-eval-xyz \\
    --dataset porto_hoser \\
    --real-data-dir data/porto_hoser \\
    --skip-detection \\
    --num-traj 50 \\
    --max-pairs 20 \\
    --seed 42

  # Cross-dataset evaluation (BJUT)
  uv run python tools/run_abnormal_od_workflow.py \\
    --eval-dir hoser-distill-optuna-porto-eval-xyz \\
    --dataset BJUT_Beijing \\
    --real-data-dir data/BJUT_Beijing \\
    --detection-config config/abnormal_detection_statistical.yaml \\
    --num-traj 50 \\
    --seed 42
        """,
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory containing models and detection results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'porto_hoser', 'BJUT_Beijing')",
    )
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        required=False,
        help="Directory containing train.csv and test.csv (optional, will use data_dir from config/evaluation.yaml if not provided)",
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=50,
        help="Number of trajectories to generate per OD pair (default: 50)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20,
        help="Max OD pairs per abnormality category (default: 20, use 0 for all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for generation (default: 42)"
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip Phase 0 detection if results already exist",
    )
    parser.add_argument(
        "--detection-config",
        type=Path,
        help="Path to detection config YAML (required unless --skip-detection)",
    )

    args = parser.parse_args()

    # Convert max-pairs 0 to None (all pairs)
    max_pairs = args.max_pairs if args.max_pairs > 0 else None

    try:
        analysis_dir = run_abnormal_od_workflow(
            eval_dir=args.eval_dir,
            dataset=args.dataset,
            real_data_dir=args.real_data_dir,
            num_trajectories=args.num_traj,
            max_pairs_per_category=max_pairs,
            seed=args.seed,
            skip_detection=args.skip_detection,
            detection_config=args.detection_config,
        )

        print(f"\n✅ Workflow complete! Analysis results: {analysis_dir}")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
