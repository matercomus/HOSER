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
from typing import Optional

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
from tools.plot_abnormal_evaluation import plot_evaluation_from_files
from tools.plot_abnormal_analysis import plot_analysis_from_files

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
            Path to translated and filtered OD pairs JSON file (or original if no translation)
        """
        # Determine source and target datasets for translation
        source_dataset = getattr(self.config, "source_dataset", self.dataset)
        target_dataset = getattr(self.config, "target_dataset", self.config.dataset)

        # Check if translation is configured
        if not source_dataset or not target_dataset:
            logger.info(
                "🟢 No translation configuration found - using OD pairs directly"
            )
            logger.info("   → Within-dataset evaluation mode")
            return self.od_pairs_file

        # Assert translation is needed (always required)
        assert source_dataset and target_dataset, (
            "Source and target datasets must be configured for translation"
        )

        # Within-dataset evaluation - no translation needed
        if source_dataset == target_dataset:
            logger.info("🟢 Within-dataset evaluation: Skipping translation entirely")
            logger.info("   → OD pairs will be used directly for trajectory generation")
            logger.info("   → Models trained on the same dataset as abnormal detection")
            return self.od_pairs_file

        # Cross-dataset evaluation with translation
        logger.info("=" * 80)
        logger.info("🗺️  Translation: Translating OD Pairs")
        logger.info("=" * 80)
        logger.info(f"Source dataset: {source_dataset}")
        logger.info(f"Target dataset: {target_dataset}")
        logger.info(f"Max distance threshold: {self.config.translation_max_distance}")

        # Check if mapping is needed and feasible
        if source_dataset != target_dataset:
            # For cross-dataset evaluation, check if mapping already exists
            mapping_file = self.config.get_translation_mapping_file()
            if mapping_file and mapping_file.exists():
                # Load mapping to check if it's valid (has mappings, not empty)
                logger.info(f"Checking existing mapping file: {mapping_file}")
                try:
                    import json

                    with open(mapping_file, "r") as f:
                        mapping_data = json.load(f)

                    if not mapping_data or len(mapping_data) == 0:
                        logger.warning(
                            "⚠️  Mapping file exists but is empty - cross-continental datasets detected"
                        )
                        logger.warning(
                            "🗺️  Cross-continental mapping not feasible, using alternative approach"
                        )
                        logger.info(
                            "Continuing without translation for cross-continental evaluation"
                        )
                        return self.od_pairs_file

                except Exception as e:
                    logger.warning(f"Could not check mapping file: {e}")

            # Create mapping file if it doesn't exist
            self.create_road_mapping()

            # After creating mapping, check if it has any valid mappings
            mapping_file = self.config.get_translation_mapping_file()
            if mapping_file and mapping_file.exists():
                try:
                    import json

                    with open(mapping_file, "r") as f:
                        mapping_data = json.load(f)

                    if not mapping_data or len(mapping_data) == 0:
                        logger.warning(
                            "⚠️  Mapping failed - no roads could be mapped between datasets"
                        )
                        logger.warning(
                            "🗺️  Cross-continental evaluation detected, skipping translation"
                        )
                        logger.info(
                            "Continuing with original OD pairs for cross-continental evaluation"
                        )
                        return self.od_pairs_file

                except Exception as e:
                    logger.warning(f"Could not validate mapping file: {e}")

        # Continue with translation logic
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

    def create_road_mapping(self) -> None:
        """Create road network mapping file if it doesn't exist

        Automatically creates the mapping file needed for cross-dataset translation
        using the source and target datasets specified in the config.
        """
        source_dataset = getattr(self.config, "source_dataset", self.dataset)
        target_dataset = getattr(self.config, "target_dataset", self.config.dataset)

        # Check if mapping already exists
        mapping_file = self.config.get_translation_mapping_file()
        if mapping_file and mapping_file.exists():
            logger.info(f"🗺️  Road mapping file already exists: {mapping_file}")
            return

        logger.info("🗺️  Road mapping file not found - creating it...")
        logger.info(f"Source dataset: {source_dataset}")
        logger.info(f"Target dataset: {target_dataset}")

        # Import here to avoid circular imports
        import sys

        sys.path.append(str(Path(__file__).parent.parent))

        from tools.map_road_networks import create_road_mapping_programmatic

        # Create mapping file programmatically
        source_roadmap = Path(self.real_data_dir) / "roadmap.geo"
        target_roadmap = Path(self.config.data_dir) / "roadmap.geo"
        output_file = (
            self.eval_dir
            / f"road_mapping_{self.config.source_dataset}_to_{self.config.target_dataset}.json"
        )

        # Use default max distance from config or 50m if not specified
        max_distance = (
            getattr(self.config, "translation_max_distance", 20.0) * 2.5
        )  # Use 2.5x threshold for mapping

        logger.info("Creating mapping:")
        logger.info(f"  Source roadmap: {source_roadmap}")
        logger.info(f"  Target roadmap: {target_roadmap}")
        logger.info(f"  Output file: {output_file}")
        logger.info(f"  Max distance: {max_distance}m")

        create_road_mapping_programmatic(
            source_roadmap=source_roadmap,
            target_roadmap=target_roadmap,
            output_file=output_file,
            max_distance_m=max_distance,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
        )

        logger.info(f"✅ Road mapping created: {output_file}")

        # Update config mapping file if it was auto-detected
        if not mapping_file:
            # This would require updating the config object, but for now
            # the get_translation_mapping_file should find the auto-generated file
            pass

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

        If generation was skipped, focuses on analysis of real abnormal detection data.
        """
        logger.info("=" * 80)
        logger.info("📊 Analysis: Aggregating and Visualizing Abnormal Results")
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

        # Generate abnormal OD analysis plots
        if getattr(self.config, "skip_generation", False):
            logger.info("Generating real abnormal data analysis...")
            self._generate_real_abnormal_analysis()
        else:
            logger.info("Generating abnormal OD evaluation plots...")
            self._generate_abnormal_od_plots()

        # Create summary report
        self._create_summary_report()

    def _generate_abnormal_od_plots(self):
        """Generate plots for abnormal OD evaluation results"""
        comparison_file = self.eval_output_dir / "comparison_report.json"

        assert comparison_file.exists(), (
            f"Comparison report not found: {comparison_file}. "
            f"Run evaluation step first."
        )

        # Use plotting module
        plot_evaluation_from_files(
            comparison_report_file=comparison_file,
            output_dir=self.figures_dir,
            dataset=self.dataset,
        )

        logger.info(f"✅ Generated evaluation plots in {self.figures_dir}")

    def _generate_real_abnormal_analysis(self):
        """Generate analysis plots for real abnormal data (no generation)"""
        assert self.od_pairs_file.exists(), (
            f"OD pairs file not found: {self.od_pairs_file}. Run extraction step first."
        )

        logger.info("Analyzing real abnormal trajectory data...")

        # Prepare detection results files
        detection_results_files = [
            self.abnormal_dir / "train" / "real_data" / "detection_results.json",
            self.abnormal_dir / "test" / "real_data" / "detection_results.json",
        ]

        # Use plotting module
        plot_analysis_from_files(
            abnormal_od_pairs_file=self.od_pairs_file,
            real_data_files=[self.train_csv, self.test_csv],
            detection_results_files=detection_results_files,
            samples_dir=self.abnormal_dir,
            output_dir=self.figures_dir,
            dataset=self.dataset,
            include_normal=False,  # Can be made configurable later
        )

        logger.info(f"✅ Generated analysis plots in {self.figures_dir}")

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

            # Translation step - only if configured
            if (
                hasattr(self.config, "source_dataset")
                and hasattr(self.config, "target_dataset")
                and self.config.source_dataset
                and self.config.target_dataset
            ):
                # Translation is configured, perform the step
                self.translate_od_pairs()
            else:
                # No translation configured, use OD pairs directly
                logger.info(
                    "🟢 No translation configuration found - using OD pairs directly"
                )
                logger.info("   → Within-dataset evaluation mode")

            # Check if we should skip generation and focus on analysis
            if getattr(self.config, "skip_generation", False):
                logger.info(
                    "🟢 Skipping trajectory generation - focusing on analysis of real abnormal data"
                )
                logger.info(
                    "   → Using existing abnormal detection results and OD pairs"
                )
            else:
                # Generate trajectories
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
