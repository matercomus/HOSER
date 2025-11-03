#!/usr/bin/env python3
"""
Bulletproof Python Pipeline for HOSER Evaluation

This script imports gene.py and evaluation.py directly to orchestrate
trajectory generation and evaluation. Designed to work with evaluation
directories created by setup_evaluation.py.

Usage:
    # From inside an evaluation directory created by setup_evaluation.py:
    cd hoser-evaluation-xyz-20241024_123456
    uv run python ../python_pipeline.py [OPTIONS]

    # Or with explicit paths:
    uv run python python_pipeline.py --eval-dir path/to/eval/dir [OPTIONS]

Options:
    --eval-dir DIR           Evaluation directory (default: current directory)
    --seed SEED              Random seed (default: from config)
    --models MODEL1,MODEL2   Models to run (default: auto-detect all)
    --od-source SOURCE      OD source: train or test (default: from config)
    --skip-gene             Skip generation (use existing trajectories)
    --skip-eval             Skip evaluation
    --force                 Force re-run even if results exist
    --cuda DEVICE           CUDA device (default: from config)
    --num-gene N            Number of trajectories (default: from config)
    --wandb-project PROJECT WandB project name (default: from config)
    --no-wandb              Disable WandB logging entirely
    --verbose               Enable verbose output
    --run-scenarios         Run scenario analysis after evaluation
    --scenarios-config PATH Path to scenarios config YAML
"""

import os
import sys
import argparse
import signal
import threading
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Set
import logging

# Detect if we're running from inside an eval directory or from project root
SCRIPT_DIR = Path(__file__).parent
CURRENT_DIR = Path.cwd()

# Add appropriate paths for imports based on location
if (CURRENT_DIR / "models").exists() and (CURRENT_DIR / "config").exists():
    # Running from inside an evaluation directory
    EVAL_DIR = CURRENT_DIR
    PROJECT_ROOT = SCRIPT_DIR  # python_pipeline.py is in project root
else:
    # Running from project root or elsewhere
    EVAL_DIR = None  # Will be set from args
    PROJECT_ROOT = SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))

# Import the programmatic interfaces
from gene import generate_trajectories_programmatic  # noqa: E402
from evaluation import evaluate_trajectories_programmatic  # noqa: E402
from utils import set_seed  # noqa: E402
import yaml  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
logger = logging.getLogger(__name__)


# Phase Decorator Infrastructure
# Phase registry - auto-populated by decorator
PHASE_REGISTRY: Dict[str, Callable] = {}


def phase(name: str, critical: bool = False):
    """Decorator to register a pipeline phase

    Args:
        name: Phase identifier (used in CLI)
        critical: If True, pipeline stops on failure
    """

    def decorator(func):
        PHASE_REGISTRY[name] = {"func": func, "critical": critical, "name": name}

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class PipelineConfig:
    """Configuration for the evaluation pipeline"""

    def __init__(self, config_path: str = None, eval_dir: Path = None):
        # Store evaluation directory
        self.eval_dir = eval_dir

        # Default values
        self.wandb_project = "hoser-evaluation"
        self.dataset = "Beijing"
        self.cuda_device = 0
        self.num_gene = 100
        self.seed = 42
        self.models = []  # Auto-detect
        self.od_sources = ["train", "test"]

        # NEW: Phase-based control (replaces skip_* flags)
        self.phases: Set[str] = {
            "generation",
            "base_eval",
            "cross_dataset",
            "abnormal",
            "scenarios",
        }

        # DEPRECATED: Keep for backward compatibility
        self.skip_gene = False
        self.skip_eval = False
        self.force = False
        self.enable_wandb = True
        self.verbose = False
        self.beam_width = 4
        self.grid_size = 0.001
        self.edr_eps = 100.0
        self.background_sync = True  # Background WandB sync
        self.run_scenarios = False  # NEW: Run scenario analysis after eval
        self.scenarios_config = None  # NEW: Path to scenarios config
        self.cross_dataset_eval = None  # NEW: Path to cross-dataset for evaluation
        self.cross_dataset_name = (
            None  # NEW: Name of cross-dataset (e.g., BJUT_Beijing)
        )
        self.run_abnormal_detection = False  # NEW: Run abnormal trajectory detection
        self.abnormal_config = None  # NEW: Path to abnormal detection config

        # Load from YAML if provided
        if config_path:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Update attributes from config
        for key, value in config_data.items():
            if key == "wandb":
                wandb_config = value
                self.wandb_project = wandb_config.get("project", self.wandb_project)
                self.enable_wandb = wandb_config.get("enable", self.enable_wandb)
                self.background_sync = wandb_config.get(
                    "background_sync", self.background_sync
                )
            elif key == "logging":
                logging_config = value
                self.verbose = logging_config.get("verbose", self.verbose)
            elif hasattr(self, key):
                setattr(self, key, value)


class ModelDetector:
    """Auto-detect models in the models directory"""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def detect_models(self) -> List[str]:
        """Detect all available models and return unique model types"""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        model_files = list(self.models_dir.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No .pth files found in {self.models_dir}")

        model_types = []
        for model_file in model_files:
            model_name = model_file.stem
            # Extract model type (handle naming patterns)
            if "_25epoch_seed" in model_name:
                # Pattern: vanilla_25epoch_seed42 -> vanilla
                # Pattern: distilled_25epoch_seed44 -> distilled_seed44
                # Pattern: vanilla_25epoch_seed43 -> vanilla_seed43
                base_model = model_name.split("_25epoch_seed")[0]
                seed = model_name.split("_seed")[-1]

                # For seed 42, use base name only (vanilla, distill)
                # For other seeds, append seed to make unique (vanilla_seed43, distill_seed44)
                if seed == "42":
                    model_type = base_model
                else:
                    model_type = f"{base_model}_seed{seed}"
            else:
                # Fallback: remove everything after first underscore
                model_type = model_name.split("_")[0]

            model_types.append(model_type)

        # Remove duplicates and sort
        unique_types = sorted(list(set(model_types)))
        logger.info(f"Detected models: {unique_types}")
        return unique_types

    def find_model_file(self, model_type: str) -> Optional[Path]:
        """Find the actual model file for a given model type"""
        for model_file in self.models_dir.glob("*.pth"):
            model_name = model_file.stem

            # Extract model type using same logic as detect_models
            if "_25epoch_seed" in model_name:
                base_model = model_name.split("_25epoch_seed")[0]
                seed = model_name.split("_seed")[-1]

                # For seed 42, use base name only
                # For other seeds, append seed to make unique
                if seed == "42":
                    extracted_type = base_model
                else:
                    extracted_type = f"{base_model}_seed{seed}"
            else:
                extracted_type = model_name.split("_")[0]

            if extracted_type == model_type:
                return model_file

        return None


class TrajectoryGenerator:
    """Generate trajectories using HOSER models"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate_trajectories(
        self, model_path: Path, model_type: str, od_source: str
    ) -> tuple[Path, dict]:
        """Generate trajectories for a specific model and OD source

        Returns:
            Tuple of (output_path, performance_metrics)
        """
        logger.info(f"Generating trajectories: {model_type} ({od_source} OD)")

        # Use the programmatic interface (returns dict with output_file, num_generated, performance)
        result = generate_trajectories_programmatic(
            dataset=self.config.dataset,
            model_path=str(model_path),
            od_source=od_source,
            seed=self.config.seed,
            num_gene=self.config.num_gene,
            cuda_device=self.config.cuda_device,
            beam_search=True,
            beam_width=self.config.beam_width,
            enable_wandb=False,  # We'll handle WandB separately
            wandb_project=None,
            wandb_run_name=None,
            wandb_tags=None,
            model_type=model_type,
        )

        output_path = Path(result["output_file"])
        performance = result.get("performance", {})

        # Log performance summary
        if performance:
            logger.info(
                f"Generation performance: {performance.get('throughput_traj_per_sec', 0):.2f} traj/s, "
                + f"mean time: {performance.get('total_time_mean', 0):.3f}s"
            )

        logger.info(f"Trajectories saved to: {output_path}")
        return output_path, performance


class TrajectoryEvaluator:
    """Evaluate generated trajectories against real data with smart caching"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.evaluation_cache = {}  # Cache full evaluations by (file, od_source)

    def evaluate_trajectories(
        self,
        generated_file: Path,
        model_type: str,
        od_source: str,
        generation_performance: dict = None,
    ) -> Dict[str, Any]:
        """Evaluate generated trajectories with caching and optional performance metrics

        Args:
            generated_file: Path to generated trajectories
            model_type: Model type identifier
            od_source: OD source (train/test)
            generation_performance: Optional performance metrics from generation
        """
        cache_key = (str(generated_file), od_source)

        # Check if we've already evaluated this exact combination
        if cache_key in self.evaluation_cache:
            logger.info(
                f"Using cached evaluation results for {model_type} ({od_source} OD)"
            )
            cached_results = self.evaluation_cache[cache_key]
            # Update with new performance metrics if provided
            if generation_performance:
                cached_results["generation_performance"] = generation_performance
            return cached_results

        logger.info(f"Evaluating trajectories: {model_type} ({od_source} OD)")

        # Use the programmatic interface with performance metrics
        results = evaluate_trajectories_programmatic(
            generated_file=str(generated_file),
            dataset=self.config.dataset,
            od_source=od_source,  # Pass OD source to load correct real data
            grid_size=self.config.grid_size,
            edr_eps=self.config.edr_eps,
            enable_wandb=False,  # We'll handle WandB separately
            wandb_project=None,
            wandb_run_name=None,
            wandb_tags=None,
            generation_performance=generation_performance,  # Pass performance metrics
        )

        # Cache the results
        self.evaluation_cache[cache_key] = results

        # Add our metadata
        results["metadata"]["model_type"] = model_type
        results["metadata"]["od_source"] = od_source
        results["metadata"]["seed"] = self.config.seed

        # Re-save the results file with updated metadata
        # Find the results file path from the eval directory structure
        eval_dirs = sorted(Path("./eval").glob("20*"), key=lambda x: x.stat().st_mtime)
        if eval_dirs:
            latest_results_file = eval_dirs[-1] / "results.json"
            if latest_results_file.exists():
                import json

                with open(latest_results_file, "w") as f:
                    json.dump(results, f, indent=4)
                logger.debug(
                    f"Updated results file with model metadata: {latest_results_file}"
                )

        logger.info(f"Evaluation completed for {model_type} ({od_source} OD)")
        return results


class WandBManager:
    """Manage WandB logging efficiently with background uploads"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.runs = {}  # Store run objects
        self.completed_runs = []  # Track completed runs for background sync
        self.sync_thread = None

    def init_run(
        self, run_name: str, tags: List[str], config_dict: Dict[str, Any]
    ) -> str:
        """Initialize a WandB run and return run ID"""
        if not self.config.enable_wandb:
            return "wandb_disabled"

        try:
            # Use offline mode - no network delays
            run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                tags=tags,
                config=config_dict,
                reinit=True,
                mode="offline",
            )
            self.runs[run_name] = run
            logger.info(f"WandB run initialized (offline): {run_name}")
            return run.id
        except Exception as e:
            logger.warning(f"Failed to initialize WandB run: {e}")
            return "wandb_failed"

    def log_metrics(self, run_name: str, metrics: Dict[str, Any]):
        """Log metrics to WandB (offline, no upload delay)"""
        if not self.config.enable_wandb or run_name not in self.runs:
            return

        try:
            # Filter out non-scalar metrics
            scalar_metrics = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "metadata"
            }
            if scalar_metrics:
                self.runs[run_name].log(scalar_metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to WandB: {e}")

    def finish_run(self, run_name: str):
        """Finish a WandB run and track for background sync"""
        if not self.config.enable_wandb or run_name not in self.runs:
            return

        try:
            run = self.runs[run_name]
            run_dir = run.dir  # Get the run directory before finishing
            run.finish()
            del self.runs[run_name]

            # Track for background sync
            self.completed_runs.append(run_dir)
            logger.info(f"WandB run completed (offline): {run_name}")
        except Exception as e:
            logger.warning(f"Failed to finish WandB run: {e}")

    def finish_all_runs(self):
        """Finish all remaining WandB runs"""
        for run_name in list(self.runs.keys()):
            self.finish_run(run_name)

    def start_background_sync(self):
        """Start background thread to sync offline runs to WandB"""
        if not self.config.enable_wandb or not self.completed_runs:
            return

        if not self.config.background_sync:
            logger.info(f"📤 {len(self.completed_runs)} WandB runs saved offline")
            logger.info("   To upload: uv run wandb sync wandb/offline-run-*")
            return

        def sync_worker():
            """Background worker to sync runs"""
            logger.info(
                f"📤 Starting background sync of {len(self.completed_runs)} WandB runs..."
            )

            import subprocess

            synced = 0
            failed = 0

            for run_dir in self.completed_runs:
                try:
                    # Use subprocess to run wandb sync via uv
                    result = subprocess.run(
                        ["uv", "run", "wandb", "sync", run_dir],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 min timeout per run
                    )
                    if result.returncode == 0:
                        synced += 1
                        logger.info(
                            f"✅ Synced {synced}/{len(self.completed_runs)}: {os.path.basename(run_dir)}"
                        )
                    else:
                        failed += 1
                        logger.warning(f"⚠️  Sync failed: {os.path.basename(run_dir)}")
                except Exception as e:
                    failed += 1
                    logger.warning(f"Error syncing {os.path.basename(run_dir)}: {e}")

            logger.info(
                f"📤 Background sync complete! {synced} synced, {failed} failed"
            )

        # Start background thread (daemon=False so it completes even if main exits)
        self.sync_thread = threading.Thread(target=sync_worker, daemon=False)
        self.sync_thread.start()
        logger.info("📤 Background WandB sync started (non-blocking)")
        logger.info("   Pipeline will exit immediately. Sync continues in background.")


class EvaluationPipeline:
    """Main evaluation pipeline"""

    def __init__(self, config: PipelineConfig, eval_dir: Path):
        self.config = config
        self.eval_dir = eval_dir
        self.models_dir = self.eval_dir / "models"
        self.detector = ModelDetector(self.models_dir)
        self.generator = TrajectoryGenerator(config)
        self.evaluator = TrajectoryEvaluator(config)
        self.wandb_manager = WandBManager(config)
        self.interrupted = False

        # Change to eval directory for all operations
        os.chdir(self.eval_dir)
        logger.info(f"Working directory: {self.eval_dir}")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Validate configuration
        self._validate_config()

        # Set random seed
        set_seed(config.seed)

        # Set PyTorch settings
        torch.set_num_threads(torch.get_num_threads())
        torch.backends.cudnn.benchmark = True

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("\n⚠️  Interrupt received. Cleaning up...")
        self.interrupted = True
        # Don't raise immediately - let the loop check self.interrupted

    def _validate_config(self):
        """Validate pipeline configuration"""
        logger.info("Validating pipeline configuration...")

        # Check models directory
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        # Check data directory
        # First check if data_dir is specified in config as absolute path
        if hasattr(self.config, "data_dir") and self.config.data_dir:
            data_dir = Path(self.config.data_dir)
            if not data_dir.is_absolute():
                # Relative to eval dir
                data_dir = self.eval_dir / data_dir
        else:
            # Default: ../data/{dataset} from eval dir
            data_dir = self.eval_dir.parent / "data" / self.config.dataset

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Resolve symlink if needed
        if data_dir.is_symlink():
            data_dir = data_dir.resolve()

        # Check required data files
        required_files = ["test.csv", "roadmap.geo"]
        for file in required_files:
            file_path = data_dir / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required data file not found: {file_path}")

        # Validate cross-dataset evaluation if configured
        if self.config.cross_dataset_eval:
            cross_data_dir = Path(self.config.cross_dataset_eval)
            if not cross_data_dir.is_absolute():
                # Relative to eval dir
                cross_data_dir = self.eval_dir / cross_data_dir

            if not cross_data_dir.exists():
                raise FileNotFoundError(
                    f"Cross-dataset directory not found: {cross_data_dir}"
                )

            # Resolve symlink if needed
            if cross_data_dir.is_symlink():
                cross_data_dir = cross_data_dir.resolve()

            # Check required cross-dataset files
            for file in required_files:
                file_path = cross_data_dir / file
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Required cross-dataset file not found: {file_path}"
                    )

            logger.info(
                f"✓ Cross-dataset evaluation enabled: {self.config.cross_dataset_name or cross_data_dir.name}"
            )

        # Check dataset config file (not required, just for validation)
        # Config is in the eval directory itself

        # Validate CUDA device
        if torch.cuda.is_available():
            if self.config.cuda_device >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {self.config.cuda_device} not available. Only {torch.cuda.device_count()} devices found."
                )
        else:
            logger.warning("CUDA not available, using CPU")

        # Validate parameters
        if self.config.num_gene <= 0:
            raise ValueError("num_gene must be positive")

        if self.config.beam_width <= 0:
            raise ValueError("beam_width must be positive")

        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be positive")

        if self.config.edr_eps <= 0:
            raise ValueError("edr_eps must be positive")

        logger.info("Configuration validation passed")

    def _check_existing_results(
        self, model_type: str, od_source: str
    ) -> Optional[Path]:
        """Check if results already exist and return path if found"""
        if self.config.force:
            return None

        # Check for existing generated file
        gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")
        generated_files = list(gene_dir.glob(f"*{model_type}*{od_source}od*.csv"))

        if generated_files:
            latest_file = max(generated_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found existing generated file: {latest_file}")
            return latest_file

        return None

    def _handle_error(
        self,
        error: Exception,
        context: str,
        model_type: str = None,
        od_source: str = None,
    ):
        """Handle errors with comprehensive logging"""
        error_msg = f"Error in {context}"
        if model_type:
            error_msg += f" for {model_type}"
        if od_source:
            error_msg += f" ({od_source} OD)"
        error_msg += f": {str(error)}"

        logger.error(error_msg)
        logger.error(f"Error type: {type(error).__name__}")

        # Log stack trace for debugging
        import traceback

        logger.error(f"Stack trace:\n{traceback.format_exc()}")

        # Continue execution unless it's a critical error
        if isinstance(error, (FileNotFoundError, ValueError, KeyError)):
            logger.error("Critical error encountered, stopping pipeline")
            raise
        else:
            logger.warning(
                "Non-critical error encountered, continuing with next operation"
            )

    def _build_file_to_model_mapping(self) -> Dict[str, tuple]:
        """Build mapping of generated file -> (model_type, od_source) from eval results"""
        import json

        mapping = {}
        eval_dir = Path("./eval")

        if not eval_dir.exists():
            logger.warning("No eval directory found, cannot build file mapping")
            return mapping

        # Scan all evaluation result directories
        for eval_subdir in eval_dir.iterdir():
            if not eval_subdir.is_dir():
                continue

            results_file = eval_subdir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results = json.load(f)

                    metadata = results.get("metadata", {})
                    generated_file = metadata.get("generated_file", "")
                    model_type = metadata.get("model_type", "")
                    od_source = metadata.get("od_source", "")

                    if generated_file and model_type and od_source:
                        # Normalize path to just the filename
                        file_path = Path(generated_file)
                        mapping[file_path.name] = (model_type, od_source)
                        logger.debug(
                            f"Mapped {file_path.name} -> {model_type} ({od_source} OD)"
                        )
                except Exception as e:
                    logger.warning(f"Failed to read {results_file}: {e}")

        logger.info(f"Built mapping for {len(mapping)} generated files")
        return mapping

    def _run_scenario_analysis(self):
        """Run scenario-based analysis on all generated trajectories"""
        from tools.analyze_scenarios import (
            run_scenario_analysis,
            run_cross_model_scenario_analysis,
        )

        # Auto-detect scenarios config if not provided
        if not self.config.scenarios_config:
            # Try config/scenarios_{dataset}.yaml first
            scenarios_config = Path(
                f"./config/scenarios_{self.config.dataset.lower()}.yaml"
            )
            if not scenarios_config.exists():
                # Fallback to config/scenarios.yaml
                scenarios_config = Path("./config/scenarios.yaml")

            if not scenarios_config.exists():
                logger.warning("No scenarios config found, skipping scenario analysis")
                return
        else:
            scenarios_config = Path(self.config.scenarios_config)

        # Copy config to eval directory for reproducibility
        config_copy = Path("./config") / scenarios_config.name
        config_copy.parent.mkdir(exist_ok=True)

        # Only copy if source and destination are different
        import shutil

        if scenarios_config.resolve() != config_copy.resolve():
            shutil.copy(scenarios_config, config_copy)
            logger.info(f"Copied scenarios config to {config_copy}")
        else:
            logger.info(f"Using existing scenarios config: {config_copy}")

        # Build mapping of generated files to models
        file_mapping = self._build_file_to_model_mapping()

        # Output directory for scenarios
        scenarios_output = Path("./scenarios")

        # Step 1: Run individual model analysis for each OD source
        logger.info("\n🎯 Step 1: Analyzing individual models...")
        for od_source in self.config.od_sources:
            logger.info(f"\nRunning scenario analysis for {od_source} OD...")

            try:
                # Find generated files for this OD source
                gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")
                if not gene_dir.exists():
                    logger.warning(f"Gene directory not found: {gene_dir}")
                    continue

                generated_files = list(gene_dir.glob("*.csv"))

                if not generated_files:
                    logger.warning(f"No generated files found in {gene_dir}")
                    continue

                # Filter by OD source using the mapping
                od_files = []
                for gen_file in generated_files:
                    if gen_file.name in file_mapping:
                        _, file_od = file_mapping[gen_file.name]
                        if file_od == od_source:
                            od_files.append(gen_file)

                if not od_files:
                    logger.warning(f"No generated files found for {od_source} OD")
                    continue

                logger.info(f"Found {len(od_files)} files for {od_source} OD")

                # Run analysis on each generated file (one per model)
                for gen_file in od_files:
                    # Get model name from mapping
                    if gen_file.name in file_mapping:
                        model_name, _ = file_mapping[gen_file.name]
                    else:
                        logger.warning(
                            f"No model mapping for {gen_file.name}, skipping"
                        )
                        continue

                    output_dir = scenarios_output / od_source / model_name

                    logger.info(f"  Analyzing {model_name} ({od_source} OD)...")

                    run_scenario_analysis(
                        generated_file=gen_file,
                        dataset=self.config.dataset,
                        od_source=od_source,
                        config_path=config_copy,
                        output_dir=output_dir,
                        model_name=model_name,
                    )

                    logger.info(f"  ✅ Results saved to {output_dir}")

            except Exception as e:
                logger.error(f"Scenario analysis failed for {od_source} OD: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Step 2: Run cross-model aggregation analysis
        logger.info("\n🔄 Step 2: Running cross-model scenario aggregation...")
        try:
            run_cross_model_scenario_analysis(
                eval_dir=self.eval_dir,
                config_path=config_copy,
                od_sources=self.config.od_sources,
            )
            logger.info("✅ Cross-model analysis complete")
        except Exception as e:
            logger.error(f"Cross-model analysis failed: {e}")
            import traceback

            traceback.print_exc()

        logger.info(f"\n✅ Scenario analysis complete! Results in {scenarios_output}/")

    def _run_cross_dataset_evaluation(self):
        """Run cross-dataset evaluation on all trained models"""
        from tools.analyze_scenarios import run_scenario_analysis

        logger.info("\n🌐 Starting cross-dataset evaluation...")
        logger.info(
            f"Evaluating models trained on {self.config.dataset} using {self.config.cross_dataset_name} data"
        )

        # Setup cross-dataset paths
        cross_data_dir = Path(self.config.cross_dataset_eval)
        if not cross_data_dir.is_absolute():
            cross_data_dir = self.eval_dir / cross_data_dir

        # Output directory structure: cross_dataset_eval/{cross_dataset_name}/{od_source}/{model}/
        cross_eval_output = (
            Path("./cross_dataset_eval") / self.config.cross_dataset_name
        )

        # Get scenarios config for cross-dataset
        scenarios_config = None
        if self.config.run_scenarios:
            # Try config/scenarios_{cross_dataset_name}.yaml
            config_path = Path(
                f"./config/scenarios_{self.config.cross_dataset_name.lower()}.yaml"
            )
            if config_path.exists():
                scenarios_config = config_path
                logger.info(f"Using cross-dataset scenarios config: {scenarios_config}")

        # Loop through all models and OD sources
        for od_source in self.config.od_sources:
            logger.info(f"\n📊 Processing {od_source} OD for cross-dataset...")

            for model_file in self.models_dir.iterdir():
                if not model_file.suffix == ".pth":
                    continue

                model_type = self._extract_model_from_filename(model_file.name)
                logger.info(f"  Model: {model_type}")

                try:
                    # Step 1: Generate trajectories on cross-dataset
                    logger.info("  🔄 Generating trajectories on cross-dataset...")
                    generated_file, gen_perf = self.generator.generate_trajectories(
                        model_file, model_type, od_source
                    )

                    # Move generated file to cross-dataset directory
                    cross_gene_dir = cross_eval_output / od_source / model_type / "gene"
                    cross_gene_dir.mkdir(parents=True, exist_ok=True)
                    cross_gen_file = cross_gene_dir / generated_file.name

                    import shutil

                    shutil.copy(generated_file, cross_gen_file)
                    logger.info(f"  💾 Generated: {cross_gen_file}")

                    # Step 2: Evaluate on cross-dataset
                    logger.info("  📈 Evaluating on cross-dataset...")
                    eval_results = evaluate_trajectories_programmatic(
                        generated_file=str(cross_gen_file),
                        dataset=self.config.dataset,
                        od_source=od_source,
                        grid_size=self.config.grid_size,
                        edr_eps=self.config.edr_eps,
                        enable_wandb=self.config.enable_wandb,
                        wandb_project=self.config.wandb_project,
                        wandb_run_name=f"cross-{self.config.cross_dataset_name}-{model_type}-{od_source}",
                        wandb_tags=[
                            "cross-dataset",
                            self.config.cross_dataset_name,
                            model_type,
                            od_source,
                        ],
                        generation_performance=gen_perf,
                        cross_dataset=True,
                        cross_dataset_name=self.config.cross_dataset_name,
                        trained_on_dataset=self.config.dataset,
                    )

                    # Save evaluation results
                    eval_output_dir = (
                        cross_eval_output / od_source / model_type / "eval"
                    )
                    eval_output_dir.mkdir(parents=True, exist_ok=True)

                    import json
                    from datetime import datetime

                    results_file = (
                        eval_output_dir
                        / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    with open(results_file, "w") as f:
                        json.dump(eval_results, f, indent=2)
                    logger.info(f"  💾 Evaluation results: {results_file}")

                    # Step 3: Run scenario analysis if configured
                    if scenarios_config:
                        logger.info("  🎯 Running scenario analysis...")
                        scenario_output = (
                            cross_eval_output / od_source / model_type / "scenarios"
                        )

                        run_scenario_analysis(
                            generated_file=cross_gen_file,
                            dataset=self.config.cross_dataset_name,
                            od_source=od_source,
                            config_path=scenarios_config,
                            output_dir=scenario_output,
                            model_name=model_type,
                            cross_dataset=True,
                            trained_on_dataset=self.config.dataset,
                        )
                        logger.info(f"  ✅ Scenarios: {scenario_output}")

                except Exception as e:
                    logger.error(
                        f"  ❌ Cross-dataset evaluation failed for {model_type} ({od_source}): {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        logger.info(
            f"\n✅ Cross-dataset evaluation complete! Results in {cross_eval_output}/"
        )

    def _run_abnormal_detection_analysis(self):
        """Run abnormal trajectory detection and model performance analysis"""
        from tools.analyze_abnormal import run_abnormal_analysis

        logger.info("\n🔍 Starting abnormal trajectory detection analysis...")

        # Auto-detect abnormal config if not provided
        if not self.config.abnormal_config:
            abnormal_config = Path("./config/abnormal_detection.yaml")
            if not abnormal_config.exists():
                logger.warning(
                    "No abnormal detection config found, skipping abnormal analysis"
                )
                return
        else:
            abnormal_config = Path(self.config.abnormal_config)

        if not abnormal_config.exists():
            logger.error(f"Abnormal detection config not found: {abnormal_config}")
            return

        # Determine which dataset to analyze
        # If cross_dataset_eval is configured, analyze the cross-dataset
        # Otherwise analyze the main dataset
        if self.config.cross_dataset_eval and self.config.cross_dataset_name:
            analysis_dataset = self.config.cross_dataset_name
            data_dir = Path(self.config.cross_dataset_eval)
            if not data_dir.is_absolute():
                data_dir = self.eval_dir / data_dir
        else:
            analysis_dataset = self.config.dataset
            data_dir = self.eval_dir.parent / "data" / self.config.dataset

        logger.info(f"Analyzing dataset: {analysis_dataset}")

        # Output directory for abnormal analysis
        abnormal_output = Path("./abnormal") / analysis_dataset

        # Step 1: Run abnormal detection on real data
        logger.info("\n🔍 Step 1: Detecting abnormal trajectories in real data...")
        for od_source in self.config.od_sources:
            real_file = data_dir / f"{od_source}.csv"
            if not real_file.exists():
                logger.warning(f"Real data file not found: {real_file}, skipping")
                continue

            detection_output = abnormal_output / od_source / "detection"
            logger.info(f"  Processing {od_source} data...")

            try:
                detection_results = run_abnormal_analysis(
                    real_file=real_file,
                    dataset=analysis_dataset,
                    config_path=abnormal_config,
                    output_dir=detection_output,
                )

                logger.info(
                    f"  ✅ Detection complete: {len(detection_results.get('abnormal_indices', {}))} abnormal categories found"
                )

                # Step 2: Evaluate models on abnormal trajectories
                logger.info(
                    f"\n📈 Step 2: Evaluating models on abnormal trajectories ({od_source})..."
                )

                # Get abnormal trajectory indices by category
                abnormal_indices = detection_results.get("abnormal_indices", {})
                all_abnormal_traj_ids = set()
                for category, indices in abnormal_indices.items():
                    all_abnormal_traj_ids.update(indices)

                if not all_abnormal_traj_ids:
                    logger.info(
                        "  No abnormal trajectories found, skipping model evaluation"
                    )
                    continue

                logger.info(
                    f"  Found {len(all_abnormal_traj_ids)} total abnormal trajectories"
                )

                # Evaluate each model on abnormal trajectories
                for model_file in self.models_dir.iterdir():
                    if not model_file.suffix == ".pth":
                        continue

                    model_type = self._extract_model_from_filename(model_file.name)
                    logger.info(f"    Evaluating {model_type}...")

                    # Find the generated trajectories for this model
                    gene_dir = Path(
                        f"./gene/{self.config.dataset}/seed{self.config.seed}"
                    )
                    if not gene_dir.exists():
                        logger.warning(f"    Gene directory not found: {gene_dir}")
                        continue

                    # Find generated file for this model and OD source
                    generated_files = list(
                        gene_dir.glob(f"*{model_type}*{od_source}*.csv")
                    )
                    if not generated_files:
                        logger.warning(
                            f"    No generated files found for {model_type} {od_source}"
                        )
                        continue

                    generated_file = generated_files[0]

                    # Calculate metrics for abnormal trajectories
                    # Note: This is a simplified version - full implementation would
                    # filter to abnormal OD pairs and compute metrics
                    model_output = (
                        abnormal_output / od_source / "model_performance" / model_type
                    )
                    model_output.mkdir(parents=True, exist_ok=True)

                    # Save model evaluation metadata
                    import json
                    from datetime import datetime

                    model_eval = {
                        "model": model_type,
                        "dataset": analysis_dataset,
                        "od_source": od_source,
                        "generated_file": str(generated_file),
                        "abnormal_trajectory_count": len(all_abnormal_traj_ids),
                        "abnormal_categories": {
                            cat: len(indices)
                            for cat, indices in abnormal_indices.items()
                        },
                        "timestamp": datetime.now().isoformat(),
                    }

                    with open(model_output / "abnormal_eval_metadata.json", "w") as f:
                        json.dump(model_eval, f, indent=2)

                    logger.info(f"    ✅ Saved evaluation metadata to {model_output}/")

            except Exception as e:
                logger.error(f"  ❌ Abnormal analysis failed for {od_source}: {e}")
                import traceback

                traceback.print_exc()
                continue

        logger.info(
            f"\n✅ Abnormal trajectory detection complete! Results in {abnormal_output}/"
        )

    def _extract_model_from_filename(self, filename: str) -> str:
        """Extract model type from generated file name"""
        # Example: hoser_vanilla_testod_gene_20241024_123456.csv -> vanilla
        if "vanilla" in filename:
            return "vanilla"
        elif "distilled" in filename:
            return "distilled"
        else:
            return "unknown"

    @phase("base_eval", critical=True)
    def run_base_eval(self):
        """Evaluate on base dataset (Beijing)"""
        logger.info("📊 Evaluating on base dataset...")

        results_summary = {}
        failed_operations = []

        for model_type in self.config.models:
            for od_source in self.config.od_sources:
                # Check for interruption
                if self.interrupted:
                    logger.info("Pipeline interrupted by user")
                    raise KeyboardInterrupt()

                logger.info(f"Evaluating {model_type} ({od_source} OD)")

                # Find generated file
                gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")

                # Try new naming pattern first: {timestamp}_{model_type}_{od_source}.csv
                pattern = f"*_{model_type}_{od_source}.csv"
                generated_files = list(gene_dir.glob(pattern))

                if not generated_files:
                    # Fallback: try old timestamp-only pattern for backward compatibility
                    generated_files = list(gene_dir.glob("*.csv"))
                    if not generated_files:
                        error_msg = f"No existing generated file found for {model_type} ({od_source} OD)"
                        logger.error(error_msg)
                        failed_operations.append(error_msg)
                        continue
                    # Use latest if multiple matches
                    generated_file = max(
                        generated_files, key=lambda x: x.stat().st_mtime
                    )
                    logger.warning(
                        f"Using legacy filename format: {generated_file.name}"
                    )
                else:
                    # Use most recent file matching pattern
                    generated_file = max(
                        generated_files, key=lambda x: x.stat().st_mtime
                    )
                    logger.info(f"Found existing file: {generated_file.name}")

                # Evaluation phase
                try:
                    eval_results = self.evaluator.evaluate_trajectories(
                        generated_file,
                        model_type,
                        od_source,
                        None,  # generation_performance not available in standalone eval
                    )

                    # Log to WandB
                    if self.config.enable_wandb:
                        run_name = (
                            f"eval_{model_type}_seed{self.config.seed}_{od_source}od"
                        )
                        tags = [
                            model_type,
                            f"seed{self.config.seed}",
                            f"{od_source}_od",
                            "evaluation",
                        ]
                        config_dict = {
                            "dataset": self.config.dataset,
                            "seed": self.config.seed,
                            "model_type": model_type,
                            "od_source": od_source,
                            "generated_file": str(generated_file),
                        }

                        self.wandb_manager.init_run(run_name, tags, config_dict)
                        self.wandb_manager.log_metrics(run_name, eval_results)
                        self.wandb_manager.finish_run(run_name)

                    # Store results for summary
                    key = f"{model_type}_{od_source}"
                    results_summary[key] = {
                        "generated_file": str(generated_file),
                        "metrics": eval_results,
                    }

                    logger.info(
                        f"✅ Evaluation complete for {model_type} ({od_source} OD)"
                    )

                except Exception as e:
                    self._handle_error(e, "evaluation", model_type, od_source)
                    failed_operations.append(
                        f"Evaluation failed for {model_type} ({od_source} OD): {str(e)}"
                    )
                    continue

        # Log summary
        if results_summary:
            logger.info("\n=== Base Dataset Evaluation Results ===")
            for key, result in results_summary.items():
                logger.info(f"{key}: {result['metrics']}")

        if failed_operations:
            logger.warning(f"\n⚠️  {len(failed_operations)} operations failed:")
            for op in failed_operations:
                logger.warning(f"  - {op}")

        logger.info("✅ Base dataset evaluation complete!")
        return results_summary

    def run(self):
        """Run the complete evaluation pipeline"""
        logger.info("Starting HOSER Distillation Evaluation Pipeline")
        logger.info(f"Configuration: {self.config.__dict__}")

        try:
            # Auto-detect models if not specified
            if not self.config.models:
                self.config.models = self.detector.detect_models()

            logger.info(f"Models to process: {self.config.models}")
            logger.info(f"OD sources: {self.config.od_sources}")

            # Calculate total operations (only count enabled phases)
            operations_per_combo = 0
            if not self.config.skip_gene:
                operations_per_combo += 1  # Generation
            if not self.config.skip_eval:
                operations_per_combo += 1  # Evaluation

            total_operations = (
                len(self.config.models)
                * len(self.config.od_sources)
                * operations_per_combo
            )
            current_operation = 0

            results_summary = {}
            failed_operations = []

            for model_type in self.config.models:
                try:
                    model_file = self.detector.find_model_file(model_type)
                    if not model_file:
                        error_msg = f"Model file not found for type: {model_type}"
                        logger.error(error_msg)
                        failed_operations.append(error_msg)
                        continue

                    logger.info(f"Processing model: {model_type} ({model_file.name})")

                    for od_source in self.config.od_sources:
                        # Check for interruption
                        if self.interrupted:
                            logger.info("Pipeline interrupted by user")
                            raise KeyboardInterrupt()

                        current_operation += 1
                        logger.info(
                            f"[{current_operation}/{total_operations}] Processing {model_type} ({od_source} OD)"
                        )

                        # Check for existing results
                        existing_file = self._check_existing_results(
                            model_type, od_source
                        )

                        # Generation phase
                        generation_performance = (
                            None  # Store performance metrics for evaluation
                        )
                        if not self.config.skip_gene:
                            try:
                                if existing_file:
                                    generated_file = existing_file
                                    generation_performance = (
                                        None  # No metrics for existing files
                                    )
                                    logger.info(
                                        f"Using existing generated file: {generated_file}"
                                    )
                                else:
                                    generated_file, generation_performance = (
                                        self.generator.generate_trajectories(
                                            model_file, model_type, od_source
                                        )
                                    )

                                # Log to WandB
                                if self.config.enable_wandb:
                                    run_name = f"gene_{model_type}_seed{self.config.seed}_{od_source}od"
                                    tags = [
                                        model_type,
                                        f"seed{self.config.seed}",
                                        f"{od_source}_od",
                                        "generation",
                                        "beam4",
                                    ]
                                    config_dict = {
                                        "dataset": self.config.dataset,
                                        "seed": self.config.seed,
                                        "num_gene": self.config.num_gene,
                                        "model_type": model_type,
                                        "od_source": od_source,
                                        "beam_width": self.config.beam_width,
                                    }

                                    self.wandb_manager.init_run(
                                        run_name, tags, config_dict
                                    )

                                    # Log generation metrics including performance
                                    log_data = {
                                        "num_trajectories_generated": self.config.num_gene,
                                        "output_file": str(generated_file),
                                    }
                                    if generation_performance:
                                        # Add performance metrics with 'perf/' prefix
                                        perf_log = {
                                            f"perf/{k}": v
                                            for k, v in generation_performance.items()
                                            if isinstance(v, (int, float))
                                        }
                                        log_data.update(perf_log)

                                    self.wandb_manager.log_metrics(run_name, log_data)
                                    self.wandb_manager.finish_run(run_name)

                            except Exception as e:
                                self._handle_error(
                                    e, "generation", model_type, od_source
                                )
                                failed_operations.append(
                                    f"Generation failed for {model_type} ({od_source} OD): {str(e)}"
                                )
                                continue
                        else:
                            # Find existing generated file by pattern
                            if existing_file:
                                generated_file = existing_file
                            else:
                                gene_dir = Path(
                                    f"./gene/{self.config.dataset}/seed{self.config.seed}"
                                )

                                # Try new naming pattern first: {timestamp}_{model_type}_{od_source}.csv
                                pattern = f"*_{model_type}_{od_source}.csv"
                                generated_files = list(gene_dir.glob(pattern))

                                if not generated_files:
                                    # Fallback: try old timestamp-only pattern for backward compatibility
                                    generated_files = list(gene_dir.glob("*.csv"))
                                    if not generated_files:
                                        error_msg = f"No existing generated file found for {model_type} ({od_source} OD)"
                                        logger.error(error_msg)
                                        failed_operations.append(error_msg)
                                        continue
                                    # Use latest if multiple matches
                                    generated_file = max(
                                        generated_files, key=lambda x: x.stat().st_mtime
                                    )
                                    logger.warning(
                                        f"Using legacy filename format: {generated_file.name}"
                                    )
                                else:
                                    # Use most recent file matching pattern
                                    generated_file = max(
                                        generated_files, key=lambda x: x.stat().st_mtime
                                    )
                                    logger.info(
                                        f"Found existing file: {generated_file.name}"
                                    )

                        # Evaluation phase
                        if not self.config.skip_eval:
                            try:
                                eval_results = self.evaluator.evaluate_trajectories(
                                    generated_file,
                                    model_type,
                                    od_source,
                                    generation_performance,
                                )

                                # Log to WandB
                                if self.config.enable_wandb:
                                    run_name = f"eval_{model_type}_seed{self.config.seed}_{od_source}od"
                                    tags = [
                                        model_type,
                                        f"seed{self.config.seed}",
                                        f"{od_source}_od",
                                        "evaluation",
                                    ]
                                    config_dict = {
                                        "dataset": self.config.dataset,
                                        "seed": self.config.seed,
                                        "model_type": model_type,
                                        "od_source": od_source,
                                        "generated_file": str(generated_file),
                                    }

                                    self.wandb_manager.init_run(
                                        run_name, tags, config_dict
                                    )
                                    self.wandb_manager.log_metrics(
                                        run_name, eval_results
                                    )
                                    self.wandb_manager.finish_run(run_name)

                                # Store results for summary
                                key = f"{model_type}_{od_source}"
                                results_summary[key] = {
                                    "generated_file": str(generated_file),
                                    "metrics": eval_results,
                                }

                            except Exception as e:
                                self._handle_error(
                                    e, "evaluation", model_type, od_source
                                )
                                failed_operations.append(
                                    f"Evaluation failed for {model_type} ({od_source} OD): {str(e)}"
                                )
                                continue

                except Exception as e:
                    self._handle_error(e, f"model processing for {model_type}")
                    failed_operations.append(
                        f"Model processing failed for {model_type}: {str(e)}"
                    )
                    continue

            # Print summary
            logger.info("Pipeline completed!")
            logger.info(f"Successful operations: {len(results_summary)}")
            logger.info(f"Failed operations: {len(failed_operations)}")

            if results_summary:
                logger.info("Results summary:")
                for key, result in results_summary.items():
                    logger.info(f"  {key}: {result['generated_file']}")
                    metrics = result["metrics"]
                    for metric, value in metrics.items():
                        if isinstance(value, float) and metric != "metadata":
                            logger.info(f"    {metric}: {value:.4f}")

            if failed_operations:
                logger.warning("Failed operations:")
                for failure in failed_operations:
                    logger.warning(f"  - {failure}")

            # NEW: Optional scenario analysis
            if self.config.run_scenarios:
                logger.info("Starting scenario analysis...")
                try:
                    self._run_scenario_analysis()
                except Exception as e:
                    logger.error(f"Scenario analysis failed: {e}")
                    # Don't fail entire pipeline if scenarios fail

            # NEW: Optional cross-dataset evaluation
            if self.config.cross_dataset_eval:
                logger.info("Starting cross-dataset evaluation...")
                try:
                    self._run_cross_dataset_evaluation()
                except Exception as e:
                    logger.error(f"Cross-dataset evaluation failed: {e}")
                    # Don't fail entire pipeline if cross-dataset fails

            # NEW: Optional abnormal trajectory detection
            if self.config.run_abnormal_detection:
                logger.info("Starting abnormal trajectory detection...")
                try:
                    self._run_abnormal_detection_analysis()
                except Exception as e:
                    logger.error(f"Abnormal trajectory detection failed: {e}")
                    # Don't fail entire pipeline if abnormal detection fails

            # Return success status
            return len(failed_operations) == 0

        except Exception as e:
            logger.error(f"Pipeline failed with critical error: {str(e)}")
            import traceback

            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
        finally:
            # Cleanup: finish all WandB runs
            try:
                self.wandb_manager.finish_all_runs()

                # Start background sync (non-blocking)
                self.wandb_manager.start_background_sync()
            except Exception as e:
                logger.warning(f"Error during WandB cleanup: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="HOSER Evaluation Pipeline")
    parser.add_argument(
        "--eval-dir", type=str, help="Evaluation directory (default: current directory)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (default: config/evaluation.yaml in eval dir)",
    )
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to run (overrides config)",
    )
    parser.add_argument(
        "--od-source", type=str, help="OD sources: train,test (overrides config)"
    )
    parser.add_argument(
        "--skip-gene", action="store_true", help="Skip generation (overrides config)"
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation (overrides config)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-run (overrides config)"
    )
    parser.add_argument("--cuda", type=int, help="CUDA device (overrides config)")
    parser.add_argument(
        "--num-gene", type=int, help="Number of trajectories (overrides config)"
    )
    parser.add_argument(
        "--wandb-project", type=str, help="WandB project (overrides config)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging (overrides config)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output (overrides config)"
    )
    parser.add_argument(
        "--run-scenarios",
        action="store_true",
        help="Run scenario analysis after evaluation (optional)",
    )
    parser.add_argument(
        "--scenarios-config",
        type=str,
        help="Path to scenarios config YAML (default: auto-detect from config/)",
    )
    parser.add_argument(
        "--cross-dataset",
        type=str,
        help="Path to cross-dataset for evaluation (e.g., ../data/BJUT_Beijing)",
    )
    parser.add_argument(
        "--cross-dataset-name",
        type=str,
        default="BJUT_Beijing",
        help="Name of the cross-dataset (default: BJUT_Beijing)",
    )
    parser.add_argument(
        "--run-abnormal",
        action="store_true",
        help="Run abnormal trajectory detection analysis",
    )
    parser.add_argument(
        "--abnormal-config",
        type=str,
        help="Path to abnormal detection config YAML (default: config/abnormal_detection.yaml)",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine evaluation directory
    if args.eval_dir:
        eval_dir = Path(args.eval_dir).resolve()
    else:
        # Use current directory if it looks like an eval dir
        current_dir = Path.cwd()
        if (current_dir / "models").exists() and (current_dir / "config").exists():
            eval_dir = current_dir
        else:
            parser.error(
                "No --eval-dir specified and current directory doesn't appear to be an evaluation directory"
            )

    if not eval_dir.exists():
        parser.error(f"Evaluation directory not found: {eval_dir}")

    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = eval_dir / "config" / "evaluation.yaml"
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")

    # Create configuration from YAML file
    config = PipelineConfig(str(config_path), eval_dir)

    # Override with command line arguments if provided
    if args.seed is not None:
        config.seed = args.seed
    if args.models is not None:
        config.models = args.models.split(",")
    if args.od_source is not None:
        config.od_sources = args.od_source.split(",")
    if args.skip_gene:
        config.skip_gene = True
    if args.skip_eval:
        config.skip_eval = True
    if args.force:
        config.force = True
    if args.cuda is not None:
        config.cuda_device = args.cuda
    if args.num_gene is not None:
        config.num_gene = args.num_gene
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.no_wandb:
        config.enable_wandb = False
    if args.verbose:
        config.verbose = True
    if args.run_scenarios:
        config.run_scenarios = True
    if args.scenarios_config:
        config.scenarios_config = args.scenarios_config
    if args.cross_dataset:
        config.cross_dataset_eval = args.cross_dataset
    if args.cross_dataset_name:
        config.cross_dataset_name = args.cross_dataset_name
    if args.run_abnormal:
        config.run_abnormal_detection = True
    if args.abnormal_config:
        config.abnormal_config = args.abnormal_config

    # Run pipeline
    try:
        pipeline = EvaluationPipeline(config, eval_dir)
        success = pipeline.run()

        if success:
            logger.info("✅ Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.warning("⚠️  Pipeline completed with some failures")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n🛑 Pipeline interrupted by user")
        logger.info(
            "   Partial results saved offline. Background sync will upload them."
        )
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
