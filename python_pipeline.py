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
            "paired_analysis",
            "cross_dataset",
            "road_network_translate",
            "abnormal",
            "abnormal_od_extract",
            "abnormal_od_generate",
            "abnormal_od_evaluate",
            "scenarios",
        }

        # Other settings
        self.force = False
        self.enable_wandb = True
        self.verbose = False
        self.beam_width = 4
        self.beam_search = True  # Use beam search by default (set False for A* search)
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
            elif key == "phases":
                # Ensure phases is always a set (YAML may load as list)
                self.phases = set(value) if isinstance(value, (list, set)) else value
            elif hasattr(self, key):
                setattr(self, key, value)


class ModelDetector:
    """Auto-detect models in the models directory"""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    @staticmethod
    def _extract_model_type_from_filename(filename: str) -> str:
        """Robust model type extraction supporting multiple naming patterns"""
        stem = filename.replace(".pth", "")

        # Pattern 1: _25epoch_seed (old naming convention)
        if "_25epoch_seed" in stem:
            base = stem.split("_25epoch_seed")[0]
            seed = stem.split("_seed")[-1]
            return base if seed == "42" else f"{base}_seed{seed}"

        # Pattern 2: _phase{N}_seed (new phase-based naming)
        elif "_phase" in stem and "_seed" in stem:
            base_with_phase = stem.split("_seed")[0]
            seed = stem.split("_seed")[1]
            return base_with_phase if seed == "42" else f"{base_with_phase}_seed{seed}"

        # Pattern 3: Fallback for simple names
        else:
            return stem.split("_")[0]

    def detect_models(self) -> List[str]:
        """Detect all available models and return unique model types"""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        model_files = list(self.models_dir.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No .pth files found in {self.models_dir}")

        model_types = []
        for model_file in model_files:
            model_type = self._extract_model_type_from_filename(model_file.name)
            model_types.append(model_type)

        # Remove duplicates and sort
        unique_types = sorted(list(set(model_types)))
        logger.info(f"Detected models: {unique_types}")
        return unique_types

    def find_model_file(self, model_type: str) -> Optional[Path]:
        """Find the actual model file for a given model type"""
        for model_file in self.models_dir.glob("*.pth"):
            extracted_type = self._extract_model_type_from_filename(model_file.name)

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
            beam_search=self.config.beam_search,
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
        """Check if A* generated file exists and return path if found.

        Only returns files that are confirmed to be A* search (beam_search_enabled: false).
        If only beam search files exist, returns None to trigger A* generation.
        """
        if self.config.force:
            return None

        # Check for existing generated file
        gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")

        # Try multiple patterns to support both Porto and Beijing naming conventions
        patterns = [
            f"*{model_type}*{od_source}.csv",  # Primary: Porto and most Beijing
            f"*{model_type}*{od_source}od*.csv",  # Backward compatibility
            f"{model_type}*{od_source}*.csv",  # Beijing: model_type first
        ]

        generated_files = []
        for pattern in patterns:
            generated_files.extend(gene_dir.glob(pattern))

        if not generated_files:
            return None

        # Check each file to see if it's A* (beam_search_enabled: false)
        # Sort by modification time, newest first
        generated_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        import json

        for csv_file in generated_files:
            # Check for corresponding perf.json file
            perf_file = csv_file.parent / f"{csv_file.stem}_perf.json"

            if perf_file.exists():
                try:
                    with open(perf_file) as f:
                        perf_data = json.load(f)

                    # Check if this is an A* file (beam_search_enabled: false)
                    beam_search_enabled = perf_data.get("beam_search_enabled", True)

                    if not beam_search_enabled:
                        logger.info(
                            f"Found existing A* generated file: {csv_file.name}"
                        )
                        return csv_file
                    else:
                        logger.debug(
                            f"Skipping beam search file: {csv_file.name} "
                            f"(looking for A* file)"
                        )
                        continue
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        f"Could not parse perf.json for {csv_file.name}: {e}. "
                        f"Assuming beam search, skipping."
                    )
                    continue
            else:
                # No perf.json file - assume it's an old file or beam search
                # Skip it and continue looking
                logger.debug(
                    f"No perf.json found for {csv_file.name}, skipping "
                    f"(looking for A* file)"
                )
                continue

        # No A* files found, only beam search files exist
        logger.info(
            f"No A* generated file found for {model_type} ({od_source} OD). "
            f"Will generate with A* search."
        )
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

        # Analyze BOTH datasets: main evaluation dataset AND cross-dataset
        datasets_to_analyze = []

        # Always analyze the main evaluation dataset (Beijing/HOSER)
        main_data_dir = self.eval_dir.parent / "data" / self.config.dataset
        if main_data_dir.exists():
            datasets_to_analyze.append(
                {
                    "name": self.config.dataset,
                    "data_dir": main_data_dir,
                    "is_main": True,
                }
            )

        # Also analyze cross-dataset if configured (BJUT_Beijing)
        if self.config.cross_dataset_eval and self.config.cross_dataset_name:
            cross_data_dir = Path(self.config.cross_dataset_eval)
            if not cross_data_dir.is_absolute():
                cross_data_dir = self.eval_dir / cross_data_dir

            if cross_data_dir.exists():
                datasets_to_analyze.append(
                    {
                        "name": self.config.cross_dataset_name,
                        "data_dir": cross_data_dir,
                        "is_main": False,
                    }
                )

        if not datasets_to_analyze:
            logger.error("No datasets found for abnormal analysis")
            return

        logger.info(
            f"Will analyze {len(datasets_to_analyze)} dataset(s): {[d['name'] for d in datasets_to_analyze]}"
        )

        # Analyze each dataset
        for dataset_info in datasets_to_analyze:
            analysis_dataset = dataset_info["name"]
            data_dir = dataset_info["data_dir"]
            is_main = dataset_info["is_main"]

            logger.info(f"\n{'=' * 70}")
            logger.info(
                f"📊 Analyzing dataset: {analysis_dataset} ({'main' if is_main else 'cross-dataset'})"
            )
            logger.info(f"{'=' * 70}")

            # Output directory for abnormal analysis
            abnormal_output = Path("./abnormal") / analysis_dataset

            # Step 1: Run abnormal detection on real data
            logger.info("\n🔍 Step 1: Detecting abnormal trajectories in real data...")
            self._analyze_dataset_abnormalities(
                analysis_dataset, data_dir, abnormal_config, abnormal_output
            )

    def _get_generated_file_for_analysis(
        self, model_type: str, od_source: str, analysis_dataset: str
    ) -> Optional[Path]:
        """Get appropriate generated file (translated if cross-dataset, original otherwise)

        Args:
            model_type: Model type identifier
            od_source: train or test
            analysis_dataset: Dataset being analyzed (Beijing, BJUT_Beijing, Porto, etc.)

        Returns:
            Path to generated file, or None if not found
        """
        # Determine if this is cross-dataset analysis
        is_cross_dataset = analysis_dataset != self.config.dataset

        if is_cross_dataset:
            # Check for translated files
            translated_dir = Path(
                f"./gene_translated/{analysis_dataset}/seed{self.config.seed}"
            )

            if translated_dir.exists():
                translated_files = list(
                    translated_dir.glob(f"*{model_type}*{od_source}*.csv")
                )
                if translated_files:
                    logger.info(
                        f"      ✅ Using translated file for cross-dataset: {translated_files[0].name}"
                    )
                    return translated_files[0]
                else:
                    logger.warning(
                        f"      ⚠️  No translated file for {model_type}/{od_source}. "
                        f"Run 'road_network_translate' phase first!"
                    )
                    return None
            else:
                logger.warning(
                    f"      ⚠️  Translated directory not found: {translated_dir}. "
                    f"Cross-dataset analysis will use original files (may cause ID mismatch!)"
                )
                # Fall through to original files with warning

        # Same-dataset analysis OR fallback: use original generated files
        gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")
        if not gene_dir.exists():
            logger.error(f"      ❌ Gene directory not found: {gene_dir}")
            return None

        generated_files = list(gene_dir.glob(f"*{model_type}*{od_source}*.csv"))
        if not generated_files:
            logger.warning(f"      ⚠️  No generated file for {model_type}/{od_source}")
            return None

        if not is_cross_dataset:
            logger.debug(
                f"      Using original file (same dataset): {generated_files[0].name}"
            )

        return generated_files[0]

    def _analyze_dataset_abnormalities(
        self, dataset_name: str, data_dir: Path, config_path: Path, output_dir: Path
    ):
        """Analyze abnormal trajectories for a single dataset (real + generated)"""
        from tools.analyze_abnormal import run_abnormal_analysis

        import json
        from datetime import datetime

        # Step 1: Detect abnormal trajectories in real data
        for od_source in self.config.od_sources:
            real_file = data_dir / f"{od_source}.csv"
            if not real_file.exists():
                logger.warning(f"Real data file not found: {real_file}, skipping")
                continue

            logger.info(f"\n  📂 Processing {od_source} data...")

            # 1A: Run detection on real data
            detection_output_real = output_dir / od_source / "real_data"
            logger.info("    🔍 Detecting abnormal trajectories in REAL data...")

            try:
                real_results = run_abnormal_analysis(
                    real_file=real_file,
                    dataset=dataset_name,
                    config_path=config_path,
                    output_dir=detection_output_real,
                )

                # Count UNIQUE abnormal trajectories (not sum of categories to avoid double-counting)
                abnormal_indices = real_results.get("abnormal_indices", {})
                unique_abnormal_ids = set()
                for indices in abnormal_indices.values():
                    unique_abnormal_ids.update(indices)
                real_abnormal_count = len(unique_abnormal_ids)
                real_total = real_results.get("total_trajectories", 0)
                real_rate = (
                    (real_abnormal_count / real_total * 100) if real_total > 0 else 0
                )

                logger.info(
                    f"    ✅ Real data: {real_abnormal_count}/{real_total} ({real_rate:.2f}%) abnormal"
                )

                # Step 2: Detect abnormal trajectories in GENERATED data
                logger.info(
                    "\n    🤖 Detecting abnormal trajectories in GENERATED data..."
                )

                # Store results for comparison
                model_results = {}

                for model_type in self.config.models:
                    # Get appropriate file (translated for cross-dataset, original otherwise)
                    generated_file = self._get_generated_file_for_analysis(
                        model_type=model_type,
                        od_source=od_source,
                        analysis_dataset=dataset_name,
                    )

                    if not generated_file:
                        # Helper already logged appropriate warning
                        continue

                    logger.info(f"      Analyzing {model_type}...")

                    # Run detection on generated trajectories
                    detection_output_gen = (
                        output_dir / od_source / "generated" / model_type
                    )

                    try:
                        gen_results = run_abnormal_analysis(
                            real_file=generated_file,
                            dataset=dataset_name,
                            config_path=config_path,
                            output_dir=detection_output_gen,
                            is_real_data=False,  # Generated data format
                        )

                        # Count UNIQUE abnormal trajectories (not sum of categories to avoid double-counting)
                        abnormal_indices = gen_results.get("abnormal_indices", {})
                        unique_abnormal_ids = set()
                        for indices in abnormal_indices.values():
                            unique_abnormal_ids.update(indices)
                        gen_abnormal_count = len(unique_abnormal_ids)
                        gen_total = gen_results.get("total_trajectories", 0)
                        gen_rate = (
                            (gen_abnormal_count / gen_total * 100)
                            if gen_total > 0
                            else 0
                        )

                        logger.info(
                            f"        Generated: {gen_abnormal_count}/{gen_total} ({gen_rate:.2f}%) abnormal"
                        )

                        # Store for comparison
                        model_results[model_type] = {
                            "model": model_type,  # Fix: Add model name to results
                            "abnormal_count": gen_abnormal_count,
                            "total_trajectories": gen_total,
                            "abnormal_rate": gen_rate,
                            "abnormal_by_category": {
                                cat: len(indices)
                                for cat, indices in gen_results.get(
                                    "abnormal_indices", {}
                                ).items()
                            },
                        }

                    except Exception as e:
                        logger.error(
                            f"        ❌ Detection failed for {model_type}: {e}"
                        )
                        continue

                # Step 3: Compare and report
                logger.info(f"\n    📊 Comparison Report ({od_source}):")
                logger.info(f"    {'=' * 60}")
                logger.info(
                    f"    Real data:   {real_abnormal_count}/{real_total} ({real_rate:.2f}%)"
                )

                for model_type, results in model_results.items():
                    diff = results["abnormal_rate"] - real_rate
                    symbol = "⚠️ " if diff > 5 else "✅"
                    logger.info(
                        f"    {symbol} {model_type:12s}: {results['abnormal_count']}/{results['total_trajectories']} "
                        f"({results['abnormal_rate']:.2f}%, {diff:+.2f}% vs real)"
                    )

                # Save comparison report
                comparison_output = output_dir / od_source / "comparison_report.json"
                comparison_output.parent.mkdir(parents=True, exist_ok=True)

                comparison_data = {
                    "dataset": dataset_name,
                    "od_source": od_source,
                    "timestamp": datetime.now().isoformat(),
                    "real_data": {
                        "abnormal_count": real_abnormal_count,
                        "total_trajectories": real_total,
                        "abnormal_rate": real_rate,
                        "abnormal_by_category": {
                            cat: len(indices)
                            for cat, indices in real_results.get(
                                "abnormal_indices", {}
                            ).items()
                        },
                    },
                    "generated_data": model_results,
                }

                with open(comparison_output, "w") as f:
                    json.dump(comparison_data, f, indent=2)

                logger.info(f"    💾 Saved comparison report to {comparison_output}")

            except Exception as e:
                logger.error(f"  ❌ Abnormal analysis failed for {od_source}: {e}")
                import traceback

                traceback.print_exc()
                continue

        logger.info(
            f"\n✅ Abnormal analysis complete for {dataset_name}! Results in {output_dir}/"
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

    @phase("generation", critical=True)
    def run_generation(self):
        """Generate trajectories for all models"""
        logger.info("🔄 Generating trajectories...")

        for model_type in self.config.models:
            try:
                model_file = self.detector.find_model_file(model_type)
                if not model_file:
                    error_msg = f"Model file not found for type: {model_type}"
                    logger.error(error_msg)
                    continue

                logger.info(f"Processing model: {model_type} ({model_file.name})")

                for od_source in self.config.od_sources:
                    # Check for interruption
                    if self.interrupted:
                        logger.info("Pipeline interrupted by user")
                        raise KeyboardInterrupt()

                    # Check for existing results
                    existing_file = self._check_existing_results(model_type, od_source)

                    # Generation logic
                    try:
                        if existing_file:
                            generated_file = existing_file
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

                                self.wandb_manager.init_run(run_name, tags, config_dict)

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
                        self._handle_error(e, "generation", model_type, od_source)
                        logger.error(
                            f"Generation failed for {model_type} ({od_source} OD): {str(e)}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Error processing model {model_type}: {str(e)}")
                continue

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

                # Find generated file (prefer A* files)
                gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")

                # Try new naming pattern first: {timestamp}_{model_type}_{od_source}.csv
                pattern = f"*_{model_type}_{od_source}.csv"
                generated_files = list(gene_dir.glob(pattern))

                if not generated_files:
                    # Fallback: try other patterns
                    patterns = [
                        f"*{model_type}*{od_source}.csv",
                        f"*{model_type}*{od_source}od*.csv",
                        f"{model_type}*{od_source}*.csv",
                    ]
                    for p in patterns:
                        generated_files.extend(gene_dir.glob(p))

                    if not generated_files:
                        error_msg = f"No existing generated file found for {model_type} ({od_source} OD)"
                        logger.error(error_msg)
                        failed_operations.append(error_msg)
                        continue

                # Filter for A* files only (check perf.json for beam_search_enabled: false)
                import json

                astar_files = []

                for csv_file in generated_files:
                    perf_file = csv_file.parent / f"{csv_file.stem}_perf.json"

                    if perf_file.exists():
                        try:
                            with open(perf_file) as f:
                                perf_data = json.load(f)

                            beam_search_enabled = perf_data.get(
                                "beam_search_enabled", True
                            )
                            if not beam_search_enabled:
                                astar_files.append(csv_file)
                        except (json.JSONDecodeError, KeyError):
                            # Skip files with invalid perf.json
                            continue

                if astar_files:
                    # Use most recent A* file
                    generated_file = max(astar_files, key=lambda x: x.stat().st_mtime)
                    logger.info(f"Found existing A* file: {generated_file.name}")
                else:
                    # No A* files found - this shouldn't happen if generation phase worked
                    # But log a warning and use the most recent file anyway
                    generated_file = max(
                        generated_files, key=lambda x: x.stat().st_mtime
                    )
                    logger.warning(
                        f"No A* file found for {model_type} ({od_source} OD), "
                        f"using most recent file: {generated_file.name}. "
                        f"This may be a beam search file."
                    )

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

        return results_summary

    @phase("cross_dataset", critical=False)
    def run_cross_dataset(self):
        """Evaluate on cross-dataset (BJUT)"""
        if not self.config.cross_dataset_eval:
            logger.info("Cross-dataset not configured, skipping")
            return

        logger.info("🌐 Evaluating on cross-dataset...")
        self._run_cross_dataset_evaluation()

    @phase("road_network_translate", critical=False)
    def run_road_network_translate(self):
        """Translate generated trajectories to cross-dataset road network"""
        logger.info("🗺️  Translating road networks for cross-dataset analysis...")

        # Only needed when cross-dataset is configured
        if not self.config.cross_dataset_eval:
            logger.info("No cross-dataset configured, skipping translation")
            return

        # Check if mapping exists, create if not
        mapping_file = Path(
            f"road_mapping_{self.config.dataset.lower()}_to_{self.config.cross_dataset_name.lower().replace(' ', '_')}.json"
        )

        if not mapping_file.exists():
            logger.info(f"  📍 Mapping file not found, creating: {mapping_file}")

            # Import mapping tool
            from tools.map_road_networks import (
                load_road_network_with_coords,
                find_nearest_road_batch,
                save_comprehensive_output,
            )

            # Determine .geo file paths
            source_geo = Path(f"data/{self.config.dataset}/roadmap.geo")
            if not source_geo.exists():
                source_geo = (
                    self.eval_dir.parent / "data" / self.config.dataset / "roadmap.geo"
                )

            target_data_dir = Path(self.config.cross_dataset_eval)
            if not target_data_dir.is_absolute():
                target_data_dir = self.eval_dir / target_data_dir
            target_geo = target_data_dir / "roadmap.geo"

            if not source_geo.exists():
                logger.error(f"❌ Source .geo file not found: {source_geo}")
                return
            if not target_geo.exists():
                logger.error(f"❌ Target .geo file not found: {target_geo}")
                return

            # Create mapping
            source_roads = load_road_network_with_coords(source_geo)
            target_roads = load_road_network_with_coords(target_geo)

            result = find_nearest_road_batch(
                source_roads=source_roads,
                target_roads=target_roads,
                max_distance_m=50.0,
            )

            save_comprehensive_output(
                result=result,
                source_dataset=self.config.dataset,
                target_dataset=self.config.cross_dataset_name,
                source_geo=source_geo,
                target_geo=target_geo,
                max_distance=50.0,
                output_file=mapping_file,
            )

            # Check mapping quality
            mapping_rate = result["statistics"]["mapping_rate_pct"]
            if mapping_rate < 70:
                logger.error(
                    f"❌ Poor mapping quality ({mapping_rate:.1f}%), aborting translation"
                )
                return
            elif mapping_rate < 85:
                logger.warning(f"⚠️  Fair mapping quality ({mapping_rate:.1f}%)")
        else:
            logger.info(f"  ✅ Using existing mapping: {mapping_file}")

        # Load mapping
        import json

        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        mapping_int = {int(k): int(v) for k, v in mapping.items()}

        # Translate all generated files
        from tools.translate_trajectories import translate_trajectory_file

        gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")
        if not gene_dir.exists():
            logger.warning(
                f"Gene directory not found: {gene_dir}, skipping translation"
            )
            return

        gene_files = list(gene_dir.glob("*.csv"))
        if not gene_files:
            logger.warning("No generated files to translate")
            return

        logger.info(f"\n  📦 Found {len(gene_files)} generated files to translate")

        output_dir = Path(
            f"./gene_translated/{self.config.cross_dataset_name}/seed{self.config.seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        for gene_file in gene_files:
            output_file = output_dir / gene_file.name

            try:
                translate_trajectory_file(
                    input_file=gene_file,
                    mapping=mapping_int,
                    output_file=output_file,
                )
            except Exception as e:
                logger.error(f"  ❌ Translation failed for {gene_file.name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        logger.info(f"\n✅ Translation complete! Files in {output_dir}/")

    @phase("abnormal", critical=False)
    def run_abnormal(self):
        """Detect abnormal trajectories"""
        if not self.config.run_abnormal_detection:
            logger.info("Abnormal detection not configured, skipping")
            return

        logger.info("🔍 Running abnormal detection...")
        self._run_abnormal_detection_analysis()

    @phase("abnormal_od_extract", critical=False)
    def run_abnormal_od_extract(self):
        """Extract abnormal OD pairs from cross-dataset detection results"""
        logger.info("📊 Extracting abnormal OD pairs from cross-dataset...")

        # Check dependency: Need abnormal detection results
        if not self.config.cross_dataset_name:
            logger.warning(
                "No cross-dataset configured, skipping abnormal OD extraction"
            )
            return

        abnormal_results_dir = Path(f"./abnormal/{self.config.cross_dataset_name}")
        if not abnormal_results_dir.exists():
            logger.error(
                f"❌ Dependency not met: Run 'abnormal' phase first. "
                f"Missing: {abnormal_results_dir}"
            )
            return

        # Find detection result files (try both old and new directory structures)
        detection_files = list(
            abnormal_results_dir.glob("*/real_data/detection_results.json")
        )
        if not detection_files:
            # Try old directory structure
            detection_files = list(
                abnormal_results_dir.glob("*/detection/detection_results.json")
            )

        if not detection_files:
            logger.error(f"❌ No detection results found in {abnormal_results_dir}")
            return

        logger.info(f"  Found {len(detection_files)} detection result files")

        # Prepare data file paths
        data_dir = Path(self.config.cross_dataset_eval)
        if not data_dir.is_absolute():
            data_dir = self.eval_dir / data_dir

        data_files = []
        for det_file in detection_files:
            # Extract split name (train/test) from path
            split = det_file.parent.parent.name
            data_file = data_dir / f"{split}.csv"
            if data_file.exists():
                data_files.append(data_file)

        if len(data_files) != len(detection_files):
            logger.warning(
                "⚠️  Some data files not found, continuing with available files"
            )

        # Import and run extraction
        from tools.extract_abnormal_od_pairs import extract_abnormal_od_pairs

        output_file = Path(
            f"./abnormal_od_pairs_{self.config.cross_dataset_name.lower().replace(' ', '_')}.json"
        )

        try:
            result = extract_abnormal_od_pairs(
                detection_results_files=detection_files,
                real_data_files=data_files,
                dataset_name=self.config.cross_dataset_name,
            )

            # Save result
            import json

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(
                f"✅ Extracted {result['total_unique_od_pairs']} abnormal OD pairs"
            )
            logger.info(f"💾 Saved to {output_file}")

        except Exception as e:
            logger.error(f"❌ OD pair extraction failed: {e}")
            import traceback

            traceback.print_exc()

    @phase("abnormal_od_generate", critical=False)
    def run_abnormal_od_generate(self):
        """Generate trajectories for abnormal OD pairs"""
        logger.info("🤖 Generating trajectories for abnormal OD pairs...")

        # Check dependency: Need OD pairs file
        od_pairs_files = list(Path(".").glob("abnormal_od_pairs_*.json"))
        if not od_pairs_files:
            logger.error(
                "❌ Dependency not met: Run 'abnormal_od_extract' phase first. "
                "Missing: abnormal_od_pairs_*.json"
            )
            return

        od_pairs_file = od_pairs_files[0]
        logger.info(f"  Using OD pairs: {od_pairs_file}")

        # Load OD pairs
        import json

        with open(od_pairs_file, "r") as f:
            od_pairs_data = json.load(f)

        # Create flat list of OD pairs
        all_pairs = []
        for category, pairs in od_pairs_data.get("od_pairs_by_category", {}).items():
            # Limit to 20 pairs per category for reasonable runtime
            pairs_limited = pairs[:20]
            all_pairs.extend(pairs_limited)
            logger.info(f"  {category}: {len(pairs_limited)} OD pairs")

        # Deduplicate
        unique_pairs = list(set(tuple(p) for p in all_pairs))
        logger.info(f"  Total unique OD pairs: {len(unique_pairs)}")

        if not unique_pairs:
            logger.warning("No OD pairs to generate for")
            return

        # Output directory
        output_dir = Path(
            f"./gene_abnormal/{self.config.dataset}/seed{self.config.seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate for each model
        from gene import generate_trajectories_programmatic

        num_traj_per_od = 50  # Reasonable default

        for model_type in self.config.models:
            try:
                model_file = self.detector.find_model_file(model_type)
                if not model_file:
                    logger.warning(f"  ⚠️  Model file not found for {model_type}")
                    continue

                logger.info(f"\n  {'=' * 60}")
                logger.info(f"  🚀 Generating with {model_type}")
                logger.info(f"  {'=' * 60}")

                # Expand OD pairs (repeat each pair num_traj_per_od times)
                od_list_expanded = []
                for origin, dest in unique_pairs:
                    for _ in range(num_traj_per_od):
                        od_list_expanded.append((origin, dest))

                logger.info(f"    Generating {len(od_list_expanded)} trajectories...")

                output_file = output_dir / f"{model_type}_abnormal_od.csv"

                result = generate_trajectories_programmatic(
                    model_path=str(model_file),
                    dataset=self.config.dataset,
                    num_generate=len(od_list_expanded),
                    od_list=od_list_expanded,
                    output_file=str(output_file),
                    seed=self.config.seed,
                    cuda_device=self.config.cuda_device,
                    beam_search=self.config.beam_search,
                    beam_width=self.config.beam_width,
                )

                if result.get("output_file"):
                    traj_count = result.get("num_generated", 0)
                    logger.info(
                        f"    ✅ Generated {traj_count} trajectories → {result['output_file']}"
                    )
                else:
                    logger.error("    ❌ Generation failed: No output file produced")

            except Exception as e:
                logger.error(f"  ❌ Error generating with {model_type}: {e}")
                import traceback

                traceback.print_exc()
                continue

        logger.info(f"\n✅ Abnormal OD generation complete! Results in {output_dir}/")

    @phase("abnormal_od_evaluate", critical=False)
    def run_abnormal_od_evaluate(self):
        """Evaluate model performance on abnormal OD pairs"""
        logger.info("📊 Evaluating performance on abnormal OD pairs...")

        # Check dependency: Need generated files
        gen_dir = Path(f"./gene_abnormal/{self.config.dataset}/seed{self.config.seed}")
        if not gen_dir.exists():
            logger.error(
                "❌ Dependency not met: Run 'abnormal_od_generate' phase first. "
                f"Missing: {gen_dir}"
            )
            return

        gen_files = list(gen_dir.glob("*_abnormal_od.csv"))
        if not gen_files:
            logger.error(f"❌ No generated files found in {gen_dir}")
            return

        logger.info(f"  Found {len(gen_files)} generated files")

        # Check dependency: Need OD pairs file
        od_pairs_files = list(Path(".").glob("abnormal_od_pairs_*.json"))
        if not od_pairs_files:
            logger.error("❌ Missing abnormal OD pairs file")
            return

        od_pairs_file = od_pairs_files[0]

        # Get real abnormal data file
        if not self.config.cross_dataset_eval:
            logger.error("❌ No cross-dataset configured")
            return

        data_dir = Path(self.config.cross_dataset_eval)
        if not data_dir.is_absolute():
            data_dir = self.eval_dir / data_dir

        real_abnormal_file = data_dir / "train.csv"
        if not real_abnormal_file.exists():
            logger.error(f"❌ Real abnormal file not found: {real_abnormal_file}")
            return

        # Output directory
        output_dir = Path(f"./eval_abnormal/{self.config.dataset}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Import evaluation functions
        import json
        from tools.analyze_abnormal import run_abnormal_analysis
        from evaluation import evaluate_trajectories_programmatic

        # Evaluate each model
        all_results = {}

        for gen_file in gen_files:
            model_name = gen_file.stem.replace("_abnormal_od", "")
            logger.info(f"\n  {'=' * 60}")
            logger.info(f"  📊 Evaluating {model_name}")
            logger.info(f"  {'=' * 60}")

            try:
                # Step 1: Abnormal detection
                logger.info("    🔍 Running abnormal detection...")
                detection_output = output_dir / model_name / "detection"
                detection_output.mkdir(parents=True, exist_ok=True)

                detection_results = run_abnormal_analysis(
                    real_file=gen_file,
                    dataset=self.config.dataset,
                    config_path=Path("config/abnormal_detection.yaml"),
                    output_dir=detection_output,
                    is_real_data=False,  # Analyzing generated abnormal OD data
                )

                # Calculate rates
                total_traj = detection_results.get("total_trajectories", 0)
                abnormal_by_category = {}

                for category, indices in detection_results.get(
                    "abnormal_indices", {}
                ).items():
                    count = len(indices)
                    rate = (count / total_traj * 100) if total_traj > 0 else 0
                    abnormal_by_category[category] = {"count": count, "rate": rate}
                    logger.info(f"      {category}: {count}/{total_traj} ({rate:.2f}%)")

                # Step 2: Similarity metrics
                logger.info("    📏 Computing similarity metrics...")
                eval_output = output_dir / model_name / "metrics"
                eval_output.mkdir(parents=True, exist_ok=True)

                eval_results = evaluate_trajectories_programmatic(
                    real_file=str(real_abnormal_file),
                    generated_file=str(gen_file),
                    dataset=self.config.dataset,
                    output_dir=str(eval_output),
                )

                metrics = {}
                if eval_results.get("status") == "success":
                    eval_data = eval_results.get("results", {})
                    metrics = {
                        "edr": eval_data.get("edr", 0.0),
                        "dtw": eval_data.get("dtw", 0.0),
                        "hausdorff": eval_data.get("hausdorff", 0.0),
                    }
                    logger.info(f"      EDR: {metrics['edr']:.4f}")

                # Save results
                model_results = {
                    "model": model_name,
                    "total_trajectories": total_traj,
                    "abnormality_detection": abnormal_by_category,
                    "similarity_metrics": metrics,
                }

                all_results[model_name] = model_results

                result_file = output_dir / model_name / "abnormal_od_evaluation.json"
                with open(result_file, "w") as f:
                    json.dump(model_results, f, indent=2)

            except Exception as e:
                logger.error(f"    ❌ Evaluation failed: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Comparison report
        logger.info(f"\n  {'=' * 60}")
        logger.info("  📊 Comparison Report")
        logger.info(f"  {'=' * 60}")

        for model_name, results in all_results.items():
            total_abnormal = sum(
                cat["count"] for cat in results["abnormality_detection"].values()
            )
            total_traj = results["total_trajectories"]
            rate = (total_abnormal / total_traj * 100) if total_traj > 0 else 0
            logger.info(
                f"    {model_name:15s}: {total_abnormal:4d}/{total_traj:4d} ({rate:5.2f}%)"
            )

        # Save comparison
        comparison_report = {
            "dataset": self.config.dataset,
            "od_pairs_file": str(od_pairs_file),
            "model_results": all_results,
        }

        comparison_file = output_dir / "comparison_report.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_report, f, indent=2)

        logger.info(f"\n✅ Evaluation complete! Results in {output_dir}/")

    @phase("scenarios", critical=False)
    def run_scenarios(self):
        """Run scenario analysis"""
        if not self.config.run_scenarios:
            logger.info("Scenarios not configured, skipping")
            return

        logger.info("🎯 Running scenario analysis...")
        self._run_scenario_analysis()

    @phase("paired_analysis", critical=False)
    def run_paired_analysis(self):
        """Run paired statistical tests comparing models on same OD pairs"""
        logger.info("📊 Running paired statistical analysis...")

        # Check dependency: need base_eval results with trajectory metrics
        eval_dir = Path("./eval")
        if not eval_dir.exists():
            logger.warning("No eval directory found, skipping paired analysis")
            return

        # Find all evaluation results with trajectory metrics
        eval_subdirs = [d for d in eval_dir.iterdir() if d.is_dir()]
        if not eval_subdirs:
            logger.warning(
                "No evaluation subdirectories found, skipping paired analysis"
            )
            return

        # Check for trajectory_metrics.json in at least one eval dir
        has_trajectory_metrics = any(
            (subdir / "trajectory_metrics.json").exists() for subdir in eval_subdirs
        )
        if not has_trajectory_metrics:
            logger.warning(
                "No trajectory_metrics.json files found. "
                "Run base_eval phase first to generate trajectory-level metrics."
            )
            return

        # Auto-detect model pairs to compare
        # Group evaluations by (model_type, od_source)
        from collections import defaultdict
        import json

        evals_by_od = defaultdict(lambda: defaultdict(list))

        for subdir in eval_subdirs:
            results_file = subdir / "results.json"
            traj_metrics_file = subdir / "trajectory_metrics.json"

            if not results_file.exists() or not traj_metrics_file.exists():
                continue

            try:
                with open(results_file) as f:
                    results = json.load(f)

                metadata = results.get("metadata", {})
                model_type = metadata.get("model_type", "unknown")
                od_source = metadata.get("od_source", "unknown")

                if model_type != "unknown" and od_source != "unknown":
                    evals_by_od[od_source][model_type].append(subdir)

            except Exception as e:
                logger.warning(f"Failed to read {results_file}: {e}")
                continue

        if not evals_by_od:
            logger.warning("No valid evaluation results found for paired analysis")
            return

        # Import the paired comparison tool
        from tools.compare_models_paired_analysis import (
            load_trajectory_metrics,
            match_trajectory_pairs,
            perform_paired_comparison,
            generate_markdown_summary,
        )

        # Output directory for paired analysis results
        paired_output_dir = Path("./paired_analysis")
        paired_output_dir.mkdir(exist_ok=True)

        # For each OD source, compare models pairwise
        comparison_count = 0
        for od_source, models_dict in evals_by_od.items():
            logger.info(f"\n📂 Processing {od_source} OD")

            model_types = list(models_dict.keys())
            if len(model_types) < 2:
                logger.info(
                    f"  Only 1 model found for {od_source} OD, skipping comparisons"
                )
                continue

            # Compare each pair of models
            for i in range(len(model_types)):
                for j in range(i + 1, len(model_types)):
                    model1_type = model_types[i]
                    model2_type = model_types[j]

                    # Get evaluation directories (use most recent if multiple)
                    eval1_dir = sorted(
                        models_dict[model1_type], key=lambda x: x.stat().st_mtime
                    )[-1]
                    eval2_dir = sorted(
                        models_dict[model2_type], key=lambda x: x.stat().st_mtime
                    )[-1]

                    logger.info(f"  Comparing {model1_type} vs {model2_type}")

                    try:
                        # Load trajectory metrics
                        data1 = load_trajectory_metrics(eval1_dir)
                        data2 = load_trajectory_metrics(eval2_dir)

                        # Match trajectory pairs
                        matched1, matched2 = match_trajectory_pairs(
                            data1["trajectory_metrics"], data2["trajectory_metrics"]
                        )

                        if len(matched1) == 0:
                            logger.warning(
                                "    No matching trajectory pairs found, skipping"
                            )
                            continue

                        # Perform paired comparison
                        metrics_to_compare = [
                            "hausdorff_km",
                            "hausdorff_norm",
                            "dtw_km",
                            "dtw_norm",
                            "edr",
                        ]
                        comparison_results = perform_paired_comparison(
                            matched_metrics1=matched1,
                            matched_metrics2=matched2,
                            model1_name=model1_type,
                            model2_name=model2_type,
                            metrics_to_compare=metrics_to_compare,
                            alpha=0.05,
                        )

                        # Prepare output
                        output_data = {
                            "model1_name": model1_type,
                            "model2_name": model2_type,
                            "model1_eval_dir": str(eval1_dir),
                            "model2_eval_dir": str(eval2_dir),
                            "n_matched_pairs": len(matched1),
                            "alpha": 0.05,
                            "metrics": comparison_results,
                            "metadata": {
                                "model1_metadata": data1.get("metadata", {}),
                                "model2_metadata": data2.get("metadata", {}),
                            },
                        }

                        # Save results
                        comparison_output_dir = (
                            paired_output_dir
                            / od_source
                            / f"{model1_type}_vs_{model2_type}"
                        )
                        comparison_output_dir.mkdir(parents=True, exist_ok=True)

                        # Convert numpy types to native Python types for JSON serialization
                        def convert_numpy_types(obj):
                            """Recursively convert numpy types to native Python types."""
                            import numpy as np

                            if isinstance(obj, dict):
                                return {
                                    k: convert_numpy_types(v) for k, v in obj.items()
                                }
                            elif isinstance(obj, list):
                                return [convert_numpy_types(item) for item in obj]
                            elif isinstance(obj, (np.integer, np.floating)):
                                return float(obj)
                            elif isinstance(obj, np.bool_):
                                return bool(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            return obj

                        output_data_serializable = convert_numpy_types(output_data)

                        output_file = comparison_output_dir / "paired_comparison.json"
                        with open(output_file, "w") as f:
                            json.dump(output_data_serializable, f, indent=2)

                        # Generate markdown summary
                        generate_markdown_summary(output_data, output_file)

                        logger.info(f"    ✅ Results saved to {comparison_output_dir}")
                        comparison_count += 1

                    except Exception as e:
                        logger.error(
                            f"    ❌ Paired analysis failed for {model1_type} vs {model2_type}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        continue

        if comparison_count > 0:
            logger.info(
                f"\n✅ Paired analysis complete! {comparison_count} comparisons saved to {paired_output_dir}/"
            )
        else:
            logger.warning("No paired comparisons were completed")

    def run(self):
        """Execute all enabled phases in order"""
        logger.info("Starting HOSER Distillation Evaluation Pipeline")
        logger.info(f"Configuration: {self.config.__dict__}")

        try:
            self._prepare_execution_context()
            failures = self._execute_phases()

            if failures:
                self._log_phase_failures(failures)
                return False

            logger.info("\n✅ Pipeline completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed with critical error: {str(e)}")
            import traceback

            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
        finally:
            self._cleanup_after_run()

    def _prepare_execution_context(self) -> None:
        """Prepare models and phase configuration before execution."""
        self._normalize_phase_configuration()
        self._ensure_models_loaded()
        logger.info(f"Models to process: {self.config.models}")
        logger.info(f"OD sources: {self.config.od_sources}")

    def _normalize_phase_configuration(self) -> None:
        """Ensure phase configuration is consistent."""
        phases = getattr(self.config, "phases", set())
        if phases is None:
            phases = set()

        self.config.phases = set(phases)
        logger.info(f"Enabled phases: {sorted(self.config.phases)}")

    def _ensure_models_loaded(self) -> None:
        """Populate model list if necessary before executing phases."""
        if not self.config.models:
            self.config.models = self.detector.detect_models()

        if not self.config.models:
            raise ValueError("No models configured for evaluation")

    def _execute_phases(self) -> List[str]:
        """Execute phases in defined order and collect non-critical failures."""
        failures: List[str] = []
        for phase_name in self._phase_execution_order():
            if not self._should_run_phase(phase_name):
                logger.info(f"⏭️  Skipping phase: {phase_name}")
                continue

            phase_info = PHASE_REGISTRY.get(phase_name)
            if not phase_info:
                logger.warning(f"⚠️  Phase not registered: {phase_name}")
                continue

            logger.info(f"\n{'=' * 70}")
            logger.info(f"🚀 Running phase: {phase_name}")
            logger.info(f"{'=' * 70}")

            try:
                phase_info["func"](self)
                logger.info(f"✅ Phase {phase_name} completed")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"❌ Phase {phase_name} failed: {exc}")
                if phase_info.get("critical"):
                    logger.error("Critical phase failed, stopping pipeline")
                    raise

                logger.warning("Non-critical phase failed, continuing")
                failures.append(f"{phase_name}: {exc}")

        return failures

    def _phase_execution_order(self) -> List[str]:
        """Return the canonical execution order for phases."""
        default_order = [
            "generation",
            "base_eval",
            "paired_analysis",
            "cross_dataset",
            "road_network_translate",
            "abnormal",
            "abnormal_od_extract",
            "abnormal_od_generate",
            "abnormal_od_evaluate",
            "scenarios",
        ]
        extras = [phase for phase in self.config.phases if phase not in default_order]
        return default_order + sorted(extras)

    def _should_run_phase(self, phase_name: str) -> bool:
        """Determine whether a phase is enabled for execution."""
        return phase_name in self.config.phases

    def _log_phase_failures(self, failures: List[str]) -> None:
        """Summarize non-critical phase failures for operator visibility."""
        logger.warning(f"\n⚠️  {len(failures)} phase(s) failed:")
        for failure in failures:
            logger.warning(f"  - {failure}")

    def _cleanup_after_run(self) -> None:
        """Ensure WandB resources are finalized regardless of outcome."""
        try:
            self.wandb_manager.finish_all_runs()
            self.wandb_manager.start_background_sync()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error during WandB cleanup: {exc}")


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
        "--only",
        type=str,
        help=(
            "Run only these phases (comma-separated). Available: "
            "generation,base_eval,paired_analysis,cross_dataset,road_network_translate,abnormal,"
            "abnormal_od_extract,abnormal_od_generate,abnormal_od_evaluate,scenarios."
        ),
    )
    parser.add_argument(
        "--skip",
        type=str,
        help=(
            "Skip these phases (comma-separated). Example: --skip generation,base_eval"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-run (overrides config)"
    )
    parser.add_argument("--cuda", type=int, help="CUDA device (overrides config)")
    parser.add_argument(
        "--num-gene", type=int, help="Number of trajectories (overrides config)"
    )
    parser.add_argument(
        "--use-astar",
        action="store_true",
        help="Use A* search instead of beam search (original HOSER method)",
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

    # Apply phase control (BEFORE other overrides)
    if hasattr(args, "only") and args.only:
        config.phases = {p.strip() for p in args.only.split(",") if p.strip()}
        logger.info(f"Running only phases: {config.phases}")

    if hasattr(args, "skip") and args.skip:
        skip_phases = {p.strip() for p in args.skip.split(",") if p.strip()}
        config.phases -= skip_phases
        logger.info(f"Skipping phases: {skip_phases}")

    # Backward compatibility shortcuts
    if args.skip_gene:
        config.phases.discard("generation")
    if args.skip_eval:
        config.phases -= {"base_eval", "cross_dataset", "abnormal", "scenarios"}

    # Override with command line arguments if provided
    if args.seed is not None:
        config.seed = args.seed
    if args.models is not None:
        config.models = args.models.split(",")
    if args.od_source is not None:
        config.od_sources = args.od_source.split(",")
    if args.force:
        config.force = True
    if args.cuda is not None:
        config.cuda_device = args.cuda
    if args.num_gene is not None:
        config.num_gene = args.num_gene
    if args.use_astar:
        config.beam_search = False
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
