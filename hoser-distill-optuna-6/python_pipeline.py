#!/usr/bin/env python3
"""
Bulletproof Python Pipeline for HOSER Distillation Evaluation

This script imports gene.py and evaluation.py directly to avoid CLI overhead
and WandB upload delays. Designed for thesis-critical evaluation runs.

Usage:
    python python_pipeline.py [OPTIONS]

Options:
    --seed SEED              Random seed (default: 42)
    --models MODEL1,MODEL2   Models to run (default: auto-detect all)
    --od-source SOURCE      OD source: train or test (default: train,test)
    --skip-gene             Skip generation (use existing trajectories)
    --skip-eval             Skip evaluation
    --force                 Force re-run even if results exist
    --cuda DEVICE           CUDA device (default: 0)
    --num-gene N            Number of trajectories (default: 100)
    --wandb-project PROJECT WandB project name (default: hoser-distill-optuna-6)
    --no-wandb              Disable WandB logging entirely
    --verbose               Enable verbose output
"""

import os
import sys
import argparse
import json
import time
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the programmatic interfaces
from gene import generate_trajectories_programmatic
from evaluation import evaluate_trajectories_programmatic
from utils import set_seed, create_nested_namespace
from models.hoser import HOSER
import yaml
import polars as pl
import numpy as np
import torch
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the evaluation pipeline"""
    
    def __init__(self, config_path: str = None):
        # Default values
        self.wandb_project = "hoser-distill-optuna-6"
        self.dataset = "Beijing"
        self.cuda_device = 0
        self.num_gene = 100
        self.seed = 42
        self.models = []  # Auto-detect
        self.od_sources = ["train", "test"]
        self.skip_gene = False
        self.skip_eval = False
        self.force = False
        self.enable_wandb = True
        self.verbose = False
        self.beam_width = 4
        self.grid_size = 0.001
        self.edr_eps = 100.0
        self.background_sync = True  # Background WandB sync
        
        # Load from YAML if provided
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update attributes from config
        for key, value in config_data.items():
            if key == 'wandb':
                wandb_config = value
                self.wandb_project = wandb_config.get('project', self.wandb_project)
                self.enable_wandb = wandb_config.get('enable', self.enable_wandb)
                self.background_sync = wandb_config.get('background_sync', self.background_sync)
            elif key == 'logging':
                logging_config = value
                self.verbose = logging_config.get('verbose', self.verbose)
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
                model_type = model_name.split("_25epoch_seed")[0]
                
                # For multiple seeds of same model type, append seed to make unique
                if "_25epoch_seed" in model_name:
                    seed = model_name.split("_seed")[-1]
                    if model_type == "distilled" and seed != "42":
                        model_type = f"{model_type}_seed{seed}"
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
                extracted_type = model_name.split("_25epoch_seed")[0]
                if "_25epoch_seed" in model_name:
                    seed = model_name.split("_seed")[-1]
                    if extracted_type == "distilled" and seed != "42":
                        extracted_type = f"{extracted_type}_seed{seed}"
            else:
                extracted_type = model_name.split("_")[0]
            
            if extracted_type == model_type:
                return model_file
        
        return None


class TrajectoryGenerator:
    """Generate trajectories using HOSER models"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def generate_trajectories(self, model_path: Path, model_type: str, od_source: str) -> Path:
        """Generate trajectories for a specific model and OD source"""
        logger.info(f"Generating trajectories: {model_type} ({od_source} OD)")
        
        # Use the programmatic interface
        output_path = generate_trajectories_programmatic(
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
            wandb_tags=None
        )
        
        logger.info(f"Trajectories saved to: {output_path}")
        return Path(output_path)


class TrajectoryEvaluator:
    """Evaluate generated trajectories against real data"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def evaluate_trajectories(self, generated_file: Path, model_type: str, od_source: str) -> Dict[str, Any]:
        """Evaluate generated trajectories"""
        logger.info(f"Evaluating trajectories: {model_type} ({od_source} OD)")
        
        # Use the programmatic interface
        results = evaluate_trajectories_programmatic(
            generated_file=str(generated_file),
            dataset=self.config.dataset,
            grid_size=self.config.grid_size,
            edr_eps=self.config.edr_eps,
            enable_wandb=False,  # We'll handle WandB separately
            wandb_project=None,
            wandb_run_name=None,
            wandb_tags=None
        )
        
        # Add our metadata
        results["metadata"]["model_type"] = model_type
        results["metadata"]["od_source"] = od_source
        results["metadata"]["seed"] = self.config.seed
        
        logger.info(f"Evaluation completed for {model_type} ({od_source} OD)")
        return results


class WandBManager:
    """Manage WandB logging efficiently with background uploads"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.runs = {}  # Store run objects
        self.completed_runs = []  # Track completed runs for background sync
        self.sync_thread = None
    
    def init_run(self, run_name: str, tags: List[str], config_dict: Dict[str, Any]) -> str:
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
                mode="offline"
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
            scalar_metrics = {k: v for k, v in metrics.items() 
                            if isinstance(v, (int, float)) and k != 'metadata'}
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
            logger.info("   To upload: wandb sync wandb/offline-run-*")
            return
        
        def sync_worker():
            """Background worker to sync runs"""
            logger.info(f"📤 Starting background sync of {len(self.completed_runs)} WandB runs...")
            
            import subprocess
            synced = 0
            failed = 0
            
            for run_dir in self.completed_runs:
                try:
                    # Use subprocess to run wandb sync
                    result = subprocess.run(
                        ['wandb', 'sync', run_dir],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 min timeout per run
                    )
                    if result.returncode == 0:
                        synced += 1
                        logger.info(f"✅ Synced {synced}/{len(self.completed_runs)}: {os.path.basename(run_dir)}")
                    else:
                        failed += 1
                        logger.warning(f"⚠️  Sync failed: {os.path.basename(run_dir)}")
                except Exception as e:
                    failed += 1
                    logger.warning(f"Error syncing {os.path.basename(run_dir)}: {e}")
            
            logger.info(f"📤 Background sync complete! {synced} synced, {failed} failed")
        
        # Start background thread (daemon=False so it completes even if main exits)
        self.sync_thread = threading.Thread(target=sync_worker, daemon=False)
        self.sync_thread.start()
        logger.info("📤 Background WandB sync started (non-blocking)")
        logger.info("   Pipeline will exit immediately. Sync continues in background.")


class EvaluationPipeline:
    """Main evaluation pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models_dir = Path("./models")
        self.detector = ModelDetector(self.models_dir)
        self.generator = TrajectoryGenerator(config)
        self.evaluator = TrajectoryEvaluator(config)
        self.wandb_manager = WandBManager(config)
        self.interrupted = False
        
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
        
        # Check data directory (handle symlink)
        data_dir = Path(f'../data/{self.config.dataset}')
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Resolve symlink if needed
        if data_dir.is_symlink():
            data_dir = data_dir.resolve()
        
        # Check required data files
        required_files = ['test.csv', 'roadmap.geo']
        for file in required_files:
            file_path = data_dir / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        # Check config file
        config_file = Path(f'../config/{self.config.dataset}.yaml')
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Validate CUDA device
        if torch.cuda.is_available():
            if self.config.cuda_device >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {self.config.cuda_device} not available. Only {torch.cuda.device_count()} devices found.")
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
    
    def _check_existing_results(self, model_type: str, od_source: str) -> Optional[Path]:
        """Check if results already exist and return path if found"""
        if self.config.force:
            return None
        
        # Check for existing generated file
        gene_dir = Path(f'./gene/{self.config.dataset}/seed{self.config.seed}')
        generated_files = list(gene_dir.glob(f'*{model_type}*{od_source}od*.csv'))
        
        if generated_files:
            latest_file = max(generated_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found existing generated file: {latest_file}")
            return latest_file
        
        return None
    
    def _handle_error(self, error: Exception, context: str, model_type: str = None, od_source: str = None):
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
            logger.warning("Non-critical error encountered, continuing with next operation")
    
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
            
            total_operations = len(self.config.models) * len(self.config.od_sources) * 2  # gene + eval
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
                        logger.info(f"[{current_operation}/{total_operations}] Processing {model_type} ({od_source} OD)")
                        
                        # Check for existing results
                        existing_file = self._check_existing_results(model_type, od_source)
                        
                        # Generation phase
                        if not self.config.skip_gene:
                            try:
                                if existing_file:
                                    generated_file = existing_file
                                    logger.info(f"Using existing generated file: {generated_file}")
                                else:
                                    generated_file = self.generator.generate_trajectories(
                                        model_file, model_type, od_source
                                    )
                                
                                # Log to WandB
                                if self.config.enable_wandb:
                                    run_name = f"gene_{model_type}_seed{self.config.seed}_{od_source}od"
                                    tags = [model_type, f"seed{self.config.seed}", f"{od_source}_od", "generation", "beam4"]
                                    config_dict = {
                                        'dataset': self.config.dataset,
                                        'seed': self.config.seed,
                                        'num_gene': self.config.num_gene,
                                        'model_type': model_type,
                                        'od_source': od_source,
                                        'beam_width': self.config.beam_width,
                                    }
                                    
                                    wandb_id = self.wandb_manager.init_run(run_name, tags, config_dict)
                                    self.wandb_manager.log_metrics(run_name, {
                                        'num_trajectories_generated': self.config.num_gene,
                                        'output_file': str(generated_file)
                                    })
                                    self.wandb_manager.finish_run(run_name)
                                
                            except Exception as e:
                                self._handle_error(e, "generation", model_type, od_source)
                                failed_operations.append(f"Generation failed for {model_type} ({od_source} OD): {str(e)}")
                                continue
                        else:
                            # Find existing generated file
                            if existing_file:
                                generated_file = existing_file
                            else:
                                gene_dir = Path(f'./gene/{self.config.dataset}/seed{self.config.seed}')
                                # Files are timestamped, just take the latest one
                                generated_files = list(gene_dir.glob('*.csv'))
                                if not generated_files:
                                    error_msg = f"No existing generated file found for {model_type} ({od_source} OD)"
                                    logger.error(error_msg)
                                    failed_operations.append(error_msg)
                                    continue
                                generated_file = max(generated_files, key=lambda x: x.stat().st_mtime)
                                logger.warning(f"Cannot determine model/od_source from filename, using latest: {generated_file.name}")
                        
                        # Evaluation phase
                        if not self.config.skip_eval:
                            try:
                                eval_results = self.evaluator.evaluate_trajectories(
                                    generated_file, model_type, od_source
                                )
                                
                                # Log to WandB
                                if self.config.enable_wandb:
                                    run_name = f"eval_{model_type}_seed{self.config.seed}_{od_source}od"
                                    tags = [model_type, f"seed{self.config.seed}", f"{od_source}_od", "evaluation"]
                                    config_dict = {
                                        'dataset': self.config.dataset,
                                        'seed': self.config.seed,
                                        'model_type': model_type,
                                        'od_source': od_source,
                                        'generated_file': str(generated_file),
                                    }
                                    
                                    wandb_id = self.wandb_manager.init_run(run_name, tags, config_dict)
                                    self.wandb_manager.log_metrics(run_name, eval_results)
                                    self.wandb_manager.finish_run(run_name)
                                
                                # Store results for summary
                                key = f"{model_type}_{od_source}"
                                results_summary[key] = {
                                    'generated_file': str(generated_file),
                                    'metrics': eval_results
                                }
                                
                            except Exception as e:
                                self._handle_error(e, "evaluation", model_type, od_source)
                                failed_operations.append(f"Evaluation failed for {model_type} ({od_source} OD): {str(e)}")
                                continue
                
                except Exception as e:
                    self._handle_error(e, f"model processing for {model_type}")
                    failed_operations.append(f"Model processing failed for {model_type}: {str(e)}")
                    continue
            
            # Print summary
            logger.info("Pipeline completed!")
            logger.info(f"Successful operations: {len(results_summary)}")
            logger.info(f"Failed operations: {len(failed_operations)}")
            
            if results_summary:
                logger.info("Results summary:")
                for key, result in results_summary.items():
                    logger.info(f"  {key}: {result['generated_file']}")
                    metrics = result['metrics']
                    for metric, value in metrics.items():
                        if isinstance(value, float) and metric != 'metadata':
                            logger.info(f"    {metric}: {value:.4f}")
            
            if failed_operations:
                logger.warning("Failed operations:")
                for failure in failed_operations:
                    logger.warning(f"  - {failure}")
            
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
    parser = argparse.ArgumentParser(description="HOSER Distillation Evaluation Pipeline")
    parser.add_argument('--config', type=str, default='config/evaluation.yaml', help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to run (overrides config)')
    parser.add_argument('--od-source', type=str, help='OD sources: train,test (overrides config)')
    parser.add_argument('--skip-gene', action='store_true', help='Skip generation (overrides config)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation (overrides config)')
    parser.add_argument('--force', action='store_true', help='Force re-run (overrides config)')
    parser.add_argument('--cuda', type=int, help='CUDA device (overrides config)')
    parser.add_argument('--num-gene', type=int, help='Number of trajectories (overrides config)')
    parser.add_argument('--wandb-project', type=str, help='WandB project (overrides config)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging (overrides config)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output (overrides config)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration from YAML file
    config = PipelineConfig(args.config)
    
    # Override with command line arguments if provided
    if args.seed is not None:
        config.seed = args.seed
    if args.models is not None:
        config.models = args.models.split(',')
    if args.od_source is not None:
        config.od_sources = args.od_source.split(',')
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
    
    # Run pipeline
    try:
        pipeline = EvaluationPipeline(config)
        success = pipeline.run()
        
        if success:
            logger.info("✅ Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.warning("⚠️  Pipeline completed with some failures")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Pipeline interrupted by user")
        logger.info("   Partial results saved offline. Background sync will upload them.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
