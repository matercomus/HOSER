#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuning for HOSER with Optuna

This script optimizes both vanilla HOSER and distillation configurations,
allowing Optuna to discover whether distillation actually improves performance.

TUNED HYPERPARAMETERS:
- distill_enable: Whether to use distillation (categorical: True/False)
                 Optuna will explore both vanilla and distilled variants

If distillation is enabled:
- lambda: KL divergence weight (0.001-0.1, log scale)
         Controls balance between teacher guidance and student loss
- temperature: Softmax temperature (1.0-5.0, linear)
              Higher values create softer probability distributions
- window: Context window size (2, 4, 8, categorical)
         Number of neighboring road segments to consider

FIXED PARAMETERS (architectural choices):
- grid_size: 0.001 (from base config)
- downsample: 1 (from base config)
- All HOSER model architecture parameters

TRAINING STRATEGY (Meaningful Results):
  • 10 epochs per trial for meaningful convergence (~8h per full trial)
  • Minimum 5 epochs before any pruning decisions
  • This ensures we see real learning patterns, not just initialization noise

PRUNING STRATEGY (MedianPruner - Conservative):
  Trial 1-3:  Run full 10 epochs (establish stable baseline, ~24h)
  Trial 4+:   Allow 5 epochs before pruning, then prune if val_acc < median
  
  Conservative approach: Trials get 5 full epochs to show potential.
  Comparison is fair: epoch 5+ performance vs median at same epoch.
  Saves ~4h per pruned trial (vs ~8h wasted on bad trial).

SAMPLING STRATEGY (GPSampler):
  Trial 1-10: Random/TPE sampling (exploration)
  Trial 11+:  Gaussian Process Bayesian optimization (exploitation)

STORAGE & CRASH RECOVERY:
  Default storage: /mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db
  Heartbeat: 60s | Grace period: 120s
  Study can be resumed after crashes/interruptions.

TIME ESTIMATE (50 trials with meaningful epochs):
  • Trial 1-3 (startup): 3 × 8h = 24h
  • Trial 4-10 (GP init): 7 trials, ~20% pruned
    - 6 complete: 6 × 8h = 48h
    - 1 pruned: 1 × 4h = 4h
    - Subtotal: 52h
  • Trial 11-50 (GP optimized): 40 trials, ~50% pruned
    - 20 complete: 20 × 8h = 160h
    - 20 pruned: 20 × 4h = 80h
    - Subtotal: 240h
  • Total: 24 + 52 + 240 = 316h (~13 days for full 50 trials)

For shorter runs, interrupt early once you see convergence.
Expected: ~10-15 meaningful trials in 48-72h.

CONFIGURATION:
  All settings in config/Beijing.yaml under 'optuna' section.
  CLI args override YAML defaults.

USAGE:
  # Standard run (uses Beijing.yaml defaults: 50 trials, 10 epochs)
  uv run python tune_hoser.py --data_dir /home/matt/Dev/HOSER-dataset

  # Shorter exploration run (fewer trials)
  uv run python tune_hoser.py --n_trials 20 --data_dir /home/matt/Dev/HOSER-dataset

  # Resume existing study
  uv run python tune_hoser.py --study_name my_study --data_dir /home/matt/Dev/HOSER-dataset

  # Use local storage
  uv run python tune_hoser.py --storage sqlite:///optuna_hoser.db --data_dir /path/to/data
"""

import os
import sys
import argparse
import tempfile
import shutil
import gc
import math
import copy
import torch
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from typing import Dict, Any, Optional
import yaml
import json
from datetime import datetime


class HOSERObjective:
    """Objective function for Optuna optimization of HOSER distillation hyperparameters."""

    def __init__(self, base_config_path: str, data_dir: str, max_epochs: int = 3):
        """
        Initialize HOSER distillation optimization objective.
        
        Args:
            base_config_path: Path to base YAML config
            data_dir: Path to HOSER dataset
            max_epochs: Max epochs per trial (default 3 for ultra-fast tuning)
        """
        self.base_config_path = base_config_path
        self.data_dir = data_dir
        self.max_epochs = max_epochs
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.
        
        Suggests hyperparameters including whether to use distillation.
        Returns validation accuracy to maximize.
        """
        # Step 1: Suggest all hyperparameters (defines search space)
        hparams = self._suggest_hyperparameters(trial)
        
        # Show what mode we're running
        mode = "distilled" if hparams['distill_enable'] else "vanilla"
        print(f"🔬 Running trial {trial.number} ({mode})")
        
        # Step 2: Create trial config from hyperparameters
        config = self._create_trial_config(trial, hparams)

        # Step 3: Run training and return metric
        temp_config_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                temp_config_path = f.name

            result = self._run_training_trial(trial, temp_config_path)
            return result
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            self._cleanup_trial_artifacts(trial.number)
            gc.collect()
            torch.cuda.empty_cache()
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space - all tunable parameters in one place.
        
        Search space includes:
        - distill_enable: Whether to use distillation (categorical: True/False)
                         This allows Optuna to discover if distillation helps
        
        If distillation is enabled, additional parameters are suggested:
        - lambda: KL divergence weight (log scale: 0.001 to 0.1)
                 Controls how much to weight teacher guidance vs student loss
        - temperature: Softmax temperature (linear: 1.0 to 5.0)
                      Higher = softer distributions, more knowledge transfer
        - window: Context window size (categorical: 2, 4, 8)
                 Number of neighboring road segments to consider
        
        CONDITIONAL SEARCH SPACE PATTERN (Optuna best practice):
        We only suggest distillation parameters when distill_enable=True.
        This creates a hierarchical/conditional search space where certain parameters
        only exist in specific branches of the configuration tree.
        
        Returns:
            Dict of hyperparameter names to suggested values
        """
        # First, decide whether to use distillation
        distill_enable = trial.suggest_categorical('distill_enable', [True, False])
        
        hparams = {'distill_enable': distill_enable}
        
        if distill_enable:
            # Only suggest distillation hyperparameters when distillation is enabled
            # This is the correct way to handle conditional parameters in Optuna
            hparams['distill_lambda'] = trial.suggest_float('distill_lambda', 0.001, 0.1, log=True)
            hparams['distill_temperature'] = trial.suggest_float('distill_temperature', 1.0, 5.0)
            hparams['distill_window'] = trial.suggest_categorical('distill_window', [2, 4, 8])
        else:
            # When distillation is disabled, don't suggest or set these params at all
            # The sampler will understand this branch doesn't have these parameters
            hparams['distill_lambda'] = None
            hparams['distill_temperature'] = None
            hparams['distill_window'] = None
        
        return hparams

    def _create_trial_config(self, trial: optuna.Trial, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create trial configuration from suggested hyperparameters.
        
        Args:
            trial: Optuna trial object
            hparams: Dict of hyperparameter names to values from _suggest_hyperparameters
        
        Returns:
            Complete config dict for this trial
        """
        config = copy.deepcopy(self.base_config)
        
        # Set training parameters
        config['optimizer_config']['max_epoch'] = self.max_epochs
        config['data_dir'] = self.data_dir
        
        # Configure distillation based on hyperparameters
        distill_enable = hparams['distill_enable']
        config['distill']['enable'] = distill_enable
        
        if distill_enable:
            # Apply distillation hyperparameters
            config['distill']['lambda'] = hparams['distill_lambda']
            config['distill']['temperature'] = hparams['distill_temperature']
            config['distill']['window'] = hparams['distill_window']
            # Keep grid_size and downsample fixed (architectural choices, not tuned)
            trial_mode = "distilled"
        else:
            # Vanilla training - distillation params don't matter
            trial_mode = "vanilla"
        
        # WandB config for trial
        config['wandb']['run_name'] = f"trial_{trial.number:03d}_{trial_mode}"
        existing_tags = config['wandb'].get('tags', [])
        config['wandb']['tags'] = existing_tags + ['hoser-tuning', trial_mode]
        
        return config
    
    def _run_training_trial(self, trial: optuna.Trial, config_path: str) -> float:
        """
        Run a single training trial and return the metric to optimize.
        
        Args:
            trial: Optuna trial object (for intermediate reporting)
            config_path: Path to temporary config file for this trial
        
        Returns:
            Validation accuracy (metric to maximize)
        """
        from train_with_distill import main as train_main
        
        try:
            result = train_main(
                dataset='Beijing',
                config_path=config_path,
                seed=42 + trial.number,  # Deterministic but different per trial
                cuda=0,
                data_dir=self.data_dir,
                return_metrics=True,
                optuna_trial=trial  # Pass trial for intermediate reporting & pruning
            )

            if result is None:
                raise optuna.TrialPruned("Training returned no metrics")

            # Validate metrics (catch NaN/inf)
            if (math.isnan(result['best_val_acc']) or math.isinf(result['best_val_acc']) or
                math.isnan(result['final_val_acc']) or math.isinf(result['final_val_acc'])):
                print(f"⚠️  Trial {trial.number}: Invalid validation metrics detected, pruning")
                raise optuna.TrialPruned("Invalid validation metrics (NaN/inf)")

            # Log additional metrics to trial
            trial.set_user_attr('final_val_acc', result['final_val_acc'])
            trial.set_user_attr('final_val_mape', result['final_val_mape'])
            trial.set_user_attr('best_val_acc', result['best_val_acc'])
            trial.set_user_attr('final_lr', result.get('final_lr', 0.0))

            # Return best validation accuracy (metric to maximize)
            return result['best_val_acc']

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned(str(e))
    
    def _cleanup_trial_artifacts(self, trial_number: int):
        """
        Clean up artifacts from a completed trial to save disk space.
        Best trial will be preserved later in main().
        """
        trial_dirs = [
            f"./save/Beijing/seed{42 + trial_number}_distill",
            f"./tensorboard_log/Beijing/seed{42 + trial_number}_distill",
            f"./log/Beijing/seed{42 + trial_number}_distill",
            f"./optuna_trials/trial_{trial_number:03d}_distilled"
        ]
        
        for dir_path in trial_dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)


def validate_and_prepare_storage(storage_arg: str) -> Optional[str]:
    """
    Validate and prepare storage URL, ensuring paths are properly handled.
    
    Args:
        storage_arg: Storage argument from CLI (can be 'memory' or SQLite URL)
    
    Returns:
        Validated storage URL or None for in-memory
        
    Raises:
        ValueError: If storage URL is invalid
    """
    if storage_arg.lower() == 'memory':
        return None
    
    # Handle SQLite URLs
    if storage_arg.startswith('sqlite:///'):
        # Extract path after sqlite:///
        db_path = storage_arg[10:]  # Remove 'sqlite:///'
        
        if not db_path:
            raise ValueError("SQLite database path cannot be empty")
        
        # Convert to absolute path if relative
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
            
        # Ensure parent directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                print(f"📁 Created directory for database: {db_dir}")
            except OSError as e:
                raise ValueError(f"Cannot create directory for database: {e}")
        
        # Reconstruct URL with absolute path
        storage_url = f"sqlite:///{db_path}"
        return storage_url
    
    # If it's another DB type (postgresql, mysql, etc.), pass through
    if '://' in storage_arg:
        return storage_arg
    
    raise ValueError(f"Invalid storage URL: {storage_arg}. Use 'memory' or 'sqlite:///path/to/db.db'")


def create_study_with_wandb(
    project_name: str, 
    study_name: str, 
    storage_url: Optional[str] = None,
    pruner_cfg: Optional[Dict[str, Any]] = None,
    sampler_cfg: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Create Optuna study with WandB integration and optional persistent storage.
    
    Args:
        project_name: WandB project name
        study_name: Unique study identifier
        storage_url: Validated SQLite database URL (from validate_and_prepare_storage)
                    If None, uses in-memory storage (not crash-safe!)
        pruner_cfg: Pruner configuration dict from YAML
        sampler_cfg: Sampler configuration dict from YAML
    
    Returns:
        tuple: (study, wandb_callback)
        
    Raises:
        RuntimeError: If study creation fails
    """
    pruner_cfg = pruner_cfg or {}
    sampler_cfg = sampler_cfg or {}

    # WandB callback configuration
    wandb_kwargs = {
        "project": project_name,
        "group": study_name,
        "job_type": "optuna-optimization"
    }

    try:
        wandbc = WeightsAndBiasesCallback(
            wandb_kwargs=wandb_kwargs,
            as_multirun=True,
            metric_name="validation_accuracy"
        )
    except Exception as e:
        print(f"⚠️  Warning: WandB callback creation failed: {e}")
        print("   Continuing without WandB integration")
        wandbc = None

    # Use GP sampler with settings from config
    gp_startup = sampler_cfg.get('n_startup_trials', 10)
    try:
        sampler = optuna.samplers.GPSampler(
            seed=42,
            n_startup_trials=gp_startup,
            deterministic_objective=False,  # Training has noise/stochasticity
            independent_sampler=optuna.samplers.TPESampler(seed=42)  # Use TPE for initial sampling
        )
        print(f"🔬 Using GP Sampler (n_startup_trials={gp_startup} from config)")
    except (ImportError, AttributeError) as e:
        print(f"⚠️  GP Sampler not available ({e}), falling back to TPE")
        sampler = optuna.samplers.TPESampler(seed=42)

    # Configure storage with crash recovery
    storage = None
    if storage_url:
        try:
            storage = RDBStorage(
                url=storage_url,
                heartbeat_interval=60,  # Update heartbeat every 60 seconds
                grace_period=120,  # Allow 120 seconds before marking trial as failed
                failed_trial_callback=RetryFailedTrialCallback(max_retry=0)  # Don't auto-retry failed trials
            )
            print(f"💾 Using persistent storage: {storage_url}")
            print("   Heartbeat interval: 60s | Grace period: 120s")
            print("   ✅ Study will survive crashes and can be resumed")
        except Exception as e:
            raise RuntimeError(f"Failed to create storage: {e}") from e
    else:
        print("⚠️  WARNING: Using in-memory storage (no crash recovery)")
        print("   Consider using --storage sqlite:///optuna_hoser.db for persistence")

    # Create Optuna study with robust error handling
    # MedianPruner with settings from config
    pruner_startup = pruner_cfg.get('n_startup_trials', 5)
    pruner_warmup = pruner_cfg.get('n_warmup_steps', 1)
    pruner_interval = pruner_cfg.get('interval_steps', 1)
    
    try:
        study = optuna.create_study(
            storage=storage,
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=pruner_startup,
                n_warmup_steps=pruner_warmup,
                interval_steps=pruner_interval
            ),
            sampler=sampler,
            load_if_exists=True  # Resume existing study if found
        )
        print("🔪 MedianPruner configured from YAML:")
        print(f"   n_startup_trials={pruner_startup}, n_warmup_steps={pruner_warmup}, interval_steps={pruner_interval}")
    except Exception as e:
        raise RuntimeError(f"Failed to create Optuna study: {e}") from e

    # Log study info
    if storage:
        existing_trials = len(study.trials)
        if existing_trials > 0:
            print(f"📊 Resuming existing study with {existing_trials} completed trials")
            if existing_trials >= 15:  # Past GP startup phase
                print("   ℹ️  GP surrogate model will use existing trial data")
        else:
            print("📊 Starting new study")

    return study, wandbc


def main():
    parser = argparse.ArgumentParser(description='Distillation hyperparameter tuning for HOSER with Optuna')
    parser.add_argument('--dataset', type=str, default='Beijing', help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HOSER dataset')
    parser.add_argument('--config', type=str, default='config/Beijing.yaml', help='Base config file (contains optuna settings)')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:////mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db', 
                       help='Optuna storage URL (default: backup drive). Use "memory" for in-memory (no persistence)')
    
    # Optional overrides (if not provided, read from config YAML)
    parser.add_argument('--n_trials', type=int, default=None, help='Override number of trials (default: from config)')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs per trial (default: from config)')
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Load config to get Optuna settings
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get Optuna settings from config, with CLI overrides
    optuna_cfg = config.get('optuna', {})
    n_trials = args.n_trials if args.n_trials is not None else optuna_cfg.get('n_trials', 30)
    max_epochs = args.max_epochs if args.max_epochs is not None else optuna_cfg.get('max_epochs', 3)
    
    # Get pruner and sampler settings from config
    pruner_cfg = optuna_cfg.get('pruner', {})
    sampler_cfg = optuna_cfg.get('sampler', {})
    
    # Create study name with timestamp (if not provided)
    if args.study_name:
        study_name = args.study_name
        print(f"📌 Using specified study name: {study_name}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"hoser_tuning_{timestamp}"
        print(f"📌 Generated study name: {study_name}")
    
    # Validate and prepare storage URL with robust path handling
    try:
        storage_url = validate_and_prepare_storage(args.storage)
    except ValueError as e:
        print(f"❌ Storage validation error: {e}")
        sys.exit(1)
    
    print()
    print(f"🔍 Starting Optuna study: {study_name}")
    print("🧪 Mode: Distillation-only (tuning lambda, temperature, window)")
    print(f"📊 Trials: {n_trials} (from {'CLI' if args.n_trials else 'config'})")
    print(f"📈 Epochs: {max_epochs} per trial (from {'CLI' if args.max_epochs else 'config'})")
    print(f"📁 Dataset: {args.data_dir}")
    print(f"⚙️  Base config: {args.config}")
    print()

    # Create study and WandB callback with error handling
    try:
        study, wandbc = create_study_with_wandb(
            config.get('wandb', {}).get('project', 'hoser-optuna-tuning'),
            study_name, 
            storage_url,
            pruner_cfg=pruner_cfg,
            sampler_cfg=sampler_cfg
        )
    except RuntimeError as e:
        print(f"❌ Failed to create study: {e}")
        sys.exit(1)
    
    # Create objective function
    objective = HOSERObjective(
        base_config_path=args.config,
        data_dir=args.data_dir,
        max_epochs=max_epochs
    )
    
    # Run optimization with proper callback handling
    try:
        callbacks = [wandbc] if wandbc is not None else []
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=callbacks,
            gc_after_trial=True,
            show_progress_bar=True
        )
        
        # Print results
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION COMPLETE!")
        print(f"📊 Best trial: {study.best_trial.number}")
        print(f"📈 Best value: {study.best_value:.4f}")
        print("🔧 Best parameters:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        print("="*60)
        
        # Preserve best trial artifacts
        best_trial_num = study.best_trial.number
        best_seed = 42 + best_trial_num
        
        # Create preserved directories with absolute paths
        results_base = os.path.abspath(f"./optuna_results/{study_name}")
        preserved_dir = os.path.join(results_base, "preserved_models")
        os.makedirs(preserved_dir, exist_ok=True)
        print(f"📁 Creating preserved models directory: {preserved_dir}")
        
        # Copy best trial artifacts with robust path handling
        best_trial_dirs = {
            os.path.abspath(f"./save/Beijing/seed{best_seed}_distill"): 
                os.path.join(preserved_dir, f"best_trial_{best_trial_num}_model"),
            os.path.abspath(f"./tensorboard_log/Beijing/seed{best_seed}_distill"): 
                os.path.join(preserved_dir, f"best_trial_{best_trial_num}_tensorboard"),
            os.path.abspath(f"./log/Beijing/seed{best_seed}_distill"): 
                os.path.join(preserved_dir, f"best_trial_{best_trial_num}_logs")
        }
        
        for src, dst in best_trial_dirs.items():
            if os.path.exists(src):
                try:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    print(f"🔒 Preserved best trial artifacts: {dst}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not copy {src} to {dst}: {e}")
        
        # Save study results with robust file handling
        try:
            with open(os.path.join(results_base, "best_params.json"), 'w') as f:
                json.dump(study.best_params, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save best_params.json: {e}")

        preserved_models = {
            "best_trial": os.path.join(preserved_dir, f"best_trial_{best_trial_num}_model", "best.pth")
        }

        try:
            with open(os.path.join(results_base, "study_summary.json"), 'w') as f:
                json.dump({
                    "study_name": study_name,
                    "n_trials": len(study.trials),
                    "best_value": study.best_value,
                    "best_trial": study.best_trial.number,
                    "best_params": study.best_params,
                    "mode": "distillation_only",
                    "preserved_models": preserved_models,
                    "storage_url": storage_url if storage_url else "in-memory"
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save study_summary.json: {e}")
        
        print(f"\n💾 Results saved to: {results_base}")
        print("🔒 Preserved models:")
        print(f"   Best trial ({best_trial_num}): {preserved_models['best_trial']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


if __name__ == '__main__':
    main()

