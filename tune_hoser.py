#!/usr/bin/env python3
"""
ULTRA-FAST Optuna hyperparameter tuning for HOSER - 24 HOUR TARGET

This script runs hyperparameter optimization optimized for speed:
  • 3 epochs per trial (~2.4 hours each, down from 19h)
  • 80 trials (maximum coverage in 24h)
  • AGGRESSIVE MedianPruner (no warmup, prune after epoch 1)
  • Intermediate reporting every epoch for maximum efficiency

Tuned parameters for distillation:
- lambda: KL divergence weight (0.001-0.1, log scale)
- temperature: Softmax temperature (1.0-5.0)
- window: Context window size (2, 4)

Fixed parameters (architectural choices):
- grid_size: 0.001 (from base config)
- downsample: 1 (from base config)

AGGRESSIVE Pruning Strategy:
  Trial 1-3: Run to completion (establish baseline, ~7h)
  Trial 4+:  Prune IMMEDIATELY if val_acc < median after epoch 1
  
  No warmup period - if you're bad at epoch 1, you're out!
  Saves ~1.6h per pruned trial (vs ~0.8h wasted on bad trial)

Storage & Crash Recovery:
  By default, stores Optuna study on backup drive for extra safety:
    /mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db
  The study can be resumed after crashes/interruptions.
  Heartbeat interval: 60s | Grace period: 120s

Time Estimate (24h target):
  • Trial 1-3 (startup): 3 × 2.4h = 7.2h
  • Trial 4-8 (GP init): 5 × 2.4h = 12h  
  • Trial 9-80 (GP optimized): 72 trials, ~50% pruned after epoch 1
    - 36 complete: 36 × 2.4h = 86.4h
    - 36 pruned: 36 × 0.8h = 28.8h
    - Subtotal: 115.2h
  • Total: 7.2 + 12 + 115.2 = 134.4h (~5.6 days)

WAIT - still too long! Let's be more aggressive:
  With 60% pruning rate:
    - 3 startup: 7.2h
    - 5 GP init: 12h
    - 72 remaining: 29 complete (69.6h) + 43 pruned (34.4h) = 104h
    - Total: 123.4h (~5.1 days)

For TRUE 24h, need to:
  1. Skip vanilla baseline (--skip_baseline)
  2. Reduce to 3 epochs (done)
  3. Target 70-80% pruning rate (very aggressive)
  4. Run 20-25 trials only (not 80)

Realistic 24h settings:
  uv run python tune_hoser.py --n_trials 25 --skip_baseline --data_dir ...
  
  Expected: 3 startup (7h) + 22 trials @ 70% pruned = 17h
  Total: 24h

Configuration:
  All tuning settings are in config/Beijing.yaml under the 'optuna' section:
    - n_trials: Number of trials (default: 25 for 24h target)
    - max_epochs: Epochs per trial (default: 3 = ~2.4h/trial)
    - skip_baseline: Skip vanilla baseline (default: true)
    - pruner: MedianPruner settings
    - sampler: GPSampler settings

  CLI args can override YAML settings if needed.

Usage:
  # Standard 24-hour run (RECOMMENDED - uses Beijing.yaml defaults)
  uv run python tune_hoser.py --data_dir /home/matt/Dev/HOSER-dataset

  # Override trials from CLI (overrides YAML)
  uv run python tune_hoser.py --n_trials 50 --data_dir /home/matt/Dev/HOSER-dataset

  # Resume existing study
  uv run python tune_hoser.py --data_dir /home/matt/Dev/HOSER-dataset --study_name my_study

  # Use local storage instead of backup drive
  uv run python tune_hoser.py --data_dir /home/matt/Dev/HOSER-dataset --storage sqlite:///optuna_hoser.db
"""

import os
import sys
import argparse
import tempfile
import shutil
import gc
import math
import torch
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from typing import Dict, Any, Optional
import yaml
import json
from datetime import datetime

# Training logic will be imported dynamically to avoid circular imports


class HOSERObjective:
    """Objective function for Optuna optimization of HOSER models."""

    def __init__(self, base_config_path: str, data_dir: str, max_epochs: int = 3, skip_baseline: bool = False):
        """
        Initialize HOSER optimization objective.
        
        Args:
            base_config_path: Path to base YAML config
            data_dir: Path to HOSER dataset
            max_epochs: Max epochs per trial (default 3 for ultra-fast tuning)
            skip_baseline: Skip vanilla baseline trial
        """
        self.base_config_path = base_config_path
        self.data_dir = data_dir
        self.max_epochs = max_epochs
        self.skip_baseline = skip_baseline
        self.trial_counter = 0
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function called by Optuna for each trial."""
        self.trial_counter += 1

        # For trial 0, force vanilla (baseline) unless skip_baseline is enabled
        if trial.number == 0 and not self.skip_baseline:
            trial_type = 'vanilla'
            print(f"🔬 Running vanilla baseline (trial {trial.number})")
        else:
            # Suggest trial type: vanilla HOSER or distilled HOSER
            trial_type = trial.suggest_categorical('trial_type', ['vanilla', 'distilled'])
            trial_mode = "distilled" if trial_type == 'distilled' else "vanilla"
            print(f"🔬 Running {trial_mode} trial (trial {trial.number})")

        # Create trial-specific config
        config = self._create_trial_config(trial, trial_type)

        # Create temporary config file for this trial
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name

        try:
            # Run training with trial config
            result = self._run_training_trial(trial, temp_config_path, trial_type)
            return result
        finally:
            # Cleanup
            os.unlink(temp_config_path)
            self._cleanup_trial_artifacts(trial.number)
            gc.collect()
            torch.cuda.empty_cache()
    
    def _create_trial_config(self, trial: optuna.Trial, trial_type: str) -> Dict[str, Any]:
        """Create configuration for a specific trial."""
        config = self.base_config.copy()
        
        # Keep all original HOSER parameters from base config - only tune distillation
        # Set max_epoch for tuning (shorter than production)
        config['optimizer_config']['max_epoch'] = self.max_epochs
        
        # Set data directory
        config['data_dir'] = self.data_dir
        
        # Configure distillation or vanilla mode
        if trial_type == 'distilled':
            config['distill']['enable'] = True
            # Tune key distillation parameters
            config['distill']['lambda'] = trial.suggest_float(
                'distill_lambda', 0.001, 0.1, log=True
            )
            config['distill']['temperature'] = trial.suggest_float(
                'distill_temperature', 1.0, 5.0
            )
            config['distill']['window'] = trial.suggest_categorical(
                'distill_window', [2, 4]
            )
            # Keep grid_size and downsample fixed (not tuned)
            # These are architectural choices rather than hyperparameters
        else:
            config['distill']['enable'] = False
        
        # WandB config for trial (use settings from base config, just override run_name)
        config['wandb']['run_name'] = f"trial_{trial.number:03d}_{trial_type}"
        # Add trial-specific tags to existing tags
        existing_tags = config['wandb'].get('tags', [])
        config['wandb']['tags'] = existing_tags + ['distill-tuning', trial_type]
        
        return config
    
    def _run_training_trial(self, trial: optuna.Trial, config_path: str, trial_type: str) -> float:
        """Run a single training trial and return the metric to optimize."""
        
        # Import training function
        from train_with_distill import main as train_main
        
        try:
            # Call training with clean API - no more sys.argv monkeypatching!
            result = train_main(
                dataset='Beijing',
                config_path=config_path,
                seed=42 + trial.number,  # Deterministic but different per trial
                cuda=0,
                data_dir=self.data_dir,
                return_metrics=True,
                optuna_trial=trial  # Pass trial for intermediate reporting
            )

            if result is None:
                raise optuna.TrialPruned("Training returned no metrics")

            # Validate that metrics are reasonable (not NaN or infinite)
            if (math.isnan(result['best_val_acc']) or math.isinf(result['best_val_acc']) or
                math.isnan(result['final_val_acc']) or math.isinf(result['final_val_acc'])):
                print(f"⚠️  Trial {trial.number}: Invalid validation metrics detected, pruning")
                raise optuna.TrialPruned("Invalid validation metrics (NaN/inf)")

            # Return final metric to maximize (validation accuracy)
            final_metric = result['best_val_acc']

            # Log additional metrics to trial
            trial.set_user_attr('final_val_acc', result['final_val_acc'])
            trial.set_user_attr('final_val_mape', result['final_val_mape'])
            trial.set_user_attr('best_val_acc', result['best_val_acc'])
            trial.set_user_attr('final_lr', result.get('final_lr', 0.0))

            return final_metric

        except optuna.TrialPruned:
            # Re-raise pruned exceptions
            raise
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned(str(e))
    
    def _cleanup_trial_artifacts(self, trial_number: int):
        """Clean up artifacts from a completed trial, but preserve vanilla baseline (trial 0) and best trial."""
        # Always preserve trial 0 (vanilla baseline)
        if trial_number == 0:
            print(f"🔒 Preserving vanilla baseline trial {trial_number}")
            return
        
        # We'll preserve the best trial later in the main function
        trial_dirs = [
            f"./save/Beijing/seed{42 + trial_number}_distill",
            f"./tensorboard_log/Beijing/seed{42 + trial_number}_distill",
            f"./log/Beijing/seed{42 + trial_number}_distill",
            f"./optuna_trials/trial_{trial_number:03d}_vanilla",
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
    # Note: Uses same project as individual trials for unified dashboard
    wandb_kwargs = {
        "project": "hoser-distill-optuna",  # Match config/Beijing.yaml
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
    gp_startup = sampler_cfg.get('n_startup_trials', 8)
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
    pruner_startup = pruner_cfg.get('n_startup_trials', 3)
    pruner_warmup = pruner_cfg.get('n_warmup_steps', 0)
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
        print(f"🔪 MedianPruner configured from YAML:")
        print(f"   n_startup_trials={pruner_startup}, n_warmup_steps={pruner_warmup}, interval_steps={pruner_interval}")
    except Exception as e:
        raise RuntimeError(f"Failed to create Optuna study: {e}") from e

    # Log study info
    if storage:
        existing_trials = len(study.trials)
        if existing_trials > 0:
            print(f"📊 Resuming existing study with {existing_trials} completed trials")
            # Validate resumed study has compatible sampler/pruner
            if existing_trials >= 15:  # Past GP startup phase
                print("   ℹ️  GP surrogate model will use existing trial data")
        else:
            print("📊 Starting new study")

    return study, wandbc


def main():
    parser = argparse.ArgumentParser(description='Ultra-fast hyperparameter tuning for HOSER with Optuna')
    parser.add_argument('--dataset', type=str, default='Beijing', help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HOSER dataset')
    parser.add_argument('--config', type=str, default='config/Beijing.yaml', help='Base config file (contains optuna settings)')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:////mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db', 
                       help='Optuna storage URL (default: backup drive). Use "memory" for in-memory (no persistence)')
    
    # Optional overrides (if not provided, read from config YAML)
    parser.add_argument('--n_trials', type=int, default=None, help='Override number of trials (default: from config)')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs per trial (default: from config)')
    parser.add_argument('--skip_baseline', action='store_true', help='Override skip_baseline setting (default: from config)')
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
    n_trials = args.n_trials if args.n_trials is not None else optuna_cfg.get('n_trials', 25)
    max_epochs = args.max_epochs if args.max_epochs is not None else optuna_cfg.get('max_epochs', 3)
    skip_baseline = args.skip_baseline or optuna_cfg.get('skip_baseline', False)
    
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
    print(f"📊 Trials: {n_trials} (from {'CLI' if args.n_trials else 'config'})")
    print(f"📈 Epochs: {max_epochs} per trial (from {'CLI' if args.max_epochs else 'config'})")
    print(f"📁 Dataset: {args.data_dir}")
    print(f"⚙️  Base config: {args.config}")
    if skip_baseline:
        print("🚫 Skipping vanilla baseline (trial 0) - starting directly with distillation")
    else:
        print("✅ Including vanilla baseline (trial 0) for comparison")
    print()

    # Create study and WandB callback with error handling
    try:
        study, wandbc = create_study_with_wandb(
            "hoser-optuna-tuning", 
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
        max_epochs=max_epochs,
        skip_baseline=skip_baseline
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
        
        # Copy vanilla baseline (trial 0) artifacts if it exists (unless skipped)
        if not args.skip_baseline:
            vanilla_seed = 42
            vanilla_dirs = {
                os.path.abspath(f"./save/Beijing/seed{vanilla_seed}_distill"): 
                    os.path.join(preserved_dir, "vanilla_trial_0_model"),
                os.path.abspath(f"./tensorboard_log/Beijing/seed{vanilla_seed}_distill"): 
                    os.path.join(preserved_dir, "vanilla_trial_0_tensorboard"),
                os.path.abspath(f"./log/Beijing/seed{vanilla_seed}_distill"): 
                    os.path.join(preserved_dir, "vanilla_trial_0_logs")
            }

            for src, dst in vanilla_dirs.items():
                if os.path.exists(src):
                    try:
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                        print(f"🔒 Preserved vanilla baseline artifacts: {dst}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not copy {src} to {dst}: {e}")

        # Save study results with robust file handling
        try:
            with open(os.path.join(results_base, "best_params.json"), 'w') as f:
                json.dump(study.best_params, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save best_params.json: {e}")

        # Determine vanilla trial number (0 if not skipped, -1 if skipped)
        vanilla_trial_num = 0 if not args.skip_baseline else -1

        preserved_models = {
            "best_trial": os.path.join(preserved_dir, f"best_trial_{best_trial_num}_model", "best.pth")
        }

        if not args.skip_baseline:
            preserved_models["vanilla_baseline"] = os.path.join(preserved_dir, "vanilla_trial_0_model", "best.pth")

        try:
            with open(os.path.join(results_base, "study_summary.json"), 'w') as f:
                json.dump({
                    "study_name": study_name,
                    "n_trials": len(study.trials),
                    "best_value": study.best_value,
                    "best_trial": study.best_trial.number,
                    "best_params": study.best_params,
                    "vanilla_trial": vanilla_trial_num,
                    "skip_baseline": args.skip_baseline,
                    "preserved_models": preserved_models,
                    "storage_url": storage_url if storage_url else "in-memory"
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save study_summary.json: {e}")
        
        print(f"\n💾 Results saved to: {results_base}")
        print("🔒 Preserved models:")
        print(f"   Best trial ({best_trial_num}): {preserved_models['best_trial']}")
        if not args.skip_baseline:
            print(f"   Vanilla baseline (0): {preserved_models['vanilla_baseline']}")
        else:
            print("   Vanilla baseline: Skipped (--skip_baseline was used)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


if __name__ == '__main__':
    main()
