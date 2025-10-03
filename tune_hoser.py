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

TRAINING STRATEGY (Fast Iteration - Option B):
  • 5 epochs per trial for rapid exploration (~4.5h per full trial @ 54min/epoch)
  • Minimum 3 epochs before pruning (~2.7h minimum)
  • Focus: Many trials to explore hyperparameter space quickly
  • Trade-off: Less convergence per trial, but more comprehensive search

PRUNING STRATEGY (HyperbandPruner - Moderate):
  • min_resource=3: All trials run at least 3 epochs before pruning
  • reduction_factor=3: Keeps top 1/3 of trials at each evaluation rung
  • Adaptive: Allocates more resources to promising trials
  
  Expected pruning rate: ~50-60% of trials stopped at 3-4 epochs
  This saves ~2h per pruned trial while gathering early performance data

SAMPLING STRATEGY (TPESampler - Adaptive):
  • Trials 0-4: Random/startup trials (exploration)
  • Trial 5+: TPE algorithm active (Bayesian optimization)
  • Note: Only SUCCESSFUL trials count toward n_startup_trials
  • Conditional search space: distill_enable branches the parameter tree

STORAGE & CRASH RECOVERY:
  Default storage: /mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db
  Heartbeat: 60s | Grace period: 120s
  Study can be resumed after crashes/interruptions.

TIME ESTIMATE (25 trials, Option B - Fast Iteration):
  Baseline: 54 minutes/epoch (9835 batches @ 3 it/s)
  
  • Full trial (5 epochs): ~4.5 hours
  • Pruned trial (3 epochs avg): ~2.7 hours
  
  Expected breakdown (assuming 50% pruned):
  • 13 complete trials: 13 × 4.5h = 58.5h
  • 12 pruned trials: 12 × 2.7h = 32.4h
  • Total: ~91 hours (~3.8 days)
  
  REALISTIC 24-HOUR PLAN:
  • ~6-7 trials (mix of complete and pruned)
  • Enough for TPE to activate (need 5 successful)
  • Run study for 3-4 days for full 25 trials
  
  For true 24h completion, reduce n_trials to 8-10 in config.

CONFIGURATION:
  All settings in config/Beijing.yaml under 'optuna' section.
  CLI args override YAML defaults.

USAGE:
  # Standard run (uses Beijing.yaml defaults: 25 trials, 5 epochs)
  uv run python tune_hoser.py --data_dir /home/matt/Dev/HOSER-dataset
  
  # Quick 24-hour run (8-10 trials)
  uv run python tune_hoser.py --n_trials 10 --data_dir /home/matt/Dev/HOSER-dataset

  # Resume existing study
  uv run python tune_hoser.py --study_name my_study --data_dir /home/matt/Dev/HOSER-dataset

  # Override epochs per trial
  uv run python tune_hoser.py --max_epochs 8 --data_dir /home/matt/Dev/HOSER-dataset
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
        self.base_seed = self.base_config.get('training', {}).get('seed', 42)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.
        
        Suggests hyperparameters including whether to use distillation.
        Returns validation accuracy to maximize.
        """
        # Step 1: Suggest all hyperparameters (defines search space)
        hparams = self._suggest_hyperparameters(trial)
        
        # Show what mode we're running
        mode = "distilled" if hparams['distill_enable'] else "vanilla baseline"
        print(f"🔬 Running trial {trial.number} ({mode})")
        
        # Step 2: Create trial config from hyperparameters
        config = self._create_trial_config(trial, hparams)

        # Step 3: Run training and return metric
        temp_config_path = None
        trial_succeeded = False
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                temp_config_path = f.name

            result = self._run_training_trial(trial, temp_config_path)
            trial_succeeded = True  # Mark trial as successful
            
            # ✅ PRESERVE MODEL IMMEDIATELY AFTER SUCCESSFUL TRIAL
            self._preserve_trial_artifacts(trial.number, mode)
            
            return result
        except optuna.TrialPruned:
            # Pruned trials are also successful (completed some epochs)
            trial_succeeded = True
            self._preserve_trial_artifacts(trial.number, mode, pruned=True)
            raise
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            
            # Only cleanup if trial failed (not succeeded or pruned)
            if not trial_succeeded:
                print(f"🧹 Trial {trial.number} failed - cleaning up artifacts")
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
        # Trial 0: vanilla baseline (no distillation)
        # Trials 1+: tune distillation hyperparameters
        if trial.number == 0:
            distill_enable = False
        else:
            distill_enable = True
        
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
                seed=self.base_seed + trial.number,  # Deterministic but different per trial
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
    
    def _preserve_trial_artifacts(self, trial_number: int, mode: str, pruned: bool = False):
        """
        Preserve model artifacts immediately after a successful or pruned trial.
        Creates a unique directory for each trial to avoid overwriting.
        
        Args:
            trial_number: The trial number
            mode: "vanilla baseline" or "distilled"
            pruned: Whether this trial was pruned (still saved, just stopped early)
        """
        trial_seed = self.base_seed + trial_number
        status = "pruned" if pruned else "complete"
        mode_short = "vanilla" if "vanilla" in mode else "distilled"
        
        # Source directories (where training saves models)
        src_dirs = {
            "model": os.path.abspath(f"./save/Beijing/seed{trial_seed}_distill"),
            "tensorboard": os.path.abspath(f"./tensorboard_log/Beijing/seed{trial_seed}_distill"),
            "logs": os.path.abspath(f"./log/Beijing/seed{trial_seed}_distill")
        }
        
        # Destination: optuna_trials/trial_NNN_{mode}_{status}/
        trial_dir = os.path.abspath(f"./optuna_trials/trial_{trial_number:03d}_{mode_short}_{status}")
        os.makedirs(trial_dir, exist_ok=True)
        
        print(f"💾 Preserving trial {trial_number} artifacts to {trial_dir}")
        
        preserved_count = 0
        for artifact_type, src_path in src_dirs.items():
            if os.path.exists(src_path):
                dst_path = os.path.join(trial_dir, artifact_type)
                try:
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    preserved_count += 1
                except Exception as e:
                    print(f"⚠️  Warning: Could not preserve {artifact_type}: {e}")
        
        if preserved_count > 0:
            print(f"✅ Trial {trial_number}: Preserved {preserved_count} artifact type(s)")
        else:
            print(f"⚠️  Trial {trial_number}: No artifacts found to preserve")
        
        # After preserving, clean up original directories to save space
        for src_path in src_dirs.values():
            if os.path.exists(src_path):
                shutil.rmtree(src_path, ignore_errors=True)
    
    def _cleanup_trial_artifacts(self, trial_number: int):
        """
        Clean up artifacts from a FAILED trial only.
        Successful/pruned trials are preserved via _preserve_trial_artifacts().
        """
        trial_seed = self.base_seed + trial_number
        trial_dirs = [
            os.path.abspath(f"./save/Beijing/seed{trial_seed}_distill"),
            os.path.abspath(f"./tensorboard_log/Beijing/seed{trial_seed}_distill"),
            os.path.abspath(f"./log/Beijing/seed{trial_seed}_distill")
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
    max_epochs: int,
    storage_url: Optional[str] = None,
    pruner_cfg: Optional[Dict[str, Any]] = None,
    sampler_cfg: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Create Optuna study with WandB integration and optional persistent storage.
    
    Args:
        project_name: WandB project name
        study_name: Unique study identifier
        max_epochs: Max epochs per trial (used for HyperbandPruner)
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
            metric_name="best_val_acc"  # Must match the value returned by objective function
        )
    except Exception as e:
        print(f"⚠️  Warning: WandB callback creation failed: {e}")
        print("   Continuing without WandB integration")
        wandbc = None

    # Use TPE sampler with settings from config
    tpe_startup = sampler_cfg.get('n_startup_trials', 10)
    seed = sampler_cfg.get('seed', 42)
    multivariate = sampler_cfg.get('multivariate', True)
    group = sampler_cfg.get('group', True)

    try:
        # TPESampler is efficient and supports pruning and conditional search spaces well.
        # group=True and multivariate=True are recommended for conditional search spaces.
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=tpe_startup,
            multivariate=multivariate,
            group=group
        )
        print(f"🔬 Using TPE Sampler (n_startup_trials={tpe_startup}, seed={seed}, multivariate={multivariate}, group={group} from config)")
    except (ImportError, AttributeError) as e:
        print(f"⚠️  TPESampler not available ({e}), falling back to RandomSampler")
        sampler = optuna.samplers.RandomSampler(seed=seed)

    # Configure storage with crash recovery
    storage = None
    if storage_url:
        try:
            storage = RDBStorage(
                url=storage_url,
                heartbeat_interval=60,  # Update heartbeat every 60 seconds
                grace_period=120,  # Allow 120 seconds before marking trial as failed
                failed_trial_callback=RetryFailedTrialCallback(max_retry=1)  # Retry failed trials once
            )
            print(f"💾 Using persistent storage: {storage_url}")
            print("   Heartbeat interval: 60s | Grace period: 120s | Retries: 1")
            print("   ✅ Study will survive crashes and can be resumed")
        except Exception as e:
            raise RuntimeError(f"Failed to create storage: {e}") from e
    else:
        print("⚠️  WARNING: Using in-memory storage (no crash recovery)")
        print("   Consider using --storage sqlite:///optuna_hoser.db for persistence")

    # Create Optuna study with robust error handling
    # HyperbandPruner with settings from config
    min_resource = pruner_cfg.get('min_resource', 1)
    reduction_factor = pruner_cfg.get('reduction_factor', 3)
    
    try:
        study = optuna.create_study(
            storage=storage,
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_epochs,
                reduction_factor=reduction_factor
            ),
            sampler=sampler,
            load_if_exists=True  # Resume existing study if found
        )
        print("🔪 HyperbandPruner configured from YAML:")
        print(f"   min_resource={min_resource}, max_resource={max_epochs}, reduction_factor={reduction_factor}")
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
    print("🧪 Mode: Distillation tuning (trial 0 = vanilla baseline, trials 1+ = distillation hyperparameters)")
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
            max_epochs,
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
        
        # All successful trials are already preserved in optuna_trials/ directory
        # Create a summary directory with quick access to the best trial
        best_trial_num = study.best_trial.number
        
        # Create results summary directory
        results_base = os.path.abspath(f"./optuna_results/{study_name}")
        os.makedirs(results_base, exist_ok=True)
        
        # Find the preserved best trial directory
        best_trial_mode = "vanilla" if not study.best_trial.params.get('distill_enable', True) else "distilled"
        best_trial_status = "pruned" if study.best_trial.state == optuna.trial.TrialState.PRUNED else "complete"
        best_trial_src = os.path.abspath(f"./optuna_trials/trial_{best_trial_num:03d}_{best_trial_mode}_{best_trial_status}")
        best_trial_link = os.path.join(results_base, "best_trial")
        
        # Create symlink or copy to best trial for quick access
        if os.path.exists(best_trial_src):
            try:
                # Try symlink first (more efficient)
                if os.path.islink(best_trial_link) or os.path.exists(best_trial_link):
                    if os.path.islink(best_trial_link):
                        os.unlink(best_trial_link)
                    else:
                        shutil.rmtree(best_trial_link)
                os.symlink(best_trial_src, best_trial_link, target_is_directory=True)
                print(f"✅ Created symlink to best trial: {best_trial_link} → trial_{best_trial_num:03d}")
            except (OSError, NotImplementedError) as e:
                # Symlink failed (e.g., Windows without admin), fallback to copy
                print(f"⚠️  Symlink not supported, copying instead: {e}")
                try:
                    shutil.copytree(best_trial_src, best_trial_link, dirs_exist_ok=True)
                    print(f"✅ Copied best trial to: {best_trial_link}")
                except Exception as e2:
                    print(f"⚠️  Could not create best trial reference: {e2}")
        else:
            print(f"⚠️  Warning: Best trial artifacts not found at {best_trial_src}")
        
        # Save study results with robust file handling
        try:
            with open(os.path.join(results_base, "best_params.json"), 'w') as f:
                json.dump(study.best_params, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save best_params.json: {e}")

        # Count successful trials
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        try:
            with open(os.path.join(results_base, "study_summary.json"), 'w') as f:
                json.dump({
                    "study_name": study_name,
                    "n_trials_total": len(study.trials),
                    "n_trials_complete": n_complete,
                    "n_trials_pruned": n_pruned,
                    "n_trials_failed": n_failed,
                    "best_value": study.best_value,
                    "best_trial": study.best_trial.number,
                    "best_params": study.best_params,
                    "preserved_trials_dir": os.path.abspath("./optuna_trials"),
                    "best_trial_link": best_trial_link,
                    "storage_url": storage_url if storage_url else "in-memory"
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save study_summary.json: {e}")
        
        print(f"\n💾 Results saved to: {results_base}")
        print(f"📁 Preserved trials: ./optuna_trials/ ({n_complete + n_pruned} successful trials)")
        print(f"🏆 Best trial: {best_trial_link}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        # Successful trials are already preserved in optuna_trials/
        # Just create a reference to the best trial
        if study.best_trial is not None:
            print("💾 Creating reference to best trial...")
            try:
                best_trial_num = study.best_trial.number
                results_base = os.path.abspath(f"./optuna_results/{study_name}")
                os.makedirs(results_base, exist_ok=True)
                
                best_trial_mode = "vanilla" if not study.best_trial.params.get('distill_enable', True) else "distilled"
                best_trial_status = "pruned" if study.best_trial.state == optuna.trial.TrialState.PRUNED else "complete"
                best_trial_src = os.path.abspath(f"./optuna_trials/trial_{best_trial_num:03d}_{best_trial_mode}_{best_trial_status}")
                best_trial_link = os.path.join(results_base, "best_trial")
                
                if os.path.exists(best_trial_src):
                    shutil.copytree(best_trial_src, best_trial_link, dirs_exist_ok=True)
                    print(f"✅ Best trial preserved at: {best_trial_link}")
                else:
                    print(f"⚠️  Best trial artifacts not found at {best_trial_src}")
            except Exception as e:
                print(f"⚠️  Could not create best trial reference: {e}")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


if __name__ == '__main__':
    main()

