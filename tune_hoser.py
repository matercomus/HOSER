#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuning for HOSER with Optuna

This script optimizes distillation hyperparameters for HOSER using CMA-ES.
Vanilla baseline runs separately for WandB comparison (Phase 0).

TUNED HYPERPARAMETERS (all trials are distilled):
- lambda: KL divergence weight (0.001-0.1, log scale)
         Controls balance between teacher guidance and student loss
- temperature: Softmax temperature (1.0-5.0, linear)
              Higher values create softer probability distributions
- window: Context window size (2-8, integer)
         Number of neighboring timesteps to consider for teacher guidance

FIXED PARAMETERS (architectural choices):
- grid_size: 0.001 (from base config)
- downsample: 1 (from base config)
- All HOSER model architecture parameters

TRAINING STRATEGY (Fast Iteration):
  • 8 epochs per trial for convergence trends (~7.3h per full trial @ 54min/epoch)
  • Minimum 5 epochs before pruning (~4.6h minimum)
  • Focus: Many trials to explore hyperparameter space efficiently
  • Trade-off: Faster exploration with adaptive early stopping

PRUNING STRATEGY (HyperbandPruner - Moderate):
  • min_resource=5: All trials run at least 5 epochs before pruning
  • reduction_factor=3: Keeps top 1/3 of trials at each evaluation rung
  • Adaptive: Allocates more resources to promising trials
  
  Expected pruning rate: ~50-60% of trials stopped at 5-6 epochs
  This saves ~2h per pruned trial while gathering early performance data

SAMPLING STRATEGY (CmaEsSampler - Evolution Strategy):
  • All parameters continuous/integer (fully compatible with CMA-ES)
  • Adaptive covariance matrix evolution from trial 0
  • Optimal for 10-100 trials with continuous optimization
  • Consistent 3-parameter space enables CMA-ES from trial 0
  • No wasted random trials - all 12 trials use CMA-ES optimization

STORAGE & CRASH RECOVERY:
  Default storage: /mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db
  Heartbeat: 60s | Grace period: 120s
  Study can be resumed after crashes/interruptions.

TIME ESTIMATE (3-Phase Approach):
  Baseline: 54 minutes/epoch (9835 batches @ 3 it/s)
  
  PHASE 0: Vanilla Baseline (optional, for WandB comparison)
  • 1 run, 8 epochs, single seed: ~7.3 hours
  • Optional - can be disabled in config
  
  PHASE 1: Hyperparameter Search (12 distilled trials, 8 max epochs)
  • Full trial (8 epochs): ~7.3 hours
  • Pruned trial (5 epochs avg): ~4.6 hours
  • Expected: 7 complete + 5 pruned
  • Total Phase 1: ~74 hours (~3.1 days)
  
  PHASE 2: Final Evaluation (best config, 25 epochs)
  • 3 runs with different seeds: 3 × 23h = 69h (~2.9 days)
  • Or 1 run for quick results: 1 × 23h = 23h (~1 day)
  
  TOTAL TIMELINE:
  • Phase 0 + Phase 1 + Phase 2 (1 seed): ~104 hours (~4.3 days)
  • Phase 0 + Phase 1 + Phase 2 (3 seeds): ~150 hours (~6.3 days)
  • Skip Phase 0: saves ~7 hours
  
  CmaEsSampler optimizes all 12 trials - no random baseline needed!

CONFIGURATION:
  All settings in config/Beijing.yaml or config/porto_hoser.yaml under 'optuna' section.
  CLI args override YAML defaults.

USAGE:
  # Standard run (uses config defaults: 12 trials, 8 epochs)
  uv run python tune_hoser.py --config config/porto_hoser.yaml --data_dir /home/matt/Dev/HOSER-dataset-porto

  # Skip Phase 0 vanilla baseline (faster start)
  # Edit config: vanilla_baseline_pretune.enabled: false

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
from optuna.trial import TrialState
from typing import Dict, Any, Optional
import yaml
import json
from datetime import datetime


class HOSERObjective:
    """Objective function for Optuna optimization of HOSER distillation hyperparameters."""

    def __init__(self, base_config_path: str, data_dir: str, max_epochs: int = 3, dataset: str = 'Beijing'):
        """
        Initialize HOSER distillation optimization objective.
        
        Args:
            base_config_path: Path to base YAML config
            data_dir: Path to HOSER dataset
            max_epochs: Max epochs per trial (default 3 for ultra-fast tuning)
            dataset: Dataset name (default 'Beijing', can be 'porto_hoser', etc.)
        """
        self.base_config_path = base_config_path
        self.data_dir = data_dir
        self.max_epochs = max_epochs
        self.dataset = dataset
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        self.base_seed = self.base_config.get('training', {}).get('seed', 42)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.
        
        Suggests distillation hyperparameters and returns validation accuracy to maximize.
        All trials are distillation trials (vanilla baseline runs separately in Phase 0).
        """
        # Step 1: Suggest all hyperparameters (defines search space)
        hparams = self._suggest_hyperparameters(trial)
        
        print(f"🔬 Running trial {trial.number} (distilled)")
        
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
            self._preserve_trial_artifacts(trial.number, pruned=False)
            
            return result
        except optuna.TrialPruned:
            # Pruned trials are also successful (completed some epochs)
            trial_succeeded = True
            self._preserve_trial_artifacts(trial.number, pruned=True)
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
        
        All trials optimize distillation hyperparameters (vanilla baseline runs separately in Phase 0).
        
        Search space includes:
        - distill_lambda: KL divergence weight (log scale: 0.001 to 0.1)
                         Controls how much to weight teacher guidance vs student loss
        - distill_temperature: Softmax temperature (linear: 1.0 to 5.0)
                              Higher = softer distributions, more knowledge transfer
        - distill_window: Context window size (integer: 2-8)
                         Number of neighboring road segments to consider for teacher guidance
        
        All parameters are continuous/integer - fully compatible with CmaEsSampler.
        The consistent 3-parameter space allows CmaEsSampler to optimize from trial 0.
        
        Returns:
            Dict of hyperparameter names to suggested values
        """
        # All trials are distillation trials (vanilla baseline runs separately in Phase 0)
        hparams = {
            'distill_enable': True,
            'distill_lambda': trial.suggest_float('distill_lambda', 0.001, 0.1, log=True),
            'distill_temperature': trial.suggest_float('distill_temperature', 1.0, 5.0),
            'distill_window': trial.suggest_int('distill_window', 2, 8)
        }
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
        
        # All trials use distillation (vanilla baseline runs separately in Phase 0)
        config['distill']['enable'] = True
        config['distill']['lambda'] = hparams['distill_lambda']
        config['distill']['temperature'] = hparams['distill_temperature']
        config['distill']['window'] = hparams['distill_window']
        # Keep grid_size and downsample fixed (architectural choices, not tuned)
        
        # WandB config for trial
        config['wandb']['run_name'] = f"trial_{trial.number:03d}_distilled"
        existing_tags = config['wandb'].get('tags', [])
        config['wandb']['tags'] = existing_tags + ['hoser-tuning', 'distilled']
        
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
                dataset=self.dataset,
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
    
    def _preserve_trial_artifacts(self, trial_number: int, pruned: bool = False):
        """
        Preserve model artifacts immediately after a successful or pruned trial.
        Creates a unique directory for each trial to avoid overwriting.
        
        Args:
            trial_number: The trial number
            pruned: Whether this trial was pruned (still saved, just stopped early)
        """
        trial_seed = self.base_seed + trial_number
        status = "pruned" if pruned else "complete"
        
        # All trials are distilled
        trial_dir = os.path.abspath(f"./optuna_trials/trial_{trial_number:03d}_distilled_{status}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Source directories (where training saves models)
        src_dirs = {
            "model": os.path.abspath(f"./save/{self.dataset}/seed{trial_seed}_distill"),
            "tensorboard": os.path.abspath(f"./tensorboard_log/{self.dataset}/seed{trial_seed}_distill"),
            "logs": os.path.abspath(f"./log/{self.dataset}/seed{trial_seed}_distill")
        }
        
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
        
        # After preserving, clean up original directories and checkpoints to save space
        for src_path in src_dirs.values():
            if os.path.exists(src_path):
                # This also removes checkpoint_latest.pth inside save_dir
                shutil.rmtree(src_path, ignore_errors=True)
        
        print(f"🧹 Cleaned up trial {trial_number} working directories (checkpoints removed)")
    
    def _cleanup_trial_artifacts(self, trial_number: int):
        """
        Clean up artifacts from a FAILED trial only.
        Successful/pruned trials are preserved via _preserve_trial_artifacts().
        """
        trial_seed = self.base_seed + trial_number
        trial_dirs = [
            os.path.abspath(f"./save/{self.dataset}/seed{trial_seed}_distill"),
            os.path.abspath(f"./tensorboard_log/{self.dataset}/seed{trial_seed}_distill"),
            os.path.abspath(f"./log/{self.dataset}/seed{trial_seed}_distill")
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

    # Configure sampler based on config
    seed = sampler_cfg.get('seed', 42)
    sampler_type = sampler_cfg.get('sampler_type', 'tpe').lower()

    try:
        if sampler_type == 'cmaes':
            # CmaEsSampler: Best for continuous optimization with 10-100 trials
            # No need for n_startup_trials, works from trial 0
            sampler = optuna.samplers.CmaEsSampler(seed=seed)
            print(f"🔬 Using CmaES Sampler (seed={seed})")
            print("   Optimized for continuous hyperparameters with limited trial budget")
        elif sampler_type == 'random':
            sampler = optuna.samplers.RandomSampler(seed=seed)
            print(f"🔬 Using Random Sampler (seed={seed})")
        else:  # Default to TPE
            tpe_startup = sampler_cfg.get('n_startup_trials', 10)
            multivariate = sampler_cfg.get('multivariate', True)
            group = sampler_cfg.get('group', True)
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=tpe_startup,
                multivariate=multivariate,
                group=group
            )
            print(f"🔬 Using TPE Sampler (n_startup_trials={tpe_startup}, seed={seed})")
    except (ImportError, AttributeError) as e:
        print(f"⚠️  Requested sampler not available ({e}), falling back to RandomSampler")
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


def _run_final_evaluation(study: optuna.Study, base_config: Dict[str, Any], data_dir: str, 
                          final_run_cfg: Dict[str, Any], study_name: str, dataset: str = 'Beijing'):
    """
    Run final evaluation with best hyperparameters from the study.
    
    This runs a full training (e.g., 25 epochs) with the optimized hyperparameters
    to get proper evaluation metrics for publication/reporting.
    
    Args:
        study: Completed Optuna study with best trial
        base_config: Base configuration dict
        data_dir: Path to dataset
        final_run_cfg: Configuration for final run (max_epochs, seeds, etc.)
        study_name: Study name for organizing results
    """
    from train_with_distill import main as train_main
    
    max_epochs_final = final_run_cfg.get('max_epochs', 25)
    seeds = final_run_cfg.get('seeds', [42])  # Default: single run with seed 42
    
    print("\n📊 Running final evaluation with best hyperparameters:")
    print(f"   Best trial: {study.best_trial.number}")
    print(f"   Best val_acc (from search): {study.best_value:.4f}")
    print(f"   Hyperparameters: {study.best_params}")
    print(f"   Final epochs: {max_epochs_final}")
    print(f"   Seeds: {seeds}")
    print()
    
    # Create config with best hyperparameters
    final_config = copy.deepcopy(base_config)
    final_config['optimizer_config']['max_epoch'] = max_epochs_final
    final_config['data_dir'] = data_dir
    
    # Apply best hyperparameters
    if study.best_params.get('distill_enable', True):
        final_config['distill']['enable'] = True
        final_config['distill']['lambda'] = study.best_params['distill_lambda']
        final_config['distill']['temperature'] = study.best_params['distill_temperature']
        final_config['distill']['window'] = study.best_params['distill_window']
        mode = "distilled"
    else:
        final_config['distill']['enable'] = False
        mode = "vanilla"
    
    # Run training for each seed
    final_results = []
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"🎯 Final Run {seed_idx + 1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")
        
        # Update WandB config
        final_config['wandb']['run_name'] = f"final_{mode}_seed{seed}"
        final_config['wandb']['tags'] = ['beijing', 'distillation', 'final-eval', mode]
        
        # Save config
        final_config_path = f"./optuna_results/{study_name}/final_config_seed{seed}.yaml"
        os.makedirs(os.path.dirname(final_config_path), exist_ok=True)
        with open(final_config_path, 'w') as f:
            yaml.dump(final_config, f)
        
        try:
            result = train_main(
                dataset=dataset,
                config_path=final_config_path,
                seed=seed,
                cuda=0,
                data_dir=data_dir,
                return_metrics=True,
                optuna_trial=None  # No pruning for final run
            )
            
            final_results.append({
                'seed': seed,
                'best_val_acc': result['best_val_acc'],
                'final_val_acc': result['final_val_acc'],
                'final_val_mape': result['final_val_mape']
            })
            
            print(f"\n✅ Final Run {seed_idx + 1} Complete:")
            print(f"   Best val_acc: {result['best_val_acc']:.4f}")
            print(f"   Final val_acc: {result['final_val_acc']:.4f}")
            print(f"   Final val_mape: {result['final_val_mape']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Final run {seed_idx + 1} failed: {e}")
            final_results.append({'seed': seed, 'error': str(e)})
    
    # Save final results summary
    results_file = f"./optuna_results/{study_name}/final_evaluation_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'study_name': study_name,
                'best_hyperparameters': study.best_params,
                'search_phase_best_val_acc': study.best_value,
                'final_evaluation_runs': final_results,
                'final_config': {
                    'max_epochs': max_epochs_final,
                    'seeds': seeds
                }
            }, f, indent=2)
        print(f"\n💾 Final evaluation results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save final results: {e}")
    
    # Print summary
    successful_runs = [r for r in final_results if 'error' not in r]
    if successful_runs:
        avg_best_val_acc = sum(r['best_val_acc'] for r in successful_runs) / len(successful_runs)
        avg_final_val_acc = sum(r['final_val_acc'] for r in successful_runs) / len(successful_runs)
        avg_final_val_mape = sum(r['final_val_mape'] for r in successful_runs) / len(successful_runs)
        
        print("\n" + "="*60)
        print("📊 FINAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Successful runs: {len(successful_runs)}/{len(seeds)}")
        print(f"Average best val_acc: {avg_best_val_acc:.4f}")
        print(f"Average final val_acc: {avg_final_val_acc:.4f}")
        print(f"Average final val_mape: {avg_final_val_mape:.4f}")
        print("="*60)


def _run_vanilla_baseline(base_config: Dict[str, Any], data_dir: str, 
                          vanilla_cfg: Dict[str, Any], study_name: str, dataset: str = 'Beijing'):
    """
    Run full vanilla baseline for fair comparison with distilled models.
    
    This runs vanilla HOSER (distill.enable=False) for the same number of epochs
    as the final distilled runs, using the base configuration without any tuned
    hyperparameters (since vanilla has no distillation hyperparameters to tune).
    
    Args:
        base_config: Base configuration dict (uses optimizer_config as-is)
        data_dir: Path to dataset  
        vanilla_cfg: Configuration for vanilla baseline (max_epochs, seeds)
        study_name: Study name for organizing results
        dataset: Dataset name (default 'Beijing')
    """
    from train_with_distill import main as train_main
    
    max_epochs = vanilla_cfg.get('max_epochs', 25)
    seeds = vanilla_cfg.get('seeds', [42])
    
    print("\n📊 Running vanilla baseline (no distillation):")
    print(f"   Dataset: {dataset}")
    print(f"   Epochs: {max_epochs} (same as final distilled runs)")
    print(f"   Seeds: {seeds}")
    print("   Config: Uses base optimizer_config (no tuned hyperparameters)")
    print()
    
    # Create vanilla config
    vanilla_config = copy.deepcopy(base_config)
    vanilla_config['optimizer_config']['max_epoch'] = max_epochs
    vanilla_config['data_dir'] = data_dir
    vanilla_config['distill']['enable'] = False  # Critical: disable distillation
    
    # Run training for each seed
    vanilla_results = []
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"🎯 Vanilla Baseline {seed_idx + 1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")
        
        # Update WandB config
        vanilla_config['wandb']['run_name'] = f"vanilla_baseline_seed{seed}"
        vanilla_config['wandb']['tags'] = vanilla_config['wandb']['tags'][:] + ['vanilla', 'baseline', 'full-training']
        
        # Save config
        config_path = f"./optuna_results/{study_name}/vanilla_baseline/vanilla_config_seed{seed}.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(vanilla_config, f)
        
        try:
            result = train_main(
                dataset=dataset,
                config_path=config_path,
                seed=seed,
                cuda=0,
                data_dir=data_dir,
                return_metrics=True,
                optuna_trial=None  # No pruning for baseline run
            )
            
            vanilla_results.append({
                'seed': seed,
                'best_val_acc': result['best_val_acc'],
                'final_val_acc': result['final_val_acc'],
                'final_val_mape': result['final_val_mape']
            })
            
            print(f"\n✅ Vanilla Baseline {seed_idx + 1} Complete:")
            print(f"   Best val_acc: {result['best_val_acc']:.4f}")
            print(f"   Final val_acc: {result['final_val_acc']:.4f}")
            print(f"   Final val_mape: {result['final_val_mape']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Vanilla baseline {seed_idx + 1} failed: {e}")
            vanilla_results.append({'seed': seed, 'error': str(e)})
    
    # Save results
    results_file = f"./optuna_results/{study_name}/vanilla_baseline_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'study_name': study_name,
                'dataset': dataset,
                'vanilla_baseline_runs': vanilla_results,
                'config': {
                    'max_epochs': max_epochs,
                    'seeds': seeds,
                    'distillation_enabled': False
                }
            }, f, indent=2)
        print(f"\n💾 Vanilla baseline results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save results: {e}")
    
    # Print summary
    successful_runs = [r for r in vanilla_results if 'error' not in r]
    if successful_runs:
        avg_best_val_acc = sum(r['best_val_acc'] for r in successful_runs) / len(successful_runs)
        avg_final_val_acc = sum(r['final_val_acc'] for r in successful_runs) / len(successful_runs)
        avg_final_val_mape = sum(r['final_val_mape'] for r in successful_runs) / len(successful_runs)
        
        print("\n" + "="*60)
        print("📊 VANILLA BASELINE SUMMARY")
        print("="*60)
        print(f"Successful runs: {len(successful_runs)}/{len(seeds)}")
        print(f"Average best val_acc: {avg_best_val_acc:.4f}")
        print(f"Average final val_acc: {avg_final_val_acc:.4f}")
        print(f"Average final val_mape: {avg_final_val_mape:.4f}")
        print("="*60)


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
        max_epochs=max_epochs,
        dataset=args.dataset
    )
    
    # Phase 0: Optional vanilla baseline (for WandB comparison)
    pretune_vanilla_cfg = optuna_cfg.get('vanilla_baseline_pretune', {})
    if pretune_vanilla_cfg.get('enabled', True):  # Default: enabled
        print("\n" + "="*60)
        print("🚀 PHASE 0: VANILLA BASELINE (Pre-Tuning)")
        print("="*60)
        print("ℹ️  Running vanilla baseline before tuning for WandB comparison")
        print(f"   Epochs: {pretune_vanilla_cfg.get('max_epochs', max_epochs)}")
        print(f"   Seeds: {pretune_vanilla_cfg.get('seeds', [config.get('training', {}).get('seed', 42)])}")
        _run_vanilla_baseline(config, args.data_dir, pretune_vanilla_cfg, f"{study_name}_phase0", dataset=args.dataset)
    
    # Calculate remaining trials (only count successful ones)
    completed_trials = len([t for t in study.trials 
                            if t.state in [TrialState.COMPLETE, TrialState.PRUNED]])
    remaining_trials = max(0, n_trials - completed_trials)
    
    print(f"📊 Study progress: {completed_trials}/{n_trials} trials complete")
    
    if remaining_trials == 0:
        print("✅ All trials already completed! Skipping Phase 1.")
    else:
        # Phase 1: Hyperparameter tuning (all trials are distilled)
        print("\n" + "="*60)
        print("🚀 PHASE 1: HYPERPARAMETER SEARCH")
        print("="*60)
        print(f"ℹ️  Running {remaining_trials} more trials with CmaEsSampler")
        
        # Clean up stale/stuck trials before starting
        from optuna.storages import fail_stale_trials
        print('🧹 Cleaning up stale trials...')
        fail_stale_trials(study)
        
        # Run optimization with proper callback handling
        callbacks = [wandbc] if wandbc is not None else []
        study.optimize(
            objective,
            n_trials=remaining_trials,  # ← FIXED: only remaining trials
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
        best_trial_mode = "distilled"  # All trials are distilled now
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
        
        # Phase 2: Run final evaluation with best hyperparameters
        final_run_cfg = optuna_cfg.get('final_run', {})
        if final_run_cfg.get('enabled', False):
            print("\n" + "="*60)
            print("🚀 PHASE 2: FINAL EVALUATION RUN")
            print("="*60)
            _run_final_evaluation(study, config, args.data_dir, final_run_cfg, study_name, dataset=args.dataset)
        
        # Phase 3: Run vanilla baseline for fair comparison
        vanilla_cfg = optuna_cfg.get('vanilla_baseline', {})
        if vanilla_cfg.get('enabled', True):  # Default: enabled
            print("\n" + "="*60)
            print("🚀 PHASE 3: VANILLA BASELINE RUN")
            print("="*60)
            print("ℹ️  Running full vanilla baseline (no distillation) for fair comparison")
            print("   with Phase 2 distilled models (same epochs, same base config)")
            _run_vanilla_baseline(config, args.data_dir, vanilla_cfg, study_name, dataset=args.dataset)


if __name__ == '__main__':
    main()

