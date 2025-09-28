#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for HOSER with WandB integration.

This script runs hyperparameter optimization for both vanilla HOSER and 
distilled HOSER variants, logging all trials to WandB for comprehensive tracking.

Usage:
  uv run python tune_hoser.py --n_trials 20 --dataset Beijing --data_dir /home/matt/Dev/HOSER-dataset
"""

import os
import sys
import argparse
import tempfile
import shutil
import gc
import torch
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from typing import Dict, Any
import yaml
import json
from datetime import datetime

# Training logic will be imported dynamically to avoid circular imports


class HOSERObjective:
    """Objective function for Optuna optimization of HOSER models."""
    
    def __init__(self, base_config_path: str, data_dir: str, max_epochs: int = 25):
        self.base_config_path = base_config_path
        self.data_dir = data_dir
        self.max_epochs = max_epochs
        self.trial_counter = 0
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function called by Optuna for each trial."""
        self.trial_counter += 1
        
        # Suggest trial type: vanilla HOSER or distilled HOSER
        trial_type = trial.suggest_categorical('trial_type', ['vanilla', 'distilled'])
        
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
            # Only tune distillation-specific parameters
            config['distill']['lambda'] = trial.suggest_float(
                'distill_lambda', 0.001, 0.1, log=True
            )
            config['distill']['temperature'] = trial.suggest_float(
                'distill_temperature', 1.0, 4.0
            )
            config['distill']['window'] = trial.suggest_categorical(
                'distill_window', [16, 24, 32, 48, 64]
            )
            config['distill']['grid_size'] = trial.suggest_categorical(
                'distill_grid_size', [0.0005, 0.001, 0.002]
            )
            config['distill']['downsample'] = trial.suggest_categorical(
                'distill_downsample', [2, 4, 8]
            )
        else:
            config['distill']['enable'] = False
        
        # WandB config for trial
        config['wandb']['enable'] = True
        config['wandb']['project'] = 'hoser-distill-optuna'
        config['wandb']['run_name'] = f"trial_{trial.number:03d}_{trial_type}"
        config['wandb']['tags'] = ['optuna', 'distill-tuning', trial_type, 'beijing']
        
        return config
    
    def _run_training_trial(self, trial: optuna.Trial, config_path: str, trial_type: str) -> float:
        """Run a single training trial and return the metric to optimize."""
        
        # Create trial-specific save directory
        trial_save_dir = f"./optuna_trials/trial_{trial.number:03d}_{trial_type}"
        os.makedirs(trial_save_dir, exist_ok=True)
        
        # Prepare arguments for training
        train_args = argparse.Namespace(
            dataset='Beijing',
            config=config_path,
            seed=42 + trial.number,  # Deterministic but different per trial
            cuda=0,
            data_dir=self.data_dir
        )
        
        # Monkey-patch sys.argv for the training script
        original_argv = sys.argv
        sys.argv = [
            'train_with_distill.py',
            '--dataset', train_args.dataset,
            '--config', train_args.config,
            '--seed', str(train_args.seed),
            '--cuda', str(train_args.cuda),
            '--data_dir', train_args.data_dir
        ]
        
        try:
            # Import and run training with validation tracking
            validation_metrics = []
            
            # We'll need to modify train_with_distill.py to return validation metrics
            # For now, we'll use a placeholder approach
            result_metric = self._run_training_with_pruning(trial, train_args, validation_metrics)
            
            return result_metric
            
        finally:
            sys.argv = original_argv
    
    def _run_training_with_pruning(self, trial: optuna.Trial, train_args: argparse.Namespace, validation_metrics: list) -> float:
        """Run training with intermediate pruning based on validation metrics."""
        
        try:
            # Import and run the modified training function
            from train_with_distill import main as train_main
            
            # Run training and get validation metrics
            result = train_main(args=train_args, return_metrics=True)
            
            if result is None:
                raise optuna.TrialPruned("Training returned no metrics")
            
            # Report intermediate values for pruning
            for epoch_metrics in result['validation_history']:
                trial.report(epoch_metrics['val_acc'], epoch_metrics['epoch'])
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Return final metric to maximize (validation accuracy)
            final_metric = result['best_val_acc']
            
            # Log additional metrics to trial
            trial.set_user_attr('final_val_acc', result['final_val_acc'])
            trial.set_user_attr('final_val_mape', result['final_val_mape'])
            trial.set_user_attr('best_val_acc', result['best_val_acc'])
            
            return final_metric
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
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


def create_study_with_wandb(project_name: str, study_name: str) -> tuple:
    """Create Optuna study with WandB integration."""
    
    # WandB callback configuration
    wandb_kwargs = {
        "project": project_name,
        "group": study_name,
        "job_type": "optuna-optimization"
    }
    
    wandbc = WeightsAndBiasesCallback(
        wandb_kwargs=wandb_kwargs,
        as_multirun=True,
        metric_name="validation_accuracy"
    )
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    return study, wandbc


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for HOSER with Optuna')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials to run')
    parser.add_argument('--dataset', type=str, default='Beijing', help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HOSER dataset')
    parser.add_argument('--config', type=str, default='config/Beijing.yaml', help='Base config file')
    parser.add_argument('--max_epochs', type=int, default=25, help='Max epochs per trial')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name')
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Create study name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"hoser_tuning_{timestamp}"
    
    print(f"🔍 Starting Optuna study: {study_name}")
    print(f"📊 Trials: {args.n_trials}")
    print(f"📁 Dataset: {args.data_dir}")
    print(f"⚙️  Base config: {args.config}")
    
    # Create study and WandB callback
    study, wandbc = create_study_with_wandb("hoser-optuna-tuning", study_name)
    
    # Create objective function
    objective = HOSERObjective(
        base_config_path=args.config,
        data_dir=args.data_dir,
        max_epochs=args.max_epochs
    )
    
    # Ensure vanilla HOSER is included in first few trials
    def enhanced_objective(trial):
        # Force first trial to be vanilla for baseline
        if trial.number == 0:
            trial.suggest_categorical('trial_type', ['vanilla'])
        return objective(trial)
    
    # Run optimization
    try:
        study.optimize(
            enhanced_objective,
            n_trials=args.n_trials,
            callbacks=[wandbc],
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
        
        # Create preserved directories
        preserved_dir = f"./optuna_results/{study_name}/preserved_models"
        os.makedirs(preserved_dir, exist_ok=True)
        
        # Copy best trial artifacts
        best_trial_dirs = {
            f"./save/Beijing/seed{best_seed}_distill": f"{preserved_dir}/best_trial_{best_trial_num}_model",
            f"./tensorboard_log/Beijing/seed{best_seed}_distill": f"{preserved_dir}/best_trial_{best_trial_num}_tensorboard",
            f"./log/Beijing/seed{best_seed}_distill": f"{preserved_dir}/best_trial_{best_trial_num}_logs"
        }
        
        for src, dst in best_trial_dirs.items():
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"🔒 Preserved best trial artifacts: {dst}")
        
        # Copy vanilla baseline (trial 0) artifacts
        vanilla_seed = 42
        vanilla_dirs = {
            f"./save/Beijing/seed{vanilla_seed}_distill": f"{preserved_dir}/vanilla_trial_0_model",
            f"./tensorboard_log/Beijing/seed{vanilla_seed}_distill": f"{preserved_dir}/vanilla_trial_0_tensorboard",
            f"./log/Beijing/seed{vanilla_seed}_distill": f"{preserved_dir}/vanilla_trial_0_logs"
        }
        
        for src, dst in vanilla_dirs.items():
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"🔒 Preserved vanilla baseline artifacts: {dst}")
        
        # Save study results
        results_dir = f"./optuna_results/{study_name}"
        
        with open(f"{results_dir}/best_params.json", 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        with open(f"{results_dir}/study_summary.json", 'w') as f:
            json.dump({
                "study_name": study_name,
                "n_trials": len(study.trials),
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "best_params": study.best_params,
                "vanilla_trial": 0,
                "preserved_models": {
                    "best_trial": f"{preserved_dir}/best_trial_{best_trial_num}_model/best.pth",
                    "vanilla_baseline": f"{preserved_dir}/vanilla_trial_0_model/best.pth"
                }
            }, f, indent=2)
        
        print(f"💾 Results saved to: {results_dir}")
        print("🔒 Preserved models:")
        print(f"   Best trial ({best_trial_num}): {preserved_dir}/best_trial_{best_trial_num}_model/best.pth")
        print(f"   Vanilla baseline (0): {preserved_dir}/vanilla_trial_0_model/best.pth")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


if __name__ == '__main__':
    main()
