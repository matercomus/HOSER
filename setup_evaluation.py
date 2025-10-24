#!/usr/bin/env python3
"""
Setup HOSER Evaluation Workspace

Creates a ready-to-run evaluation directory with models and config.

Usage:
    uv run python setup_evaluation.py [--dataset DATASET] [--name NAME] [--dry-run]
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import uuid
import yaml
import re


class SetupError(Exception):
    """Setup error with helpful hint"""
    def __init__(self, message, hint=None):
        self.message = message
        self.hint = hint
        super().__init__(message)


class EvaluationSetup:
    def __init__(self, dataset, name, source_dir, dry_run=False):
        self.dataset = dataset
        self.name = name
        self.source_dir = Path(source_dir)
        self.dry_run = dry_run
        
        # Generate unique directory name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        uid = str(uuid.uuid4())[:8]
        self.eval_dir = Path(f"hoser-distill-optuna-{name}-{uid}-{timestamp}")
        self.models_dir = self.eval_dir / "models"
        self.config_dir = self.eval_dir / "config"
    
    def find_trained_models(self):
        """Find trained models in save directory"""
        print(f"🔍 Scanning: {self.source_dir}/{self.dataset}...")
        
        dataset_dir = self.source_dir / self.dataset
        
        # Basic check
        if not dataset_dir.exists():
            available = [d.name for d in self.source_dir.iterdir() if d.is_dir()]
            raise SetupError(
                f"Dataset '{self.dataset}' not found in {self.source_dir}",
                hint=f"Available datasets: {', '.join(available) if available else 'none'}"
            )
        
        models = {}
        
        # Find seed directories with best.pth
        for seed_dir in dataset_dir.glob("seed*"):
            if not seed_dir.is_dir():
                continue
            
            best_pth = seed_dir / "best.pth"
            if not best_pth.exists():
                continue
            
            # Extract model type: seed42_distill -> distill
            match = re.match(r'seed(\d+)_(distill|vanilla)', seed_dir.name)
            if match:
                seed_num, model_type = match.groups()
                
                # Create key
                key = model_type if seed_num == "42" else f"{model_type}_seed{seed_num}"
                models[key] = best_pth
                print(f"  ✅ {key}: {best_pth.relative_to(self.source_dir)}")
        
        if not models:
            raise SetupError(
                f"No trained models found in {dataset_dir}",
                hint="Expected directories like: seed42_vanilla/, seed42_distill/\n"
                     "       Each with a best.pth checkpoint file"
            )
        
        print(f"✅ Found {len(models)} model(s)\n")
        return models
    
    def create_directories(self):
        """Create directory structure"""
        print(f"📁 Creating: {self.eval_dir}/")
        
        dirs = [
            self.eval_dir,
            self.models_dir,
            self.config_dir,
            self.eval_dir / "gene" / self.dataset / "seed42",
            self.eval_dir / "eval",
        ]
        
        for d in dirs:
            if self.dry_run:
                print(f"  [dry-run] {d}")
            else:
                d.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ {d.relative_to(self.eval_dir)}/")
        
        print()
    
    def copy_models(self, models):
        """Copy models to evaluation directory"""
        print(f"📦 Copying {len(models)} model(s)...\n")
        
        for model_key, source_path in models.items():
            # Target filename
            if "_seed" in model_key:
                base, seed_part = model_key.rsplit("_seed", 1)
                target_name = f"{base}_25epoch_seed{seed_part}.pth"
            else:
                target_name = f"{model_key}_25epoch_seed42.pth"
            
            target_path = self.models_dir / target_name
            size_mb = source_path.stat().st_size / (1024**2)
            
            if self.dry_run:
                print(f"  [dry-run] {target_name} ({size_mb:.1f} MB)")
            else:
                shutil.copy2(source_path, target_path)
                print(f"  ✅ {target_name} ({size_mb:.1f} MB)")
        
        print()
    
    def create_config(self):
        """Create config files"""
        print("⚙️  Creating config...\n")
        
        # Look for evaluation template in main config directory
        eval_template = Path("config/evaluation.yaml")
        
        if eval_template.exists():
            # Load template and copy it
            with open(eval_template) as f:
                config = yaml.safe_load(f)
            
            # Update dataset-specific fields
            config['dataset'] = self.dataset
            config['data_dir'] = f'../data/{self.dataset}'
            
            # Update wandb project if specified
            if 'wandb' in config and isinstance(config['wandb'], dict):
                config['wandb']['project'] = f'hoser-{self.name}'
            
            target = self.config_dir / "evaluation.yaml"
            
            if self.dry_run:
                print(f"  [dry-run] {target} (from template)")
            else:
                with open(target, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  ✅ {target.relative_to(self.eval_dir)} (from config/evaluation.yaml)")
                print(f"     Dataset: {self.dataset}")
        else:
            # Create default config if no template exists
            print(f"  ⚠️  No template found at {eval_template}, creating default config")
            
            config = {
                'dataset': self.dataset,
                'data_dir': f'../data/{self.dataset}',
                'num_gene': 5000,
                'beam_width': 4,
                'seed': 42,
                'cuda_device': 0,
                'od_sources': ['train', 'test'],
                'wandb': {
                    'enable': True, 
                    'project': f'hoser-{self.name}',
                    'background_sync': True
                },
                'logging': {
                    'verbose': False
                }
            }
            
            target = self.config_dir / "evaluation.yaml"
            
            if self.dry_run:
                print(f"  [dry-run] {target} (default)")
            else:
                with open(target, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                print(f"  ✅ {target.relative_to(self.eval_dir)} (default config)")
                print(f"     Dataset: {self.dataset}")
        
        # NEW: Copy scenarios config template if available
        scenarios_template = Path(f"config/scenarios_{self.dataset.lower()}.yaml")
        if scenarios_template.exists():
            target = self.config_dir / scenarios_template.name
            if self.dry_run:
                print(f"  [dry-run] {target}")
            else:
                shutil.copy2(scenarios_template, target)
                print(f"  ✅ {target.relative_to(self.eval_dir)} (scenarios)")
        else:
            print(f"  ℹ️  No scenarios config found (optional): {scenarios_template}")
        
        print()
    
    def create_readme(self, models):
        """Create README"""
        content = f"""# HOSER Evaluation: {self.name}

**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: {self.dataset}
**Models**: {', '.join(sorted(models.keys()))}

## Quick Start

```bash
cd {self.eval_dir.name}
uv run python ../python_pipeline.py
```

## Test Run

```bash
cd {self.eval_dir.name}
uv run python ../python_pipeline.py --num-gene 10
```

## With Scenario Analysis

```bash
cd {self.eval_dir.name}
uv run python ../python_pipeline.py --run-scenarios
```

## Configuration

Edit `config/evaluation.yaml` to change settings.
CLI arguments override config values.

## Output

- `gene/{self.dataset}/seed42/` - Generated trajectories
- `eval/` - Evaluation results
- `scenarios/` - Scenario-based analysis (if enabled)
"""
        
        target = self.eval_dir / "README.md"
        
        if self.dry_run:
            print("📝 [dry-run] README.md\n")
        else:
            target.write_text(content)
            print("📝 Created README.md\n")
    
    def print_summary(self):
        """Print completion summary"""
        print("=" * 60)
        print("✅ SETUP COMPLETE!")
        print("=" * 60)
        print(f"Directory: {self.eval_dir}")
        print()
        print("Next steps:")
        print(f"  cd {self.eval_dir}")
        print("  uv run python ../python_pipeline.py")
        print()
        print("Quick test:")
        print(f"  cd {self.eval_dir} && uv run python ../python_pipeline.py --num-gene 10")
        print("=" * 60)
    
    def run(self):
        """Run setup"""
        print("🚀 HOSER Evaluation Setup")
        print(f"Dataset: {self.dataset}")
        print(f"Name: {self.name}")
        
        if self.dry_run:
            print("⚠️  DRY RUN - no files will be created\n")
        else:
            print()
        
        try:
            # Find models
            models = self.find_trained_models()
            
            # Create structure
            self.create_directories()
            
            # Copy models
            self.copy_models(models)
            
            # Create config
            self.create_config()
            
            # Create README
            self.create_readme(models)
            
            # Summary
            if not self.dry_run:
                self.print_summary()
        
        except SetupError as e:
            print(f"\n❌ Error: {e.message}")
            if e.hint:
                print(f"💡 Hint: {e.hint}")
            sys.exit(1)
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Setup HOSER evaluation workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Beijing (default)
  uv run python setup_evaluation.py
  
  # Porto
  uv run python setup_evaluation.py --dataset porto_hoser
  
  # Preview
  uv run python setup_evaluation.py --dry-run
        """
    )
    
    parser.add_argument('--dataset', default='Beijing', help='Dataset name')
    parser.add_argument('--name', default='evaluation', help='Evaluation name')
    parser.add_argument('--source-dir', default='./save', help='Source directory')
    parser.add_argument('--dry-run', action='store_true', help='Preview only')
    
    args = parser.parse_args()
    
    setup = EvaluationSetup(
        dataset=args.dataset,
        name=args.name,
        source_dir=args.source_dir,
        dry_run=args.dry_run
    )
    
    setup.run()


if __name__ == '__main__':
    main()

