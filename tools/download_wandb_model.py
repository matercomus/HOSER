#!/usr/bin/env python3
"""
Download model checkpoints from Weights & Biases runs.

This tool allows you to download model files from WandB runs by specifying
the run ID, or search for runs matching specific criteria.

Examples:
    # Download a specific run by ID
    uv run python tools/download_wandb_model.py --run_id 0vw2ywd9
    
    # Download and rename to a specific output path
    uv run python tools/download_wandb_model.py --run_id 0vw2ywd9 --output vanilla_25epoch.pth
    
    # Search for vanilla runs with 25 epochs
    uv run python tools/download_wandb_model.py --search --distill false --epochs 25
    
    # Download from a different project
    uv run python tools/download_wandb_model.py --run_id xyz123 --project hoser-eval
"""

import argparse
import os
import sys
from typing import Optional, List
import wandb


def download_model_from_run(
    run_id: str,
    entity: str = "matercomus",
    project: str = "hoser-distill-optuna-6",
    output_path: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Download model checkpoint from a WandB run.
    
    Args:
        run_id: WandB run ID (the part after the timestamp in run directory name)
        entity: WandB entity/username
        project: WandB project name
        output_path: Where to save the model (if None, saves to ./downloaded_models/)
        verbose: Print detailed information
        
    Returns:
        Path to the downloaded model file
    """
    if verbose:
        print(f"🔍 Fetching run: {entity}/{project}/{run_id}")
    
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        if verbose:
            print(f"\n📊 Run info:")
            print(f"   Name: {run.name}")
            print(f"   State: {run.state}")
            print(f"   Created: {run.created_at}")
            
            # Print relevant config
            config = run.config
            if 'distill' in config and 'enable' in config['distill']:
                distill_enabled = config['distill']['enable']
                print(f"   Distillation: {'✅ Enabled' if distill_enabled else '❌ Disabled (vanilla)'}")
            
            if 'optimizer_config' in config and 'max_epoch' in config['optimizer_config']:
                epochs = config['optimizer_config']['max_epoch']
                print(f"   Epochs: {epochs}")
            
            # Print final metrics
            summary = run.summary
            if 'val/next_step_acc' in summary:
                val_acc = summary['val/next_step_acc']
                print(f"   Final val_acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            if 'epoch' in summary:
                print(f"   Completed epochs: {summary['epoch']}")
        
        # List available files
        model_files = [f for f in run.files() if f.name.endswith('.pth')]
        
        if not model_files:
            print(f"\n❌ No .pth model files found in this run!")
            print(f"📁 Available files:")
            for file in run.files():
                print(f"   - {file.name}")
            return None
        
        if verbose:
            print(f"\n📁 Model files found:")
            for file in model_files:
                print(f"   - {file.name} ({file.size / 1024 / 1024:.1f} MB)")
        
        # Download the model
        model_file = model_files[0]  # Usually there's only one
        
        if output_path:
            output_dir = os.path.dirname(output_path) or "."
            output_filename = os.path.basename(output_path)
        else:
            output_dir = "./downloaded_models"
            output_filename = f"{run.name}_{run_id}.pth"
            output_path = os.path.join(output_dir, output_filename)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose:
            print(f"\n⬇️  Downloading: {model_file.name}")
        
        # Download to temp location then move
        temp_dir = os.path.join(output_dir, ".tmp")
        os.makedirs(temp_dir, exist_ok=True)
        model_file.download(root=temp_dir, replace=True)
        
        # Find the downloaded file (WandB preserves directory structure)
        downloaded_path = os.path.join(temp_dir, model_file.name)
        if os.path.exists(downloaded_path):
            # Move to final location
            os.rename(downloaded_path, output_path)
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
        
        if verbose:
            print(f"✅ Model saved to: {os.path.abspath(output_path)}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None


def search_runs(
    entity: str = "matercomus",
    project: str = "hoser-distill-optuna-6",
    distill_enabled: Optional[bool] = None,
    epochs: Optional[int] = None,
    min_val_acc: Optional[float] = None,
    limit: int = 10
) -> List[dict]:
    """
    Search for WandB runs matching specific criteria.
    
    Args:
        entity: WandB entity/username
        project: WandB project name
        distill_enabled: Filter by distillation (True/False/None for all)
        epochs: Filter by number of epochs
        min_val_acc: Minimum validation accuracy
        limit: Maximum number of results
        
    Returns:
        List of matching runs with metadata
    """
    print(f"🔍 Searching runs in {entity}/{project}")
    filters = []
    
    if distill_enabled is not None:
        filters.append(f"   - Distillation: {'enabled' if distill_enabled else 'disabled'}")
    if epochs is not None:
        filters.append(f"   - Epochs: {epochs}")
    if min_val_acc is not None:
        filters.append(f"   - Min val_acc: {min_val_acc:.4f}")
    
    if filters:
        print("Filters:")
        for f in filters:
            print(f)
    
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        
        matching_runs = []
        for run in runs:
            # Apply filters
            if distill_enabled is not None:
                config_distill = run.config.get('distill', {}).get('enable')
                if config_distill != distill_enabled:
                    continue
            
            if epochs is not None:
                config_epochs = run.config.get('optimizer_config', {}).get('max_epoch')
                summary_epochs = run.summary.get('epoch')
                # Match either configured epochs or completed epochs
                if config_epochs != epochs and summary_epochs != epochs:
                    continue
            
            if min_val_acc is not None:
                val_acc = run.summary.get('val/next_step_acc')
                if val_acc is None or val_acc < min_val_acc:
                    continue
            
            # Extract metadata
            matching_runs.append({
                'id': run.id,
                'name': run.name,
                'state': run.state,
                'created': run.created_at,
                'distill': run.config.get('distill', {}).get('enable'),
                'epochs': run.summary.get('epoch'),
                'val_acc': run.summary.get('val/next_step_acc'),
            })
            
            if len(matching_runs) >= limit:
                break
        
        print(f"\n📊 Found {len(matching_runs)} matching runs:")
        print(f"{'ID':<15} {'Name':<30} {'Distill':<10} {'Epochs':<8} {'Val Acc':<10} {'State':<10}")
        print("-" * 90)
        
        for run_info in matching_runs:
            distill_str = "✅ Yes" if run_info['distill'] else "❌ No"
            val_acc_str = f"{run_info['val_acc']:.4f}" if run_info['val_acc'] else "N/A"
            print(f"{run_info['id']:<15} {run_info['name']:<30} {distill_str:<10} "
                  f"{run_info['epochs'] or 'N/A':<8} {val_acc_str:<10} {run_info['state']:<10}")
        
        return matching_runs
        
    except Exception as e:
        print(f"❌ Error searching runs: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints from Weights & Biases runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Common arguments
    parser.add_argument("--entity", type=str, default="matercomus",
                        help="WandB entity/username (default: matercomus)")
    parser.add_argument("--project", type=str, default="hoser-distill-optuna-6",
                        help="WandB project name (default: hoser-distill-optuna-6)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    # Download-specific arguments
    parser.add_argument("--run_id", type=str,
                        help="WandB run ID to download (e.g., 0vw2ywd9)")
    parser.add_argument("--output", "-o", type=str,
                        help="Output path for downloaded model (default: ./downloaded_models/<run_name>_<run_id>.pth)")
    
    # Search-specific arguments
    parser.add_argument("--search", action="store_true",
                        help="Search for runs instead of downloading")
    parser.add_argument("--distill", type=str, choices=['true', 'false'],
                        help="Filter by distillation enabled (true) or disabled (false)")
    parser.add_argument("--epochs", type=int,
                        help="Filter by number of epochs")
    parser.add_argument("--min_val_acc", type=float,
                        help="Minimum validation accuracy")
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum number of search results (default: 10)")
    
    args = parser.parse_args()
    
    if args.search:
        # Search mode
        distill_enabled = None
        if args.distill == 'true':
            distill_enabled = True
        elif args.distill == 'false':
            distill_enabled = False
        
        matching_runs = search_runs(
            entity=args.entity,
            project=args.project,
            distill_enabled=distill_enabled,
            epochs=args.epochs,
            min_val_acc=args.min_val_acc,
            limit=args.limit
        )
        
        if matching_runs:
            print(f"\n💡 To download a specific run, use:")
            print(f"   uv run python tools/download_wandb_model.py --run_id <ID>")
        
    elif args.run_id:
        # Download mode
        model_path = download_model_from_run(
            run_id=args.run_id,
            entity=args.entity,
            project=args.project,
            output_path=args.output,
            verbose=not args.quiet
        )
        
        if model_path:
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        parser.print_help()
        print("\n❌ Error: Must specify either --run_id or --search")
        sys.exit(1)


if __name__ == "__main__":
    main()

