#!/usr/bin/env python3
"""
Collect artifacts for a distilled HOSER run:

- Convert generated trajectories to GeoJSON using hoser_to_geojson.py
- Organize outputs under backup root: <backup_root>/<wandb_run_name>/{geojson_output,model,eval,wandb,meta}
- Copy model checkpoint, latest eval results, config, and the matching wandb run dir

Usage (via uv):
  uv run python tools/collect_distill_artifacts.py \
    --run_name Beijing_b24_acc4 \
    --run_dir /path/to/run_YYYYMMDD_HHMMSS \
    --generated_csv /home/matt/Dev/HOSER/gene/Beijing/seed0/2025-09-26_19-25-56.csv \
    --backup_root /mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil

Notes:
- This script assumes checkpoints saved by train_with_distill.py under save/<dataset>/seed<seed>_distill/best.pth
- The wandb directory is searched under /home/matt/Dev/HOSER/wandb by default.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def run_geojson_conversion(run_dir: Path, generated_csv: Path, output_dir: Path) -> None:
    """Invoke the GeoJSON converter with given inputs and output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hoser_dir = run_dir / 'hoser_format'
    hoser_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the generated CSV is available within hoser_format
    dst_generated = hoser_dir / generated_csv.name
    if str(generated_csv.resolve()) != str(dst_generated.resolve()):
        shutil.copy2(generated_csv, dst_generated)

    # Build command
    cmd = (
        f"uv run python /home/matt/Dev/Bigscity-LibCity-Datasets/hoser_to_geojson.py "
        f"{run_dir} --file {generated_csv.name} --output_dir {output_dir} --force-regenerate --individual"
    )
    # Execute
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"GeoJSON conversion failed with exit code {ret}")


def try_copy(src: Path, dst: Path) -> bool:
    try:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        if src.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            return True
    except Exception:
        return False
    return False


def find_latest_eval_dir(run_dir: Path) -> Path | None:
    eval_dirs = sorted(run_dir.glob('eval_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    return eval_dirs[0] if eval_dirs else None


def find_wandb_run_dir(wandb_root: Path, run_name: str) -> Path | None:
    # Search metadata for matching name
    for meta_path in wandb_root.glob('run-*/wandb-metadata.json'):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('name') == run_name:
                return meta_path.parent.parent
        except Exception:
            continue
    # Fallback to latest
    runs = sorted(wandb_root.glob('run-*'), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def main():
    parser = argparse.ArgumentParser(description='Collect artifacts for a distilled HOSER run')
    parser.add_argument('--run_name', required=True, help='WandB run name of the training (e.g., Beijing_b24_acc4)')
    parser.add_argument('--run_dir', required=True, help='Path to the dataset run directory with hoser_format')
    parser.add_argument('--generated_csv', required=True, help='Path to the generated CSV file to convert')
    parser.add_argument('--backup_root', required=True, help='Backup root directory to collect artifacts under')
    parser.add_argument('--dataset', default='Beijing', help='Dataset name used for checkpoint path resolution')
    parser.add_argument('--seed', type=int, default=0, help='Seed used in training (for checkpoint path)')
    parser.add_argument('--wandb_root', default='/home/matt/Dev/HOSER/wandb', help='Path to local wandb runs')
    parser.add_argument('--config_path', default='/home/matt/Dev/HOSER/config/Beijing.yaml', help='Config file to copy into backup/meta')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    generated_csv = Path(args.generated_csv).resolve()
    backup_root = Path(args.backup_root).resolve()

    backup_dir = backup_root / args.run_name
    geojson_dir = backup_dir / 'geojson_output'
    model_dir = backup_dir / 'model'
    eval_dir = backup_dir / 'eval'
    wandb_dir = backup_dir / 'wandb'
    meta_dir = backup_dir / 'meta'

    # Create folder structure
    for p in (geojson_dir, model_dir, eval_dir, wandb_dir, meta_dir):
        p.mkdir(parents=True, exist_ok=True)

    # 1) Convert to GeoJSON
    run_geojson_conversion(run_dir, generated_csv, geojson_dir)

    # 2) Copy model checkpoint
    ckpt_src = Path(f'/home/matt/Dev/HOSER/save/{args.dataset}/seed{args.seed}_distill/best.pth')
    try_copy(ckpt_src, model_dir / 'best.pth')

    # 3) Copy latest evaluation directory if present
    latest_eval = find_latest_eval_dir(run_dir)
    if latest_eval is not None:
        try_copy(latest_eval, eval_dir / latest_eval.name)

    # 4) Copy config and metadata
    try_copy(Path(args.config_path), meta_dir / Path(args.config_path).name)
    with open(meta_dir / 'context.json', 'w') as f:
        json.dump({
            'run_name': args.run_name,
            'run_dir': str(run_dir),
            'generated_csv': str(generated_csv),
            'timestamp': datetime.now().isoformat(),
            'dataset': args.dataset,
            'seed': args.seed,
        }, f, indent=2)

    # 5) Copy matching wandb run (or latest as fallback)
    wandb_root = Path(args.wandb_root)
    if wandb_root.exists():
        wandb_src = find_wandb_run_dir(wandb_root, args.run_name)
        if wandb_src is not None:
            try_copy(wandb_src, wandb_dir / wandb_src.name)

    print(f"✅ Collected artifacts under: {backup_dir}")


if __name__ == '__main__':
    main()


