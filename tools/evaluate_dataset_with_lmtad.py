#!/usr/bin/env python3
"""
Evaluate raw dataset CSV splits (train/val/test) with LM-TAD teacher.

This script is a lightweight wrapper around the project's LM-TAD evaluation
utilities and is intended for evaluating the raw CSV splits that live under
`data/<dataset>/` (for example `data/Beijing_abnormal/train.csv`) while
pointing to a roadmap file from another dataset directory (for example
`data/Beijing/roadmap.geo`).

Usage example:
  uv run python tools/evaluate_dataset_with_lmtad.py \
    --dataset Beijing_abnormal \
    --data-dir data/Beijing_abnormal \
    --roadmap data/Beijing/roadmap.geo \
    --lmtad-checkpoint /path/to/ckpt_best.pt
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simple_evaluate_with_lmtad import (  # noqa: E402
    load_hoser_trajectories,
    evaluate_trajectories_direct,
    load_road_centroids,
)  # noqa: E402
from critics.lmtad_teacher import LMTADTeacher  # noqa: E402
from critics.grid_mapper import GridMapper, GridConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_splits(
    data_dir: Path,
    roadmap_file: Path,
    lmtad_ckpt: Path,
    lmtad_repo: Path,
    device: str = "cuda:0",
    batch_size: int = 128,
    splits: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
):
    if splits is None:
        splits = ["train", "val", "test"]

    if output_dir is None:
        output_dir = Path("tools_eval_lmtad") / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not roadmap_file.exists():
        raise FileNotFoundError(f"Roadmap file not found: {roadmap_file}")
    if not lmtad_ckpt.exists():
        raise FileNotFoundError(f"LM-TAD checkpoint not found: {lmtad_ckpt}")
    if not lmtad_repo.exists():
        raise FileNotFoundError(f"LM-TAD repo not found: {lmtad_repo}")

    # Load model once
    logger.info(f"Loading LM-TAD teacher from {lmtad_ckpt}...")
    model = LMTADTeacher(
        repo_path=str(lmtad_repo),
        ckpt_path=str(lmtad_ckpt),
        device=device,
        dtype="float16",
        window=256,
    )
    logger.info("LM-TAD teacher loaded successfully")

    # Load roadmap centroids and create grid mapper
    # Note: `load_road_centroids` returns [lng, lat] (x, y). GridMapper expects
    # centroids in (lat, lng) order. Swap columns accordingly.
    road_centroids = load_road_centroids(roadmap_file)
    if road_centroids.shape[1] != 2:
        raise ValueError("Road centroids must be shape (N,2)")

    # Swap [lng, lat] -> [lat, lng]
    road_centroids_swapped = road_centroids[:, [1, 0]]

    grid_config = GridConfig(
        min_lat=float(road_centroids_swapped[:, 0].min()),
        max_lat=float(road_centroids_swapped[:, 0].max()),
        min_lng=float(road_centroids_swapped[:, 1].min()),
        max_lng=float(road_centroids_swapped[:, 1].max()),
        grid_size=0.001,
    )

    mapper = GridMapper(
        boundary=grid_config, road_centroids=road_centroids_swapped, verify_hw=None
    )
    road_to_token = torch.from_numpy(mapper.map_all()).to(device)
    logger.info(f"Created grid mapper: {mapper.grid_h}x{mapper.grid_w} cells")

    all_results = {}

    for split in splits:
        csv_file = data_dir / f"{split}.csv"
        if not csv_file.exists():
            logger.warning(f"Split file not found, skipping: {csv_file}")
            continue

        logger.info(f"Loading trajectories from {csv_file}...")
        trajectories = load_hoser_trajectories(csv_file)
        if len(trajectories) == 0:
            logger.warning(f"No valid trajectories in {csv_file}, skipping")
            continue

        logger.info(
            f"Evaluating {len(trajectories)} trajectories for split '{split}'..."
        )
        perplexities, outliers, _ = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=model,
            road_to_token=road_to_token,
            device=device,
            batch_size=batch_size,
        )

        all_results[split] = {
            "num_trajectories": len(trajectories),
            "mean_log_perplexity": float(perplexities.mean()),
            "median_log_perplexity": float(np.median(perplexities)),
            "std_log_perplexity": float(perplexities.std()),
            "min_log_perplexity": float(perplexities.min()),
            "max_log_perplexity": float(perplexities.max()),
            "outlier_rate": float(outliers.mean()),
            "log_perplexity_values": perplexities.tolist(),
            "outlier_labels": outliers.tolist(),
        }

        logger.info(
            f"Split '{split}': mean_log_perplexity={all_results[split]['mean_log_perplexity']:.4f}, outlier_rate={all_results[split]['outlier_rate']:.2%}"
        )

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate dataset CSV splits with LM-TAD"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (used for default paths)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to dataset CSVs (overrides default)",
    )
    parser.add_argument(
        "--roadmap",
        type=Path,
        required=True,
        help="Path to roadmap.geo used for mapping roads to grid tokens",
    )
    parser.add_argument(
        "--lmtad-checkpoint", type=Path, required=True, help="LM-TAD checkpoint path"
    )
    parser.add_argument(
        "--lmtad-repo",
        type=Path,
        default=Path("/home/matt/Dev/LMTAD"),
        help="LM-TAD repo path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for evaluation"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to evaluate",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Directory to write results"
    )

    args = parser.parse_args()

    data_dir = (
        args.data_dir if args.data_dir is not None else Path("data") / args.dataset
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    evaluate_splits(
        data_dir=data_dir,
        roadmap_file=args.roadmap,
        lmtad_ckpt=args.lmtad_checkpoint,
        lmtad_repo=args.lmtad_repo,
        device=args.device,
        batch_size=args.batch_size,
        splits=splits,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
