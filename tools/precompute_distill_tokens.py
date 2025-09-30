#!/usr/bin/env python3
"""Precompute LM-TAD grid tokens for HOSER caches.

This script augments the existing *.pt cache files produced by dataset.py
with two additional keys:
  - trace_grid_token: np.ndarray[int64] of shape (trace_len,)
  - candidate_grid_token: np.ndarray[object], mirroring candidate_road_id
    but containing the mapped grid tokens instead of road IDs.

Precomputing these tokens allows the training loop to avoid expensive
per-batch CPU work when preparing inputs for the LM-TAD teacher.

Uses multiprocessing for efficient parallel processing like dataset.py (all available CPU cores).
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Allow importing GridMapper without relative path issues
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from critics.grid_mapper import GridMapper, GridConfig  # noqa: E402


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def build_road_to_token_mapping(config: dict, data_dir: Path = None) -> np.ndarray:
    if data_dir is None:
        data_dir = Path(config.get("data_dir", ROOT / "data/Beijing"))
    geo_path = data_dir / "roadmap.geo"

    if not geo_path.exists():
        raise FileNotFoundError(f"roadmap.geo not found at {geo_path}")

    geo = pd.read_csv(geo_path)

    # Compute boundaries
    coords_series = geo["coordinates"].apply(eval)
    min_lng = min(min(pt[0] for pt in coords) for coords in coords_series)
    max_lng = max(max(pt[0] for pt in coords) for coords in coords_series)
    min_lat = min(min(pt[1] for pt in coords) for coords in coords_series)
    max_lat = max(max(pt[1] for pt in coords) for coords in coords_series)

    # Road centroids (lat, lng)
    road_centroids_lat = np.array([
        np.mean([pt[1] for pt in coords]) for coords in coords_series
    ], dtype=np.float32)
    road_centroids_lng = np.array([
        np.mean([pt[0] for pt in coords]) for coords in coords_series
    ], dtype=np.float32)

    grid_size = float(config["distill"]["grid_size"])
    downsample = int(config["distill"].get("downsample", 1))

    grid_cfg = GridConfig(
        min_lat=float(min_lat),
        max_lat=float(max_lat),
        min_lng=float(min_lng),
        max_lng=float(max_lng),
        grid_size=grid_size,
        downsample_factor=downsample,
    )

    mapper = GridMapper(grid_cfg, np.stack([road_centroids_lat, road_centroids_lng], axis=1))
    road_to_token = mapper.map_all()
    return road_to_token.astype(np.int64)


# Global variable for multiprocessing
_road_to_token_global = None


def init_worker(road_to_token: np.ndarray):
    """Initialize worker process with shared road_to_token mapping."""
    global _road_to_token_global
    _road_to_token_global = road_to_token


def process_single_file(args: Tuple[str, str]) -> None:
    """Process a single cache file to add grid tokens."""
    file_path, cache_dir = args

    try:
        data = torch.load(file_path, weights_only=False)

        if "trace_grid_token" in data and "candidate_grid_token" in data:
            return  # Already processed

        trace_ids = np.asarray(data["trace_road_id"], dtype=np.int64)
        candidate_ids = data["candidate_road_id"]  # object array

        # Use shared mapping for efficiency
        trace_tokens = _road_to_token_global[trace_ids]

        candidate_tokens = np.empty_like(candidate_ids)
        for i, cand_array in enumerate(candidate_ids):
            cand_array = np.asarray(cand_array, dtype=np.int64)
            if cand_array.size == 0:
                candidate_tokens[i] = np.array([], dtype=np.int64)
            else:
                candidate_tokens[i] = _road_to_token_global[cand_array]

        data["trace_grid_token"] = trace_tokens
        data["candidate_grid_token"] = candidate_tokens

        torch.save(data, file_path)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def augment_cache(cache_dir: Path, road_to_token: np.ndarray) -> None:
    """Process all cache files using multiprocessing for efficiency."""

    pt_files = sorted(cache_dir.glob("data_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not pt_files:
        print(f"No cache files found in {cache_dir}")
        return

    print(f"🔄 Processing {len(pt_files)} files in {cache_dir} using {multiprocessing.cpu_count()} processes...")

    # Prepare tasks for multiprocessing
    tasks = [(str(file_path), str(cache_dir)) for file_path in pt_files]

    # Use multiprocessing for efficient parallel processing (use all available cores like dataset.py)
    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(),  # Use all available cores like vanilla HOSER
        initializer=init_worker,
        initargs=(road_to_token,)
    ) as pool:
        # Process files in parallel with progress tracking
        for _ in tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks), desc=f'Precomputing grid tokens'):
            pass

    print(f"✅ Completed processing {len(pt_files)} files in {cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute LM-TAD grid tokens for HOSER caches")
    parser.add_argument("--config", default="config/Beijing.yaml", help="Path to training config YAML")
    parser.add_argument("--data_dir", help="Override data directory (optional, defaults to config value)")
    parser.add_argument("--splits", default="train,val", help="Comma separated splits to process (train,val)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config = load_config(config_path)

    # Use provided data_dir or fall back to config
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(config.get("data_dir", ROOT / "data/Beijing"))

    road_to_token = build_road_to_token_mapping(config, data_dir)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    for split in splits:
        cache_dir = data_dir / f"{split}_cache"
        if not cache_dir.exists():
            print(f"Cache directory {cache_dir} does not exist, skipping")
            continue
        print(f"Processing cache: {cache_dir}")
        augment_cache(cache_dir, road_to_token)

    print("Precomputation complete.")


if __name__ == "__main__":
    main()
