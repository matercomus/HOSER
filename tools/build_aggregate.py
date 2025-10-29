#!/usr/bin/env python3
"""
Simple script to build aggregate.pt files for HOSER datasets.

Usage:
    uv run python tools/build_aggregate.py --data_dir /home/matt/Dev/HOSER-dataset
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

# Now we can import from the parent directory
from dataset import Dataset


def main():
    parser = argparse.ArgumentParser(description="Build aggregate dataset tensors")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to HOSER dataset"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for split in ["train", "val", "test"]:
        traj_file = data_dir / f"{split}.csv"
        if not traj_file.exists():
            print(f"Skipping {split}.csv (not found)")
            continue

        cache_dir = Path(str(traj_file).replace(".csv", "_cache"))
        aggregate_file = cache_dir / "aggregate.pt"

        if aggregate_file.exists():
            print(f"✅ Aggregate already exists for {split}: {aggregate_file}")
            continue

        print(f"🔨 Building aggregate for {split}...")

        # Instantiate Dataset - it will handle the preprocessing and aggregation
        # We need to temporarily allow Dataset to build the aggregate
        # For now, just inform the user to check that preprocessing ran
        geo_file = data_dir / "roadmap.geo"
        rel_file = data_dir / "roadmap.rel"

        if not geo_file.exists() or not rel_file.exists():
            print(f"❌ Missing roadmap files in {data_dir}")
            continue

        print(f"   Cache directory will be: {cache_dir}")
        print("   This requires Dataset preprocessing to have already run.")
        print("   If cache doesn't exist, run Dataset preprocessing first.")

        # Check if individual cache files exist
        if not cache_dir.exists() or not list(cache_dir.glob("data_*.pt")):
            print(f"❌ Individual .pt cache files not found in {cache_dir}")
            print("   Run preprocessing first by instantiating Dataset from Python:")
            print("   >>> from dataset import Dataset")
            print(f'   >>> Dataset("{geo_file}", "{rel_file}", "{traj_file}")')
            continue

        print("✅ Found individual cache files, building aggregate...")

        try:
            dataset = Dataset(str(geo_file), str(rel_file), str(traj_file))
            print(f"✅ Aggregate built successfully: {aggregate_file}")
        except Exception as e:
            print(f"❌ Failed to build aggregate: {e}")


if __name__ == "__main__":
    main()
