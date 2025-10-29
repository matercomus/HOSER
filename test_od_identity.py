#!/usr/bin/env python3
"""
Test script to verify that generated trajectories use identical OD pairs from training data.
"""

import polars as pl
import sys
from pathlib import Path


def extract_od_pairs_from_trajectories(df, rid_col="rid_list"):
    """Extract unique OD pairs from trajectory data."""
    od_pairs = set()

    for rid_list_str in df.get_column(rid_col):
        try:
            # Handle comma-separated string format
            rid_list = (
                rid_list_str.split(",")
                if isinstance(rid_list_str, str)
                else rid_list_str
            )
            if len(rid_list) >= 2:
                od_pair = (rid_list[0], rid_list[-1])
                od_pairs.add(od_pair)
        except Exception:
            continue

    return od_pairs


def test_od_identity(dataset="Beijing", num_gene=100):
    """Test that generated ODs are identical to training ODs."""

    print(f"🧪 Testing OD identity for {dataset} dataset...")

    # Load training data
    train_file = f"./data/{dataset}/train.csv"
    if not Path(train_file).exists():
        print(f"❌ Training file not found: {train_file}")
        return

    train_df = pl.read_csv(train_file)
    print(f"📊 Loaded {len(train_df)} training trajectories")

    # Extract training OD pairs
    train_ods = extract_od_pairs_from_trajectories(train_df)
    print(f"🎯 Found {len(train_ods)} unique OD pairs in training data")

    # Generate synthetic trajectories
    print(f"🎲 Generating {num_gene} synthetic trajectories...")

    # Run gene.py to generate trajectories
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "gene.py",
            "--dataset",
            dataset,
            "--nx_astar",
            "--num_gene",
            str(num_gene),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Generation failed: {result.stderr}")
        return

    # Find generated file
    output_dir = Path(f"./gene/{dataset}/seed0")
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return

    # Get the most recent generated file
    generated_files = list(output_dir.glob("*.csv"))
    if not generated_files:
        print(f"❌ No generated files found in {output_dir}")
        return

    latest_file = max(generated_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using generated file: {latest_file}")

    # Load generated data
    gen_df = pl.read_csv(latest_file)
    print(f"📊 Loaded {len(gen_df)} generated trajectories")

    # Extract generated OD pairs
    gen_ods = set()
    for i in range(len(gen_df)):
        origin = gen_df["origin_road_id"][i]
        dest = gen_df["destination_road_id"][i]
        gen_ods.add((origin, dest))

    print(f"🎯 Found {len(gen_ods)} unique OD pairs in generated data")

    # Compare OD pairs
    print("\n" + "=" * 60)
    print("🔍 OD PAIR COMPARISON RESULTS")
    print("=" * 60)

    # Check if all generated ODs are in training ODs
    all_generated_in_training = gen_ods.issubset(train_ods)

    print(f"✅ All generated ODs in training ODs: {all_generated_in_training}")

    if not all_generated_in_training:
        # Find ODs that are in generated but not in training
        novel_ods = gen_ods - train_ods
        print(f"❌ Found {len(novel_ods)} novel OD pairs in generated data:")
        for od in list(novel_ods)[:5]:  # Show first 5
            print(f"   {od}")
        if len(novel_ods) > 5:
            print(f"   ... and {len(novel_ods) - 5} more")

    # Check if all training ODs are in generated ODs
    all_training_in_generated = train_ods.issubset(gen_ods)
    print(f"✅ All training ODs in generated ODs: {all_training_in_generated}")

    # Calculate overlap
    overlap = len(gen_ods.intersection(train_ods))
    overlap_pct = (overlap / len(gen_ods)) * 100 if gen_ods else 0

    print(f"📊 Overlap: {overlap}/{len(gen_ods)} ({overlap_pct:.1f}%)")

    # Show some examples
    print("\n🔍 Sample generated OD pairs:")
    for i, od in enumerate(list(gen_ods)[:5]):
        in_training = "✅" if od in train_ods else "❌"
        print(f"   {i + 1}. {od} {in_training}")

    print("\n" + "=" * 60)

    if all_generated_in_training:
        print("🎉 CONFIRMED: All generated ODs are identical to training ODs")
        print("   This validates the privacy concern - ODs are preserved!")
    else:
        print("🤔 MIXED: Some generated ODs are novel")
        print("   This suggests the model might be generating new ODs")

    return all_generated_in_training


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "Beijing"
    num_gene = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    test_od_identity(dataset, num_gene)
