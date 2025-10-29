#!/usr/bin/env python3
"""
Rename generated trajectory files to include model_type and od_source.

Uses backup eval results to map timestamped files to their correct model/OD source.

Usage:
    uv run python tools/rename_generated_files.py <eval-dir>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def load_file_mappings(eval_dir: Path) -> Dict[str, Tuple[str, str]]:
    """
    Load mappings from backup eval results.

    Returns:
        Dict mapping generated filename -> (model_type, od_source)
    """
    backup_dir = eval_dir / "eval.backup"
    if not backup_dir.exists():
        print(f"❌ No eval.backup directory found in {eval_dir}")
        sys.exit(1)

    mappings = {}

    # Read all results.json files
    for eval_subdir in sorted(backup_dir.iterdir()):
        if not eval_subdir.is_dir():
            continue

        results_file = eval_subdir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file) as f:
                results = json.load(f)

            metadata = results.get("metadata", {})
            generated_file = metadata.get("generated_file", "")
            od_source = metadata.get("od_source", "")

            if not generated_file or not od_source:
                continue

            # Extract just the filename
            filename = Path(generated_file).name

            # Determine model_type from eval directory order
            # Order was: vanilla (train/test), vanilla_seed43 (train/test),
            #            vanilla_seed44 (train/test), distill (train/test),
            #            distill_seed43 (train/test), distill_seed44 (train/test)

            # We need to infer from the sequence
            # For now, use a simple heuristic based on timestamp
            # Since files are sequential, we can count and assign

            mappings[filename] = (eval_subdir.name, od_source)

        except Exception as e:
            print(f"⚠️  Warning: Failed to read {results_file}: {e}")
            continue

    return mappings


def infer_model_from_sequence(eval_timestamp_to_od: Dict[str, str]) -> Dict[str, str]:
    """
    Infer model type from the sequence of evaluations.

    Expected order (12 evals):
    0-1: vanilla (train, test)
    2-3: vanilla_seed43 (train, test)
    4-5: vanilla_seed44 (train, test)
    6-7: distill (train, test)
    8-9: distill_seed43 (train, test)
    10-11: distill_seed44 (train, test)
    """
    sorted_timestamps = sorted(eval_timestamp_to_od.keys())

    model_sequence = [
        "vanilla",
        "vanilla",
        "vanilla_seed43",
        "vanilla_seed43",
        "vanilla_seed44",
        "vanilla_seed44",
        "distill",
        "distill",
        "distill_seed43",
        "distill_seed43",
        "distill_seed44",
        "distill_seed44",
    ]

    timestamp_to_model = {}
    for i, timestamp in enumerate(sorted_timestamps):
        if i < len(model_sequence):
            timestamp_to_model[timestamp] = model_sequence[i]
        else:
            timestamp_to_model[timestamp] = "unknown"

    return timestamp_to_model


def rename_files(eval_dir: Path, dry_run: bool = False):
    """Rename generated files to include model_type and od_source."""

    # Load mappings
    print("📂 Loading file mappings from backup eval results...")
    file_mappings = load_file_mappings(eval_dir)

    if not file_mappings:
        print("❌ No file mappings found")
        sys.exit(1)

    print(f"✅ Found {len(file_mappings)} evaluation results")

    # Infer models from sequence
    eval_timestamp_to_od = {}
    filename_to_eval_timestamp = {}
    for filename, (eval_timestamp, od_source) in file_mappings.items():
        eval_timestamp_to_od[eval_timestamp] = od_source
        filename_to_eval_timestamp[filename] = eval_timestamp

    timestamp_to_model = infer_model_from_sequence(eval_timestamp_to_od)

    # Build final mapping: filename -> (model_type, od_source)
    final_mappings = {}
    for filename, (eval_timestamp, od_source) in file_mappings.items():
        model_type = timestamp_to_model.get(eval_timestamp, "unknown")
        final_mappings[filename] = (model_type, od_source)

    # Find gene directory
    gene_dir = eval_dir / "gene" / "porto_hoser" / "seed42"
    if not gene_dir.exists():
        print(f"❌ Gene directory not found: {gene_dir}")
        sys.exit(1)

    print(f"📁 Gene directory: {gene_dir}")
    print(f"\n{'DRY RUN - ' if dry_run else ''}Renaming files:\n")

    renamed_count = 0
    for old_filename, (model_type, od_source) in sorted(final_mappings.items()):
        old_path = gene_dir / old_filename

        if not old_path.exists():
            print(f"⚠️  File not found: {old_filename}")
            continue

        # Extract timestamp from old filename
        timestamp = old_filename.replace(".csv", "")
        new_filename = f"{timestamp}_{model_type}_{od_source}.csv"
        new_path = gene_dir / new_filename

        print(f"  {old_filename}")
        print(f"  → {new_filename}")
        print()

        if not dry_run:
            try:
                old_path.rename(new_path)
                renamed_count += 1
            except Exception as e:
                print(f"❌ Failed to rename: {e}")
                continue

    if dry_run:
        print(f"\n📋 DRY RUN: Would rename {len(final_mappings)} files")
        print("Run without --dry-run to actually rename files")
    else:
        print(f"\n✅ Successfully renamed {renamed_count}/{len(final_mappings)} files")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python tools/rename_generated_files.py <eval-dir> [--dry-run]"
        )
        print("\nExample:")
        print(
            "  uv run python tools/rename_generated_files.py hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732"
        )
        sys.exit(1)

    eval_dir = Path(sys.argv[1])
    if not eval_dir.exists():
        print(f"❌ Directory not found: {eval_dir}")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv

    rename_files(eval_dir, dry_run=dry_run)


if __name__ == "__main__":
    main()
