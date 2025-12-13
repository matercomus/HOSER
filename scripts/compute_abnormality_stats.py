#!/usr/bin/env python3
"""
Compute abnormality statistics for HOSER datasets.

This script compares original HOSER CSV splits with generated abnormal CSV splits
and produces a JSON and CSV summary suitable for plotting and research reporting.

Usage examples:
  uv run python scripts/compute_abnormality_stats.py \
    --original data/porto_hoser --abnormal data/porto_hoser_abnormal --out results/porto_stats

  uv run python scripts/compute_abnormality_stats.py \
    --original data/Beijing --abnormal data/Beijing_abnormal --out results/beijing_stats

The script writes `abnormality_stats.json` and `abnormality_stats.csv` into the `--out` directory.
"""

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute abnormality stats for HOSER datasets"
    )
    p.add_argument(
        "--original", required=True, help="Directory with original HOSER CSV splits"
    )
    p.add_argument(
        "--abnormal", required=True, help="Directory with abnormal HOSER CSV splits"
    )
    p.add_argument(
        "--out", required=True, help="Output directory for stats (JSON + CSV)"
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Splits to analyze",
    )
    p.add_argument(
        "--id-col",
        default="traj_id",
        help="Unique id column present in CSVs (default: traj_id)",
    )
    p.add_argument(
        "--rid-col", default="rid_list", help="RID list column name (default: rid_list)"
    )
    return p.parse_args()


def load_original_map(original_csv: str, id_col: str, rid_col: str) -> Dict[str, str]:
    """Return mapping id -> rid_list (string) for original CSV."""
    mapping = {}
    with open(original_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if id_col not in reader.fieldnames or rid_col not in reader.fieldnames:
            # fallback: try to find a trajectory id-like column
            # we'll still return what we can
            pass
        for row in reader:
            key = row.get(id_col) or row.get("mm_id") or row.get("traj_id")
            if key is None:
                # skip rows we can't map
                continue
            mapping[key] = row.get(rid_col, "")
    return mapping


def analyze_split(
    orig_map: Dict[str, str], abnormal_csv: str, id_col: str, rid_col: str
):
    stats = {
        "total_rows": 0,
        "original_rows": 0,
        "abnormal_rows": 0,
        "modified_abnormal_rows": 0,
        "per_type_counts": Counter(),
        "per_type_modified": Counter(),
    }

    with open(abnormal_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["total_rows"] += 1
            info = row.get("abnormality_info", "normal")
            if info == "normal":
                stats["original_rows"] += 1
                continue
            # abnormal row
            stats["abnormal_rows"] += 1
            # attempt to parse type (info is stored as Python-dict-like string)
            # simple heuristic: look for "'type': 'X'" or '"type": "X"'
            typ = None
            if "type":
                # try common patterns
                if "type': '" in info:
                    typ = info.split("type': '")[1].split("'", 1)[0]
                elif '"type": "' in info:
                    typ = info.split('"type": "')[1].split('"', 1)[0]
                else:
                    # fallback lookup
                    for candidate in ["detour", "route_switch", "perturb"]:
                        if candidate in info:
                            typ = candidate
                            break
            if typ is None:
                typ = "unknown"
            stats["per_type_counts"][typ] += 1

            # check whether the rid_list differs from the original (modified)
            key = row.get(id_col) or row.get("mm_id") or row.get("traj_id")
            orig_rids = orig_map.get(key)
            if orig_rids is None:
                # cannot verify modification for this row
                continue
            if row.get(rid_col, "") != orig_rids:
                stats["modified_abnormal_rows"] += 1
                stats["per_type_modified"][typ] += 1

    return stats


def save_results(out_dir: str, dataset: str, results: Dict[str, Dict]):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{dataset}_abnormality_stats.json")
    csv_path = os.path.join(out_dir, f"{dataset}_abnormality_stats.csv")

    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # flatten results to CSV rows per split
    rows = []
    for split, s in results.items():
        row = {
            "dataset": dataset,
            "split": split,
            "total_rows": s["total_rows"],
            "original_rows": s["original_rows"],
            "abnormal_rows": s["abnormal_rows"],
            "abnormal_fraction": s["abnormal_rows"] / s["total_rows"]
            if s["total_rows"]
            else 0,
            "modified_abnormal_rows": s["modified_abnormal_rows"],
            "modified_fraction_of_abnormal": s["modified_abnormal_rows"]
            / s["abnormal_rows"]
            if s["abnormal_rows"]
            else 0,
        }
        # add per-type counts
        for t, c in s["per_type_counts"].items():
            row[f"count_{t}"] = c
        for t, c in s["per_type_modified"].items():
            row[f"modified_{t}"] = c
        rows.append(row)

    # write CSV with union of keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = list(sorted(keys))
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return json_path, csv_path


def main():
    args = parse_args()
    results = {}
    for split in args.splits:
        orig_csv = os.path.join(args.original, f"{split}.csv")
        ab_csv = os.path.join(args.abnormal, f"{split}.csv")
        if not os.path.exists(orig_csv):
            print(f"Warning: original split not found: {orig_csv}, skipping")
            continue
        if not os.path.exists(ab_csv):
            print(f"Warning: abnormal split not found: {ab_csv}, skipping")
            continue

        print(f"Analyzing {split} ...")
        orig_map = load_original_map(orig_csv, args.id_col, args.rid_col)
        stats = analyze_split(orig_map, ab_csv, args.id_col, args.rid_col)
        results[split] = stats

    json_path, csv_path = save_results(
        args.out, os.path.basename(args.original.rstrip("/")), results
    )
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")


if __name__ == "__main__":
    main()
