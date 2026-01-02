#!/usr/bin/env python3
"""Build a balanced evaluation dataset CSV.

Goal
- Create a split CSV containing all abnormal rows plus a matched set of normal
  rows, to disambiguate detectability vs thresholding.

Matching strategy (simple + deterministic)
- Identify abnormal rows using `abnormality_info` (abnormal iff non-empty and
  not in {normal, null-like}).
- Compute trajectory length as number of road IDs in `rid_list`.
- Bucket length by `--length-bucket` (default 5) and match normals to each
  abnormal by selecting rows from the same bucket when possible; otherwise use
  the nearest bucket.

This script writes:
- `<out-dir>/<split>.csv` (balanced)
- `<out-dir>/balanced_manifest.json` (counts + settings)

The output directory is compatible with `tools/evaluate_dataset_with_lmtad.py`
via `--data-dir <out-dir>`.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_NULL_LIKE = {"", "nan", "none", "null"}


@dataclass(frozen=True)
class BalancedManifest:
    source_dataset_dir: str
    split: str
    out_dir: str
    seed: int
    normal_per_abnormal: int
    length_bucket: int
    allow_replacement: bool
    num_rows_out: int
    num_abnormal_out: int
    num_normal_out: int


def _is_abnormal(raw: str) -> bool:
    s = str(raw or "").strip().lower()
    if s in _NULL_LIKE:
        return False
    if s == "normal":
        return False
    return True


def _rid_list_length(rid_list: str) -> int:
    items = [x.strip() for x in str(rid_list or "").split(",") if x.strip()]
    return int(len(items))


def _bucket(length: int, bucket_size: int) -> int:
    if bucket_size <= 1:
        return int(length)
    return int(length // int(bucket_size))


def _nearest_bucket(target: int, candidates: Sequence[int]) -> Optional[int]:
    if not candidates:
        return None
    return min(candidates, key=lambda b: (abs(int(b) - int(target)), int(b)))


def _sample_without_replacement(
    *,
    pool: List[int],
    k: int,
    rng,
) -> List[int]:
    if k <= 0:
        return []
    if len(pool) <= k:
        rng.shuffle(pool)
        out = pool[:]
        pool.clear()
        return out
    # Fisher-Yates via shuffle of indices is fine at these sizes.
    rng.shuffle(pool)
    out = pool[:k]
    del pool[:k]
    return out


def _select_normals_for_abnormals(
    *,
    abnormal_buckets: List[int],
    normals_by_bucket: Dict[int, List[int]],
    normals_needed_per_abnormal: int,
    rng,
    allow_replacement: bool,
) -> List[int]:
    selected: List[int] = []

    available_buckets = sorted(normals_by_bucket.keys())
    for b in abnormal_buckets:
        for _ in range(int(normals_needed_per_abnormal)):
            chosen_bucket = None
            if b in normals_by_bucket and normals_by_bucket[b]:
                chosen_bucket = b
            else:
                # Find nearest bucket that still has normals.
                nonempty = [x for x in available_buckets if normals_by_bucket[x]]
                chosen_bucket = _nearest_bucket(b, nonempty)

            if chosen_bucket is None:
                if not allow_replacement:
                    raise ValueError(
                        "Not enough normal rows to match abnormals without replacement. "
                        "Re-run with --allow-replacement or lower --normal-per-abnormal."
                    )
                # Replacement: sample from any bucket.
                nonempty_all = [x for x in available_buckets if normals_by_bucket[x]]
                if not nonempty_all:
                    # All pools empty: cannot even sample with replacement.
                    raise ValueError("No normal rows available for matching")
                chosen_bucket = rng.choice(nonempty_all)

            pool = normals_by_bucket[int(chosen_bucket)]
            if pool:
                picked = _sample_without_replacement(pool=pool, k=1, rng=rng)
                selected.extend(picked)
            else:
                # Pool empty: only possible under replacement mode.
                if not allow_replacement:
                    raise ValueError("Internal error: empty pool without replacement")

    return selected


def build_balanced_rows(
    *,
    rows: List[Dict[str, str]],
    normal_per_abnormal: int,
    length_bucket: int,
    seed: int,
    allow_replacement: bool,
) -> Tuple[List[Dict[str, str]], int, int]:
    """Return (balanced_rows, num_abnormal, num_normal)."""

    import random

    if normal_per_abnormal < 0:
        raise ValueError("normal_per_abnormal must be >= 0")

    rng = random.Random(int(seed))

    abnormal_idx: List[int] = []
    abnormal_buckets: List[int] = []

    normals_by_bucket: Dict[int, List[int]] = defaultdict(list)

    for i, row in enumerate(rows):
        is_abn = _is_abnormal(row.get("abnormality_info", ""))
        length = _rid_list_length(row.get("rid_list", ""))
        b = _bucket(length, int(length_bucket))
        if is_abn:
            abnormal_idx.append(i)
            abnormal_buckets.append(b)
        else:
            normals_by_bucket[b].append(i)

    # Deterministic order for bucket pools.
    for b in list(normals_by_bucket.keys()):
        normals_by_bucket[b] = list(normals_by_bucket[b])

    normal_idx = _select_normals_for_abnormals(
        abnormal_buckets=abnormal_buckets,
        normals_by_bucket=normals_by_bucket,
        normals_needed_per_abnormal=int(normal_per_abnormal),
        rng=rng,
        allow_replacement=bool(allow_replacement),
    )

    # Combine and shuffle.
    chosen = abnormal_idx + normal_idx
    rng.shuffle(chosen)
    balanced = [rows[i] for i in chosen]

    return balanced, int(len(abnormal_idx)), int(len(normal_idx))


def _read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_csv_rows(
    *,
    out_path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a balanced split CSV: all abnormal + matched normals"
    )
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        required=True,
        help="Dataset dir containing <split>.csv (must include abnormality_info).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name (train/val/test). Default: train",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output dataset dir (will contain <split>.csv + balanced_manifest.json).",
    )
    parser.add_argument(
        "--normal-per-abnormal",
        type=int,
        default=1,
        help="How many normal rows to sample per abnormal row (default: 1).",
    )
    parser.add_argument(
        "--length-bucket",
        type=int,
        default=5,
        help="Trajectory-length bucket size used for matching (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help="Allow sampling normals with replacement if insufficient normals exist.",
    )
    parser.add_argument(
        "--copy-roadmaps",
        action="store_true",
        help="Copy roadmap.geo and roadmap.rel into out-dir if present.",
    )

    args = parser.parse_args()

    source = Path(args.source_dataset_dir)
    split = str(args.split).strip()
    out_dir = Path(args.out_dir)

    csv_in = source / f"{split}.csv"
    if not csv_in.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_in}")

    fieldnames, rows = _read_csv_rows(csv_in)
    if "rid_list" not in fieldnames:
        raise ValueError(f"Missing 'rid_list' column in {csv_in}")
    if "abnormality_info" not in fieldnames:
        raise ValueError(
            f"Missing 'abnormality_info' column in {csv_in}. "
            "Balanced evaluation requires labels."
        )

    balanced, n_abn, n_norm = build_balanced_rows(
        rows=rows,
        normal_per_abnormal=int(args.normal_per_abnormal),
        length_bucket=int(args.length_bucket),
        seed=int(args.seed),
        allow_replacement=bool(args.allow_replacement),
    )

    out_csv = out_dir / f"{split}.csv"
    _write_csv_rows(out_path=out_csv, fieldnames=fieldnames, rows=balanced)

    if bool(args.copy_roadmaps):
        _copy_if_exists(source / "roadmap.geo", out_dir / "roadmap.geo")
        _copy_if_exists(source / "roadmap.rel", out_dir / "roadmap.rel")

    manifest = BalancedManifest(
        source_dataset_dir=str(source),
        split=str(split),
        out_dir=str(out_dir),
        seed=int(args.seed),
        normal_per_abnormal=int(args.normal_per_abnormal),
        length_bucket=int(args.length_bucket),
        allow_replacement=bool(args.allow_replacement),
        num_rows_out=int(len(balanced)),
        num_abnormal_out=int(n_abn),
        num_normal_out=int(n_norm),
    )

    (out_dir / "balanced_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
