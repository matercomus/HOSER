#!/usr/bin/env python3
"""Analyze LM-TAD scores with baseline-calibrated thresholds.

Why this exists
---------------
The current evaluator (`simple_evaluate_with_lmtad.evaluate_trajectories_direct`) is
hard-coded to mark the top 5% of the *evaluated set* as outliers:

    threshold = np.percentile(scores, 95)
    outliers = score > threshold

That produces an outlier rate ~5% by construction and is not suitable for
comparing an abnormal dataset against a normal baseline.

This script instead:
- Loads per-trajectory log-perplexity scores from existing
  `tools_eval_lmtad/<dataset>/evaluation_results.json` outputs.
- Computes a fixed threshold from a baseline (normal) dataset split.
- Applies the fixed threshold to a target dataset split.
- Optionally derives synthetic ground-truth labels from the target CSV
  `abnormality_info` column (when available), producing precision/recall.

It is designed for research reproducibility: the code used to obtain summary
numbers can be cited directly.

Example
-------
uv run python tools/analyze_lmtad_baseline_threshold.py \
  --baseline-eval tools_eval_lmtad/porto_hoser/evaluation_results.json \
  --target-eval tools_eval_lmtad/porto_hoser_abnormal_2/evaluation_results.json \
  --target-csv tools_eval_lmtad/porto_hoser_abnormal_2/train_sampled.csv \
  --split train \
  --quantiles 0.95 0.99
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ThresholdResult:
    quantile: float
    threshold: float
    baseline_outlier_rate: float
    target_outlier_rate: float
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tn: Optional[int] = None

    @property
    def precision(self) -> Optional[float]:
        if self.tp is None or self.fp is None:
            return None
        denom = self.tp + self.fp
        return (self.tp / denom) if denom else None

    @property
    def recall(self) -> Optional[float]:
        if self.tp is None or self.fn is None:
            return None
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else None


def _read_scores(eval_json: Path, split: str) -> List[float]:
    """Load log-perplexity scores from an evaluator JSON file."""
    data = json.loads(eval_json.read_text())
    if split not in data:
        raise KeyError(f"Split '{split}' not found in {eval_json}")
    scores = data[split].get("log_perplexity_values")
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"No scores found for split '{split}' in {eval_json}")
    return [float(x) for x in scores]


def _percentile(values: Sequence[float], q: float) -> float:
    """Compute percentile using numpy-like linear interpolation (p=0..1)."""
    if not values:
        raise ValueError("values must be non-empty")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")

    xs = sorted(values)
    # numpy default: linear interpolation between closest ranks
    # pos in [0, n-1]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _predict_outliers(scores: Sequence[float], threshold: float) -> List[bool]:
    return [float(s) > float(threshold) for s in scores]


def _safe_int_list_from_string(value: str) -> List[int]:
    """Parse road-id sequences from several formats without using eval().

    Supported:
    - "[1, 2, 3]"
    - "(1, 2, 3)"
    - "1,2,3" (comma-separated ints)
    - "[[rid, ts], ...]" (extract rid as first element)
    """
    s = (value or "").strip()
    if not s:
        return []

    # Fast path: plain comma-separated ints (no brackets)
    if (
        "," in s
        and not any(ch in s for ch in "[](){}")
        and all(
            part.strip().lstrip("-").isdigit() for part in s.split(",") if part.strip()
        )
    ):
        return [int(p.strip()) for p in s.split(",") if p.strip()]

    # Try literal_eval for list/tuple/nested structures
    try:
        obj = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

    if obj is None:
        return []

    # List/tuple of ints
    if isinstance(obj, (list, tuple)) and obj:
        first = obj[0]
        if isinstance(first, int):
            return [int(x) for x in obj]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            # list of pairs like [[rid, ts], ...]
            out: List[int] = []
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    try:
                        out.append(int(item[0]))
                    except (TypeError, ValueError):
                        continue
            return out

    return []


def _labels_from_target_csv(
    csv_path: Path,
    rid_col_hint: Optional[str] = None,
) -> Tuple[List[bool], int, int]:
    """Derive injected/normal labels from a target CSV.

    Returns
    -------
    labels: List[bool]
        True for injected rows, False for normal rows.
    used_rows: int
        Number of rows that produced a valid non-empty trajectory.
    skipped_rows: int
        Rows skipped because trajectory parsing produced empty list.

    Notes
    -----
    We intentionally mirror the evaluator behavior of skipping empty trajectories.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Target CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        # Determine road-id column
        rid_col: Optional[str] = None
        if rid_col_hint and rid_col_hint in reader.fieldnames:
            rid_col = rid_col_hint
        elif "gene_trace_road_id" in reader.fieldnames:
            rid_col = "gene_trace_road_id"
        elif "rid_list" in reader.fieldnames:
            rid_col = "rid_list"
        else:
            # fallback: any col containing 'rid'
            rid_col = next((c for c in reader.fieldnames if "rid" in c.lower()), None)

        if rid_col is None:
            raise ValueError(
                f"Cannot find rid column in {csv_path}. Columns={reader.fieldnames}"
            )

        has_info = "abnormality_info" in reader.fieldnames
        labels: List[bool] = []
        used = 0
        skipped = 0

        for row in reader:
            road_str = row.get(rid_col, "")
            road_ids = _safe_int_list_from_string(road_str)
            if not road_ids:
                skipped += 1
                continue

            info_val = (row.get("abnormality_info") if has_info else None) or "normal"
            is_injected = str(info_val).strip() != "normal"
            labels.append(is_injected)
            used += 1

    return labels, used, skipped


def _confusion(pred: Sequence[bool], true: Sequence[bool]) -> Tuple[int, int, int, int]:
    if len(pred) != len(true):
        raise ValueError(f"pred/true length mismatch: {len(pred)} vs {len(true)}")
    tp = sum(1 for p, t in zip(pred, true) if p and t)
    fp = sum(1 for p, t in zip(pred, true) if p and not t)
    fn = sum(1 for p, t in zip(pred, true) if (not p) and t)
    tn = sum(1 for p, t in zip(pred, true) if (not p) and (not t))
    return tp, fp, fn, tn


def analyze(
    baseline_eval: Path,
    target_eval: Path,
    split: str,
    quantiles: Sequence[float],
    target_csv: Optional[Path] = None,
) -> List[ThresholdResult]:
    baseline_scores = _read_scores(baseline_eval, split=split)
    target_scores = _read_scores(target_eval, split=split)

    labels: Optional[List[bool]] = None
    if target_csv is not None:
        labels, used_rows, skipped_rows = _labels_from_target_csv(target_csv)
        if len(labels) != len(target_scores):
            raise ValueError(
                "Target CSV labels do not align with target scores. "
                f"labels={len(labels)} scores={len(target_scores)} "
                f"(used_rows={used_rows} skipped_rows={skipped_rows}). "
                "Ensure you pass the exact CSV used for evaluation (e.g., *_sampled.csv)."
            )

    results: List[ThresholdResult] = []
    for q in quantiles:
        thr = _percentile(baseline_scores, q)
        base_pred = _predict_outliers(baseline_scores, thr)
        targ_pred = _predict_outliers(target_scores, thr)

        base_rate = sum(base_pred) / len(base_pred)
        targ_rate = sum(targ_pred) / len(targ_pred)

        if labels is not None:
            tp, fp, fn, tn = _confusion(targ_pred, labels)
            results.append(
                ThresholdResult(
                    quantile=q,
                    threshold=float(thr),
                    baseline_outlier_rate=float(base_rate),
                    target_outlier_rate=float(targ_rate),
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                )
            )
        else:
            results.append(
                ThresholdResult(
                    quantile=q,
                    threshold=float(thr),
                    baseline_outlier_rate=float(base_rate),
                    target_outlier_rate=float(targ_rate),
                )
            )

    return results


def _format_markdown(
    baseline_name: str,
    target_name: str,
    split: str,
    results: Sequence[ThresholdResult],
    labels_available: bool,
) -> str:
    lines: List[str] = []
    lines.append(f"## Baseline-calibrated thresholds ({split})")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_name}`")
    lines.append(f"- Target: `{target_name}`")
    lines.append("")

    if labels_available:
        lines.append(
            "| Baseline quantile | Threshold | Baseline outlier rate | "
            "Target outlier rate | Precision | Recall | TP | FP | FN | TN |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            prec = r.precision
            rec = r.recall
            lines.append(
                "| "
                f"{r.quantile:.2f} | {r.threshold:.6f} | {r.baseline_outlier_rate:.4f} | "
                f"{r.target_outlier_rate:.4f} | "
                f"{'' if prec is None else f'{prec:.4f}'} | "
                f"{'' if rec is None else f'{rec:.4f}'} | "
                f"{r.tp} | {r.fp} | {r.fn} | {r.tn} |"
            )
    else:
        lines.append(
            "| Baseline quantile | Threshold | Baseline outlier rate | "
            "Target outlier rate |"
        )
        lines.append("|---:|---:|---:|---:|")
        for r in results:
            lines.append(
                "| "
                f"{r.quantile:.2f} | {r.threshold:.6f} | {r.baseline_outlier_rate:.4f} | "
                f"{r.target_outlier_rate:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute baseline-calibrated thresholds for LM-TAD log-perplexity",
    )
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        required=True,
        help="Path to baseline evaluation_results.json",
    )
    parser.add_argument(
        "--target-eval",
        type=Path,
        required=True,
        help="Path to target evaluation_results.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split key in the JSON (train/val/test)",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.95],
        help="One or more quantiles in (0,1), e.g. 0.95 0.99",
    )
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=None,
        help=(
            "Optional: CSV file used for target evaluation (e.g. train_sampled.csv). "
            "If provided and it contains abnormality_info, the script reports precision/recall."
        ),
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default=None,
        help="Optional display name for baseline dataset",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=None,
        help="Optional display name for target dataset",
    )

    args = parser.parse_args()

    baseline_name = args.baseline_name or args.baseline_eval.parent.name
    target_name = args.target_name or args.target_eval.parent.name

    results = analyze(
        baseline_eval=args.baseline_eval,
        target_eval=args.target_eval,
        split=args.split,
        quantiles=args.quantiles,
        target_csv=args.target_csv,
    )

    print(
        _format_markdown(
            baseline_name=baseline_name,
            target_name=target_name,
            split=args.split,
            results=results,
            labels_available=args.target_csv is not None,
        )
    )


if __name__ == "__main__":
    main()
