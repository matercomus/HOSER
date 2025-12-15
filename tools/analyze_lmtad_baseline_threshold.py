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
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    method: str
    quantile: float
    threshold: float
    baseline_outlier_rate: float
    target_outlier_rate: float
    baseline_split_fprs: Optional[Dict[str, float]] = None
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    tn: Optional[int] = None
    auroc: Optional[float] = None
    average_precision: Optional[float] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None

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


def _read_all_splits(eval_json: Path) -> Dict[str, List[float]]:
    """Load scores for all splits from an evaluator JSON file."""
    data = json.loads(eval_json.read_text())
    out: Dict[str, List[float]] = {}
    for split, payload in data.items():
        if not isinstance(payload, dict):
            continue
        vals = payload.get("log_perplexity_values")
        if isinstance(vals, list) and vals:
            out[split] = [float(x) for x in vals]
    if not out:
        raise ValueError(f"No splits with scores found in {eval_json}")
    return out


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


def _median(values: Sequence[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        raise ValueError("values must be non-empty")
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _mad(values: Sequence[float], center: float) -> float:
    """Median absolute deviation (unscaled)."""
    dev = [abs(float(x) - float(center)) for x in values]
    return _median(dev)


def _predict_outliers(scores: Sequence[float], threshold: float) -> List[bool]:
    return [float(s) > float(threshold) for s in scores]


def _rankdata_average(values: Sequence[float]) -> List[float]:
    """Compute 1-based average ranks with tie handling."""
    pairs = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(pairs):
        start = pos
        v = pairs[pos][0]
        while pos < len(pairs) and pairs[pos][0] == v:
            pos += 1
        # ranks in [start+1, pos]
        avg_rank = (start + 1 + pos) / 2.0
        for _, idx in pairs[start:pos]:
            ranks[idx] = avg_rank
    return ranks


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Compute AUROC via the Mann–Whitney U statistic (tie-aware).

    Returns None if labels are degenerate.
    """
    n = len(scores)
    if n == 0 or n != len(labels):
        return None
    n_pos = sum(bool(x) for x in labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = _rankdata_average(scores)
    sum_ranks_pos = sum(r for r, y in zip(ranks, labels) if y)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _average_precision(
    scores: Sequence[float], labels: Sequence[bool]
) -> Optional[float]:
    """Compute Average Precision (area under precision-recall curve).

    Uses the standard "step" integration (a.k.a. average precision).
    Returns None if labels are degenerate.
    """
    if not scores or len(scores) != len(labels):
        return None
    n_pos = sum(bool(x) for x in labels)
    if n_pos == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        if labels[idx]:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return float(ap)


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
    method: Literal["quantile", "zscore", "mad_z"] = "quantile",
    z: Optional[float] = None,
    report_baseline_splits: bool = True,
    bootstrap: int = 0,
    target_csv: Optional[Path] = None,
    seed: int = 0,
) -> List[ThresholdResult]:
    baseline_scores = _read_scores(baseline_eval, split=split)
    target_scores = _read_scores(target_eval, split=split)

    baseline_all: Optional[Dict[str, List[float]]] = None
    if report_baseline_splits:
        baseline_all = _read_all_splits(baseline_eval)

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

    rng = np.random.default_rng(seed)

    for q in quantiles:
        if method == "quantile":
            thr = _percentile(baseline_scores, q)
        elif method == "zscore":
            if z is None:
                raise ValueError("--z is required when --method=zscore")
            mu = float(np.mean(baseline_scores))
            sigma = float(np.std(baseline_scores, ddof=1))
            if sigma <= 0:
                raise ValueError("Baseline std is zero; cannot use zscore method")
            thr = mu + float(z) * sigma
        elif method == "mad_z":
            if z is None:
                raise ValueError("--z is required when --method=mad_z")
            med = _median(baseline_scores)
            mad = _mad(baseline_scores, center=med)
            if mad <= 0:
                raise ValueError("Baseline MAD is zero; cannot use mad_z method")
            # Consistent scaling for normal distributions: sigma ~= 1.4826 * MAD
            thr = float(med + float(z) * (1.4826 * mad))
        else:
            raise ValueError(f"Unknown method: {method}")

        base_pred = _predict_outliers(baseline_scores, thr)
        targ_pred = _predict_outliers(target_scores, thr)

        base_rate = sum(base_pred) / len(base_pred)
        targ_rate = sum(targ_pred) / len(targ_pred)

        baseline_split_fprs: Optional[Dict[str, float]] = None
        if baseline_all is not None:
            baseline_split_fprs = {
                s: float(sum(_predict_outliers(v, thr)) / len(v))
                for s, v in baseline_all.items()
            }

        ci: Optional[Dict[str, Tuple[float, float]]] = None
        if bootstrap and bootstrap > 0:
            thr_samples: List[float] = []
            base_rate_samples: List[float] = []
            targ_rate_samples: List[float] = []
            ap_samples: List[float] = []
            auroc_samples: List[float] = []

            base_arr = np.asarray(baseline_scores, dtype=np.float64)
            targ_arr = np.asarray(target_scores, dtype=np.float64)
            labels_arr: Optional[np.ndarray] = None
            if labels is not None:
                labels_arr = np.asarray(labels, dtype=bool)

            for _ in range(int(bootstrap)):
                b_idx = rng.integers(0, len(base_arr), size=len(base_arr))
                t_idx = rng.integers(0, len(targ_arr), size=len(targ_arr))
                b = base_arr[b_idx]
                t = targ_arr[t_idx]

                if method == "quantile":
                    thr_b = float(np.quantile(b, q, method="linear"))
                elif method == "zscore":
                    mu_b = float(np.mean(b))
                    sigma_b = float(np.std(b, ddof=1))
                    thr_b = mu_b + float(z) * sigma_b
                else:
                    med_b = float(np.median(b))
                    mad_b = float(np.median(np.abs(b - med_b)))
                    thr_b = med_b + float(z) * (1.4826 * mad_b)

                thr_samples.append(thr_b)
                base_rate_samples.append(float(np.mean(b > thr_b)))
                targ_rate_samples.append(float(np.mean(t > thr_b)))

                if labels_arr is not None:
                    # Resample labels with the same indices as scores.
                    t_labels = labels_arr[t_idx]
                    au = _auroc(t.tolist(), t_labels.tolist())
                    ap = _average_precision(t.tolist(), t_labels.tolist())
                    if au is not None:
                        auroc_samples.append(float(au))
                    if ap is not None:
                        ap_samples.append(float(ap))

            def _ci(xs: Sequence[float]) -> Tuple[float, float]:
                if not xs:
                    return (float("nan"), float("nan"))
                lo, hi = np.quantile(np.asarray(xs), [0.025, 0.975])
                return (float(lo), float(hi))

            ci = {
                "threshold": _ci(thr_samples),
                "baseline_outlier_rate": _ci(base_rate_samples),
                "target_outlier_rate": _ci(targ_rate_samples),
            }
            if auroc_samples:
                ci["auroc"] = _ci(auroc_samples)
            if ap_samples:
                ci["average_precision"] = _ci(ap_samples)

        if labels is not None:
            tp, fp, fn, tn = _confusion(targ_pred, labels)
            auroc = _auroc(target_scores, labels)
            ap = _average_precision(target_scores, labels)
            results.append(
                ThresholdResult(
                    method=method,
                    quantile=q,
                    threshold=float(thr),
                    baseline_outlier_rate=float(base_rate),
                    target_outlier_rate=float(targ_rate),
                    baseline_split_fprs=baseline_split_fprs,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                    auroc=auroc,
                    average_precision=ap,
                    ci=ci,
                )
            )
        else:
            results.append(
                ThresholdResult(
                    method=method,
                    quantile=q,
                    threshold=float(thr),
                    baseline_outlier_rate=float(base_rate),
                    target_outlier_rate=float(targ_rate),
                    baseline_split_fprs=baseline_split_fprs,
                    ci=ci,
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
            "Target outlier rate | Precision | Recall | AUROC | AP | TP | FP | FN | TN |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            prec = r.precision
            rec = r.recall
            au = r.auroc
            ap = r.average_precision
            lines.append(
                "| "
                f"{r.quantile:.2f} | {r.threshold:.6f} | {r.baseline_outlier_rate:.4f} | "
                f"{r.target_outlier_rate:.4f} | "
                f"{'' if prec is None else f'{prec:.4f}'} | "
                f"{'' if rec is None else f'{rec:.4f}'} | "
                f"{'' if au is None else f'{au:.4f}'} | "
                f"{'' if ap is None else f'{ap:.4f}'} | "
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

    # Optional calibration readout on baseline held-out splits.
    split_fprs: Dict[str, List[str]] = {}
    for r in results:
        if r.baseline_split_fprs is None:
            continue
        for s, v in r.baseline_split_fprs.items():
            split_fprs.setdefault(s, []).append(f"q={r.quantile:.2f}: {v:.4f}")
    if split_fprs:
        lines.append(
            "Baseline FPR on each split (sanity check; should be near the target alpha only if the split distribution matches):"
        )
        for s in sorted(split_fprs.keys()):
            lines.append(f"- `{s}`: " + ", ".join(split_fprs[s]))
        lines.append("")

    # Confidence intervals (bootstrap) if present.
    any_ci = any(r.ci for r in results)
    if any_ci:
        lines.append("Bootstrap 95% CIs (per setting):")
        for r in results:
            if not r.ci:
                continue
            ci = r.ci
            parts = [
                f"q={r.quantile:.2f}",
                f"thr=[{ci['threshold'][0]:.6f}, {ci['threshold'][1]:.6f}]",
                f"base_rate=[{ci['baseline_outlier_rate'][0]:.4f}, {ci['baseline_outlier_rate'][1]:.4f}]",
                f"targ_rate=[{ci['target_outlier_rate'][0]:.4f}, {ci['target_outlier_rate'][1]:.4f}]",
            ]
            if "auroc" in ci:
                parts.append(f"auroc=[{ci['auroc'][0]:.4f}, {ci['auroc'][1]:.4f}]")
            if "average_precision" in ci:
                parts.append(
                    f"ap=[{ci['average_precision'][0]:.4f}, {ci['average_precision'][1]:.4f}]"
                )
            lines.append("- " + "; ".join(parts))
        lines.append("")

    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        "--method",
        type=str,
        default="quantile",
        choices=["quantile", "zscore", "mad_z"],
        help=(
            "Thresholding method. 'quantile' uses the baseline quantile. "
            "'zscore' uses mean+z*std. 'mad_z' uses median+z*(1.4826*MAD)."
        ),
    )
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Z value used by zscore/mad_z methods (e.g., 3.0).",
    )
    parser.add_argument(
        "--no-report-baseline-splits",
        action="store_true",
        help="Disable reporting baseline split FPR sanity checks.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap resamples for 95%% CIs (0 disables).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap.",
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
        "--out-json",
        type=Path,
        default=None,
        help="Optional: write a machine-readable JSON summary to this path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional: write the markdown table/output to this path.",
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
        method=args.method,  # type: ignore[arg-type]
        z=args.z,
        report_baseline_splits=not args.no_report_baseline_splits,
        bootstrap=args.bootstrap,
        target_csv=args.target_csv,
        seed=args.seed,
    )

    md = _format_markdown(
        baseline_name=baseline_name,
        target_name=target_name,
        split=args.split,
        results=results,
        labels_available=args.target_csv is not None,
    )
    print(md)

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)

    if args.out_json is not None:
        payload: Dict[str, Any] = {
            "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "baseline_eval": str(args.baseline_eval),
            "target_eval": str(args.target_eval),
            "split": args.split,
            "method": args.method,
            "z": args.z,
            "quantiles": list(args.quantiles),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "baseline_name": baseline_name,
            "target_name": target_name,
            "results": [
                {
                    "method": r.method,
                    "quantile": r.quantile,
                    "threshold": r.threshold,
                    "baseline_outlier_rate": r.baseline_outlier_rate,
                    "target_outlier_rate": r.target_outlier_rate,
                    "baseline_split_fprs": r.baseline_split_fprs,
                    "tp": r.tp,
                    "fp": r.fp,
                    "fn": r.fn,
                    "tn": r.tn,
                    "precision": r.precision,
                    "recall": r.recall,
                    "auroc": r.auroc,
                    "average_precision": r.average_precision,
                    "ci": r.ci,
                }
                for r in results
            ],
        }
        _write_json(args.out_json, payload)


if __name__ == "__main__":
    main()
