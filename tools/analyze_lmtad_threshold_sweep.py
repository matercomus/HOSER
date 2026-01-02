#!/usr/bin/env python3
"""Analyze LM-TAD scores under multiple baseline-calibrated thresholds.

This script is meant to disambiguate:
- "Method is wrong" vs
- "Abnormalities are not detectable by LM-TAD score"

It works on saved artifacts:
- Target evaluation outputs (scores) from `tools/evaluate_dataset_with_lmtad.py`
- The CSV used for scoring (must include `abnormality_info`)
- A baseline eval JSON containing baseline score samples (`baseline_eval.json`)

Outputs:
- A JSON summary (machine readable)
- Optionally a Markdown report with a compact table

Notes
- "Positive" means abnormal (`abnormality_info` non-empty and not "normal").
- Higher LM-TAD log-perplexity implies more abnormal.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_NULL_LIKE = {"", "nan", "none", "null"}


@dataclass(frozen=True)
class ThresholdMetrics:
    quantile: float
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    fpr: float


@dataclass(frozen=True)
class SweepSummary:
    split: str
    n: int
    n_pos: int
    n_neg: int
    pos_fraction: float
    auroc: Optional[float]
    auprc: Optional[float]
    thresholds: List[ThresholdMetrics]
    by_type: Dict[str, Dict[str, Any]]


def _is_abnormal(raw: str) -> bool:
    s = str(raw or "").strip().lower()
    if s in _NULL_LIKE:
        return False
    if s == "normal":
        return False
    return True


def _load_eval_scores(eval_dir: Path, split: str) -> np.ndarray:
    json_path = eval_dir / "evaluation_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing evaluation_results.json in {eval_dir}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if split not in payload:
        raise ValueError(f"Split '{split}' not found in {json_path}")

    scores = np.asarray(payload[split].get("log_perplexity_values", []), dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError(f"No finite log_perplexity_values for split '{split}'")
    return scores


def _iter_labels(csv_path: Path, label_col: str) -> Iterable[Tuple[bool, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or label_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing column '{label_col}': {csv_path} (cols={reader.fieldnames})"
            )
        for row in reader:
            raw = (row.get(label_col) or "").strip()
            yield _is_abnormal(raw), raw


def _parse_abnormal_type(raw: str) -> str:
    if not raw or raw.strip().lower() in _NULL_LIKE or raw.strip().lower() == "normal":
        return "normal"
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict) and "type" in obj:
            return str(obj["type"])
    except Exception:
        pass
    # heuristic fallback
    s = raw.lower()
    if "route_switch" in s:
        return "route_switch"
    if "detour" in s:
        return "detour"
    return "abnormal"


def _roc_auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores/labels must be non-empty and same length")
    y = labels.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / float(n_pos)
    fpr = fp / float(n_neg)

    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(np.trapezoid(tpr, fpr))


def _pr_auprc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores/labels must be non-empty and same length")
    y = labels.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)

    precision = tp / (tp + fp)
    # Average precision (matches tools/plot_lmtad_results.py)
    return float(precision[y_sorted].sum() / float(n_pos))


def _threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    thr: float,
) -> ThresholdMetrics:
    pred = scores >= float(thr)
    y = labels.astype(bool)

    tp = int(np.logical_and(pred, y).sum())
    fp = int(np.logical_and(pred, ~y).sum())
    fn = int(np.logical_and(~pred, y).sum())
    tn = int(np.logical_and(~pred, ~y).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return ThresholdMetrics(
        quantile=float("nan"),
        threshold=float(thr),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        precision=float(precision),
        recall=float(recall),
        fpr=float(fpr),
    )


def _load_baseline_scores(baseline_eval: Path, split: str) -> np.ndarray:
    path = baseline_eval
    if path.is_dir():
        # Prefer baseline_eval.json (written by evaluate tool)
        cand = path / "baseline_eval.json"
        if cand.exists():
            path = cand
        else:
            raise FileNotFoundError(f"Missing baseline_eval.json in {baseline_eval}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if split not in payload:
        raise ValueError(f"Split '{split}' not found in baseline eval: {path}")

    scores = np.asarray(payload[split].get("log_perplexity_values", []), dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError(f"No finite baseline scores for split '{split}'")
    return scores


def analyze(
    *,
    eval_dir: Path,
    split: str,
    csv_path: Path,
    baseline_eval: Path,
    baseline_split: str,
    quantiles: Sequence[float],
    label_col: str,
) -> SweepSummary:
    scores = _load_eval_scores(eval_dir, split)

    labels_list: List[bool] = []
    raw_list: List[str] = []
    for is_abn, raw in _iter_labels(csv_path, label_col):
        labels_list.append(bool(is_abn))
        raw_list.append(str(raw))

    labels = np.asarray(labels_list, dtype=bool)
    if scores.size != labels.size:
        raise ValueError(
            f"Length mismatch: scores={scores.size} labels={labels.size}. "
            "Ensure --csv is the exact CSV evaluated (same row order)."
        )

    auroc = _roc_auroc(scores, labels)
    auprc = _pr_auprc(scores, labels)

    base_scores = _load_baseline_scores(Path(baseline_eval), split=str(baseline_split))

    thr_metrics: List[ThresholdMetrics] = []
    for q in quantiles:
        thr = float(np.quantile(base_scores, float(q)))
        m = _threshold_metrics(scores, labels, thr)
        thr_metrics.append(
            ThresholdMetrics(
                quantile=float(q),
                threshold=float(thr),
                tp=m.tp,
                fp=m.fp,
                fn=m.fn,
                tn=m.tn,
                precision=m.precision,
                recall=m.recall,
                fpr=m.fpr,
            )
        )

    # Per-type breakdown at each quantile (helpful for detour vs route_switch).
    types = [_parse_abnormal_type(x) for x in raw_list]
    by_type: Dict[str, Dict[str, Any]] = {}
    for t in sorted(set(types)):
        idx = np.asarray([i for i, tt in enumerate(types) if tt == t], dtype=int)
        if idx.size == 0:
            continue
        y_t = labels[idx]
        s_t = scores[idx]
        entry: Dict[str, Any] = {
            "n": int(idx.size),
            "n_pos": int(y_t.sum()),
            "mean_score": float(np.mean(s_t)) if idx.size else None,
        }
        # add recall@q per type
        per_q: Dict[str, Any] = {}
        for m in thr_metrics:
            pred = s_t >= float(m.threshold)
            tp = int(np.logical_and(pred, y_t).sum())
            fn = int(np.logical_and(~pred, y_t).sum())
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            per_q[f"q={m.quantile:.2f}"] = {"recall": rec, "tp": tp, "fn": fn}
        entry["threshold_sweep"] = per_q
        by_type[str(t)] = entry

    n = int(labels.size)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    return SweepSummary(
        split=str(split),
        n=n,
        n_pos=n_pos,
        n_neg=n_neg,
        pos_fraction=float(n_pos / n) if n else 0.0,
        auroc=auroc,
        auprc=auprc,
        thresholds=thr_metrics,
        by_type=by_type,
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_markdown(path: Path, summary: SweepSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# LM-TAD threshold sweep ({summary.split})")
    lines.append("")
    lines.append(f"- N={summary.n} (pos={summary.n_pos}, neg={summary.n_neg}, pos_fraction={summary.pos_fraction:.2%})")
    if summary.auroc is not None:
        lines.append(f"- AUROC={summary.auroc:.4f}")
    else:
        lines.append("- AUROC: n/a (needs both positive and negative labels)")
    if summary.auprc is not None:
        lines.append(f"- AUPRC={summary.auprc:.4f}")
    else:
        lines.append("- AUPRC: n/a (needs positive labels)")
    lines.append("")

    lines.append("## Recall/precision at baseline quantiles")
    lines.append("")
    lines.append("| q | thr | recall | precision | FPR | TP | FP | FN | TN |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in summary.thresholds:
        lines.append(
            "| "
            f"{m.quantile:.2f} | {m.threshold:.6f} | {m.recall:.3f} | {m.precision:.3f} | {m.fpr:.3f} | "
            f"{m.tp} | {m.fp} | {m.fn} | {m.tn} |"
        )

    lines.append("")
    lines.append("## Per-type notes")
    lines.append("")
    for t, entry in summary.by_type.items():
        lines.append(f"- `{t}`: n={entry['n']} mean_score={entry['mean_score']:.4f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute AUROC/AUPRC and recall@{q} for LM-TAD scores"
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument(
        "--split", type=str, default="train", help="Split name (default: train)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "CSV used for scoring (must contain abnormality_info). "
            "If omitted, tries <eval-dir>/<split>_sampled.csv."
        ),
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="abnormality_info",
        help="Label column (default: abnormality_info)",
    )
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        required=True,
        help="Baseline eval JSON (or directory containing baseline_eval.json)",
    )
    parser.add_argument(
        "--baseline-split",
        type=str,
        default="train",
        help="Baseline split used to compute thresholds (default: train)",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.90,0.95,0.99",
        help="Comma-separated baseline quantiles to sweep (default: 0.90,0.95,0.99)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Where to write JSON summary (default: <eval-dir>/threshold_sweep.json)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Where to write Markdown summary (optional).",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    split = str(args.split)

    csv_path = Path(args.csv) if args.csv is not None else (eval_dir / f"{split}_sampled.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Pass --csv explicitly if you evaluated a different file."
        )

    quantiles = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    if not quantiles:
        raise ValueError("No quantiles provided")

    summary = analyze(
        eval_dir=eval_dir,
        split=split,
        csv_path=csv_path,
        baseline_eval=Path(args.baseline_eval),
        baseline_split=str(args.baseline_split),
        quantiles=quantiles,
        label_col=str(args.label_col),
    )

    out_json = Path(args.out_json) if args.out_json is not None else (eval_dir / "threshold_sweep.json")
    _write_json(out_json, asdict(summary))

    out_md = Path(args.out_md) if args.out_md is not None else (eval_dir / "threshold_sweep.md")
    _write_markdown(out_md, summary)

    print(str(out_json))
    print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
