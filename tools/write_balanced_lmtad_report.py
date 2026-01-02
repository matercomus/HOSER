#!/usr/bin/env python3
"""Write a Markdown report (with plots) for a balanced LM-TAD sweep run.

Inputs
- --eval-dir: directory produced by `tools/run_balanced_lmtad_sweep.py` (contains evaluation_results.json + threshold_sweep.json)
- --csv: the balanced split CSV used for scoring (must contain abnormality_info)

Outputs
- <eval-dir>/report.md (or --out-md)
- <eval-dir>/plots/*.png

This is intentionally lightweight and uses only numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_NULL_LIKE = {"", "nan", "none", "null"}


def _is_abnormal(raw: str) -> bool:
    s = str(raw or "").strip().lower()
    if s in _NULL_LIKE or s == "normal":
        return False
    return True


def _parse_type(raw: str) -> str:
    if not _is_abnormal(raw):
        return "normal"
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict) and "type" in obj:
            return str(obj["type"])
    except Exception:
        pass
    s = str(raw).lower()
    if "route_switch" in s:
        return "route_switch"
    if "detour" in s:
        return "detour"
    return "abnormal"


def _read_labels(csv_path: Path, *, label_col: str = "abnormality_info") -> Tuple[np.ndarray, List[str]]:
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or label_col not in reader.fieldnames:
            raise ValueError(f"CSV missing '{label_col}': {csv_path}")
        labels: List[bool] = []
        raw: List[str] = []
        for row in reader:
            v = (row.get(label_col) or "").strip()
            labels.append(_is_abnormal(v))
            raw.append(v)
    return np.asarray(labels, dtype=bool), raw


def _load_scores(eval_dir: Path, split: str) -> np.ndarray:
    path = eval_dir / "evaluation_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = np.asarray(payload[split]["log_perplexity_values"], dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError(f"No finite scores for split '{split}'")
    return scores


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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

    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return fpr, tpr


def _pr_curve(scores: np.ndarray, labels: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    y = labels.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)

    precision = tp / (tp + fp)
    recall = tp / float(n_pos)

    # Add endpoints for nicer plot.
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return recall, precision


def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    rc = _roc_curve(scores, labels)
    if rc is None:
        return None
    fpr, tpr = rc
    return float(np.trapezoid(tpr, fpr))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    y = labels.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / (tp + fp)
    return float(precision[y_sorted].sum() / float(n_pos))


def _load_threshold_sweep(eval_dir: Path) -> Dict[str, Any]:
    path = eval_dir / "threshold_sweep.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_hist(scores: np.ndarray, labels: np.ndarray, thresholds: Dict[str, float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = scores[labels]
    neg = scores[~labels]

    plt.figure(figsize=(8, 4.5))
    bins = 60
    plt.hist(neg, bins=bins, alpha=0.55, label=f"normal (n={neg.size})")
    plt.hist(pos, bins=bins, alpha=0.55, label=f"abnormal (n={pos.size})")

    for q, thr in thresholds.items():
        plt.axvline(float(thr), linestyle="--", linewidth=1.5, label=f"thr@{q}={thr:.3f}")

    plt.title("LM-TAD log-perplexity distributions (balanced set)")
    plt.xlabel("log perplexity")
    plt.ylabel("count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_roc(scores: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rc = _roc_curve(scores, labels)
    if rc is None:
        return
    fpr, tpr = rc
    auc = _auroc(scores, labels)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pr(scores: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pc = _pr_curve(scores, labels)
    if pc is None:
        return
    recall, precision = pc
    ap = _auprc(scores, labels)
    base = float(labels.mean()) if labels.size else 0.0

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, linewidth=2, label=f"PR (AP={ap:.3f})")
    plt.hlines(base, 0, 1, linestyles=":", colors="gray", label=f"base rate={base:.2%}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _thresholds_from_sweep(sweep: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in sweep.get("thresholds", []):
        q = m.get("quantile")
        thr = m.get("threshold")
        if q is None or thr is None:
            continue
        out[f"q={float(q):.2f}"] = float(thr)
    return out


def _interpretation_block(sweep: Dict[str, Any]) -> str:
    auroc = sweep.get("auroc")
    auprc = sweep.get("auprc")
    pos_frac = sweep.get("pos_fraction")

    lines: List[str] = []
    lines.append("## Interpretation")
    lines.append("")

    if pos_frac is not None:
        lines.append(f"- Balanced positive rate: {float(pos_frac):.2%} (by construction)")

    if auroc is None:
        lines.append("- AUROC: n/a (need both positive and negative labels)")
    else:
        lines.append(f"- AUROC: {float(auroc):.3f}")

    if auprc is None:
        lines.append("- AUPRC: n/a (need positive labels)")
    else:
        lines.append(f"- AUPRC (average precision): {float(auprc):.3f}")

    lines.append("")
    lines.append("### LM-TAD vs. method")
    lines.append("")
    lines.append(
        "This experiment separates two questions:\n"
        "1) *Is the LM-TAD score informative?* (ranking/separation)\n"
        "2) *Is the current method/thresholding operating point appropriate?* (decision rule)"
    )
    lines.append("")
    lines.append(
        "- AUROC/AUPRC summarize *separation*: if they are well above chance, LM-TAD is assigning higher scores to abnormals more often than normals."
    )
    lines.append(
        "- Recall/precision at a baseline-calibrated quantile (q) summarize the *operating point*: how many you actually flag when you enforce a fixed baseline false-positive rate."
    )

    # Threshold behavior: if outlier rate is ~5% on balanced data, recall can be at most ~10% unless
    # positives are heavily concentrated in tail.
    lines.append("")
    lines.append(
        "Baseline-calibrated thresholds (e.g., q=0.95) control false-positive rate on the baseline dataset; "
        "on a balanced set they will still typically flag only ~5% of all samples as outliers. "
        "So high recall requires the positives to be very concentrated in the extreme tail of the score distribution."
    )

    # Pull per-quantile metrics.
    thr = sweep.get("thresholds", [])
    if thr:
        lines.append("")
        lines.append("Practical readout:")
        for m in thr:
            q = m.get("quantile")
            rec = m.get("recall")
            prec = m.get("precision")
            fpr = m.get("fpr")
            if q is None:
                continue
            lines.append(
                f"- q={float(q):.2f}: recall={float(rec):.3f}, precision={float(prec):.3f}, FPR={float(fpr):.3f}"
            )

    lines.append("")
    # Evidence-based conclusion (avoid over-claiming).
    if auroc is not None and float(auroc) >= 0.7:
        lines.append(
            "Conclusion: the LM-TAD score is informative on this balanced set (good separation), "
            "but strict baseline-calibrated thresholds trade recall for low FPR. "
            "So if your downstream goal is *high recall*, the issue is mostly the operating point (method choice of q), "
            "not that LM-TAD is completely non-detecting."
        )
    else:
        lines.append(
            "Conclusion: if AUROC/AUPRC are close to chance, that points to weak separation (LM-TAD score itself may not encode these abnormalities well). "
            "If AUROC is decent but recall is low at strict q, the main lever is the operating point (choose a less strict q or a different decision rule)."
        )

    return "\n".join(lines)


def write_report(*, eval_dir: Path, csv_path: Path, split: str, out_md: Path) -> Path:
    sweep = _load_threshold_sweep(eval_dir)
    scores = _load_scores(eval_dir, split)
    labels, raw = _read_labels(csv_path)

    if scores.size != labels.size:
        raise ValueError(
            f"Length mismatch: scores={scores.size}, labels={labels.size}. "
            "Ensure --csv is the same file used for scoring."
        )

    plots_dir = eval_dir / "plots"
    thresholds = _thresholds_from_sweep(sweep)

    hist_path = plots_dir / "score_hist.png"
    roc_path = plots_dir / "roc.png"
    pr_path = plots_dir / "pr.png"

    _plot_hist(scores, labels, thresholds, hist_path)
    _plot_roc(scores, labels, roc_path)
    _plot_pr(scores, labels, pr_path)

    # Per-type counts.
    types = [_parse_type(x) for x in raw]
    type_counts: Dict[str, int] = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    auroc = _auroc(scores, labels)
    auprc = _auprc(scores, labels)

    lines: List[str] = []
    lines.append(f"# Balanced LM-TAD report ({split})")
    lines.append("")
    lines.append(f"Eval dir: `{eval_dir}`")
    lines.append(f"CSV: `{csv_path}`")
    lines.append("")
    lines.append(f"- N={int(labels.size)} pos={int(labels.sum())} neg={int((~labels).sum())}")
    if auroc is not None:
        lines.append(f"- AUROC={auroc:.4f}")
    if auprc is not None:
        lines.append(f"- AUPRC={auprc:.4f}")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"![Score histogram]({hist_path.relative_to(eval_dir)})")
    lines.append("")
    lines.append(f"![ROC]({roc_path.relative_to(eval_dir)})")
    lines.append("")
    lines.append(f"![PR]({pr_path.relative_to(eval_dir)})")
    lines.append("")

    lines.append("## Composition")
    lines.append("")
    for k in sorted(type_counts.keys()):
        lines.append(f"- `{k}`: {type_counts[k]}")

    lines.append("")
    lines.append("## Threshold sweep")
    lines.append("")
    lines.append("| q | thr | recall | precision | FPR | TP | FP | FN | TN |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in sweep.get("thresholds", []):
        lines.append(
            "| "
            f"{float(m['quantile']):.2f} | {float(m['threshold']):.6f} | {float(m['recall']):.3f} | {float(m['precision']):.3f} | {float(m['fpr']):.3f} | "
            f"{int(m['tp'])} | {int(m['fp'])} | {int(m['fn'])} | {int(m['tn'])} |"
        )

    lines.append("")
    lines.append(_interpretation_block(sweep))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def main() -> int:
    _ensure_repo_root_on_syspath()

    parser = argparse.ArgumentParser(description="Write a report with plots for a balanced LM-TAD run")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out-md", type=Path, default=None)

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    csv_path = Path(args.csv)
    split = str(args.split)

    out_md = Path(args.out_md) if args.out_md is not None else (eval_dir / "report.md")

    report_path = write_report(eval_dir=eval_dir, csv_path=csv_path, split=split, out_md=out_md)
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
