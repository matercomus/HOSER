#!/usr/bin/env python3
"""Compute teacher separability metrics from saved LM-TAD scores + labels.

This is intended for the distillation use-case where LM-TAD is used as a
continuous scoring signal (log-perplexity), not primarily as a binary
classifier.

Given:
- `evaluation_results.json` produced by tools/evaluate_dataset_with_lmtad.py
  (or via tools/run_lmtad_decision_benchmark.py), containing per-split
  `log_perplexity_values`.
- The sampled CSV used for evaluation, containing a ground-truth abnormality
  label column (default: `abnormality_info`).

It outputs a concise summary table with:
- AUROC (ranking separability)
- Cliff's delta (effect size; equals 2*AUROC-1 for two groups)
- Cohen's d (standardized mean separation)
- 1D Wasserstein distance (distribution shift)
- Recall@top-k% (operational: how many abnormals are in the top-scoring tail)

Bootstrap confidence intervals are computed with stratified resampling (pos/neg
separately) to keep prevalence stable.

Example:
  uv run python tools/teacher_separability.py \
    --name Beijing \
    --eval-json research_runs/_benchmarks/.../evaluation_results.json \
    --labels-csv research_runs/_benchmarks/.../train.csv \
    --split train \
    --bootstrap 500 \
    --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_NULL_LIKE = {"", "nan", "none", "null", "normal"}


def is_abnormal_label(raw: str, *, normal_value: str = "normal") -> bool:
    """Return True if a label indicates an abnormal trajectory.

    This is intentionally tolerant to common representations of "normal".

    Rules:
    - treat {"", "nan", "none", "null", "normal"} (case-insensitive) as normal
    - treat `normal_value` (case-insensitive) as normal
    - everything else is abnormal
    """

    s = str(raw or "").strip().lower()
    if s in _NULL_LIKE:
        return False
    if s == str(normal_value or "").strip().lower():
        return False
    return True


def read_bool_labels_from_csv(
    csv_path: Path,
    *,
    label_col: str,
    normal_value: str = "normal",
) -> np.ndarray:
    """Read abnormality labels from a CSV as a boolean numpy array."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    labels: list[bool] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or label_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing column '{label_col}': {csv_path} "
                f"(cols={reader.fieldnames})"
            )
        for row in reader:
            raw = (row.get(label_col) or "").strip()
            labels.append(is_abnormal_label(raw, normal_value=normal_value))

    return np.asarray(labels, dtype=bool)


def read_scores_from_eval_json(eval_json: Path, *, split: str) -> np.ndarray:
    """Read log-perplexity scores from evaluation_results.json for a split."""

    if not eval_json.exists():
        raise FileNotFoundError(f"Eval JSON not found: {eval_json}")

    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    if split not in payload:
        raise ValueError(
            f"Split '{split}' not found in {eval_json}; available={sorted(payload.keys())}"
        )

    values = payload[split].get("log_perplexity_values")
    if not isinstance(values, list):
        raise ValueError(
            f"Missing or invalid 'log_perplexity_values' for split '{split}' in {eval_json}"
        )

    scores = np.asarray(values, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    return scores


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks with average ranks for ties.

    Equivalent to scipy.stats.rankdata(values, method="average"), but implemented
    without scipy.
    """

    if values.ndim != 1:
        raise ValueError("rankdata expects a 1D array")

    n = int(values.size)
    if n == 0:
        return values.astype(np.float64)

    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        # ranks are 1-based; average rank for positions [i, j)
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC using the Mann–Whitney U statistic (handles ties)."""

    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("scores and labels must be 1D")
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores/labels must be non-empty and same length")

    y = labels.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    r = _rankdata_average_ties(scores)
    r_pos_sum = float(r[y].sum())
    u = r_pos_sum - (n_pos * (n_pos + 1) / 2.0)
    return float(u / float(n_pos * n_neg))


def cliffs_delta_from_auroc(auc: float) -> float:
    """Cliff's delta for two groups, derived from AUROC."""

    return float(2.0 * float(auc) - 1.0)


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute Cohen's d between two groups."""

    if pos.size == 0 or neg.size == 0:
        raise ValueError("cohens_d requires non-empty groups")

    m1 = float(pos.mean())
    m0 = float(neg.mean())
    v1 = float(pos.var(ddof=1)) if pos.size > 1 else 0.0
    v0 = float(neg.var(ddof=1)) if neg.size > 1 else 0.0

    denom = (pos.size + neg.size - 2)
    if denom <= 0:
        return 0.0

    pooled = math.sqrt(((pos.size - 1) * v1 + (neg.size - 1) * v0) / float(denom))
    if pooled == 0.0:
        return 0.0
    return float((m1 - m0) / pooled)


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the 1D Wasserstein distance (earth mover) between two samples.

    Implemented via integrating |CDF_x - CDF_y| over the sorted support.
    """

    if x.size == 0 or y.size == 0:
        raise ValueError("wasserstein_1d requires non-empty groups")

    xs = np.sort(x.astype(np.float64), kind="mergesort")
    ys = np.sort(y.astype(np.float64), kind="mergesort")

    support = np.sort(np.unique(np.concatenate([xs, ys])))
    if support.size == 1:
        return float(abs(xs[0] - ys[0]))

    cdf_x = np.searchsorted(xs, support, side="right") / float(xs.size)
    cdf_y = np.searchsorted(ys, support, side="right") / float(ys.size)
    diff = np.abs(cdf_x - cdf_y)

    return float(np.trapezoid(diff, support))


def recall_at_top_frac(scores: np.ndarray, labels: np.ndarray, frac: float) -> float:
    """Recall among positives captured in the top `frac` scores."""

    if not (0.0 < float(frac) <= 1.0):
        raise ValueError("frac must be in (0, 1]")
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores/labels must be non-empty and same length")

    y = labels.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("recall requires at least one positive")

    k = int(math.ceil(float(frac) * float(scores.size)))
    k = max(1, min(int(scores.size), k))

    order = np.argsort(-scores)
    top = order[:k]
    tp = int(y[top].sum())
    return float(tp / float(n_pos))


def precision_at_top_frac(scores: np.ndarray, labels: np.ndarray, frac: float) -> float:
    """Precision among the top `frac` scores."""

    if not (0.0 < float(frac) <= 1.0):
        raise ValueError("frac must be in (0, 1]")
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores/labels must be non-empty and same length")

    y = labels.astype(bool)
    k = int(math.ceil(float(frac) * float(scores.size)))
    k = max(1, min(int(scores.size), k))

    order = np.argsort(-scores)
    top = order[:k]
    tp = int(y[top].sum())
    return float(tp / float(k))


def _bootstrap_ci(
    values: np.ndarray, *, ci: float
) -> tuple[float, float]:
    alpha = (1.0 - float(ci)) / 2.0
    lo = float(np.quantile(values, alpha))
    hi = float(np.quantile(values, 1.0 - alpha))
    return lo, hi


def bootstrap_stratified(
    *,
    pos: np.ndarray,
    neg: np.ndarray,
    bootstrap: int,
    seed: int,
    top_fracs: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified bootstrap for separability metrics.

    Resamples positives and negatives separately (with replacement), then
    recombines and computes metrics on the bootstrapped sample.

    Returns:
        (auroc_samples, cohens_d_samples, wasserstein_samples, cliffs_delta_samples,
         recall_samples, precision_samples)
        where recall_samples/precision_samples have shape (bootstrap, len(top_fracs)).
    """

    if bootstrap <= 0:
        raise ValueError("bootstrap must be > 0")
    if pos.size == 0 or neg.size == 0:
        raise ValueError("bootstrap_stratified requires non-empty pos/neg")

    rng = np.random.default_rng(int(seed))
    n_pos = int(pos.size)
    n_neg = int(neg.size)

    auc_s = np.empty(int(bootstrap), dtype=np.float64)
    d_s = np.empty(int(bootstrap), dtype=np.float64)
    w_s = np.empty(int(bootstrap), dtype=np.float64)
    delta_s = np.empty(int(bootstrap), dtype=np.float64)
    r_s = np.empty((int(bootstrap), len(top_fracs)), dtype=np.float64)
    p_s = np.empty((int(bootstrap), len(top_fracs)), dtype=np.float64)

    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(n_neg)

    for i in range(int(bootstrap)):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)

        b_pos_scores = pos[b_pos]
        b_neg_scores = neg[b_neg]

        b_scores = np.concatenate([b_pos_scores, b_neg_scores])
        b_labels = np.concatenate([
            np.ones(n_pos, dtype=bool),
            np.zeros(n_neg, dtype=bool),
        ])

        perm = rng.permutation(b_scores.size)
        b_scores = b_scores[perm]
        b_labels = b_labels[perm]

        b_auc = auroc(b_scores, b_labels)
        auc_s[i] = b_auc
        delta_s[i] = cliffs_delta_from_auroc(b_auc)
        d_s[i] = cohens_d(b_pos_scores, b_neg_scores)
        w_s[i] = wasserstein_1d(b_pos_scores, b_neg_scores)
        for j, f in enumerate(top_fracs):
            r_s[i, j] = recall_at_top_frac(b_scores, b_labels, f)
            p_s[i, j] = precision_at_top_frac(b_scores, b_labels, f)

    return auc_s, d_s, w_s, delta_s, r_s, p_s


@dataclass(frozen=True)
class CaseInput:
    name: str
    eval_json: Path
    labels_csv: Path


@dataclass(frozen=True)
class MetricCI:
    value: float
    lo: float
    hi: float


@dataclass(frozen=True)
class CaseResult:
    name: str
    n: int
    prevalence: float
    auroc: MetricCI
    cliffs_delta: MetricCI
    cohens_d: MetricCI
    wasserstein: MetricCI
    recall_top_1pct: MetricCI
    recall_top_5pct: MetricCI
    recall_top_10pct: MetricCI
    precision_top_1pct: MetricCI
    precision_top_5pct: MetricCI
    precision_top_10pct: MetricCI


def compute_case(
    *,
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    bootstrap: int,
    seed: int,
    ci: float,
    top_fracs: Sequence[float],
) -> CaseResult:
    """Compute separability metrics and bootstrap CIs for one dataset."""

    if scores.size != labels.size:
        raise ValueError(
            f"Length mismatch: scores={scores.size} labels={labels.size} ({name})"
        )

    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]

    y = labels.astype(bool)
    n = int(scores.size)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both pos and neg labels ({name}); pos={n_pos} neg={n_neg}")

    pos = scores[y]
    neg = scores[~y]

    point_auc = auroc(scores, y)
    point_delta = cliffs_delta_from_auroc(point_auc)
    point_d = cohens_d(pos, neg)
    point_w = wasserstein_1d(pos, neg)

    recalls = [recall_at_top_frac(scores, y, f) for f in top_fracs]
    precisions = [precision_at_top_frac(scores, y, f) for f in top_fracs]

    if bootstrap <= 0:
        def no_ci(v: float) -> MetricCI:
            return MetricCI(value=float(v), lo=float("nan"), hi=float("nan"))

        return CaseResult(
            name=name,
            n=n,
            prevalence=float(n_pos) / float(n),
            auroc=no_ci(point_auc),
            cliffs_delta=no_ci(point_delta),
            cohens_d=no_ci(point_d),
            wasserstein=no_ci(point_w),
            recall_top_1pct=no_ci(recalls[0]),
            recall_top_5pct=no_ci(recalls[1]),
            recall_top_10pct=no_ci(recalls[2]),
            precision_top_1pct=no_ci(precisions[0]),
            precision_top_5pct=no_ci(precisions[1]),
            precision_top_10pct=no_ci(precisions[2]),
        )

    auc_s, d_s, w_s, delta_s, r_s, p_s = bootstrap_stratified(
        pos=pos,
        neg=neg,
        bootstrap=int(bootstrap),
        seed=int(seed),
        top_fracs=top_fracs,
    )

    auc_lo, auc_hi = _bootstrap_ci(auc_s, ci=ci)
    delta_lo, delta_hi = _bootstrap_ci(delta_s, ci=ci)
    d_lo, d_hi = _bootstrap_ci(d_s, ci=ci)
    w_lo, w_hi = _bootstrap_ci(w_s, ci=ci)

    r_ci = []
    for j in range(len(top_fracs)):
        lo, hi = _bootstrap_ci(r_s[:, j], ci=ci)
        r_ci.append((lo, hi))

    p_ci = []
    for j in range(len(top_fracs)):
        lo, hi = _bootstrap_ci(p_s[:, j], ci=ci)
        p_ci.append((lo, hi))

    return CaseResult(
        name=name,
        n=n,
        prevalence=float(n_pos) / float(n),
        auroc=MetricCI(point_auc, auc_lo, auc_hi),
        cliffs_delta=MetricCI(point_delta, delta_lo, delta_hi),
        cohens_d=MetricCI(point_d, d_lo, d_hi),
        wasserstein=MetricCI(point_w, w_lo, w_hi),
        recall_top_1pct=MetricCI(recalls[0], r_ci[0][0], r_ci[0][1]),
        recall_top_5pct=MetricCI(recalls[1], r_ci[1][0], r_ci[1][1]),
        recall_top_10pct=MetricCI(recalls[2], r_ci[2][0], r_ci[2][1]),
        precision_top_1pct=MetricCI(precisions[0], p_ci[0][0], p_ci[0][1]),
        precision_top_5pct=MetricCI(precisions[1], p_ci[1][0], p_ci[1][1]),
        precision_top_10pct=MetricCI(precisions[2], p_ci[2][0], p_ci[2][1]),
    )


def _fmt_ci(m: MetricCI, *, digits: int = 4) -> str:
    if math.isnan(m.lo) or math.isnan(m.hi):
        return f"{m.value:.{digits}f}"
    return f"{m.value:.{digits}f} [{m.lo:.{digits}f}, {m.hi:.{digits}f}]"


def results_to_markdown(results: Sequence[CaseResult]) -> str:
    """Render results as a concise Markdown table."""

    headers = [
        "Dataset",
        "N",
        "Prev.",
        "AUROC",
        "Cliff’s δ",
        "Cohen’s d",
        "W1",
        "Recall@top1%",
        "Recall@top5%",
        "Recall@top10%",
        "Precision@top1%",
        "Precision@top5%",
        "Precision@top10%",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in results:
        row = [
            r.name,
            str(r.n),
            f"{100.0 * r.prevalence:.2f}%",
            _fmt_ci(r.auroc, digits=4),
            _fmt_ci(r.cliffs_delta, digits=4),
            _fmt_ci(r.cohens_d, digits=4),
            _fmt_ci(r.wasserstein, digits=4),
            _fmt_ci(r.recall_top_1pct, digits=4),
            _fmt_ci(r.recall_top_5pct, digits=4),
            _fmt_ci(r.recall_top_10pct, digits=4),
            _fmt_ci(r.precision_top_1pct, digits=4),
            _fmt_ci(r.precision_top_5pct, digits=4),
            _fmt_ci(r.precision_top_10pct, digits=4),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def _parse_cases(args: argparse.Namespace) -> list[CaseInput]:
    if not args.name or not args.eval_json or not args.labels_csv:
        raise ValueError("--name, --eval-json, and --labels-csv must be provided")

    if not (len(args.name) == len(args.eval_json) == len(args.labels_csv)):
        raise ValueError(
            "--name, --eval-json, and --labels-csv must be repeated the same number of times"
        )

    cases: list[CaseInput] = []
    for name, ej, lc in zip(args.name, args.eval_json, args.labels_csv):
        cases.append(CaseInput(name=name, eval_json=Path(ej), labels_csv=Path(lc)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute teacher separability metrics from saved scores + labels"
    )
    parser.add_argument(
        "--name",
        action="append",
        help="Dataset name (repeatable; must match --eval-json/--labels-csv count)",
    )
    parser.add_argument(
        "--eval-json",
        action="append",
        help="Path to evaluation_results.json (repeatable)",
    )
    parser.add_argument(
        "--labels-csv",
        action="append",
        help="Path to sampled CSV used for evaluation (repeatable)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split key in evaluation_results.json (default: train)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="abnormality_info",
        help="CSV column name for ground-truth abnormal label (default: abnormality_info)",
    )
    parser.add_argument(
        "--normal-value",
        type=str,
        default="normal",
        help="Value to treat as normal (default: normal)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Stratified bootstrap resamples for CI (0 disables; default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap (default: 0)",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="CI level in (0,1) (default: 0.95)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional: write the markdown table to this path",
    )

    args = parser.parse_args()
    cases = _parse_cases(args)

    top_fracs = (0.01, 0.05, 0.10)
    results: list[CaseResult] = []

    for case in cases:
        scores = read_scores_from_eval_json(case.eval_json, split=str(args.split))
        labels = read_bool_labels_from_csv(
            case.labels_csv,
            label_col=str(args.label_col),
            normal_value=str(args.normal_value),
        )

        if scores.size != labels.size:
            raise ValueError(
                f"Length mismatch for '{case.name}': scores={scores.size} labels={labels.size}. "
                "Ensure you pass the exact sampled CSV used for evaluation."
            )

        results.append(
            compute_case(
                name=case.name,
                scores=scores,
                labels=labels,
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
                ci=float(args.ci),
                top_fracs=top_fracs,
            )
        )

    md = results_to_markdown(results)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")

    print(md)


if __name__ == "__main__":
    main()
