#!/usr/bin/env python3
"""Plot teacher separability metrics from saved LM-TAD scores + labels.

This complements tools/teacher_separability.py by producing figures suitable
for reports:

1) Metric summary with bootstrap confidence intervals (AUROC, Cliff's delta,
   Cohen's d, Wasserstein-1D).
2) Recall@top-k% curve (with CI bands) to show how quickly abnormals concentrate
   in the top-scoring tail.

Inputs are the same as teacher_separability:
- evaluation_results.json (scores)
- sampled CSV (labels)

Example:
  uv run python tools/plot_teacher_separability.py \
    --name Beijing_per_type_detour \
    --eval-json research_runs/_benchmarks/.../evaluation_results.json \
    --labels-csv research_runs/_benchmarks/.../train.csv \
    --name porto_hoser_per_type_detour \
    --eval-json research_runs/_benchmarks/.../evaluation_results.json \
    --labels-csv research_runs/_benchmarks/.../train.csv \
    --out-dir research_runs/_benchmarks/lmtad_teacher_separability_plots_20260109 \
    --bootstrap 500 --seed 0
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    # When imported as `tools.plot_teacher_separability` (e.g., in tests).
    from tools.teacher_separability import (  # type: ignore
        _bootstrap_ci,
        auroc,
        bootstrap_stratified,
        cliffs_delta_from_auroc,
        cohens_d,
        precision_at_top_frac,
        read_bool_labels_from_csv,
        read_scores_from_eval_json,
        recall_at_top_frac,
        wasserstein_1d,
    )
except ModuleNotFoundError:
    # When invoked as `python tools/plot_teacher_separability.py`.
    from teacher_separability import (  # type: ignore
        _bootstrap_ci,
        auroc,
        bootstrap_stratified,
        cliffs_delta_from_auroc,
        cohens_d,
        precision_at_top_frac,
        read_bool_labels_from_csv,
        read_scores_from_eval_json,
        recall_at_top_frac,
        wasserstein_1d,
    )


@dataclass(frozen=True)
class MetricSummary:
    name: str
    n: int
    prevalence: float
    auc: tuple[float, float, float]
    delta: tuple[float, float, float]
    d: tuple[float, float, float]
    w1: tuple[float, float, float]


def _quantile_band(samples: np.ndarray, *, ci: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) bands over axis=0."""

    alpha = (1.0 - float(ci)) / 2.0
    lo = np.quantile(samples, alpha, axis=0)
    hi = np.quantile(samples, 1.0 - alpha, axis=0)
    return lo, hi


def _metric_ci(value: float, samples: np.ndarray, *, ci: float) -> tuple[float, float, float]:
    lo, hi = _bootstrap_ci(samples, ci=ci)
    return float(value), float(lo), float(hi)


def _compute_metric_summary(
    *,
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    bootstrap: int,
    seed: int,
    ci: float,
    top_fracs: Sequence[float],
) -> tuple[MetricSummary, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute point+CI metrics and return recall/precision samples for plotting."""

    if scores.size != labels.size:
        raise ValueError(f"Length mismatch for {name}: scores={scores.size} labels={labels.size}")

    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]

    y = labels.astype(bool)
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

    if bootstrap <= 0:
        dummy = np.asarray([float("nan")], dtype=np.float64)
        summary = MetricSummary(
            name=name,
            n=int(scores.size),
            prevalence=float(n_pos) / float(scores.size),
            auc=(float(point_auc), float("nan"), float("nan")),
            delta=(float(point_delta), float("nan"), float("nan")),
            d=(float(point_d), float("nan"), float("nan")),
            w1=(float(point_w), float("nan"), float("nan")),
        )
        return summary, dummy, dummy, dummy, dummy

    auc_s, d_s, w_s, delta_s, r_s, p_s = bootstrap_stratified(
        pos=pos,
        neg=neg,
        bootstrap=int(bootstrap),
        seed=int(seed),
        top_fracs=top_fracs,
    )

    summary = MetricSummary(
        name=name,
        n=int(scores.size),
        prevalence=float(n_pos) / float(scores.size),
        auc=_metric_ci(point_auc, auc_s, ci=ci),
        delta=_metric_ci(point_delta, delta_s, ci=ci),
        d=_metric_ci(point_d, d_s, ci=ci),
        w1=_metric_ci(point_w, w_s, ci=ci),
    )

    # point recall curve values (same definition as in teacher_separability)
    point_recalls = np.asarray(
        [recall_at_top_frac(scores, y, f) for f in top_fracs], dtype=np.float64
    )

    point_precisions = np.asarray(
        [precision_at_top_frac(scores, y, f) for f in top_fracs], dtype=np.float64
    )

    return summary, point_recalls, r_s, point_precisions, p_s


def plot_metric_bars(
    *,
    summaries: Sequence[MetricSummary],
    out_path: Path,
) -> None:
    """Create a compact metric bar plot with bootstrap CIs."""

    metrics = [
        ("AUROC", "auc", (0.0, 1.0)),
        ("Cliff’s δ", "delta", (-1.0, 1.0)),
        ("Cohen’s d", "d", None),
        ("Wasserstein-1D", "w1", None),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.6))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    names = [s.name for s in summaries]
    x = np.arange(len(names))

    for ax, (title, field, ylim) in zip(axes, metrics):
        vals = []
        yerr = [[], []]
        for s in summaries:
            v, lo, hi = getattr(s, field)
            vals.append(v)
            if not (math.isnan(lo) or math.isnan(hi)):
                yerr[0].append(v - lo)
                yerr[1].append(hi - v)
            else:
                yerr[0].append(0.0)
                yerr[1].append(0.0)

        ax.bar(x, vals, yerr=np.asarray(yerr), capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_recall_curve(
    *,
    names: Sequence[str],
    top_fracs: np.ndarray,
    point_recalls: Sequence[np.ndarray],
    recall_samples: Sequence[np.ndarray],
    ci: float,
    out_path: Path,
) -> None:
    """Plot recall@top-k% curve with CI bands."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # Random ranking baseline: recall ~ top_frac
    ax.plot(top_fracs, top_fracs, linestyle="--", color="gray", linewidth=1, label="random baseline")

    for name, pr, rs in zip(names, point_recalls, recall_samples):
        lo, hi = _quantile_band(rs, ci=ci)
        ax.plot(top_fracs, pr, marker="o", label=name)
        ax.fill_between(top_fracs, lo, hi, alpha=0.18, label=f"{name} {int(100.0 * ci)}% CI")

    ax.set_xlabel("Top fraction of trajectories reviewed")
    ax.set_ylabel("Recall on abnormal trajectories")
    ax.set_xlim(float(top_fracs.min()), float(top_fracs.max()))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax.set_title("Teacher separability: recall in top-scoring tail")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision_curve(
    *,
    names: Sequence[str],
    prevalence: Sequence[float],
    top_fracs: np.ndarray,
    point_precisions: Sequence[np.ndarray],
    precision_samples: Sequence[np.ndarray],
    ci: float,
    out_path: Path,
) -> None:
    """Plot precision@top-k% curve with CI bands and prevalence baselines."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    baseline_labeled = False
    for name, prev, pp, ps in zip(names, prevalence, point_precisions, precision_samples):
        lo, hi = _quantile_band(ps, ci=ci)
        ax.plot(top_fracs, pp, marker="o", label=name)
        ax.fill_between(top_fracs, lo, hi, alpha=0.18, label=f"{name} {int(100.0 * ci)}% CI")
        ax.hlines(
            prev,
            xmin=float(top_fracs.min()),
            xmax=float(top_fracs.max()),
            colors="gray",
            linestyles="--",
            linewidth=1,
            label=("prevalence baseline" if not baseline_labeled else "_nolegend_"),
        )
        baseline_labeled = True

    ax.set_xlabel("Top fraction of trajectories reviewed")
    ax.set_ylabel("Precision in selected top fraction")
    ax.set_xlim(float(top_fracs.min()), float(top_fracs.max()))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax.set_title("Teacher separability: precision in top-scoring tail")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_cases(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if not args.name or not args.eval_json or not args.labels_csv:
        raise ValueError("--name, --eval-json, and --labels-csv must be provided")
    if not (len(args.name) == len(args.eval_json) == len(args.labels_csv)):
        raise ValueError(
            "--name, --eval-json, and --labels-csv must be repeated the same number of times"
        )
    return [
        (n, Path(ej), Path(lc))
        for n, ej, lc in zip(args.name, args.eval_json, args.labels_csv)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot teacher separability metrics from saved scores + labels"
    )
    parser.add_argument("--name", action="append", help="Dataset name (repeatable)")
    parser.add_argument("--eval-json", action="append", help="Path to evaluation_results.json (repeatable)")
    parser.add_argument("--labels-csv", action="append", help="Path to sampled CSV used for evaluation (repeatable)")
    parser.add_argument("--split", type=str, default="train", help="Split key (default: train)")
    parser.add_argument("--label-col", type=str, default="abnormality_info", help="Label column (default: abnormality_info)")
    parser.add_argument("--normal-value", type=str, default="normal", help="Value treated as normal (default: normal)")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap resamples (0 disables; default: 500)")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed (default: 0)")
    parser.add_argument("--ci", type=float, default=0.95, help="CI level (default: 0.95)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write plots")
    parser.add_argument(
        "--curve-max-frac",
        type=float,
        default=0.20,
        help="Max fraction for recall curve (default: 0.20)",
    )
    parser.add_argument(
        "--curve-num-points",
        type=int,
        default=20,
        help="Number of curve points in (0, max] (default: 20)",
    )

    args = parser.parse_args()

    out_dir: Path = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _parse_cases(args)

    max_frac = float(args.curve_max_frac)
    if not (0.0 < max_frac <= 1.0):
        raise ValueError("--curve-max-frac must be in (0, 1]")

    num = int(args.curve_num_points)
    if num <= 1:
        raise ValueError("--curve-num-points must be > 1")

    top_fracs = np.linspace(max(0.001, max_frac / float(num)), max_frac, num=num)

    summaries: list[MetricSummary] = []
    names: list[str] = []
    prevs: list[float] = []
    point_recalls: list[np.ndarray] = []
    recall_samples: list[np.ndarray] = []
    point_precisions: list[np.ndarray] = []
    precision_samples: list[np.ndarray] = []

    for i, (name, eval_json, labels_csv) in enumerate(cases):
        scores = read_scores_from_eval_json(eval_json, split=str(args.split))
        labels = read_bool_labels_from_csv(
            labels_csv,
            label_col=str(args.label_col),
            normal_value=str(args.normal_value),
        )
        if scores.size != labels.size:
            raise ValueError(
                f"Length mismatch for '{name}': scores={scores.size} labels={labels.size}. "
                "Ensure you pass the exact sampled CSV used for evaluation."
            )

        summary, pr, rs, pp, ps = _compute_metric_summary(
            name=name,
            scores=scores,
            labels=labels,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed) + i,
            ci=float(args.ci),
            top_fracs=top_fracs,
        )

        summaries.append(summary)
        names.append(name)
        prevs.append(float(summary.prevalence))
        point_recalls.append(pr)
        recall_samples.append(rs)
        point_precisions.append(pp)
        precision_samples.append(ps)

    plot_metric_bars(
        summaries=summaries,
        out_path=out_dir / "teacher_separability_metrics.png",
    )
    plot_recall_curve(
        names=names,
        top_fracs=top_fracs,
        point_recalls=point_recalls,
        recall_samples=recall_samples,
        ci=float(args.ci),
        out_path=out_dir / "teacher_separability_recall_curve.png",
    )
    plot_precision_curve(
        names=names,
        prevalence=prevs,
        top_fracs=top_fracs,
        point_precisions=point_precisions,
        precision_samples=precision_samples,
        ci=float(args.ci),
        out_path=out_dir / "teacher_separability_precision_curve.png",
    )

    # Also write a small README-ish note for convenience.
    note = (
        "Generated by tools/plot_teacher_separability.py\n"
        f"split={args.split} label_col={args.label_col} bootstrap={args.bootstrap} ci={args.ci}\n"
    )
    (out_dir / "README.txt").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
