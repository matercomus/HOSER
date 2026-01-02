#!/usr/bin/env python3
"""Plot LM-TAD evaluation results (histograms + boxplot).

Reads `evaluation_results.json` (aggregated) or `evaluation_results.jsonl` and
produces per-split histograms with the computed 95th percentile threshold and
an across-splits boxplot of log-perplexities.

Usage example:
  python tools/plot_lmtad_results.py --eval-dir tools_eval_lmtad/Beijing_abnormal --out plots/lmtad_eval.png

Exclude test split:
    python tools/plot_lmtad_results.py --eval-dir tools_eval_lmtad/Beijing_abnormal --out plots/ --splits train,val

Optional ROC curves (requires labels from the sampled CSV used for evaluation):
    python tools/plot_lmtad_results.py \
        --eval-dir tools_eval_lmtad/porto_hoser_abnormal_2 \
        --out tools_eval_lmtad/porto_hoser_abnormal_2 \
        --splits train,val \
        --labels-csv-template tools_eval_lmtad/porto_hoser_abnormal_2/{split}_sampled.csv

When labels are provided, this script also outputs:
- per-split ROC curves
- per-split Precision-Recall (PR) curves
- per-split normal-vs-abnormal density plots (overlaid histograms + KDE)
"""

from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def _csv_has_column(csv_path: Path, col: str) -> bool:
    """Return True if a CSV file contains a given column name."""
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames is not None and col in reader.fieldnames
    except Exception:
        return False


def _read_bool_labels_from_csv(
    csv_path: Path,
    *,
    label_col: str,
    abnormal_value: str = "normal",
) -> list[bool]:
    """Return abnormality labels from a sampled CSV.

    Label rule (matches earlier analysis tooling):
    - abnormal if `abnormality_info` exists and is not equal to `normal`.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    labels: list[bool] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or label_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing column '{label_col}': {csv_path} (cols={reader.fieldnames})"
            )
        for row in reader:
            raw = (row.get(label_col) or "").strip()
            labels.append(raw != "" and raw != abnormal_value)
    return labels


def _roc_curve_points(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve points and AUROC.

    Returns:
        (fpr, tpr, auroc)
    """
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores/labels must be non-empty and same length")

    y = labels.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC requires both positive and negative labels")

    order = np.argsort(-scores)
    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)

    tpr = tp / float(n_pos)
    fpr = fp / float(n_neg)

    # Add (0,0) start.
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    auroc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auroc


def _precision_recall_curve_points(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision-Recall curve points and AUPRC (average precision).

    Returns:
        (recall, precision, auprc)

    Notes:
        - `labels=True` is treated as the positive class.
        - `scores` should be higher for more-positive predictions.
    """
    if scores.size == 0 or scores.size != labels.size:
        raise ValueError("scores/labels must be non-empty and same length")

    y = labels.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("PR curve requires at least one positive label")

    order = np.argsort(-scores)
    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)

    precision = tp / (tp + fp)
    recall = tp / float(n_pos)

    # Average precision: mean precision at each positive example.
    # Equivalent to a step-wise area under the PR curve.
    auprc = float(precision[y_sorted].sum() / float(n_pos))

    # Add a (0,1) starting point for cleaner plots.
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))

    return recall, precision, auprc


def _bootstrap_ci_percent(
    values01: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
    ci: float,
) -> tuple[float, float]:
    """Bootstrap percentile CI for mean(values01) expressed in percent."""
    if bootstrap <= 0:
        raise ValueError("bootstrap must be > 0")
    if not (0.0 < float(ci) < 1.0):
        raise ValueError("ci must be in (0,1)")
    if values01.size == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(int(seed))
    n = int(values01.size)
    means = np.empty(int(bootstrap), dtype=np.float64)
    for i in range(int(bootstrap)):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values01[idx].mean()) * 100.0

    alpha = (1.0 - float(ci)) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi


def load_results(eval_dir: Path):
    agg = {}
    json_file = eval_dir / "evaluation_results.json"
    jsonl_file = eval_dir / "evaluation_results.jsonl"
    if json_file.exists():
        with open(json_file, "r") as f:
            agg = json.load(f)
        return agg

    if jsonl_file.exists():
        for line in open(jsonl_file, "r"):
            try:
                rec = json.loads(line)
                split = rec.get("split")
                if split:
                    agg[split] = rec.get("results", {})
            except Exception:
                continue
        return agg

    raise FileNotFoundError(f"No results file found in {eval_dir}")


def plot_results(
    agg: dict,
    out_path: Path,
    show: bool = False,
    splits: list[str] | None = None,
    labels_csv_by_split: dict[str, Path] | None = None,
    label_col: str = "abnormality_info",
    labels_required: bool = False,
    bootstrap: int = 0,
    seed: int = 0,
    ci: float = 0.95,
):
    # Prepare data
    if splits is None:
        splits = sorted(agg.keys())
    else:
        requested = [s for s in splits if s]
        missing = [s for s in requested if s not in agg]
        if missing:
            available = ", ".join(sorted(agg.keys()))
            raise ValueError(
                f"Requested splits not present in results: {missing}. Available: [{available}]"
            )
        splits = requested

    data = {}
    for s in splits:
        vals = np.array(agg[s].get("log_perplexity_values", []), dtype=np.float64)
        # Filter non-finite values for plotting/statistics
        vals = vals[np.isfinite(vals)]
        data[s] = vals

    if len(splits) == 0:
        raise ValueError("No splits found in results to plot")

    sns.set(style="whitegrid")

    # Determine output directory and base name. If out_path is a directory
    # or has no suffix, treat it as a directory. Otherwise use parent and stem
    # to build filenames.
    if out_path.exists() and out_path.is_dir():
        out_dir = out_path
        base = "lmtad_eval"
    else:
        if out_path.suffix:
            out_dir = out_path.parent
            base = out_path.stem
        else:
            out_dir = out_path
            base = "lmtad_eval"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save one histogram per split
    saved_files = []
    for s in splits:
        vals = data[s]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if vals.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center")
            ax.set_title(s)
        else:
            sns.histplot(vals, bins=50, kde=False, ax=ax)
            base_meta = agg.get(s, {}).get("baseline_calibrated") if isinstance(agg, dict) else None
            thr = None
            if isinstance(base_meta, dict) and base_meta.get("threshold") is not None:
                thr = float(base_meta["threshold"])
                q = base_meta.get("quantile")
                if q is not None:
                    ax.axvline(
                        thr,
                        color="red",
                        linestyle="--",
                        label=f"Baseline q={float(q):.3f} thr={thr:.3f}",
                    )
                else:
                    ax.axvline(
                        thr,
                        color="red",
                        linestyle="--",
                        label=f"Baseline thr={thr:.3f}",
                    )
            else:
                thr = float(np.percentile(vals, 95))
                ax.axvline(
                    thr,
                    color="red",
                    linestyle="--",
                    label=f"95th pct (within split)={thr:.3f}",
                )
            ax.set_title(f"{s} (N={len(vals)})")
            ax.set_xlabel("Log perplexity")
            ax.legend()

        out_file = out_dir / f"{base}_{s}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        saved_files.append(out_file)
        plt.close(fig)

        # Optional: ROC/PR curves if we have labels for this split.
        if labels_csv_by_split is not None and s in labels_csv_by_split:
            try:
                labels_list = _read_bool_labels_from_csv(
                    labels_csv_by_split[s], label_col=label_col
                )
            except Exception as e:
                if labels_required:
                    raise
                print(
                    "[plot_lmtad_results] Skipping ROC/PR/density for split "
                    f"'{s}': cannot read labels from {labels_csv_by_split[s]} ({e})"
                )
                continue
            scores = np.array(agg[s].get("log_perplexity_values", []), dtype=np.float64)
            labels = np.array(labels_list, dtype=bool)
            if scores.size != labels.size:
                raise ValueError(
                    f"Length mismatch for split '{s}': scores={scores.size} labels={labels.size}. "
                    "Use the exact sampled CSV used for evaluation, and ensure no rows were dropped."
                )

            finite_mask = np.isfinite(scores)
            scores = scores[finite_mask]
            labels = labels[finite_mask]

            # Normal vs abnormal density plot (overlaid histograms + KDE).
            # This complements ROC/PR by showing the score distribution separation.
            normal_scores = scores[~labels]
            abnormal_scores = scores[labels]
            fig_den, ax_den = plt.subplots(figsize=(8, 4.5))
            if scores.size == 0:
                ax_den.text(0.5, 0.5, "No finite values", ha="center")
            else:
                base_meta = agg.get(s, {}).get("baseline_calibrated") if isinstance(agg, dict) else None
                if isinstance(base_meta, dict) and base_meta.get("threshold") is not None:
                    thr = float(base_meta["threshold"])
                    q = base_meta.get("quantile")
                    thr_label = (
                        f"Baseline q={float(q):.3f} thr={thr:.3f}"
                        if q is not None
                        else f"Baseline thr={thr:.3f}"
                    )
                else:
                    thr = float(np.percentile(scores, 95))
                    thr_label = f"95th pct (within split)={thr:.3f}"
                # Use density so shapes are comparable even if class counts differ.
                if normal_scores.size > 0:
                    sns.histplot(
                        normal_scores,
                        bins=50,
                        stat="density",
                        element="step",
                        fill=True,
                        alpha=0.35,
                        ax=ax_den,
                        label=f"Normal (n={normal_scores.size})",
                        color=sns.color_palette("Blues", 3)[1],
                    )
                    if normal_scores.size >= 2:
                        sns.kdeplot(
                            normal_scores,
                            ax=ax_den,
                            color=sns.color_palette("Blues", 3)[2],
                            linewidth=2,
                        )
                if abnormal_scores.size > 0:
                    sns.histplot(
                        abnormal_scores,
                        bins=50,
                        stat="density",
                        element="step",
                        fill=True,
                        alpha=0.35,
                        ax=ax_den,
                        label=f"Abnormal (n={abnormal_scores.size})",
                        color=sns.color_palette("Reds", 3)[1],
                    )
                    if abnormal_scores.size >= 2:
                        sns.kdeplot(
                            abnormal_scores,
                            ax=ax_den,
                            color=sns.color_palette("Reds", 3)[2],
                            linewidth=2,
                        )

                ax_den.axvline(
                    thr,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label=thr_label,
                )
                ax_den.set_xlabel("Log perplexity")
                ax_den.set_ylabel("Density")
                ax_den.set_title(f"{s}: Normal vs abnormal (density)")
                ax_den.legend()

            den_file = out_dir / f"{base}_{s}_density.png"
            fig_den.tight_layout()
            fig_den.savefig(den_file, dpi=150)
            saved_files.append(den_file)
            plt.close(fig_den)

            fpr, tpr, auroc = _roc_curve_points(scores, labels)
            fig_roc, ax_roc = plt.subplots(figsize=(5.5, 5.5))
            ax_roc.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            pad = 0.02
            ax_roc.set_xlim(-pad, 1.0 + pad)
            ax_roc.set_ylim(-pad, 1.0 + pad)
            ax_roc.set_title(f"{s}: ROC curve")
            ax_roc.legend(loc="lower right")
            roc_file = out_dir / f"{base}_{s}_roc.png"
            fig_roc.tight_layout()
            fig_roc.savefig(roc_file, dpi=150)
            saved_files.append(roc_file)
            plt.close(fig_roc)

            recall, precision, auprc = _precision_recall_curve_points(scores, labels)
            fig_pr, ax_pr = plt.subplots(figsize=(5.5, 5.5))
            ax_pr.plot(recall, precision, label=f"AUPRC={auprc:.4f}")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            pad = 0.02
            ax_pr.set_xlim(-pad, 1.0 + pad)
            ax_pr.set_ylim(-pad, 1.0 + pad)
            ax_pr.set_title(f"{s}: Precision-Recall curve")
            ax_pr.legend(loc="lower left")
            pr_file = out_dir / f"{base}_{s}_pr.png"
            fig_pr.tight_layout()
            fig_pr.savefig(pr_file, dpi=150)
            saved_files.append(pr_file)
            plt.close(fig_pr)

    # Save a separate boxplot comparing splits
    fig2, ax2 = plt.subplots(figsize=(max(6, len(splits) * 0.8), 5))
    ordered_vals = [data[s] for s in splits]
    sns.boxplot(data=ordered_vals, orient="v", ax=ax2)
    ax2.set_xticks(np.arange(len(splits)))
    ax2.set_xticklabels(splits, rotation=45, ha="right")
    ax2.set_ylabel("Log perplexity")
    ax2.set_title("Per-split log perplexity distribution (boxplot)")
    boxplot_file = out_dir / f"{base}_boxplot.png"
    fig2.tight_layout()
    fig2.savefig(boxplot_file, dpi=150)
    saved_files.append(boxplot_file)
    plt.close(fig2)

    for p in saved_files:
        print(f"Saved plot to: {p}")
    if show:
        # If user asked to show, open the boxplot
        img = plt.imread(str(boxplot_file))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    # Abnormality percentages bar chart (uses outlier_rate from results)
    abnormal_rates = []
    counts = []
    outlier_labels_by_split: dict[str, np.ndarray] = {}
    for s in splits:
        r = agg[s].get("outlier_rate")
        # Expecting fraction in [0,1]; convert to percent for plotting
        if r is None:
            abnormal_rates.append(np.nan)
        else:
            abnormal_rates.append(float(r) * 100.0)
        counts.append(
            int(agg[s].get("num_trajectories", len(data[s])))
            if agg[s].get("num_trajectories") is not None
            else len(data[s])
        )

        labels = agg[s].get("outlier_labels")
        if isinstance(labels, list) and labels:
            outlier_labels_by_split[s] = np.asarray(labels, dtype=np.float64)

    # Error bars: bootstrap percentile CI when enabled, else binomial SE.
    if bootstrap and int(bootstrap) > 0:
        yerr = np.zeros((2, len(splits)), dtype=np.float64)
        ci_low: list[float] = []
        ci_high: list[float] = []
        for i, (s, pct, n) in enumerate(zip(splits, abnormal_rates, counts)):
            if np.isnan(pct) or n <= 0:
                ci_low.append(0.0)
                ci_high.append(0.0)
                continue
            if s not in outlier_labels_by_split:
                raise ValueError(
                    f"Cannot bootstrap CI for split '{s}': missing outlier_labels in results."
                )
            lo, hi = _bootstrap_ci_percent(
                outlier_labels_by_split[s],
                bootstrap=int(bootstrap),
                seed=int(seed) + i,
                ci=float(ci),
            )
            ci_low.append(lo)
            ci_high.append(hi)
            yerr[0, i] = float(pct) - lo
            yerr[1, i] = hi - float(pct)
    else:
        # Compute standard error (%) for proportion p: sqrt(p*(1-p)/n) * 100
        yerr = np.zeros(len(splits), dtype=np.float64)
        ci_low = []
        ci_high = []
        for pct, n in zip(abnormal_rates, counts):
            if np.isnan(pct) or n <= 0:
                yerr = yerr
                ci_low.append(0.0)
                ci_high.append(0.0)
                continue
            p = pct / 100.0
            se = np.sqrt(p * (1.0 - p) / float(n)) * 100.0
            ci_low.append(float(pct) - 1.96 * float(se))
            ci_high.append(float(pct) + 1.96 * float(se))
            yerr[len(ci_low) - 1] = float(se)

    fig3, ax3 = plt.subplots(figsize=(max(6, len(splits) * 0.9), 4))
    x = np.arange(len(splits))
    # Use matplotlib bar so we can draw errorbars using computed SE
    bars = ax3.bar(
        x,
        abnormal_rates,
        yerr=yerr,
        capsize=6,
        color=sns.color_palette("Reds", len(splits)),
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(splits)
    ax3.set_ylabel("Abnormality rate (%)")
    ax3.set_xlabel("Split")
    ax3.set_title("Abnormality (outlier) percentage per split", pad=14)

    # Adjust y-limits to leave space above bars for annotations and errorbars
    top = 0.0
    if isinstance(yerr, np.ndarray) and yerr.ndim == 2:
        for v, up in zip(abnormal_rates, yerr[1, :]):
            if not np.isnan(v):
                top = max(top, float(v) + float(up))
    else:
        for v, se in zip(abnormal_rates, yerr):
            if not np.isnan(v):
                top = max(top, float(v) + float(se))
    ax3.set_ylim(0, max(5.0, top * 1.25 + 1.0))

    # Annotate bars above the errorbar (or inside if tall enough)
    for i, (rect, v) in enumerate(zip(bars, abnormal_rates)):
        if np.isnan(v):
            continue
        # place the label just above the errorbar
        if isinstance(yerr, np.ndarray) and yerr.ndim == 2:
            up = float(yerr[1, i])
        else:
            up = float(yerr[i])
        y = float(v) + up + 0.5
        ax3.text(
            rect.get_x() + rect.get_width() / 2.0,
            y,
            f"{v:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    abnormal_file = out_dir / f"{base}_abnormality.png"
    fig3.tight_layout()
    fig3.savefig(abnormal_file, dpi=150)
    saved_files.append(abnormal_file)
    plt.close(fig3)

    # Also write a small CSV summary with counts and abnormality rates
    summary_file = out_dir / f"{base}_abnormality.csv"
    try:
        with open(summary_file, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(
                [
                    "split",
                    "num_trajectories",
                    "outlier_rate_fraction",
                    "outlier_rate_percent",
                    "outlier_rate_errbar_percent",
                    "outlier_rate_ci_low_percent",
                    "outlier_rate_ci_high_percent",
                ]
            )
            if isinstance(yerr, np.ndarray) and yerr.ndim == 2:
                errbars = [float(x) for x in yerr[1, :]]
            else:
                errbars = [float(x) for x in yerr]
            for s, cnt, r, err, lo, hi in zip(
                splits, counts, abnormal_rates, errbars, ci_low, ci_high
            ):
                frac = (r / 100.0) if not np.isnan(r) else ""
                pct = r if not np.isnan(r) else ""
                writer.writerow([s, cnt, frac, pct, err, lo, hi])
    except Exception:
        pass

    print(f"Saved abnormality plot to: {abnormal_file}")
    print(f"Saved abnormality summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot LM-TAD evaluation results")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory with evaluation_results.json or .jsonl",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output image path (png/pdf)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help=(
            "Comma-separated split names to plot (e.g., train,val). "
            "Default: plot all splits found in the results."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help=(
            "Optional: labels CSV for a single split (must contain abnormality_info). "
            "For multiple splits, use --labels-csv-template."
        ),
    )
    parser.add_argument(
        "--labels-csv-template",
        type=str,
        default=None,
        help=(
            "Optional: per-split labels CSV template containing '{split}', e.g. "
            "tools_eval_lmtad/porto_hoser_abnormal_2/{split}_sampled.csv."
        ),
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="abnormality_info",
        help="Column name used for ground-truth abnormal labels (default: abnormality_info).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help=(
            "Bootstrap resamples for abnormality-rate error bars (0 disables). "
            "When enabled, error bars show percentile CI of outlier_rate."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="CI level for bootstrap (default: 0.95).",
    )
    parser.add_argument("--show", action="store_true", help="Show plot after saving")
    args = parser.parse_args()

    agg = load_results(args.eval_dir)
    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    labels_csv_by_split: dict[str, Path] | None = None
    labels_required = False
    if args.labels_csv_template is not None:
        labels_required = True
        if splits is None:
            # Derive from agg keys to keep UX simple.
            split_names = sorted(agg.keys())
        else:
            split_names = splits
        labels_csv_by_split = {
            s: Path(str(args.labels_csv_template).format(split=s)) for s in split_names
        }
    elif args.labels_csv is not None:
        labels_required = True
        if splits is None or len(splits) != 1:
            raise ValueError(
                "--labels-csv requires exactly one split. Use --splits <one> or --labels-csv-template."
            )
        labels_csv_by_split = {splits[0]: args.labels_csv}
    else:
        # Default behavior: auto-detect per-split sampled CSVs in eval-dir.
        # This keeps the CLI simple for common cases where the evaluation
        # directory already contains `{split}_sampled.csv` files.
        #
        # Important: only enable label-based plots (ROC/PR/density) when the
        # sampled CSV actually contains the requested label column.
        split_names = sorted(agg.keys()) if splits is None else splits
        inferred: dict[str, Path] = {}
        for s in split_names:
            candidate = args.eval_dir / f"{s}_sampled.csv"
            if candidate.exists() and _csv_has_column(candidate, str(args.label_col)):
                inferred[s] = candidate
        if inferred:
            labels_csv_by_split = inferred

    plot_results(
        agg,
        args.out,
        show=args.show,
        splits=splits,
        labels_csv_by_split=labels_csv_by_split,
        label_col=str(args.label_col),
        labels_required=bool(labels_required),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        ci=float(args.ci),
    )


if __name__ == "__main__":
    main()
