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

Optional decision-benchmark artifacts:
- When you pass --metrics-json (from tools/run_lmtad_decision_benchmark.py), you can
    emit compact, boundary-dependent outputs (confusion matrices + per-q tables)
    into a single `decision/` directory.

Optional organized output:
- When you pass --organized-dirs, outputs are written into subdirectories under
    the plots directory (hist/, density/, roc/, pr/, summary/, abnormality/).
"""

from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


_NULL_LIKE = {"", "nan", "none", "null", "normal"}


def _is_abnormal_label(raw: str, *, normal_value: str = "normal") -> bool:
    """Return True if a CSV label indicates an abnormal trajectory.

    This is intentionally tolerant to common "null-like" values that show up
    across datasets and export pipelines.

    Rules:
    - treat {"", "nan", "none", "null", "normal"} (case-insensitive) as normal
    - also treat `normal_value` (case-insensitive) as normal
    - everything else is abnormal
    """

    s = str(raw or "").strip().lower()
    if s in _NULL_LIKE:
        return False
    if s == str(normal_value or "").strip().lower():
        return False
    return True


def _parse_q_key(q_key: str) -> float:
    """Parse q keys like 'q=0.95' into float(0.95)."""
    q_key = str(q_key).strip()
    if q_key.startswith("q="):
        q_key = q_key[2:]
    return float(q_key)


def _load_metrics_json(path: Path) -> dict:
    """Load metrics.json written by tools/run_lmtad_decision_benchmark.py."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _extract_method_rows(metrics: dict, method_key: str) -> list[dict]:
    """Extract per-q rows for a given method from metrics.json."""
    bucket = metrics.get(method_key)
    if not isinstance(bucket, dict):
        return []

    rows: list[dict] = []
    for key, entry in bucket.items():
        if not isinstance(entry, dict):
            continue
        q = _safe_float(entry.get("q") if method_key == "baseline_quantile" else entry.get("q_matched"))
        if q is None:
            try:
                q = _parse_q_key(str(key))
            except Exception:
                continue

        row = {
            "q": float(q),
            "method": str(entry.get("method") or method_key),
            "tp": _safe_int(entry.get("tp")),
            "fp": _safe_int(entry.get("fp")),
            "tn": _safe_int(entry.get("tn")),
            "fn": _safe_int(entry.get("fn")),
            "precision": _safe_float(entry.get("precision")),
            "recall": _safe_float(entry.get("recall")),
            "f1": _safe_float(entry.get("f1")),
            "fpr": _safe_float(entry.get("fpr")),
            "flag_rate": _safe_float(entry.get("flag_rate")),
        }

        # Backward compat: compute f1 if missing but precision/recall exist.
        if row.get("f1") is None:
            p = row.get("precision")
            r = row.get("recall")
            if p is not None and r is not None and (p + r) > 0:
                row["f1"] = float((2.0 * p * r) / (p + r))
        if method_key == "baseline_quantile":
            row["threshold"] = _safe_float(entry.get("threshold"))
        else:
            row["cutoff"] = _safe_float(entry.get("cutoff"))
            row["k"] = _safe_int(entry.get("k"))
        rows.append(row)

    rows.sort(key=lambda r: float(r["q"]))
    return rows


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable field order
    fieldnames: list[str] = []
    for k in rows[0].keys():
        fieldnames.append(k)
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_method_summary_md(path: Path, method_key: str, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Decision summary: {method_key}\n")
    if not rows:
        lines.append("No per-q rows found in metrics.json.\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    # Basic notes
    lines.append("This is computed from `metrics.json` (no re-scoring).\n")
    lines.append("## Per-q table\n")
    header = ["q", "precision", "recall", "f1", "fpr", "flag_rate", "tp", "fp", "tn", "fn"]
    if method_key == "baseline_quantile":
        header.insert(1, "threshold")
    else:
        header.insert(1, "cutoff")
        header.insert(2, "k")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for r in rows:
        row_vals = [fmt(r.get(k)) for k in header]
        lines.append("| " + " | ".join(row_vals) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_confusion_grid(out_path: Path, *, title: str, rows: list[dict], method_key: str):
    if not rows:
        return
    n = len(rows)
    cols = 4
    rws = int(np.ceil(n / float(cols)))
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 3.3, rws * 3.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rws, cols)

    vmax = 0
    mats = []
    for r in rows:
        tn = int(r.get("tn") or 0)
        fp = int(r.get("fp") or 0)
        fn = int(r.get("fn") or 0)
        tp = int(r.get("tp") or 0)
        mat = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        mats.append(mat)
        vmax = max(vmax, int(mat.max()))

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        r = rows[i]
        mat = mats[i]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, vmax))
        total = float(mat.sum()) if mat.sum() else 1.0
        for (rr, cc), v in np.ndenumerate(mat):
            pct = 100.0 * float(v) / total
            ax.text(cc, rr, f"{int(v)}\n{pct:.1f}%", ha="center", va="center", fontsize=8)

        q = float(r["q"])
        meta = ""
        if method_key == "baseline_quantile" and r.get("threshold") is not None:
            meta = f"thr={float(r['threshold']):.3f}"
        if method_key != "baseline_quantile" and r.get("cutoff") is not None:
            meta = f"cut={float(r['cutoff']):.3f}"
        pr = r.get("precision")
        rc = r.get("recall")
        f1 = r.get("f1")
        fpr = r.get("fpr")
        stats = []
        if pr is not None:
            stats.append(f"P={float(pr):.2f}")
        if rc is not None:
            stats.append(f"R={float(rc):.2f}")
        if f1 is not None:
            stats.append(f"F1={float(f1):.2f}")
        if fpr is not None:
            stats.append(f"FPR={float(fpr):.2f}")
        ax.set_title(f"q={q:.2f} {meta}\n" + " ".join(stats), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred N", "Pred P"], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(["True N", "True P"], fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    cbar_ax = fig.add_axes([0.985, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_metrics_vs_q(out_path: Path, *, title: str, rows: list[dict], method_key: str):
    if not rows:
        return
    q = np.array([float(r["q"]) for r in rows], dtype=np.float64)
    def _to_float_or_nan(v) -> float:
        x = _safe_float(v)
        return float(x) if x is not None else float("nan")

    precision = np.array([_to_float_or_nan(r.get("precision")) for r in rows], dtype=np.float64)
    recall = np.array([_to_float_or_nan(r.get("recall")) for r in rows], dtype=np.float64)
    f1 = np.array([_to_float_or_nan(r.get("f1")) for r in rows], dtype=np.float64)
    fpr = np.array([_to_float_or_nan(r.get("fpr")) for r in rows], dtype=np.float64)
    flag = np.array([_to_float_or_nan(r.get("flag_rate")) for r in rows], dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    ax1.plot(q, precision, marker="o", label="precision")
    ax1.plot(q, recall, marker="o", label="recall")
    ax1.plot(q, f1, marker="o", label="f1")
    ax1.set_ylabel("Score")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(q, fpr, marker="o", label="fpr")
    ax2.plot(q, flag, marker="o", label="flag_rate")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("q")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _get_out_dirs(out_dir: Path, organized: bool) -> dict[str, Path]:
    """Return output directories for each plot category."""
    if not organized:
        return {
            "root": out_dir,
            "hist": out_dir,
            "density": out_dir,
            "roc": out_dir,
            "pr": out_dir,
            "summary": out_dir,
            "abnormality": out_dir,
        }

    dirs = {
        "root": out_dir,
        "hist": out_dir / "hist",
        "density": out_dir / "density",
        "roc": out_dir / "roc",
        "pr": out_dir / "pr",
        "summary": out_dir / "summary",
        "abnormality": out_dir / "abnormality",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _get_decision_dirs(out_dir: Path) -> dict[str, Path]:
    base = out_dir / "decision"
    dirs = {
        "root": base,
        "tables": base / "tables",
        "summary": base / "summary",
        "confusion": base / "confusion",
        "curves": base / "curves",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


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
    normal_value: str = "normal",
) -> list[bool]:
    """Return abnormality labels from a sampled CSV.

    Label rule:
    - abnormal if the label is not a normal/null-like value.
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
            labels.append(_is_abnormal_label(raw, normal_value=str(normal_value)))
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
        (recall, precision, average_precision)

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
    # This matches the common ML definition of AP (e.g., sklearn's
    # average_precision_score) and may differ from trapezoidal PR-AUC.
    ap = float(precision[y_sorted].sum() / float(n_pos))

    # Add a (0,1) starting point for cleaner plots.
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))

    return recall, precision, ap


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
    metrics: dict | None = None,
    organized_dirs: bool = False,
    emit_decision_artifacts: bool = False,
    threshold_lines: str = "auto",
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

    out_dirs = _get_out_dirs(out_dir, organized=organized_dirs)
    if threshold_lines not in {"auto", "none"}:
        raise ValueError("threshold_lines must be 'auto' or 'none'")

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
            if threshold_lines == "auto":
                base_meta = (
                    agg.get(s, {}).get("baseline_calibrated")
                    if isinstance(agg, dict)
                    else None
                )
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
                        color="gray",
                        linestyle=":",
                        label=f"95th pct (within split; descriptive)={thr:.3f}",
                    )
            ax.set_title(f"{s} (N={len(vals)})")
            ax.set_xlabel("Log perplexity")
            ax.legend(fontsize=8)

        out_file = out_dirs["hist"] / f"{base}_{s}.png"
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
                thr_label = None
                if threshold_lines == "auto":
                    base_meta = (
                        agg.get(s, {}).get("baseline_calibrated")
                        if isinstance(agg, dict)
                        else None
                    )
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
                        thr_label = f"95th pct (within split; descriptive)={thr:.3f}"
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

                if thr_label is not None:
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
                ax_den.legend(fontsize=8)

            den_file = out_dirs["density"] / f"{base}_{s}_density.png"
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
            roc_file = out_dirs["roc"] / f"{base}_{s}_roc.png"
            fig_roc.tight_layout()
            fig_roc.savefig(roc_file, dpi=150)
            saved_files.append(roc_file)
            plt.close(fig_roc)

            recall, precision, ap = _precision_recall_curve_points(scores, labels)
            fig_pr, ax_pr = plt.subplots(figsize=(5.5, 5.5))
            ax_pr.plot(recall, precision, label=f"AP={ap:.4f}")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            pad = 0.02
            ax_pr.set_xlim(-pad, 1.0 + pad)
            ax_pr.set_ylim(-pad, 1.0 + pad)
            ax_pr.set_title(f"{s}: Precision-Recall curve (AP)")
            ax_pr.legend(loc="lower left")
            pr_file = out_dirs["pr"] / f"{base}_{s}_pr.png"
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
    boxplot_file = out_dirs["summary"] / f"{base}_boxplot.png"
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

    # Predicted outlier-rate bar chart (uses outlier_rate from results)
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

    # Error bars: bootstrap percentile CI when enabled.
    # If bootstrap is requested but per-example outlier labels are missing,
    # fall back to a normal-approx 95% CI using only outlier_rate and N.
    if bootstrap and int(bootstrap) > 0:
        need = [s for s in splits if not np.isnan(abnormal_rates[splits.index(s)])]
        missing = [s for s in need if s not in outlier_labels_by_split]
        if missing:
            print(
                "[plot_lmtad_results] --bootstrap requested but outlier_labels are missing for "
                f"{missing}; falling back to normal-approx CI."
            )
            bootstrap = 0

    # Error bars: bootstrap percentile CI when enabled, else normal-approx 95% CI.
    if bootstrap and int(bootstrap) > 0:
        yerr = np.zeros((2, len(splits)), dtype=np.float64)
        ci_low: list[float] = []
        ci_high: list[float] = []
        for i, (s, pct, n) in enumerate(zip(splits, abnormal_rates, counts)):
            if np.isnan(pct) or n <= 0:
                ci_low.append(0.0)
                ci_high.append(0.0)
                continue
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
        # Normal-approx 95% CI for proportion p: +/- 1.96 * sqrt(p*(1-p)/n)
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
            yerr[len(ci_low) - 1] = 1.96 * float(se)

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
    ax3.set_ylabel("Predicted outlier rate (%)")
    ax3.set_xlabel("Split")
    ax3.set_title("Predicted outlier rate per split", pad=14)

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

    abnormal_file = out_dirs["abnormality"] / f"{base}_abnormality.png"
    fig3.tight_layout()
    fig3.savefig(abnormal_file, dpi=150)
    saved_files.append(abnormal_file)
    plt.close(fig3)

    # Also write a small CSV summary with counts and abnormality rates
    summary_file = out_dirs["abnormality"] / f"{base}_abnormality.csv"
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

    # Decision artifacts (boundary-dependent) from metrics.json.
    if emit_decision_artifacts:
        if metrics is None:
            raise ValueError("emit_decision_artifacts requires --metrics-json")
        decision_dirs = _get_decision_dirs(out_dirs["root"])

        for method_key in ["baseline_quantile", "topk_matched"]:
            rows = _extract_method_rows(metrics, method_key)
            if not rows:
                continue
            _write_csv(decision_dirs["tables"] / f"{method_key}.csv", rows)
            _write_method_summary_md(
                decision_dirs["summary"] / f"{method_key}.md", method_key, rows
            )
            _plot_confusion_grid(
                decision_dirs["confusion"] / f"{method_key}_grid.png",
                title=f"Confusion matrices across q ({method_key})",
                rows=rows,
                method_key=method_key,
            )
            _plot_metrics_vs_q(
                decision_dirs["curves"] / f"{method_key}_metrics_vs_q.png",
                title=f"Decision metrics vs q ({method_key})",
                rows=rows,
                method_key=method_key,
            )


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
        "--metrics-json",
        type=Path,
        default=None,
        help=(
            "Optional: metrics.json from tools/run_lmtad_decision_benchmark.py. "
            "When provided, overlays all q-grid thresholds (baseline_quantile) and top-k cutoffs."
        ),
    )
    parser.add_argument(
        "--organized-dirs",
        action="store_true",
        help=(
            "Write plots into subdirectories under the output directory: "
            "hist/, density/, roc/, pr/, summary/, abnormality/."
        ),
    )
    parser.add_argument(
        "--emit-decision-artifacts",
        action="store_true",
        help=(
            "Write boundary-dependent artifacts from --metrics-json into decision/: "
            "per-q tables (CSV/MD), confusion-matrix grids, and per-method summary plots."
        ),
    )
    parser.add_argument(
        "--threshold-lines",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help=(
            "Whether to draw a threshold line on hist/density plots. "
            "'auto' keeps the existing behavior; 'none' draws no vertical lines."
        ),
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

    metrics = None
    if args.metrics_json is not None:
        metrics = _load_metrics_json(Path(args.metrics_json))

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
        metrics=metrics,
        organized_dirs=bool(args.organized_dirs),
        emit_decision_artifacts=bool(args.emit_decision_artifacts),
        threshold_lines=str(args.threshold_lines),
    )


if __name__ == "__main__":
    main()
