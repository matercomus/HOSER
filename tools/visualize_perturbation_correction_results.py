#!/usr/bin/env python3
"""Visualize Phase B perturbation correction results.

Reads per-model outputs written by the `perturbation_correction` pipeline phase:
- `eval_dir/perturbation_correction/{model}/summary.json`
- `eval_dir/perturbation_correction/{model}/rows.jsonl`

Generates plots using model-aware colors and display names from
`tools/model_detection.py`.

Example:
  uv run python tools/visualize_perturbation_correction_results.py \
    --eval-dir hoser-distill-beijing \
    --output-dir hoser-distill-beijing/figures/perturbation_correction
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pandas is required for this plotting script. "
        "Install with `uv add pandas` if missing."
    ) from exc

try:
    import seaborn as sns
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "seaborn is required for this plotting script. "
        "Install with `uv add seaborn` if missing."
    ) from exc

from model_detection import get_display_name, get_model_color

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSummary:
    model: str
    display: str
    color: str
    rsr: float
    total: int
    valid: int
    corrected: int
    mean_to_clean_km: Optional[float]
    mean_to_dirty_km: Optional[float]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return data


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def load_phase_b_results(eval_dir: Path) -> tuple[List[ModelSummary], pd.DataFrame]:
    """Load summaries and per-row results for all models in an eval dir."""
    root = eval_dir / "perturbation_correction"
    if not root.exists():
        raise FileNotFoundError(f"No Phase B outputs found at: {root}")

    summaries: List[ModelSummary] = []
    rows: List[Dict[str, Any]] = []

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        summary_path = model_dir / "summary.json"
        rows_path = model_dir / "rows.jsonl"

        if not summary_path.exists():
            logger.warning("Missing summary.json for %s", model)
            continue

        summary = _load_json(summary_path)
        counts = summary.get("counts", {})
        dtw = summary.get("dtw_km", {})

        ms = ModelSummary(
            model=model,
            display=get_display_name(model),
            color=get_model_color(model),
            rsr=float(summary.get("rsr", 0.0)),
            total=int(counts.get("total", 0)),
            valid=int(counts.get("valid", 0)),
            corrected=int(counts.get("corrected", 0)),
            mean_to_clean_km=(
                float(dtw["mean_to_clean"])
                if dtw.get("mean_to_clean") is not None
                else None
            ),
            mean_to_dirty_km=(
                float(dtw["mean_to_dirty"])
                if dtw.get("mean_to_dirty") is not None
                else None
            ),
        )
        summaries.append(ms)

        if rows_path.exists():
            for r in _iter_jsonl(rows_path):
                r = dict(r)
                r["model"] = model
                r["model_display"] = ms.display
                rows.append(r)
        else:
            logger.warning("Missing rows.jsonl for %s", model)

    if not summaries:
        raise ValueError(f"No model summaries found under: {root}")

    df = pd.DataFrame(rows)
    if not df.empty:
        # Derived metrics for plotting
        df["dtw_delta_km"] = df["dtw_to_dirty_km"].astype(float) - df[
            "dtw_to_clean_km"
        ].astype(float)

    return summaries, df


def plot_rsr_by_model(
    summaries: List[ModelSummary], output_dir: Path, title_prefix: str
) -> None:
    data = pd.DataFrame(
        {
            "model": [s.model for s in summaries],
            "display": [s.display for s in summaries],
            "rsr": [s.rsr for s in summaries],
            "color": [s.color for s in summaries],
            "valid": [s.valid for s in summaries],
        }
    ).sort_values("rsr", ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.2), 5))
    bars = ax.bar(data["display"], data["rsr"], color=data["color"], alpha=0.9)

    for bar, rsr, valid in zip(bars, data["rsr"], data["valid"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rsr:.2f}\n(n={valid})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("RSR (corrected rate)")
    ax.set_title(f"{title_prefix}: Correction Rate (RSR)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "rsr_by_model.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_dtw_gap_by_model(
    summaries: List[ModelSummary], output_dir: Path, title_prefix: str
) -> None:
    records = []
    for s in summaries:
        if s.mean_to_clean_km is None or s.mean_to_dirty_km is None:
            continue
        records.append(
            {
                "model": s.model,
                "display": s.display,
                "gap_km": float(s.mean_to_dirty_km - s.mean_to_clean_km),
                "color": s.color,
            }
        )

    if not records:
        logger.warning("No DTW means found in summaries; skipping dtw_gap plot")
        return

    data = pd.DataFrame(records).sort_values("gap_km", ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.2), 5))
    bars = ax.bar(data["display"], data["gap_km"], color=data["color"], alpha=0.9)

    for bar, gap in zip(bars, data["gap_km"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.02 * max(1.0, float(data["gap_km"].max()))),
            f"{gap:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Mean DTW gap (dirty - clean) [km]")
    ax.set_title(f"{title_prefix}: Mean DTW Gap (Dirty − Clean)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "dtw_gap_by_model.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_dtw_delta_distribution(
    df: pd.DataFrame, summaries: List[ModelSummary], output_dir: Path, title_prefix: str
) -> None:
    if df.empty:
        logger.warning("No per-row data found; skipping dtw_delta boxplot")
        return

    palette = {s.display: s.color for s in summaries}
    df = df.copy()
    df["model_display"] = df["model"].map({s.model: s.display for s in summaries})

    fig, ax = plt.subplots(figsize=(max(10, df["model_display"].nunique() * 1.4), 6))

    sns.boxplot(
        data=df,
        x="model_display",
        y="dtw_delta_km",
        hue="model_display",
        dodge=False,
        palette=palette,
        ax=ax,
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_ylabel("DTW delta (dirty - clean) [km]")
    ax.set_xlabel("Model")
    ax.set_title(f"{title_prefix}: Per-sample DTW Delta Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "dtw_delta_boxplot.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_rsr_by_model_and_strength(
    df: pd.DataFrame,
    summaries: List[ModelSummary],
    output_dir: Path,
    title_prefix: str,
) -> None:
    """Plot correction rate split by perturbation strength.

    Uses `ab_strength == 'strong'` vs everything else (including None).
    """

    if df.empty:
        logger.warning("No per-row data found; skipping rsr_by_strength plot")
        return

    if "ab_strength" not in df.columns or "corrected" not in df.columns:
        logger.warning(
            "Missing ab_strength/corrected columns; skipping rsr_by_strength plot"
        )
        return

    df = df.copy()
    df = df[df["corrected"].notna()].copy()
    if df.empty:
        logger.warning("No valid corrected labels; skipping rsr_by_strength plot")
        return

    df["strength_bucket"] = np.where(
        df["ab_strength"].astype(str) == "strong", "strong", "other"
    )
    display_by_model = {s.model: s.display for s in summaries}
    color_by_model = {s.model: s.color for s in summaries}
    df["model_display"] = df["model"].map(display_by_model)

    grouped = (
        df.groupby(["model", "model_display", "strength_bucket"], dropna=False)
        .agg(
            rsr=("corrected", "mean"),
            n=("corrected", "size"),
        )
        .reset_index()
    )

    # Ensure stable order by display name
    model_order = [s.model for s in sorted(summaries, key=lambda s: s.display)]
    display_order = [display_by_model[m] for m in model_order]

    # Build two aligned bars per model
    strength_order = ["strong", "other"]
    x = np.arange(len(model_order), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, len(model_order) * 1.5), 6))

    for idx, strength in enumerate(strength_order):
        offset = (-width / 2) if idx == 0 else (width / 2)
        heights = []
        ns = []
        colors = []

        for m in model_order:
            row = grouped[(grouped["model"] == m) & (grouped["strength_bucket"] == strength)]
            if row.empty:
                heights.append(np.nan)
                ns.append(0)
            else:
                heights.append(float(row["rsr"].iloc[0]))
                ns.append(int(row["n"].iloc[0]))
            colors.append(color_by_model.get(m, "#999999"))

        alpha = 0.9 if strength == "strong" else 0.45
        bars = ax.bar(
            x + offset,
            heights,
            width=width,
            color=colors,
            alpha=alpha,
            label=strength,
            edgecolor="black",
            linewidth=0.3,
        )

        for bar, h, n in zip(bars, heights, ns):
            if not np.isfinite(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(display_order, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("RSR (corrected rate)")
    ax.set_title(f"{title_prefix}: RSR by Perturbation Strength")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="ab_strength bucket")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "rsr_by_model_and_strength.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_dtw_clean_vs_dirty_scatter(
    df: pd.DataFrame,
    summaries: List[ModelSummary],
    output_dir: Path,
    title_prefix: str,
) -> None:
    """Scatter plot of DTW to clean vs DTW to dirty with y=x reference."""

    if df.empty:
        logger.warning("No per-row data found; skipping dtw_clean_vs_dirty plot")
        return

    required = {"dtw_to_clean_km", "dtw_to_dirty_km", "model"}
    if not required.issubset(df.columns):
        logger.warning("Missing DTW/model columns; skipping dtw_clean_vs_dirty plot")
        return

    palette = {s.display: s.color for s in summaries}
    display_by_model = {s.model: s.display for s in summaries}

    df = df.copy()
    df["model_display"] = df["model"].map(display_by_model)

    x = df["dtw_to_clean_km"].astype(float)
    y = df["dtw_to_dirty_km"].astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    df = df.loc[finite].copy()
    if df.empty:
        logger.warning("No finite DTW values; skipping dtw_clean_vs_dirty plot")
        return

    # Clip to a high percentile for readability; still faithful for comparisons.
    xy_max = float(
        np.nanpercentile(
            np.maximum(
                df["dtw_to_clean_km"].astype(float).to_numpy(),
                df["dtw_to_dirty_km"].astype(float).to_numpy(),
            ),
            99.5,
        )
    )
    xy_max = max(1.0, xy_max)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    sns.scatterplot(
        data=df,
        x="dtw_to_clean_km",
        y="dtw_to_dirty_km",
        hue="model_display",
        palette=palette,
        alpha=0.25,
        s=12,
        linewidth=0,
        ax=ax,
    )

    ax.plot([0, xy_max], [0, xy_max], color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)
    ax.set_xlabel("DTW to clean [km]")
    ax.set_ylabel("DTW to dirty [km]")
    ax.set_title(
        f"{title_prefix}: DTW to Clean vs Dirty (clipped to p99.5)"
    )
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            title="Model",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
        )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "dtw_clean_vs_dirty_scatter.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_dtw_delta_cdf(
    df: pd.DataFrame,
    summaries: List[ModelSummary],
    output_dir: Path,
    title_prefix: str,
) -> None:
    """Plot ECDFs of DTW delta (dirty - clean) per model."""

    if df.empty:
        logger.warning("No per-row data found; skipping dtw_delta_cdf plot")
        return

    if "dtw_delta_km" not in df.columns:
        logger.warning("Missing dtw_delta_km; skipping dtw_delta_cdf plot")
        return

    palette = {s.display: s.color for s in summaries}
    display_by_model = {s.model: s.display for s in summaries}

    df = df.copy()
    df["model_display"] = df["model"].map(display_by_model)
    df["dtw_delta_km"] = df["dtw_delta_km"].astype(float)
    df = df[np.isfinite(df["dtw_delta_km"])].copy()
    if df.empty:
        logger.warning("No finite dtw_delta_km; skipping dtw_delta_cdf plot")
        return

    # Stable order by display name
    order = [s.display for s in sorted(summaries, key=lambda s: s.display)]

    fig, ax = plt.subplots(figsize=(max(9, len(order) * 0.65), 6))
    sns.ecdfplot(
        data=df,
        x="dtw_delta_km",
        hue="model_display",
        hue_order=order,
        palette=palette,
        ax=ax,
    )

    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("DTW delta (dirty - clean) [km]")
    ax.set_ylabel("ECDF")
    ax.set_title(f"{title_prefix}: ECDF of DTW Delta (Dirty − Clean)")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            title="Model",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
        )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "dtw_delta_cdf.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Phase B perturbation correction results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory containing perturbation_correction outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots (default: <eval-dir>/figures/perturbation_correction)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix (default: eval dir name)",
    )

    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (eval_dir / "figures" / "perturbation_correction")
    )
    title_prefix = args.title or eval_dir.name

    logger.info("Loading Phase B results from %s", eval_dir)
    summaries, df = load_phase_b_results(eval_dir)

    # Prefer stable ordering by model display name
    summaries = sorted(summaries, key=lambda s: s.display)

    plot_rsr_by_model(summaries, output_dir, title_prefix)
    plot_rsr_by_model_and_strength(df, summaries, output_dir, title_prefix)
    plot_dtw_gap_by_model(summaries, output_dir, title_prefix)
    plot_dtw_delta_distribution(df, summaries, output_dir, title_prefix)
    plot_dtw_clean_vs_dirty_scatter(df, summaries, output_dir, title_prefix)
    plot_dtw_delta_cdf(df, summaries, output_dir, title_prefix)

    logger.info("Done. Wrote plots to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
