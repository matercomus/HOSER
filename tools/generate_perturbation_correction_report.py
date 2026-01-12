#!/usr/bin/env python3
"""Generate a reproducible Markdown report for Phase B (perturbation correction).

This script reads per-model outputs written by the `perturbation_correction`
pipeline phase:

- <eval_dir>/perturbation_correction/<model>/summary.json
- <eval_dir>/perturbation_correction/<model>/rows.jsonl

It aggregates metrics across models, computes per-sample DTW delta statistics,
optionally summarizes LM-TAD teacher triangulation if present, and writes a
Markdown report designed for faithful reproduction.

It can also embed plots if they exist at:

- <eval_dir>/figures/perturbation_correction/*.png

Example:
  uv run python tools/generate_perturbation_correction_report.py \
    --eval-dir hoser-perturbed-beijing-pert-type-detour-eval

Notes:
- This script does not run Phase B. It only reads already-generated outputs.
- For plots, run:
    uv run python tools/visualize_perturbation_correction_results.py --eval-dir <eval_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from model_detection import build_model_metadata, get_display_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPaths:
    model: str
    summary_path: Path
    rows_path: Path


@dataclass(frozen=True)
class SummaryMetrics:
    model: str
    display_name: str
    model_group: str
    group_display_name: str
    seed: Optional[int]
    rsr: float
    corrected: int
    valid: int
    invalid: int
    mean_to_clean_km: Optional[float]
    mean_to_dirty_km: Optional[float]


@dataclass(frozen=True)
class RowDeltaStats:
    n: int
    neg_frac: Optional[float]
    tie_frac: Optional[float]
    median: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    ge10_frac: Optional[float]
    ge20_frac: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]


def _pct(sorted_vals: Sequence[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = int(math.floor((len(sorted_vals) - 1) * p))
    return float(sorted_vals[idx])


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _model_group_and_seed(model_name: str) -> Tuple[str, Optional[int]]:
    """Derive (base_model, seed_number) using tools/model_detection.py."""

    meta = build_model_metadata(model_name)
    return meta.base_model, meta.seed_number


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return data


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


def _find_model_paths(eval_dir: Path) -> List[ModelPaths]:
    root = eval_dir / "perturbation_correction"
    if not root.exists():
        raise FileNotFoundError(f"No Phase B outputs found at: {root}")

    out: List[ModelPaths] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary_path = model_dir / "summary.json"
        rows_path = model_dir / "rows.jsonl"
        if summary_path.exists() and rows_path.exists():
            out.append(
                ModelPaths(
                    model=model_dir.name,
                    summary_path=summary_path,
                    rows_path=rows_path,
                )
            )
        else:
            logger.warning(
                "Skipping %s (missing summary.json or rows.jsonl)", model_dir.name
            )

    if not out:
        raise ValueError(f"No complete model outputs found under: {root}")
    return out


def _load_summary_metrics(paths: ModelPaths) -> Tuple[SummaryMetrics, Optional[Dict[str, Any]]]:
    s = _load_json(paths.summary_path)
    counts = s.get("counts", {})
    dtw = s.get("dtw_km", {})

    mean_to_clean = dtw.get("mean_to_clean")
    mean_to_dirty = dtw.get("mean_to_dirty")

    teacher = s.get("teacher")
    if teacher is not None and not isinstance(teacher, dict):
        teacher = None

    model_group, seed = _model_group_and_seed(paths.model)
    group_display_name = get_display_name(model_group)

    return (
        SummaryMetrics(
            model=paths.model,
            display_name=get_display_name(paths.model),
            model_group=model_group,
            group_display_name=group_display_name,
            seed=seed,
            rsr=float(s.get("rsr", 0.0)),
            corrected=int(counts.get("corrected", 0)),
            valid=int(counts.get("valid", 0)),
            invalid=int(counts.get("invalid", 0)),
            mean_to_clean_km=float(mean_to_clean) if mean_to_clean is not None else None,
            mean_to_dirty_km=float(mean_to_dirty) if mean_to_dirty is not None else None,
        ),
        teacher,
    )


def _compute_row_delta_stats(rows_path: Path) -> RowDeltaStats:
    deltas: List[float] = []
    neg = 0
    ties = 0
    ge10 = 0
    ge20 = 0

    for r in _iter_jsonl(rows_path):
        dc = float(r["dtw_to_clean_km"])
        dd = float(r["dtw_to_dirty_km"])
        d = float(dd - dc)
        deltas.append(d)
        if d < 0:
            neg += 1
        if abs(d) < 1e-9:
            ties += 1
        if d >= 10:
            ge10 += 1
        if d >= 20:
            ge20 += 1

    deltas.sort()
    n = len(deltas)
    if n == 0:
        return RowDeltaStats(
            n=0,
            neg_frac=None,
            tie_frac=None,
            median=None,
            p90=None,
            p95=None,
            p99=None,
            ge10_frac=None,
            ge20_frac=None,
            min_val=None,
            max_val=None,
        )

    return RowDeltaStats(
        n=n,
        neg_frac=neg / n,
        tie_frac=ties / n,
        median=float(statistics.median(deltas)),
        p90=_pct(deltas, 0.90),
        p95=_pct(deltas, 0.95),
        p99=_pct(deltas, 0.99),
        ge10_frac=ge10 / n,
        ge20_frac=ge20 / n,
        min_val=float(deltas[0]),
        max_val=float(deltas[-1]),
    )


def _safe_read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort YAML loader (optional dependency).

    The project typically has PyYAML installed; if not, we still produce a report.
    """

    try:
        import yaml  # type: ignore
    except ImportError:
        return None

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return None
    return data


def _relpath(from_path: Path, to_path: Path) -> str:
    """Compute a POSIX relative path for Markdown links.

    Use `os.path.relpath` to handle symlinks/mounts robustly.
    """

    rel = os.path.relpath(str(to_path), start=str(from_path))
    return Path(rel).as_posix()


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _render_report(
    *,
    eval_dir: Path,
    output_md: Path,
    title: Optional[str],
    embed_plots: bool,
) -> str:
    model_paths = _find_model_paths(eval_dir)

    summaries: List[SummaryMetrics] = []
    row_stats: Dict[str, RowDeltaStats] = {}
    teacher_by_model: Dict[str, Dict[str, Any]] = {}

    for mp in model_paths:
        s, teacher = _load_summary_metrics(mp)
        summaries.append(s)
        row_stats[mp.model] = _compute_row_delta_stats(mp.rows_path)
        if teacher is not None:
            teacher_by_model[mp.model] = teacher

    # stable order: group, seed, model
    summaries.sort(key=lambda s: (s.model_group, s.seed or 0, s.model))

    # Optional config snippet
    config_path = eval_dir / "config" / "evaluation.yaml"
    config = _safe_read_yaml(config_path) if config_path.exists() else None

    # Embedded plots
    figures_dir = eval_dir / "figures" / "perturbation_correction"
    plot_paths: List[Tuple[str, Path]] = [
        ("rsr", figures_dir / "rsr_by_model.png"),
        ("rsr_strength", figures_dir / "rsr_by_model_and_strength.png"),
        ("gap", figures_dir / "dtw_gap_by_model.png"),
        ("box", figures_dir / "dtw_delta_boxplot.png"),
        ("scatter", figures_dir / "dtw_clean_vs_dirty_scatter.png"),
        ("cdf", figures_dir / "dtw_delta_cdf.png"),
    ]

    # Build tables
    per_model_rows: List[List[str]] = []
    for s in summaries:
        rs = row_stats.get(s.model)
        gap = (
            (s.mean_to_dirty_km - s.mean_to_clean_km)
            if (s.mean_to_dirty_km is not None and s.mean_to_clean_km is not None)
            else None
        )
        per_model_rows.append(
            [
                s.model,
                s.display_name,
                s.model_group,
                s.group_display_name,
                str(s.seed) if s.seed is not None else "",
                _fmt(s.rsr, 3),
                str(s.corrected),
                str(s.valid),
                str(s.invalid),
                _fmt(s.mean_to_clean_km, 3),
                _fmt(s.mean_to_dirty_km, 3),
                _fmt(gap, 3),
                _fmt(rs.neg_frac if rs else None, 4),
                _fmt(rs.tie_frac if rs else None, 4),
                _fmt(rs.median if rs else None, 3),
                _fmt(rs.p90 if rs else None, 3),
                _fmt(rs.p95 if rs else None, 3),
                _fmt(rs.p99 if rs else None, 3),
                _fmt(rs.max_val if rs else None, 3),
            ]
        )

    group_to_items: Dict[str, List[SummaryMetrics]] = {}
    for s in summaries:
        group_to_items.setdefault(s.model_group, []).append(s)

    def avg(vals: List[Optional[float]]) -> Optional[float]:
        xs = [v for v in vals if v is not None]
        return sum(xs) / len(xs) if xs else None

    def std(vals: List[Optional[float]]) -> Optional[float]:
        xs = [v for v in vals if v is not None]
        if len(xs) < 2:
            return None
        return float(statistics.pstdev(xs))

    group_rows: List[List[str]] = []
    for g, items in sorted(group_to_items.items()):
        group_display = items[0].group_display_name if items else get_display_name(g)
        rsr_vals = [i.rsr for i in items]
        gaps = [
            (i.mean_to_dirty_km - i.mean_to_clean_km)
            if (i.mean_to_dirty_km is not None and i.mean_to_clean_km is not None)
            else None
            for i in items
        ]
        negs = [row_stats[i.model].neg_frac for i in items if i.model in row_stats]
        group_rows.append(
            [
                g,
                group_display,
                str(len(items)),
                _fmt(avg(rsr_vals), 4),
                _fmt(std(rsr_vals), 4),
                _fmt(avg(gaps), 4),
                _fmt(std(gaps), 4),
                _fmt(avg(negs), 4),
            ]
        )

    # Strength split tables
    # Per-model
    strength_per_model: List[List[str]] = []
    strength_by_group: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for mp in model_paths:
        strong_n = strong_c = 0
        other_n = other_c = 0
        for r in _iter_jsonl(mp.rows_path):
            bucket = "strong" if str(r.get("ab_strength")) == "strong" else "other"
            if bucket == "strong":
                strong_n += 1
                if r.get("corrected") is True:
                    strong_c += 1
            else:
                other_n += 1
                if r.get("corrected") is True:
                    other_c += 1

            g, _seed = _model_group_and_seed(mp.model)
            key = (g, bucket)
            a = strength_by_group.setdefault(key, {"n": 0, "corrected": 0, "deltas": []})
            a["n"] += 1
            if r.get("corrected") is True:
                a["corrected"] += 1
            dc = float(r["dtw_to_clean_km"])
            dd = float(r["dtw_to_dirty_km"])
            a["deltas"].append(float(dd - dc))

        strength_per_model.append(
            [
                mp.model,
                get_display_name(mp.model),
                _model_group_and_seed(mp.model)[0],
                get_display_name(_model_group_and_seed(mp.model)[0]),
                str(_model_group_and_seed(mp.model)[1] or ""),
                str(strong_n),
                _fmt((strong_c / strong_n) if strong_n else None, 4),
                str(other_n),
                _fmt((other_c / other_n) if other_n else None, 4),
            ]
        )

    strength_per_model.sort(key=lambda r: (r[2], int(r[4]) if r[4] else 0, r[0]))

    # Group-level
    strength_group_rows: List[List[str]] = []
    for (g, bucket), a in sorted(strength_by_group.items()):
        group_display = get_display_name(g)
        deltas = a["deltas"]
        rsr = a["corrected"] / a["n"] if a["n"] else None
        mean_d = sum(deltas) / len(deltas) if deltas else None
        median_d = float(statistics.median(deltas)) if deltas else None
        neg_frac = (sum(1 for d in deltas if d < 0) / len(deltas)) if deltas else None
        strength_group_rows.append(
            [
                g,
                group_display,
                bucket,
                str(a["n"]),
                _fmt(rsr, 4),
                _fmt(mean_d, 4),
                _fmt(median_d, 4),
                _fmt(neg_frac, 4),
            ]
        )

    # Outliers
    all_rows: List[Dict[str, Any]] = []
    for mp in model_paths:
        for r in _iter_jsonl(mp.rows_path):
            dc = float(r["dtw_to_clean_km"])
            dd = float(r["dtw_to_dirty_km"])
            r = dict(r)
            r["dtw_delta_km"] = float(dd - dc)
            r["model"] = mp.model
            all_rows.append(r)

    all_rows.sort(key=lambda r: float(r["dtw_delta_km"]))
    neg_outliers = all_rows[:10]
    pos_outliers = list(reversed(all_rows[-10:]))

    def outlier_rows(items: Sequence[Dict[str, Any]]) -> List[List[str]]:
        out: List[List[str]] = []
        for idx, r in enumerate(items, start=1):
            out.append(
                [
                    str(idx),
                    str(r.get("model", "")),
                    str(r.get("traj_id", "")),
                    str(r.get("od", "")),
                    str(r.get("ab_strength", "")),
                    _fmt(r.get("dtw_to_clean_km"), 3),
                    _fmt(r.get("dtw_to_dirty_km"), 3),
                    _fmt(r.get("dtw_delta_km"), 3),
                    str(r.get("corrected", "")),
                ]
            )
        return out

    # Teacher tables
    teacher_table = ""
    if teacher_by_model:
        teacher_rows: List[List[str]] = []
        for s in summaries:
            t = teacher_by_model.get(s.model)
            if t is None:
                continue
            teacher_rows.append(
                [
                    s.model,
                    s.display_name,
                    s.model_group,
                    s.group_display_name,
                    str(s.seed) if s.seed is not None else "",
                    _fmt(t.get("mean_log_perplexity_generated"), 4),
                    _fmt(t.get("mean_log_perplexity_clean"), 4),
                    _fmt(t.get("mean_log_perplexity_dirty"), 4),
                    _fmt(t.get("triangulation_rate"), 4),
                ]
            )

        teacher_table = _md_table(
            [
                "Model",
                "Display name",
                "Model group",
                "Group display",
                "Seed",
                "Mean log ppl (gen)",
                "Mean log ppl (clean)",
                "Mean log ppl (dirty)",
                "Triangulation rate",
            ],
            teacher_rows,
        )

    # Render markdown
    title_text = title or f"Phase B (Perturbation Correction) — {eval_dir.name}"

    lines: List[str] = []
    lines.append(f"# {title_text}")
    lines.append("")
    lines.append(f"**Eval dir:** `{eval_dir.as_posix()}`  ")
    lines.append(f"**Generated:** {date.today().isoformat()}")
    lines.append("")

    lines.append("## What this report covers")
    lines.append("")
    lines.append(
        "This report aggregates Phase B `perturbation_correction` outputs written by the pipeline. "
        "It is designed for faithful reproduction of key metrics (RSR, DTW deltas), stratifications, "
        "and outlier examples."
    )
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("Phase B is run via `python_pipeline.py` (phase name: `perturbation_correction`).")
    lines.append("")
    lines.append("Run only Phase B:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"uv run python python_pipeline.py --eval-dir {eval_dir.as_posix()} --only perturbation_correction"
    )
    lines.append("```")
    lines.append("")

    if config is None and config_path.exists():
        lines.append(
            "Note: `config/evaluation.yaml` exists, but PyYAML is not available in this environment, "
            "so the config snippet is not embedded."
        )
        lines.append("")

    if config is not None:
        lines.append("### Phase B config snapshot")
        lines.append("")
        keys = [
            "dataset",
            "data_dir",
            "perturbation_source_csv",
            "perturbation_od_source",
            "perturbation_max_entries",
            "perturbation_seed",
            "perturbation_use_astar",
            "perturbation_lmtad_checkpoint",
            "perturbation_lmtad_repo",
            "perturbation_lmtad_batch_size",
            "beam_width",
            "cuda_device",
            "force",
        ]
        lines.append("```yaml")
        for k in keys:
            if k in config:
                lines.append(f"{k}: {config[k]}")
        lines.append("```")
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    if embed_plots and figures_dir.exists():
        # Make plot links relative to where the markdown will live.
        # If output is inside repo, this yields stable relative paths.
        out_dir = output_md.parent.resolve()
        for key, path in plot_paths:
            if not path.exists():
                continue
            rel = _relpath(out_dir, path.resolve())
            if key == "rsr":
                lines.append("### Correction Rate (RSR)")
            elif key == "rsr_strength":
                lines.append("### Correction Rate (RSR) by perturbation strength")
            elif key == "gap":
                lines.append("### Mean DTW gap (dirty − clean)")
            elif key == "box":
                lines.append("### Per-sample DTW delta distribution")
            elif key == "scatter":
                lines.append("### DTW to clean vs DTW to dirty")
            elif key == "cdf":
                lines.append("### DTW delta CDF")
            lines.append("")
            lines.append(f"![]({rel})")
            lines.append("")

        if not any(p.exists() for _k, p in plot_paths):
            lines.append(
                "No plots found under `figures/perturbation_correction/`. Generate them with:"
            )
            lines.append("")
            lines.append("```bash")
            lines.append(
                f"uv run python tools/visualize_perturbation_correction_results.py --eval-dir {eval_dir.as_posix()}"
            )
            lines.append("```")
            lines.append("")
    else:
        lines.append(
            "Plots are not embedded (either missing or `--no-embed-plots` was used). "
            "To generate them:"
        )
        lines.append("")
        lines.append("```bash")
        lines.append(
            f"uv run python tools/visualize_perturbation_correction_results.py --eval-dir {eval_dir.as_posix()}"
        )
        lines.append("```")
        lines.append("")

    lines.append("## Table 1 — Per-model Phase B metrics")
    lines.append("")
    lines.append(
        _md_table(
            [
                "Model",
                "Display name",
                "Model group",
                "Group display",
                "Seed",
                "RSR",
                "Corrected",
                "Valid",
                "Invalid",
                "Mean DTW→clean (km)",
                "Mean DTW→dirty (km)",
                "Mean gap (km)",
                "Neg frac",
                "Tie frac",
                "Median Δ",
                "P90 Δ",
                "P95 Δ",
                "P99 Δ",
                "Max Δ",
            ],
            per_model_rows,
        )
    )
    lines.append("")

    lines.append("## Table 2 — Group averages (by model group)")
    lines.append("")
    lines.append(
        _md_table(
            [
                "Model group",
                "Group display",
                "N runs",
                "Mean RSR",
                "Std RSR",
                "Mean gap km",
                "Std gap km",
                "Mean neg frac",
            ],
            group_rows,
        )
    )
    lines.append("")

    lines.append("## Table 3 — RSR by perturbation strength (per model)")
    lines.append("")
    lines.append(
        _md_table(
            [
                "Model",
                "Display name",
                "Model group",
                "Group display",
                "Seed",
                "Strong n",
                "Strong RSR",
                "Other n",
                "Other RSR",
            ],
            strength_per_model,
        )
    )
    lines.append("")

    lines.append("## Table 4 — Strength-stratified aggregates (by model group)")
    lines.append("")
    lines.append(
        _md_table(
            [
                "Model group",
                "Group display",
                "Strength bucket",
                "N rows",
                "RSR",
                "Mean Δ (km)",
                "Median Δ (km)",
                "Neg frac",
            ],
            strength_group_rows,
        )
    )
    lines.append("")

    if teacher_table:
        lines.append("## Table 5 — Optional LM-TAD teacher triangulation")
        lines.append("")
        lines.append(
            "If `summary.json` contains a `teacher` field, Phase B computed LM-TAD log perplexity "
            "triangulation metrics (see `tools/perturbation_correction.py`)."
        )
        lines.append("")
        lines.append(teacher_table)
        lines.append("")

    lines.append("## Outliers")
    lines.append("")
    lines.append("### Most negative DTW deltas (generated closer to dirty)")
    lines.append("")
    lines.append(
        _md_table(
            [
                "Rank",
                "Model",
                "traj_id",
                "od",
                "ab_strength",
                "DTW clean (km)",
                "DTW dirty (km)",
                "Δ (dirty-clean) km",
                "corrected",
            ],
            outlier_rows(neg_outliers),
        )
    )
    lines.append("")

    lines.append("### Most positive DTW deltas (dirty much farther than clean)")
    lines.append("")
    lines.append(
        _md_table(
            [
                "Rank",
                "Model",
                "traj_id",
                "od",
                "ab_strength",
                "DTW clean (km)",
                "DTW dirty (km)",
                "Δ (dirty-clean) km",
                "corrected",
            ],
            outlier_rows(pos_outliers),
        )
    )
    lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "- RSR uses a strict comparison (`DTW(pred, clean) < DTW(pred, dirty)`), so ties count as not corrected."
    )
    lines.append(
        "- A large gap (dirty − clean) indicates the prediction is much closer to clean than dirty under centroid-DTW."
    )
    lines.append(
        "- Strength stratification uses `ab_strength == 'strong'` vs everything else; availability depends on the perturbation CSV metadata."
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate a Markdown report for Phase B perturbation correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory containing perturbation_correction outputs",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help=(
            "Output markdown path. Default: <eval-dir>/analysis/perturbation_correction_report.md"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title (default derived from eval dir name)",
    )
    parser.add_argument(
        "--no-embed-plots",
        action="store_true",
        help="Do not embed plots even if present under figures/perturbation_correction",
    )

    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else (eval_dir / "analysis" / "perturbation_correction_report.md")
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)

    content = _render_report(
        eval_dir=eval_dir,
        output_md=output_md,
        title=args.title,
        embed_plots=not args.no_embed_plots,
    )

    output_md.write_text(content)
    logger.info("Wrote report to %s", output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
