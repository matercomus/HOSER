#!/usr/bin/env python3
"""Summarize LM-TAD reference evaluation artifacts.

This script parses LM-TAD's `EVALUATION_ANALYSIS.md` files (produced in the
external LM-TAD repo and symlinked into this workspace under `results/LMTAD/`).

It extracts the key numbers we care about for comparison with HOSER-side
`tools_eval_lmtad/*` evaluations:
- Non-outlier mean/median log perplexity
- Detour and route-switch mean/median log perplexity
- Average precision (AP) and PR-AUC (as reported)
- The reported "optimal" threshold used in that evaluation

Example
-------
uv run python tools/summarize_lmtad_reference_eval.py \
  --paths \
    results/LMTAD/beijing_hoser_reference/**/eval/EVALUATION_ANALYSIS.md \
    results/LMTAD/porto_hoser/**/eval/EVALUATION_ANALYSIS.md
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class RefSummary:
    dataset: str
    run_id: str
    eval_dir: str
    non_outlier_mean: Optional[float]
    non_outlier_median: Optional[float]
    route_switch_mean: Optional[float]
    route_switch_median: Optional[float]
    detour_mean: Optional[float]
    detour_median: Optional[float]
    ap: Optional[float]
    pr_auc: Optional[float]
    threshold: Optional[float]


def _find_first_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def _parse_table_row_mean_median(
    text: str, row_name: str
) -> tuple[Optional[float], Optional[float]]:
    """Parse a markdown table row like:

    | **Non-outlier** | 0.5325 | 0.1547 | 0.5133 |

    Returns (mean, median).
    """
    # Capture mean in col2 and median in col4.
    pat = rf"^\|\s*\*\*{re.escape(row_name)}\*\*\s*\|\s*([0-9.]+)\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)\s*\|\s*$"
    m = re.search(pat, text, flags=re.MULTILINE)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except (TypeError, ValueError):
        return None, None


def parse_analysis(md_path: Path) -> RefSummary:
    text = md_path.read_text(encoding="utf-8", errors="replace")

    # Header fields
    run_id = "unknown"
    m = re.search(r"\*\*Run ID:\*\*\s*`([^`]+)`", text)
    if m:
        run_id = m.group(1)

    eval_dir = ""
    m = re.search(r"\*\*Evaluation Directory:\*\*\s*`([^`]+)`", text)
    if m:
        eval_dir = m.group(1)

    # Infer dataset name from eval_dir prefix if available
    dataset = (
        md_path.parent.parent.parent.parent.name
    )  # results/LMTAD/<dataset>/run_.../...
    if eval_dir:
        dataset = eval_dir.split("/", 1)[0]

    # Log Perplexity Statistics table rows
    non_mean, non_median = _parse_table_row_mean_median(text, "Non-outlier")
    rs_mean, rs_median = _parse_table_row_mean_median(text, "Route Switch")
    det_mean, det_median = _parse_table_row_mean_median(text, "Detour")

    # Performance metrics table: take the first row's Avg Precision and PR-AUC.
    # Example row:
    # | final_model | 0.05 | 3 | 0.1 | 0.9966 | ... | Avg Precision | PR-AUC |
    ap = _find_first_float(
        r"^\|\s*[^|]+\|\s*0\.05\s*\|\s*\d+\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|.*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$",
        text,
    )
    # The regex above captures only the first group; we also want PR-AUC.
    pr_auc = None
    m = re.search(
        r"^\|\s*[^|]+\|\s*0\.05\s*\|\s*\d+\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|.*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$",
        text,
        flags=re.MULTILINE,
    )
    if m:
        try:
            ap = float(m.group(1))
            pr_auc = float(m.group(2))
        except (TypeError, ValueError):
            ap = ap
            pr_auc = None

    threshold = _find_first_float(r"-\s*\*\*Detection Threshold:\*\*\s*([0-9.]+)", text)

    return RefSummary(
        dataset=dataset,
        run_id=run_id,
        eval_dir=eval_dir,
        non_outlier_mean=non_mean,
        non_outlier_median=non_median,
        route_switch_mean=rs_mean,
        route_switch_median=rs_median,
        detour_mean=det_mean,
        detour_median=det_median,
        ap=ap,
        pr_auc=pr_auc,
        threshold=threshold,
    )


def _expand_globs(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        out.extend(Path(m) for m in matches)
    # de-dupe
    uniq: List[Path] = []
    seen = set()
    for p in out:
        rp = str(p)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def to_markdown(summaries: List[RefSummary]) -> str:
    lines: List[str] = []
    lines.append("## LM-TAD reference-run summary")
    lines.append("")
    lines.append(
        "| Dataset | Run | Non-outlier mean | Route-switch mean | Detour mean | AP | PR-AUC | Threshold |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            "| "
            f"{s.dataset} | {s.run_id} | "
            f"{'' if s.non_outlier_mean is None else f'{s.non_outlier_mean:.4f}'} | "
            f"{'' if s.route_switch_mean is None else f'{s.route_switch_mean:.4f}'} | "
            f"{'' if s.detour_mean is None else f'{s.detour_mean:.4f}'} | "
            f"{'' if s.ap is None else f'{s.ap:.4f}'} | "
            f"{'' if s.pr_auc is None else f'{s.pr_auc:.4f}'} | "
            f"{'' if s.threshold is None else f'{s.threshold:.4f}'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize LM-TAD reference evaluation markdown files"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more file paths or glob patterns to EVALUATION_ANALYSIS.md",
    )
    args = parser.parse_args()

    md_files = _expand_globs(args.paths)
    if not md_files:
        raise SystemExit("No files matched")

    summaries = [parse_analysis(p) for p in md_files]
    print(to_markdown(summaries))


if __name__ == "__main__":
    main()
