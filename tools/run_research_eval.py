#!/usr/bin/env python3
"""Create a named, timestamped research run with saved outputs.

This tool is meant for research reproducibility:
- Creates a run directory under `research_runs/` by default.
- Writes a `manifest.json` capturing args + git commit (when available).
- Computes the non-linear coefficient indicator for baseline + targets.
- Runs LM-TAD evaluation in baseline-calibrated mode and saves outputs.

It intentionally reuses existing project tooling:
- `tools/nonlinear_coefficient.py` logic (imported functions)
- `tools/evaluate_dataset_with_lmtad.py::evaluate_splits`

Example:
  uv run python tools/run_research_eval.py \
    --run-name bj_detectable_dr_smoke \
    --baseline-dataset-dir data/Beijing \
    --target-dataset-dirs data/Beijing_abnormal_3_detectable_dr \
    --lmtad-checkpoint /path/to/ckpt_best.pt \
    --device cuda:0 \
    --splits train \
    --sample-frac 0.002 \
    --baseline-quantile 0.95
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tools.evaluate_dataset_with_lmtad import evaluate_splits
from tools.nonlinear_coefficient import (
    load_outgoing_edges,
    load_road_lengths_m,
    non_linear_coefficient,
    _parse_rid_list,
)


@dataclass(frozen=True)
class NonLinearSummary:
    dataset: str
    split: str
    max_rows: int
    normals_count: int
    normals_mean: float
    abnormals_count: int
    abnormals_mean: float


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    """Return a filesystem-safe run name."""
    keep = []
    for ch in text.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("-")
    out = "".join(keep).strip("-_")
    return out or "run"


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        sha = cp.stdout.strip()
        return sha or None
    except Exception:
        return None


def _nonlinear_for_dataset(
    *,
    dataset_dir: Path,
    split: str,
    max_rows: int,
) -> NonLinearSummary:
    """Compute non-linear coefficient means for normal vs abnormal rows."""

    split_csv = dataset_dir / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}")

    road_len_m = load_road_lengths_m(dataset_dir / "roadmap.geo")
    outgoing = load_outgoing_edges(dataset_dir / "roadmap.rel")

    normals: List[float] = []
    abnormals: List[float] = []

    import csv

    with split_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "rid_list" not in r.fieldnames:
            raise ValueError(f"Missing rid_list column: {split_csv}")

        for i, row in enumerate(r):
            if max_rows is not None and i >= int(max_rows):
                break

            rid_list = _parse_rid_list(row.get("rid_list") or "")
            coef = non_linear_coefficient(
                rid_list, outgoing=outgoing, road_len_m=road_len_m
            )
            if coef is None:
                continue

            ab_info = (row.get("abnormality_info") or "").strip().lower()
            if ab_info and ab_info != "normal":
                abnormals.append(float(coef))
            else:
                normals.append(float(coef))

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return NonLinearSummary(
        dataset=str(dataset_dir.name),
        split=str(split),
        max_rows=int(max_rows),
        normals_count=int(len(normals)),
        normals_mean=_mean(normals),
        abnormals_count=int(len(abnormals)),
        abnormals_mean=_mean(abnormals),
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _jsonify(obj: Any) -> Any:
    """Convert common non-JSON types (e.g., Path) into JSON-serializable values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a reproducible research evaluation (named outputs)"
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("research_runs"),
        help="Root folder to store run directories.",
    )

    parser.add_argument(
        "--baseline-dataset-dir",
        type=Path,
        required=True,
        help="Baseline dataset dir containing train/val/test CSV + roadmap.geo/rel.",
    )
    parser.add_argument(
        "--target-dataset-dirs",
        type=str,
        required=True,
        help="Comma-separated dataset dirs to evaluate (e.g., data/Foo,data/Bar).",
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma-separated splits to evaluate (e.g., train,val,test).",
    )

    # Non-linear coefficient options
    parser.add_argument(
        "--nonlinear-max-rows",
        type=int,
        default=20000,
        help="Max rows per split to use for non-linear coefficient summary.",
    )

    # LM-TAD options
    parser.add_argument(
        "--lmtad-checkpoint",
        "--ckpt",
        dest="lmtad_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lmtad-repo",
        type=Path,
        default=Path("/home/mka299/LMTAD"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-frac", type=float, default=0.01)
    parser.add_argument("--baseline-quantile", type=float, default=0.95)
    parser.add_argument("--baseline-split", type=str, default="train")
    parser.add_argument(
        "--roadmap",
        type=Path,
        default=None,
        help=(
            "Optional override for roadmap.geo used by LM-TAD mapping. "
            "Default: <baseline-dataset-dir>/roadmap.geo"
        ),
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    run_name = _safe_name(args.run_name)
    ts = _utc_timestamp()
    run_dir = Path(args.run_root) / f"{ts}__{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    targets = [
        Path(s.strip())
        for s in str(args.target_dataset_dirs).split(",")
        if s.strip()
    ]

    baseline_dir = Path(args.baseline_dataset_dir)
    roadmap_geo = Path(args.roadmap) if args.roadmap is not None else baseline_dir / "roadmap.geo"

    manifest: Dict[str, Any] = {
        "timestamp_utc": ts,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "cwd": os.getcwd(),
        "git_commit": _git_commit(repo_root),
        "args": _jsonify(vars(args)),
        "outputs": {
            "nonlinear_dir": str(run_dir / "nonlinear"),
            "lmtad_dir": str(run_dir / "lmtad"),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)

    # Non-linear coefficient summaries
    nonlinear_dir = run_dir / "nonlinear"
    nonlinear_payload: Dict[str, Any] = {"baseline": {}, "targets": {}}

    for split in splits:
        summary = _nonlinear_for_dataset(
            dataset_dir=baseline_dir,
            split=split,
            max_rows=int(args.nonlinear_max_rows),
        )
        nonlinear_payload["baseline"][split] = asdict(summary)

    for tdir in targets:
        nonlinear_payload["targets"].setdefault(str(tdir.name), {})
        for split in splits:
            summary = _nonlinear_for_dataset(
                dataset_dir=tdir,
                split=split,
                max_rows=int(args.nonlinear_max_rows),
            )
            nonlinear_payload["targets"][str(tdir.name)][split] = asdict(summary)

    _write_json(nonlinear_dir / "nonlinear_summary.json", nonlinear_payload)

    # LM-TAD evaluation outputs (baseline-calibrated)
    lmtad_root = run_dir / "lmtad"
    baseline_out = lmtad_root / baseline_dir.name

    evaluate_splits(
        data_dir=baseline_dir,
        roadmap_file=roadmap_geo,
        lmtad_ckpt=Path(args.lmtad_checkpoint),
        lmtad_repo=Path(args.lmtad_repo),
        device=str(args.device),
        batch_size=int(args.batch_size),
        splits=splits,
        output_dir=baseline_out,
        sample_frac=float(args.sample_frac),
        sample_seed=42,
        baseline_eval=None,
        baseline_quantile=float(args.baseline_quantile),
        baseline_split=str(args.baseline_split),
        write_baseline=True,
        baseline_out=baseline_out / "baseline_eval.json",
    )

    for tdir in targets:
        out_dir = lmtad_root / tdir.name
        evaluate_splits(
            data_dir=tdir,
            roadmap_file=roadmap_geo,
            lmtad_ckpt=Path(args.lmtad_checkpoint),
            lmtad_repo=Path(args.lmtad_repo),
            device=str(args.device),
            batch_size=int(args.batch_size),
            splits=splits,
            output_dir=out_dir,
            sample_frac=float(args.sample_frac),
            sample_seed=42,
            baseline_eval=baseline_out,
            baseline_quantile=float(args.baseline_quantile),
            baseline_split=str(args.baseline_split),
            write_baseline=False,
        )

    _write_json(
        run_dir / "done.json",
        {
            "ok": True,
            "run_dir": str(run_dir),
            "baseline": str(baseline_dir),
            "targets": [str(p) for p in targets],
            "splits": splits,
        },
    )

    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
