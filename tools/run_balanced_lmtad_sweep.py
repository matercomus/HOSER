#!/usr/bin/env python3
"""One-shot balanced LM-TAD evaluation + threshold sweep.

This is the repeatable "disambiguation check":
1) Build a balanced split CSV (all abnormal + matched normals)
2) Run LM-TAD scoring on the balanced split using a baseline-calibrated threshold
3) Sweep multiple baseline quantiles and report AUROC/AUPRC + recall/precision

Outputs (under --out-dir):
- balanced_data/<name>/<split>.csv
- eval/<name>/evaluation_results.json
- eval/<name>/threshold_sweep.json
- eval/<name>/threshold_sweep.md

Example
uv run python tools/run_balanced_lmtad_sweep.py \
  --name bj_detectable_dr_bal1 \
  --baseline-data-dir data/Beijing \
  --baseline-eval tools_eval_lmtad/Beijing \
  --target-data-dir data/Beijing_abnormal_3_detectable_dr \
  --split train \
  --normal-per-abnormal 1 \
  --ckpt /path/to/lmtad.ckpt \
  --lmtad-repo /home/mka299/LMTAD \
  --device cuda:0 \
  --batch-size 128 \
  --quantiles 0.90,0.95,0.99 \
  --out-dir research_runs/_balanced_checks
"""

from __future__ import annotations

import argparse
from pathlib import Path

# When executed as `python tools/run_balanced_lmtad_sweep.py`, Python sets
# `sys.path[0]` to `<repo>/tools`, which breaks `import tools.*`.
# Ensure the repo root is on sys.path so the `tools` package is importable.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.analyze_lmtad_threshold_sweep import main as sweep_main
from tools.make_balanced_eval_dataset import main as balance_main


def _run_balance_step(*, target_data_dir: Path, split: str, balanced_dir: Path, normal_per_abnormal: int,
                      length_bucket: int, seed: int, allow_replacement: bool, copy_roadmaps: bool) -> None:
    args = [
        "--source-dataset-dir",
        str(target_data_dir),
        "--split",
        str(split),
        "--out-dir",
        str(balanced_dir),
        "--normal-per-abnormal",
        str(int(normal_per_abnormal)),
        "--length-bucket",
        str(int(length_bucket)),
        "--seed",
        str(int(seed)),
    ]
    if allow_replacement:
        args.append("--allow-replacement")
    if copy_roadmaps:
        args.append("--copy-roadmaps")

    import sys

    old = sys.argv
    try:
        sys.argv = ["make_balanced_eval_dataset.py", *args]
        balance_main()
    finally:
        sys.argv = old


def _run_eval_step(*, data_dir: Path, output_dir: Path, ckpt: Path, lmtad_repo: Path, device: str, batch_size: int,
                  splits: str, sample_frac: float, sample_seed: int, baseline_eval: Path, baseline_quantile: float,
                  baseline_split: str, roadmap: Path | None) -> None:
    from tools import evaluate_dataset_with_lmtad as eval_mod

    eval_mod.evaluate_splits(
        data_dir=data_dir,
        roadmap_file=roadmap,
        lmtad_ckpt=ckpt,
        lmtad_repo=lmtad_repo,
        device=device,
        batch_size=int(batch_size),
        splits=[s.strip() for s in splits.split(",") if s.strip()],
        output_dir=output_dir,
        sample_frac=float(sample_frac),
        sample_seed=int(sample_seed),
        baseline_eval=baseline_eval,
        baseline_quantile=float(baseline_quantile),
        baseline_split=str(baseline_split),
        write_baseline=False,
    )


def _run_sweep_step(*, eval_dir: Path, split: str, csv_path: Path, baseline_eval: Path, baseline_split: str,
                   quantiles: str) -> None:
    args = [
        "--eval-dir",
        str(eval_dir),
        "--split",
        str(split),
        "--csv",
        str(csv_path),
        "--baseline-eval",
        str(baseline_eval),
        "--baseline-split",
        str(baseline_split),
        "--quantiles",
        str(quantiles),
    ]

    import sys

    old = sys.argv
    try:
        sys.argv = ["analyze_lmtad_threshold_sweep.py", *args]
        sweep_main()
    finally:
        sys.argv = old


def main() -> int:
    parser = argparse.ArgumentParser(description="Run balanced LM-TAD eval + threshold sweep")
    parser.add_argument("--name", type=str, required=True, help="Run name used for subfolders")

    parser.add_argument("--baseline-data-dir", type=Path, required=False, default=None,
                        help="Baseline dataset dir (for default roadmap lookup when --roadmap omitted).")
    parser.add_argument("--baseline-eval", type=Path, required=True,
                        help="Baseline eval dir or baseline_eval.json (used for thresholds)")

    parser.add_argument("--target-data-dir", type=Path, required=True, help="Target dataset dir with <split>.csv")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--normal-per-abnormal", type=int, default=1)
    parser.add_argument("--length-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-replacement", action="store_true")
    parser.add_argument("--copy-roadmaps", action="store_true")

    parser.add_argument("--roadmap", type=Path, default=None,
                        help="Optional roadmap.geo override; if omitted, evaluator will try <data-dir>/roadmap.geo")

    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--lmtad-repo", type=Path, default=Path("/home/mka299/LMTAD"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--splits", type=str, default=None,
                        help="Comma splits to eval (default: just --split)")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)

    parser.add_argument("--baseline-quantile", type=float, default=0.95)
    parser.add_argument("--baseline-split", type=str, default="train")
    parser.add_argument("--quantiles", type=str, default="0.90,0.95,0.99")

    parser.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args()

    name = str(args.name)
    split = str(args.split)
    splits = str(args.splits) if args.splits is not None else split

    # Fail fast on required paths.
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"LM-TAD checkpoint not found: {ckpt}. "
            "Pass the real checkpoint path via --ckpt."
        )

    lmtad_repo = Path(args.lmtad_repo)
    if not lmtad_repo.exists():
        raise FileNotFoundError(
            f"LM-TAD repo not found: {lmtad_repo}. Pass the repo path via --lmtad-repo."
        )

    baseline_eval = Path(args.baseline_eval)
    if not baseline_eval.exists():
        raise FileNotFoundError(
            f"Baseline eval not found: {baseline_eval}. "
            "Pass a directory containing baseline_eval.json or the file itself via --baseline-eval."
        )

    target_data_dir = Path(args.target_data_dir)
    split_csv = target_data_dir / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Target split CSV not found: {split_csv}. "
            "Ensure --target-data-dir points to a dataset folder containing <split>.csv."
        )

    out_dir = Path(args.out_dir)
    balanced_dir = out_dir / "balanced_data" / name
    eval_dir = out_dir / "eval" / name

    # Resolve roadmap file (evaluate_splits requires a Path).
    roadmap_file: Path | None
    if args.roadmap is not None:
        roadmap_file = Path(args.roadmap)
    else:
        candidates: list[Path] = [
            balanced_dir / "roadmap.geo",
            target_data_dir / "roadmap.geo",
        ]
        if args.baseline_data_dir is not None:
            candidates.append(Path(args.baseline_data_dir) / "roadmap.geo")
        roadmap_file = next((p for p in candidates if p.exists()), None)

    if roadmap_file is None or not roadmap_file.exists():
        raise FileNotFoundError(
            "Roadmap file not found. Pass --roadmap <path/to/roadmap.geo> "
            "or ensure roadmap.geo exists in the target dataset dir."
        )

    _run_balance_step(
        target_data_dir=target_data_dir,
        split=split,
        balanced_dir=balanced_dir,
        normal_per_abnormal=int(args.normal_per_abnormal),
        length_bucket=int(args.length_bucket),
        seed=int(args.seed),
        allow_replacement=bool(args.allow_replacement),
        copy_roadmaps=bool(args.copy_roadmaps),
    )

    csv_path = balanced_dir / f"{split}.csv"

    _run_eval_step(
        data_dir=balanced_dir,
        output_dir=eval_dir,
        ckpt=ckpt,
        lmtad_repo=lmtad_repo,
        device=str(args.device),
        batch_size=int(args.batch_size),
        splits=splits,
        sample_frac=float(args.sample_frac),
        sample_seed=int(args.sample_seed),
        baseline_eval=baseline_eval,
        baseline_quantile=float(args.baseline_quantile),
        baseline_split=str(args.baseline_split),
        roadmap=roadmap_file,
    )

    _run_sweep_step(
        eval_dir=eval_dir,
        split=split,
        csv_path=csv_path,
        baseline_eval=baseline_eval,
        baseline_split=str(args.baseline_split),
        quantiles=str(args.quantiles),
    )

    print(str(eval_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
