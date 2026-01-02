from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _write_split_csv(path: Path, *, rows: int, abnormal_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rid_list", "abnormality_info"])
        writer.writeheader()
        for i in range(rows):
            ab = "{'type': 'detour'}" if i < abnormal_rows else "normal"
            writer.writerow({"rid_list": "1,2,3", "abnormality_info": ab})


def test_run_balanced_lmtad_sweep_smoke(tmp_path, monkeypatch):
    """End-to-end smoke: balance -> eval (monkeypatched) -> sweep."""

    from tools import evaluate_dataset_with_lmtad as eval_mod

    baseline_data = tmp_path / "baseline_data"
    target_data = tmp_path / "target_data"

    _write_split_csv(baseline_data / "train.csv", rows=10, abnormal_rows=0)
    _write_split_csv(target_data / "train.csv", rows=10, abnormal_rows=4)

    # Minimal roadmap + ckpt placeholders.
    (baseline_data / "roadmap.geo").write_text("", encoding="utf-8")
    (target_data / "roadmap.geo").write_text("", encoding="utf-8")

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("", encoding="utf-8")

    lmtad_repo = tmp_path / "LMTAD"
    lmtad_repo.mkdir()

    class FakeTeacher:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(eval_mod, "LMTADTeacher", FakeTeacher)

    monkeypatch.setattr(
        eval_mod,
        "_create_grid_mapper",
        lambda **kwargs: (None, (64, 64)),
    )

    baseline_scores = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    target_scores = np.concatenate(
        [
            np.linspace(0.0, 0.2, 6, dtype=np.float64),
            np.linspace(2.0, 2.3, 4, dtype=np.float64),
        ]
    )

    baseline_prefix = str(baseline_data.resolve()) + "/"

    def fake_evaluate_csv(*, csv_path, **kwargs):
        csv_path_str = str(Path(csv_path).resolve())
        with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1  # minus header
        if csv_path_str.startswith(baseline_prefix):
            scores = baseline_scores[:n_rows]
        else:
            scores = target_scores[:n_rows]
        within = np.zeros_like(scores, dtype=np.float32)
        return scores, within, int(scores.shape[0])

    monkeypatch.setattr(eval_mod, "_evaluate_csv", fake_evaluate_csv)

    # Write a baseline eval file (normally created by a prior baseline run).
    baseline_eval_dir = tmp_path / "baseline_eval"
    baseline_eval_dir.mkdir()
    (baseline_eval_dir / "baseline_eval.json").write_text(
        json.dumps({"train": {"log_perplexity_values": baseline_scores.tolist()}}),
        encoding="utf-8",
    )

    from tools.run_balanced_lmtad_sweep import main as run_main

    out_dir = tmp_path / "out"

    import sys

    old = sys.argv
    try:
        sys.argv = [
            "run_balanced_lmtad_sweep.py",
            "--name",
            "demo",
            "--baseline-data-dir",
            str(baseline_data),
            "--baseline-eval",
            str(baseline_eval_dir),
            "--target-data-dir",
            str(target_data),
            "--split",
            "train",
            "--normal-per-abnormal",
            "1",
            "--ckpt",
            str(ckpt),
            "--lmtad-repo",
            str(lmtad_repo),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--quantiles",
            "0.90,0.95",
            "--out-dir",
            str(out_dir),
        ]
        run_main()
    finally:
        sys.argv = old

    eval_dir = out_dir / "eval" / "demo"
    assert (eval_dir / "evaluation_results.json").exists()
    assert (eval_dir / "threshold_sweep.json").exists()
    assert (eval_dir / "threshold_sweep.md").exists()
