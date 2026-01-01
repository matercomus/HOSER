"""Smoke tests for `tools.evaluate_dataset_with_lmtad`.

These tests validate the baseline-calibrated outlier thresholding logic without
loading the real LM-TAD model (monkeypatched for speed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_min_split_csv(path: Path, *, rows: int, abnormal_rows: int) -> None:
    """Write a minimal split CSV with `rid_list` and `abnormality_info`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("rid_list,abnormality_info\n")
        for i in range(rows):
            rid_list = "1,2,3"
            ab = "{'type': 'detour'}" if i < abnormal_rows else "normal"
            f.write(f"{rid_list},{ab}\n")


def test_baseline_calibrated_threshold_is_applied(tmp_path, monkeypatch):
    """Target outliers are computed using the baseline quantile threshold."""

    from tools import evaluate_dataset_with_lmtad as mod

    baseline_dir = tmp_path / "baseline"
    target_dir = tmp_path / "target"

    _write_min_split_csv(baseline_dir / "train.csv", rows=4, abnormal_rows=0)
    _write_min_split_csv(target_dir / "train.csv", rows=4, abnormal_rows=4)

    roadmap_geo = tmp_path / "roadmap.geo"
    roadmap_geo.write_text("", encoding="utf-8")

    lmtad_ckpt = tmp_path / "ckpt.pt"
    lmtad_ckpt.write_text("", encoding="utf-8")

    lmtad_repo = tmp_path / "LMTAD"
    lmtad_repo.mkdir()

    class FakeTeacher:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(mod, "LMTADTeacher", FakeTeacher)

    # Avoid reading/processing the roadmap in this smoke test.
    monkeypatch.setattr(
        mod,
        "_create_grid_mapper",
        lambda **kwargs: (None, (64, 64)),
    )

    baseline_scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    target_scores = np.array([3.0, 3.1, 3.2, 3.3], dtype=np.float64)

    baseline_prefix = str(baseline_dir.resolve()) + "/"

    def fake_evaluate_csv(*, csv_path, **kwargs):
        csv_path_str = str(Path(csv_path).resolve())
        if csv_path_str.startswith(baseline_prefix):
            scores = baseline_scores
        else:
            scores = target_scores
        within = np.zeros_like(scores, dtype=np.float32)
        return scores, within, int(scores.shape[0])

    monkeypatch.setattr(mod, "_evaluate_csv", fake_evaluate_csv)

    baseline_out_dir = tmp_path / "baseline_eval"
    baseline_eval_path = baseline_out_dir / "baseline_eval.json"

    mod.evaluate_splits(
        data_dir=baseline_dir,
        roadmap_file=roadmap_geo,
        lmtad_ckpt=lmtad_ckpt,
        lmtad_repo=lmtad_repo,
        device="cpu",
        batch_size=4,
        splits=["train"],
        output_dir=baseline_out_dir,
        sample_frac=1.0,
        write_baseline=True,
        baseline_out=baseline_eval_path,
    )

    assert baseline_eval_path.exists()

    target_out_dir = tmp_path / "target_eval"
    mod.evaluate_splits(
        data_dir=target_dir,
        roadmap_file=roadmap_geo,
        lmtad_ckpt=lmtad_ckpt,
        lmtad_repo=lmtad_repo,
        device="cpu",
        batch_size=4,
        splits=["train"],
        output_dir=target_out_dir,
        sample_frac=1.0,
        baseline_eval=baseline_out_dir,
        baseline_quantile=0.75,
        baseline_split="train",
        write_baseline=False,
    )

    results_path = target_out_dir / "evaluation_results.json"
    assert results_path.exists()

    results = json.loads(results_path.read_text(encoding="utf-8"))
    split = results["train"]

    expected_threshold = float(np.quantile(baseline_scores, 0.75))
    baseline_expected_outlier_rate = float((baseline_scores > expected_threshold).mean())

    assert split["outlier_method"] == "baseline_quantile"
    assert split["outlier_rate"] == 1.0

    meta = split["baseline_calibrated"]
    assert meta["threshold"] == expected_threshold
    assert meta["baseline_outlier_rate"] == baseline_expected_outlier_rate
    assert meta["target_outlier_rate"] == 1.0
