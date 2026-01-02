"""Smoke tests for `tools.run_research_eval`.

This verifies we create a timestamped, named run directory with a manifest and
subfolders, without running real (slow) LM-TAD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_min_dataset_dir(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Minimal split CSV with required columns.
    (dataset_dir / "train.csv").write_text(
        "rid_list,abnormality_info\n"
        '"1,2,3",normal\n'
        '"1,2,3","{\'type\': \'detour\'}"\n',
        encoding="utf-8",
    )

    # Placeholder roadmap files (wrapper's nonlinear loaders are monkeypatched).
    (dataset_dir / "roadmap.geo").write_text("", encoding="utf-8")
    (dataset_dir / "roadmap.rel").write_text("", encoding="utf-8")


def test_run_creates_named_output_tree(tmp_path, monkeypatch):
    from tools import run_research_eval as tool

    baseline_dir = tmp_path / "Beijing"
    target_dir = tmp_path / "Beijing_abnormal_3_detectable_dr"
    _write_min_dataset_dir(baseline_dir)
    _write_min_dataset_dir(target_dir)

    # Speed: monkeypatch nonlinear loaders + coefficient to be deterministic.
    monkeypatch.setattr(tool, "load_road_lengths_m", lambda *_a, **_k: {1: 1.0, 2: 1.0, 3: 1.0})
    monkeypatch.setattr(tool, "load_outgoing_edges", lambda *_a, **_k: {1: [2], 2: [3]})
    monkeypatch.setattr(tool, "non_linear_coefficient", lambda *_a, **_k: 1.25)

    # Speed: monkeypatch LM-TAD evaluation to just write expected JSON outputs.
    def fake_evaluate_splits(*, output_dir: Path, write_baseline: bool, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation_results.json").write_text(
            json.dumps({"train": {"outlier_method": "baseline_quantile", "outlier_rate": 0.1}}),
            encoding="utf-8",
        )
        if write_baseline:
            (output_dir / "baseline_eval.json").write_text(
                json.dumps({"train": {"log_perplexity_values": [0.1, 0.2]}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(tool, "evaluate_splits", fake_evaluate_splits)

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("", encoding="utf-8")

    run_root = tmp_path / "research_runs"

    argv = [
        "--run-name",
        "my research run",
        "--run-root",
        str(run_root),
        "--baseline-dataset-dir",
        str(baseline_dir),
        "--target-dataset-dirs",
        str(target_dir),
        "--lmtad-checkpoint",
        str(ckpt),
        "--device",
        "cpu",
        "--splits",
        "train",
        "--sample-frac",
        "1.0",
    ]

    monkeypatch.setattr("sys.argv", ["tools/run_research_eval.py", *argv])
    assert tool.main() == 0

    # Expect exactly one run directory.
    run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "done.json").exists()

    nonlinear = run_dir / "nonlinear" / "nonlinear_summary.json"
    assert nonlinear.exists()
    payload = json.loads(nonlinear.read_text(encoding="utf-8"))
    assert "baseline" in payload
    assert "targets" in payload

    # LM-TAD outputs are placed under run_dir/lmtad/<dataset>/
    assert (run_dir / "lmtad" / baseline_dir.name / "evaluation_results.json").exists()
    assert (run_dir / "lmtad" / baseline_dir.name / "baseline_eval.json").exists()
    assert (run_dir / "lmtad" / target_dir.name / "evaluation_results.json").exists()


@pytest.mark.parametrize("run_name", ["", "   ", "$$$", "a/b:c", "my run"])
def test_safe_name_never_empty(run_name):
    from tools.run_research_eval import _safe_name

    assert _safe_name(run_name)
