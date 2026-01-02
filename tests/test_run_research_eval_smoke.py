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
    from tools import nonlinear_coefficient as nl
    from tools import evaluate_dataset_with_lmtad as eval_tool

    baseline_dir = tmp_path / "Beijing"
    target_dir = tmp_path / "Beijing_abnormal_3_detectable_dr"
    _write_min_dataset_dir(baseline_dir)
    _write_min_dataset_dir(target_dir)

    # Speed: monkeypatch nonlinear loaders + coefficient to be deterministic.
    monkeypatch.setattr(
        nl, "load_road_lengths_m", lambda *_a, **_k: {1: 1.0, 2: 1.0, 3: 1.0}
    )
    monkeypatch.setattr(nl, "load_outgoing_edges", lambda *_a, **_k: {1: [2], 2: [3]})
    monkeypatch.setattr(nl, "non_linear_coefficient", lambda *_a, **_k: 1.25)

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

    monkeypatch.setattr(eval_tool, "evaluate_splits", fake_evaluate_splits)

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


def test_run_can_auto_plot_abnormal_od(tmp_path, monkeypatch):
    """Optional abnormal OD plotting is executed and recorded."""

    from tools import run_research_eval as tool
    from tools import nonlinear_coefficient as nl
    from tools import evaluate_dataset_with_lmtad as eval_tool

    baseline_dir = tmp_path / "Beijing"
    target_dir = tmp_path / "Beijing_abnormal_3_detectable_dr"
    _write_min_dataset_dir(baseline_dir)
    _write_min_dataset_dir(target_dir)

    # Nonlinear: deterministic.
    monkeypatch.setattr(
        nl, "load_road_lengths_m", lambda *_a, **_k: {1: 1.0, 2: 1.0, 3: 1.0}
    )
    monkeypatch.setattr(nl, "load_outgoing_edges", lambda *_a, **_k: {1: [2], 2: [3]})
    monkeypatch.setattr(nl, "non_linear_coefficient", lambda *_a, **_k: 1.25)

    # LM-TAD: stub.
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

    monkeypatch.setattr(eval_tool, "evaluate_splits", fake_evaluate_splits)

    # Abnormal plotting: provide a minimal comparison_report.json and monkeypatch
    # the plotting module to avoid matplotlib work.
    report_dir = tmp_path / "eval_abnormal" / "porto_hoser"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "comparison_report.json"
    report_path.write_text(
        json.dumps(
            {
                "model_results": {
                    "dummy": {
                        "total_trajectories": 10,
                        "abnormality_detection": {"detour": {"count": 2}},
                        "similarity_metrics": {"edr": 1.0, "dtw": 2.0, "hausdorff": 3.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    from tools import plot_abnormal_evaluation as pae

    called = {"ok": False}

    def fake_plot_evaluation_from_files(*, comparison_report_file, output_dir, dataset, config=None):
        called["ok"] = True
        output_dir.mkdir(parents=True, exist_ok=True)
        # Simulate one output file.
        out = output_dir / "abnormality_reproduction_rates.png"
        out.write_text("x", encoding="utf-8")
        return {"abnormality_reproduction_rates": out}

    monkeypatch.setattr(pae, "plot_evaluation_from_files", fake_plot_evaluation_from_files)

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("", encoding="utf-8")

    run_root = tmp_path / "research_runs"
    argv = [
        "--run-name",
        "with_abnormal_plots",
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
        "--abnormal-comparison-report",
        str(report_path),
        "--abnormal-dataset",
        "porto_hoser",
    ]

    monkeypatch.setattr("sys.argv", ["tools/run_research_eval.py", *argv])
    assert tool.main() == 0
    assert called["ok"] is True

    run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    done = json.loads((run_dir / "done.json").read_text(encoding="utf-8"))
    assert done["abnormal"]["comparison_report"] == str(report_path)
    assert done["abnormal"]["plots_dir"] is not None
    assert (Path(done["abnormal"]["plots_dir"]) / "abnormality_reproduction_rates.png").exists()


@pytest.mark.parametrize("run_name", ["", "   ", "$$$", "a/b:c", "my run"])
def test_safe_name_never_empty(run_name):
    from tools.run_research_eval import _safe_name

    assert _safe_name(run_name)


def test_run_can_skip_baseline_and_plot_lmtad(tmp_path, monkeypatch):
    """Targets can be evaluated against an existing baseline eval without re-running baseline."""

    from tools import run_research_eval as tool
    from tools import nonlinear_coefficient as nl
    from tools import evaluate_dataset_with_lmtad as eval_tool

    baseline_dir = tmp_path / "Beijing"
    target_dir = tmp_path / "Beijing_abnormal_3_detectable_dr"
    _write_min_dataset_dir(baseline_dir)
    _write_min_dataset_dir(target_dir)

    # Nonlinear: deterministic.
    monkeypatch.setattr(
        nl, "load_road_lengths_m", lambda *_a, **_k: {1: 1.0, 2: 1.0, 3: 1.0}
    )
    monkeypatch.setattr(nl, "load_outgoing_edges", lambda *_a, **_k: {1: [2], 2: [3]})
    monkeypatch.setattr(nl, "non_linear_coefficient", lambda *_a, **_k: 1.25)

    # Create an existing baseline eval file.
    baseline_eval = tmp_path / "baseline_eval.json"
    baseline_eval.write_text(
        json.dumps({"train": {"log_perplexity_values": [0.1, 0.2]}}),
        encoding="utf-8",
    )

    # LM-TAD: ensure we only evaluate the target.
    calls = {"n": 0}

    def fake_evaluate_splits(*, output_dir: Path, write_baseline: bool, baseline_eval=None, sample_frac: float, **kwargs):
        calls["n"] += 1
        # When skipping baseline, we should never write baseline.
        assert write_baseline is False
        assert baseline_eval is not None
        assert float(sample_frac) == 0.5
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation_results.json").write_text(
            json.dumps({"train": {"outlier_method": "baseline_quantile", "outlier_rate": 0.1}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(eval_tool, "evaluate_splits", fake_evaluate_splits)

    # Plotting: avoid importing heavy matplotlib.
    plotted = {"datasets": []}

    def fake_plot(*, eval_dir: Path, out_dir: Path, splits: list[str]):
        plotted["datasets"].append((eval_dir.name, out_dir.name, tuple(splits)))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "lmtad_eval_train.png").write_text("x", encoding="utf-8")

    monkeypatch.setattr(tool, "_plot_lmtad_results", fake_plot)

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("", encoding="utf-8")

    run_root = tmp_path / "research_runs"
    argv = [
        "--run-name",
        "targets_only",
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
        "--skip-baseline",
        "--baseline-eval",
        str(baseline_eval),
        "--sample-frac",
        "0.1",
        "--sample-frac-target",
        "0.5",
        "--plot-lmtad",
    ]

    monkeypatch.setattr("sys.argv", ["tools/run_research_eval.py", *argv])
    assert tool.main() == 0

    # Only one LM-TAD evaluation (target) should have happened.
    assert calls["n"] == 1

    run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    done = json.loads((run_dir / "done.json").read_text(encoding="utf-8"))
    assert done["baseline_eval_used"] == str(baseline_eval)
    assert done["lmtad_plots"]

    # Baseline LM-TAD outputs should not exist under this run.
    assert not (run_dir / "lmtad" / baseline_dir.name).exists()
    assert (run_dir / "lmtad" / target_dir.name / "evaluation_results.json").exists()

    # Plot should have been produced for the target.
    assert any(name == target_dir.name for (name, _out, _splits) in plotted["datasets"])
