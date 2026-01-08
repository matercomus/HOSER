from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _write_split_csv(path: Path, *, rows: int, abnormal_rows: int, abnormal_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rid_list", "abnormality_info"])
        writer.writeheader()
        for i in range(rows):
            if i < abnormal_rows:
                ab = f"{{'type': '{abnormal_type}'}}"
            else:
                ab = "normal"
            writer.writerow({"rid_list": "1,2,3", "abnormality_info": ab})


def test_run_lmtad_decision_benchmark_smoke(tmp_path, monkeypatch):
    from tools import evaluate_dataset_with_lmtad as eval_mod

    # Create three target dataset dirs.
    targets = []
    for name, ab_type in [
        ("A", "detour"),
        ("B", "route_switch"),
        ("C", "detour"),
    ]:
        d = tmp_path / f"target_{name}"
        _write_split_csv(d / "train.csv", rows=12, abnormal_rows=6, abnormal_type=ab_type)
        (d / "roadmap.geo").write_text("", encoding="utf-8")
        targets.append(d)

    # Baseline eval file (used to set thresholds)
    baseline_eval_dir = tmp_path / "baseline_eval"
    baseline_eval_dir.mkdir()
    baseline_scores = np.linspace(0.0, 2.0, 100, dtype=np.float64)
    (baseline_eval_dir / "baseline_eval.json").write_text(
        json.dumps({"train": {"log_perplexity_values": baseline_scores.tolist()}}),
        encoding="utf-8",
    )

    # Required ckpt/repo placeholders.
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("", encoding="utf-8")
    lmtad_repo = tmp_path / "LMTAD"
    lmtad_repo.mkdir()

    class FakeTeacher:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(eval_mod, "LMTADTeacher", FakeTeacher)
    monkeypatch.setattr(eval_mod, "_create_grid_mapper", lambda **kwargs: (None, (64, 64)))

    def fake_evaluate_csv(*, csv_path, **kwargs):
        # Return scores aligned with CSV rows.
        with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        # Make first half low, second half high to give separation.
        scores = np.concatenate(
            [np.linspace(0.0, 0.2, n // 2), np.linspace(3.0, 3.2, n - n // 2)]
        ).astype(np.float64)
        within = np.zeros_like(scores, dtype=np.float32)
        return scores, within, int(scores.size)

    monkeypatch.setattr(eval_mod, "_evaluate_csv", fake_evaluate_csv)

    from tools.run_lmtad_decision_benchmark import main as run_main

    out_dir = tmp_path / "out"

    import sys

    old = sys.argv
    try:
        sys.argv = [
            "run_lmtad_decision_benchmark.py",
            "--name",
            "demo",
            "--out-dir",
            str(out_dir),
            "--baseline-eval",
            str(baseline_eval_dir),
            "--baseline-split",
            "train",
            "--target-data-dirs",
            ",".join(str(t) for t in targets),
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
            "--q",
            "0.90,0.95",
        ]
        run_main()
    finally:
        sys.argv = old

    root = out_dir / "demo"
    assert (root / "analysis" / "summary.json").exists()
    assert (root / "analysis" / "summary.md").exists()

    # Per dataset artifacts
    for d in targets:
        ds = d.name
        assert (root / "analysis" / ds / "metrics.json").exists()
        assert (root / "analysis" / ds / "report.md").exists()
        assert (root / "analysis" / ds / "plots" / "score_hist.png").exists()
        assert (root / "analysis" / ds / "plots" / "score_by_type_box.png").exists()

        metrics = json.loads((root / "analysis" / ds / "metrics.json").read_text(encoding="utf-8"))
        for bucket_name in ["baseline_quantile", "topk_matched"]:
            bucket = metrics.get(bucket_name)
            assert isinstance(bucket, dict)
            assert bucket, f"Expected non-empty {bucket_name}"
            for _, entry in bucket.items():
                assert "f1" in entry
