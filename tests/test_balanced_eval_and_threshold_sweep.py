from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def test_build_balanced_rows_keeps_all_abnormals():
    from tools.make_balanced_eval_dataset import build_balanced_rows

    rows = [
        {"rid_list": "1,2,3", "abnormality_info": "normal"},
        {"rid_list": "1,2,3,4", "abnormality_info": "normal"},
        {"rid_list": "1,2,3,4,5,6,7,8,9,10", "abnormality_info": "normal"},
        {"rid_list": "1,2,3,4,5,6,7,8,9,10,11", "abnormality_info": "normal"},
        {"rid_list": "1,2,3", "abnormality_info": "{'type': 'detour'}"},
        {"rid_list": "1,2,3,4,5,6,7,8,9,10", "abnormality_info": "{'type': 'route_switch'}"},
    ]

    balanced, n_abn, n_norm = build_balanced_rows(
        rows=rows,
        normal_per_abnormal=1,
        length_bucket=5,
        seed=123,
        allow_replacement=False,
    )

    assert n_abn == 2
    assert n_norm == 2
    assert len(balanced) == 4

    # All abnormal rows must be present.
    ab_infos = [r["abnormality_info"] for r in balanced]
    assert "{'type': 'detour'}" in ab_infos
    assert "{'type': 'route_switch'}" in ab_infos


def _write_min_scored_csv(path: Path, abnormality_info: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rid_list", "abnormality_info"])
        writer.writeheader()
        for ab in abnormality_info:
            writer.writerow({"rid_list": "1,2,3", "abnormality_info": ab})


def test_threshold_sweep_metrics(tmp_path):
    from tools.analyze_lmtad_threshold_sweep import analyze

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Scores correspond to CSV row order.
    scores = [0.0, 2.0, 4.0, 6.0]
    (eval_dir / "evaluation_results.json").write_text(
        json.dumps({"train": {"log_perplexity_values": scores}}),
        encoding="utf-8",
    )

    csv_path = eval_dir / "train_sampled.csv"
    _write_min_scored_csv(
        csv_path,
        abnormality_info=["normal", "normal", "{'type': 'detour'}", "{'type': 'detour'}"],
    )

    baseline_eval = tmp_path / "baseline_eval.json"
    baseline_scores = [0.0, 1.0, 2.0, 3.0]
    baseline_eval.write_text(
        json.dumps({"train": {"log_perplexity_values": baseline_scores}}),
        encoding="utf-8",
    )

    summary = analyze(
        eval_dir=eval_dir,
        split="train",
        csv_path=csv_path,
        baseline_eval=baseline_eval,
        baseline_split="train",
        quantiles=[0.5],
        label_col="abnormality_info",
    )

    assert summary.n == 4
    assert summary.n_pos == 2
    assert summary.n_neg == 2

    assert summary.auroc == 1.0
    assert summary.auprc == 1.0

    m = summary.thresholds[0]
    assert m.quantile == 0.5
    assert np.isclose(m.threshold, np.quantile(np.asarray(baseline_scores), 0.5))

    # threshold=1.5 => preds for scores {2,4,6}
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 0, 1)
    assert np.isclose(m.recall, 1.0)
    assert np.isclose(m.precision, 2 / 3)
    assert np.isclose(m.fpr, 1 / 2)

    assert "detour" in summary.by_type
    assert summary.by_type["detour"]["n_pos"] == 2
