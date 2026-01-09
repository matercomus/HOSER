from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.plot_teacher_separability import (
    MetricSummary,
    plot_metric_bars,
    plot_precision_curve,
    plot_recall_curve,
)


def test_plotting_smoke(tmp_path: Path) -> None:
    summaries = [
        MetricSummary(
            name="A",
            n=10,
            prevalence=0.2,
            auc=(0.7, 0.6, 0.8),
            delta=(0.4, 0.2, 0.6),
            d=(0.5, 0.1, 0.9),
            w1=(0.3, 0.2, 0.4),
        ),
        MetricSummary(
            name="B",
            n=12,
            prevalence=0.1,
            auc=(0.8, 0.75, 0.85),
            delta=(0.6, 0.5, 0.7),
            d=(0.8, 0.6, 1.0),
            w1=(0.2, 0.15, 0.25),
        ),
    ]

    out_metrics = tmp_path / "metrics.png"
    plot_metric_bars(summaries=summaries, out_path=out_metrics)
    assert out_metrics.exists()

    top_fracs = np.array([0.01, 0.05, 0.10], dtype=np.float64)
    point_recalls = [np.array([0.02, 0.10, 0.20]), np.array([0.03, 0.12, 0.30])]
    recall_samples = [
        np.tile(point_recalls[0], (50, 1)),
        np.tile(point_recalls[1], (50, 1)),
    ]

    out_curve = tmp_path / "curve.png"
    plot_recall_curve(
        names=["A", "B"],
        top_fracs=top_fracs,
        point_recalls=point_recalls,
        recall_samples=recall_samples,
        ci=0.95,
        out_path=out_curve,
    )
    assert out_curve.exists()

    point_precisions = [np.array([0.2, 0.3, 0.4]), np.array([0.1, 0.2, 0.25])]
    precision_samples = [
        np.tile(point_precisions[0], (50, 1)),
        np.tile(point_precisions[1], (50, 1)),
    ]

    out_pcurve = tmp_path / "precision_curve.png"
    plot_precision_curve(
        names=["A", "B"],
        prevalence=[0.2, 0.1],
        top_fracs=top_fracs,
        point_precisions=point_precisions,
        precision_samples=precision_samples,
        ci=0.95,
        out_path=out_pcurve,
    )
    assert out_pcurve.exists()
