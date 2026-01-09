import numpy as np

from tools.teacher_separability import (
    auroc,
    cliffs_delta_from_auroc,
    cohens_d,
    recall_at_top_frac,
    wasserstein_1d,
)


def test_auroc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 10.0, 11.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert auroc(scores, labels) == 1.0


def test_auroc_worst_separation() -> None:
    scores = np.array([10.0, 11.0, 0.1, 0.2], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert auroc(scores, labels) == 0.0


def test_cliffs_delta_matches_auroc_relation() -> None:
    assert cliffs_delta_from_auroc(0.5) == 0.0
    assert cliffs_delta_from_auroc(1.0) == 1.0
    assert cliffs_delta_from_auroc(0.0) == -1.0


def test_cohens_d_sign() -> None:
    pos = np.array([2.0, 2.0, 3.0], dtype=np.float64)
    neg = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    assert cohens_d(pos, neg) > 0.0


def test_wasserstein_zero_when_identical() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    assert wasserstein_1d(x, y) == 0.0


def test_recall_at_top_frac_basic() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    labels = np.array([0, 1, 0, 1], dtype=bool)
    # Top 50% are scores 3.0 and 2.0 => labels [1,0], so recall = 1/2
    assert np.isclose(recall_at_top_frac(scores, labels, 0.5), 0.5)
