"""Pytest configuration for HOSER tests.

This module configures sys.path and provides test fixtures for faking the
LMTAD teacher. It needs to modify sys.path before importing test helpers so
it must execute early in the file.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402
from critics.lmtad_teacher import LMTADTeacher  # noqa: E402


def make_fake_lmtad_teacher(vocab_size=6167, grid_hw=(64, 64)):
    """Create a realistic fake LMTADTeacher mock for tests.

    Returns mock with spec of LMTADTeacher and configured methods.
    """
    fake = MagicMock(spec=LMTADTeacher)
    fake.vocab_size.return_value = vocab_size
    fake.get_grid_size_hw.return_value = grid_hw

    # Provide a predict_next_distribution stub for tests that exercise it
    def _predict_next_distribution(history_tokens):
        # Return shape (B, V) probabilities for next token, but keep simple
        import numpy as np

        B = history_tokens.shape[0]
        V = vocab_size
        # deterministic all ones for simplicity; cast to float
        return (np.ones((B, V), dtype=float), None)

    fake.predict_next_distribution.side_effect = _predict_next_distribution
    return fake


@pytest.fixture
def fake_lmtad_teacher():
    yield make_fake_lmtad_teacher()


@pytest.fixture
def patch_lmtad_teacher(monkeypatch, fake_lmtad_teacher):
    """Monkeypatch LMTADTeacher constructor across the codebase to use a fake teacher.

    Returns a function that can be used inside tests to perform the patching.
    """

    def _patch():
        # Patch in places where the real LMTADTeacher is imported
        monkeypatch.setattr(
            "critics.distill_hook.LMTADTeacher", lambda **kwargs: fake_lmtad_teacher
        )
        monkeypatch.setattr(
            "tools.evaluate_with_lmtad.LMTADTeacher",
            lambda **kwargs: fake_lmtad_teacher,
        )
        monkeypatch.setattr(
            "tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher",
            lambda **kwargs: fake_lmtad_teacher,
        )
        return fake_lmtad_teacher

    return _patch
