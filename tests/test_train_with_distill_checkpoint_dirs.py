"""Regression tests for checkpoint/model saving path robustness.

These tests ensure `train_with_distill.py` creates parent directories before
saving and that output directories are computed stably (independent of CWD).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_train_with_distill():
    """Import train_with_distill with a clear skip if heavy deps are missing."""

    try:
        import train_with_distill

        return train_with_distill
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"train_with_distill import unavailable: {exc}")


def test_ensure_parent_dir_creates_missing_parents(tmp_path: Path) -> None:
    """Ensures saving to a deep file path won't fail due to missing parents."""

    train_with_distill = _import_train_with_distill()

    target = tmp_path / "a" / "b" / "checkpoint_latest.pth"
    assert not target.parent.exists()

    train_with_distill._ensure_parent_dir(target)

    assert target.parent.is_dir()


def test_get_output_dirs_respects_explicit_project_root(tmp_path: Path) -> None:
    """Ensures callers can override project root (useful for tests)."""

    train_with_distill = _import_train_with_distill()

    fake_root = tmp_path / "proj"
    save_dir, tb_dir, log_dir = train_with_distill._get_output_dirs(
        dataset_name="porto_hoser_abnormal_3",
        seed=42,
        dir_suffix="distill_lambda0p5",
        project_root=fake_root,
    )

    assert save_dir == fake_root / "save" / "porto_hoser_abnormal_3" / "seed42_distill_lambda0p5"
    assert tb_dir == fake_root / "tensorboard_log" / "porto_hoser_abnormal_3" / "seed42_distill_lambda0p5"
    assert log_dir == fake_root / "log" / "porto_hoser_abnormal_3" / "seed42_distill_lambda0p5"


def test_get_output_dirs_is_stable_under_cwd_change(tmp_path: Path, monkeypatch) -> None:
    """Ensures output directories are stable even if CWD changes."""

    train_with_distill = _import_train_with_distill()

    monkeypatch.chdir(tmp_path)

    save_dir, tb_dir, log_dir = train_with_distill._get_output_dirs(
        dataset_name="Beijing_abnormal_3",
        seed=1,
        dir_suffix="vanilla",
    )

    module_root = Path(train_with_distill.__file__).resolve().parent

    assert save_dir.is_absolute()
    assert tb_dir.is_absolute()
    assert log_dir.is_absolute()

    assert save_dir == module_root / "save" / "Beijing_abnormal_3" / "seed1_vanilla"
    assert tb_dir == module_root / "tensorboard_log" / "Beijing_abnormal_3" / "seed1_vanilla"
    assert log_dir == module_root / "log" / "Beijing_abnormal_3" / "seed1_vanilla"
