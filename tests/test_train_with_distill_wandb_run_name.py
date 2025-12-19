"""Unit tests for WandB run name resolution in `train_with_distill.py`.

These are pure logic tests and do not require WandB connectivity.
"""

from __future__ import annotations

import pytest


def _import_train_with_distill():
    """Import train_with_distill with a clear skip if heavy deps are missing."""

    try:
        import train_with_distill

        return train_with_distill
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"train_with_distill import unavailable: {exc}")


def test_resolve_wandb_run_name_cli_override_wins() -> None:
    """CLI override should take highest precedence."""

    train_with_distill = _import_train_with_distill()

    got = train_with_distill._resolve_wandb_run_name(
        dataset_name="porto_hoser_abnormal_3",
        batch_size=64,
        accum_steps=2,
        config_run_name="from_config",
        cli_run_name="from_cli",
    )
    assert got == "from_cli"


def test_resolve_wandb_run_name_uses_config_when_no_cli() -> None:
    """Config run name should be used when CLI override is not provided."""

    train_with_distill = _import_train_with_distill()

    got = train_with_distill._resolve_wandb_run_name(
        dataset_name="Beijing_abnormal_3",
        batch_size=32,
        accum_steps=1,
        config_run_name="cfg_name",
        cli_run_name=None,
    )
    assert got == "cfg_name"


def test_resolve_wandb_run_name_falls_back_to_default() -> None:
    """Default derived name should be used when neither CLI nor config is set."""

    train_with_distill = _import_train_with_distill()

    got = train_with_distill._resolve_wandb_run_name(
        dataset_name="Beijing_abnormal_3",
        batch_size=16,
        accum_steps=4,
        config_run_name="",
        cli_run_name=None,
    )
    assert got == "Beijing_abnormal_3_b16_acc4"
