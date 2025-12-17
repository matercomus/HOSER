"""Tests for scenario-cross-model variant grouping helpers."""

from pathlib import Path

from tools.trajectory_variant_grouping import (
    VariantAxes,
    VariantGroupKey,
    build_cross_model_output_dir,
    collect_variant_seed_groups,
)


def test_beijing_no_axes_groups_by_seed_only():
    axes, groups = collect_variant_seed_groups(
        ["vanilla_seed42", "distilled_seed42"], dataset_has_phases=False
    )

    assert axes == VariantAxes(
        include_phase=False, include_lambda=False, include_abnormal=False
    )

    assert groups == {
        VariantGroupKey(phase_label=None, lambda_label=None, abnormal_label=None): {
            "seed42": {"distill": "distilled_seed42", "vanilla": "vanilla_seed42"}
        }
    }


def test_beijing_l1_axis_splits_distilled_groups():
    axes, groups = collect_variant_seed_groups(
        ["vanilla_seed42", "distilled_seed42", "distilled_l1_seed42"],
        dataset_has_phases=False,
    )

    assert axes.include_lambda is True
    assert axes.include_abnormal is False
    assert axes.include_phase is False

    assert set(groups.keys()) == {
        VariantGroupKey(phase_label=None, lambda_label="default", abnormal_label=None),
        VariantGroupKey(phase_label=None, lambda_label="L1", abnormal_label=None),
    }

    assert (
        groups[
            VariantGroupKey(
                phase_label=None, lambda_label="default", abnormal_label=None
            )
        ]["seed42"]["distill"]
        == "distilled_seed42"
    )

    assert (
        groups[
            VariantGroupKey(phase_label=None, lambda_label="L1", abnormal_label=None)
        ]["seed42"]["distill"]
        == "distilled_l1_seed42"
    )


def test_beijing_l0p001_axis_present_when_explicit_models_exist():
    axes, groups = collect_variant_seed_groups(
        [
            "vanilla_seed42",
            "distilled_seed42",
            "distilled_l0p001_seed42",
            "distilled_l1_seed42",
        ],
        dataset_has_phases=False,
    )

    assert axes.include_lambda is True
    assert set(groups.keys()) == {
        VariantGroupKey(phase_label=None, lambda_label="default", abnormal_label=None),
        VariantGroupKey(phase_label=None, lambda_label="L0p001", abnormal_label=None),
        VariantGroupKey(phase_label=None, lambda_label="L1", abnormal_label=None),
    }


def test_abnormal_axis_splits_normal_vs_abnormal():
    axes, groups = collect_variant_seed_groups(
        [
            "vanilla_seed42",
            "distilled_seed42",
            "vanilla_abnormal_seed42",
            "distilled_abnormal_seed42",
        ],
        dataset_has_phases=False,
    )

    assert axes.include_abnormal is True
    assert axes.include_lambda is False
    assert axes.include_phase is False

    assert set(groups.keys()) == {
        VariantGroupKey(phase_label=None, lambda_label=None, abnormal_label="normal"),
        VariantGroupKey(phase_label=None, lambda_label=None, abnormal_label="abnormal"),
    }


def test_porto_phase_axis_present_when_distill_phase_models_exist():
    axes, groups = collect_variant_seed_groups(
        ["vanilla_seed42", "distill_phase2_seed42"], dataset_has_phases=True
    )

    assert axes.include_phase is True
    assert axes.include_lambda is False
    assert axes.include_abnormal is False

    assert groups == {
        VariantGroupKey(phase_label="phase2", lambda_label=None, abnormal_label=None): {
            "seed42": {
                "distill": "distill_phase2_seed42",
                "vanilla": "vanilla_seed42",
            }
        }
    }


def test_build_output_dir_omits_missing_axes():
    base = Path("/tmp/figures/trajectories")

    output_dir = build_cross_model_output_dir(
        base_output_dir=base,
        od_type="train",
        scenario="rare_turns",
        axes=VariantAxes(
            include_phase=False, include_lambda=True, include_abnormal=True
        ),
        group_key=VariantGroupKey(
            phase_label=None, lambda_label="L1", abnormal_label="abnormal"
        ),
        subfolder="seed42",
    )

    assert output_dir.as_posix().endswith(
        "scenario_cross_model/train/rare_turns/abnormal/L1/seed42"
    )
