"""Helpers for grouping trajectory outputs into meaningful cross-model comparisons.

This module is intentionally pure and testable: it only inspects model identifiers
(via `tools.model_detection`) and returns grouping + output-path decisions.

Scenario-cross-model plots should compare exactly:
- Real trajectory (optional / handled by caller)
- One vanilla variant
- One distilled variant (one of the available variants)

Variant axes (phase, lambda, abnormal) are only used in directory structure when
there is evidence that axis is present in the evaluation workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tools.model_detection import (
    DEFAULT_SEED_TOKEN,
    ModelMetadata,
    build_model_metadata,
)


@dataclass(frozen=True)
class VariantAxes:
    """Which variant axes should be used in output directory structure."""

    include_phase: bool
    include_lambda: bool
    include_abnormal: bool


@dataclass(frozen=True)
class VariantGroupKey:
    """A concrete selection along enabled variant axes."""

    phase_label: Optional[str] = None
    lambda_label: Optional[str] = None
    abnormal_label: Optional[str] = None


def _is_abnormal(metadata: ModelMetadata) -> bool:
    return "abnormal" in metadata.base_model.lower()


def _infer_abnormal_label(metadata: ModelMetadata) -> str:
    return "abnormal" if _is_abnormal(metadata) else "normal"


def _infer_lambda_label(metadata: ModelMetadata) -> Optional[str]:
    """Return a stable label for distilled lambda variants.

        - Distilled L1 families map to "L1".
        - Distilled lambda=0.001 families map to "L0p001".
                - Distilled lambda=0.5 families map to "L0p5".
        - Plain distilled families (no explicit lambda token) map to "default".
    - Porto distillation phases do not map to a lambda label because they use
      phase-based training identifiers instead of lambda sweeps.
    """

    base_lower = metadata.base_model.lower()
    if base_lower.startswith("distill_phase"):
        return None

    if metadata.normalized_base != "distilled":
        return None

    if "l0p001" in base_lower or "lambda0.001" in base_lower:
        return "L0p001"

    if "l0p5" in base_lower or "lambda0.5" in base_lower or "lambda0p5" in base_lower:
        return "L0p5"

    if "l1" in base_lower:
        return "L1"

    return "default"


def _infer_phase_label(
    metadata: ModelMetadata, dataset_has_phases: bool
) -> Optional[str]:
    if not dataset_has_phases:
        return None

    base_lower = metadata.base_model.lower()
    if base_lower.startswith("distill_phase"):
        return metadata.phase_label

    return None


def derive_variant_axes(
    model_names: Iterable[str],
    *,
    dataset_has_phases: bool,
) -> VariantAxes:
    """Infer which axes are meaningful for a set of model identifiers."""

    metadatas: List[ModelMetadata] = [
        build_model_metadata(name)
        for name in model_names
        if name not in ("real", "unknown")
    ]

    phase_labels = {
        _infer_phase_label(metadata, dataset_has_phases)
        for metadata in metadatas
        if metadata.normalized_base == "distilled"
    }
    phase_labels.discard(None)

    lambda_labels = {
        _infer_lambda_label(metadata)
        for metadata in metadatas
        if metadata.normalized_base == "distilled"
    }
    lambda_labels.discard(None)

    abnormal_labels = {
        _infer_abnormal_label(metadata)
        for metadata in metadatas
        if metadata.normalized_base in {"vanilla", "distilled"}
    }

    include_phase = bool(phase_labels)
    include_lambda = len(lambda_labels) > 1
    include_abnormal = abnormal_labels == {"normal", "abnormal"}

    return VariantAxes(
        include_phase=include_phase,
        include_lambda=include_lambda,
        include_abnormal=include_abnormal,
    )


def collect_variant_seed_groups(
    models: Iterable[str],
    *,
    dataset_has_phases: bool,
) -> Tuple[VariantAxes, Dict[VariantGroupKey, Dict[str, Dict[str, str]]]]:
    """Return `VariantGroupKey -> seed -> {vanilla, distill}` for plotting.

    The caller can optionally add real trajectories separately.
    """

    model_names = [name for name in models if name not in ("real", "unknown")]
    axes = derive_variant_axes(model_names, dataset_has_phases=dataset_has_phases)

    vanilla_by_seed: Dict[Tuple[str, Optional[str]], List[str]] = {}
    distilled_by_group: Dict[VariantGroupKey, Dict[str, List[str]]] = {}

    for model_name in model_names:
        metadata = build_model_metadata(model_name)
        seed = metadata.seed_label or DEFAULT_SEED_TOKEN

        abnormal_label = (
            _infer_abnormal_label(metadata) if axes.include_abnormal else None
        )
        lambda_label = _infer_lambda_label(metadata) if axes.include_lambda else None
        phase_label = (
            _infer_phase_label(metadata, dataset_has_phases)
            if axes.include_phase
            else None
        )

        if metadata.normalized_base == "vanilla":
            vanilla_by_seed.setdefault((seed, abnormal_label), []).append(model_name)
            continue

        if metadata.normalized_base != "distilled":
            continue

        group_key = VariantGroupKey(
            phase_label=phase_label,
            lambda_label=lambda_label,
            abnormal_label=abnormal_label,
        )
        distilled_by_group.setdefault(group_key, {}).setdefault(seed, []).append(
            model_name
        )

    def select_primary(names: List[str]) -> Optional[str]:
        """Select a deterministic primary model from candidates.

        When multiple model identifiers match the same grouping criteria, choose
        the lexicographically-first name to keep output stable across runs.
        """
        return sorted(names)[0] if names else None

    def vanilla_candidates(seed: str, abnormal_label: Optional[str]) -> List[str]:
        """Return vanilla candidates for a seed, falling back to default.

        Strategy:
        1) Prefer exact seed match.
        2) If unavailable, fall back to DEFAULT_SEED_TOKEN for that same abnormal label.
        """
        direct = vanilla_by_seed.get((seed, abnormal_label))
        if direct:
            return direct
        fallback = vanilla_by_seed.get((DEFAULT_SEED_TOKEN, abnormal_label))
        if fallback:
            return fallback
        return []

    groups: Dict[VariantGroupKey, Dict[str, Dict[str, str]]] = {}

    for group_key, by_seed in distilled_by_group.items():
        for seed, distilled_models in by_seed.items():
            distilled_model = select_primary(distilled_models)
            if not distilled_model:
                continue

            vanilla_model = select_primary(
                vanilla_candidates(seed, group_key.abnormal_label)
            )
            if not vanilla_model:
                continue

            groups.setdefault(group_key, {})[seed] = {
                "distill": distilled_model,
                "vanilla": vanilla_model,
            }

    return axes, groups


def build_cross_model_output_dir(
    *,
    base_output_dir: Path,
    od_type: str,
    scenario: Optional[str],
    axes: VariantAxes,
    group_key: VariantGroupKey,
    subfolder: Optional[str],
) -> Path:
    """Build an output directory path, omitting axes that aren't present."""

    if scenario:
        output_dir = base_output_dir / "scenario_cross_model" / od_type / scenario
    else:
        output_dir = base_output_dir / "cross_model" / od_type

    if axes.include_abnormal and group_key.abnormal_label:
        output_dir = output_dir / group_key.abnormal_label

    if axes.include_lambda and group_key.lambda_label:
        output_dir = output_dir / group_key.lambda_label

    if axes.include_phase and group_key.phase_label:
        output_dir = output_dir / group_key.phase_label

    if subfolder:
        output_dir = output_dir / subfolder

    return output_dir
