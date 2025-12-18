"""Trajectory diff and alignment utilities for perturbation visualization.

This module classifies which parts of a perturbed (dirty) trajectory match the
clean (real) trajectory and which parts differ.

Primary consumer: dataset-centric perturbation visualization tooling.

Design goals:
- Pure + testable (no plotting, no file I/O)
- Extensible: metadata-first heuristics per perturbation type, with a robust
  LCS fallback for unknown/future perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple


CleanNodeLabel = Literal["shared", "missing"]
DirtyNodeLabel = Literal["shared", "perturbed"]
SegmentLabel = Literal["shared", "missing", "perturbed"]


@dataclass(frozen=True)
class AlignmentResult:
    """Alignment labels for clean vs dirty trajectories."""

    clean_node_labels: List[CleanNodeLabel]
    dirty_node_labels: List[DirtyNodeLabel]

    clean_segment_labels: List[CleanNodeLabel]
    dirty_segment_labels: List[DirtyNodeLabel]

    meta: Dict[str, Any]


def align_trajectories(
    clean_road_ids: Sequence[int],
    dirty_road_ids: Sequence[int],
    abnormality_info: Optional[Dict[str, Any]] = None,
) -> AlignmentResult:
    """Classify shared vs missing/perturbed parts of clean/dirty trajectories.

    Strategy:
    1) Apply metadata-guided labeling when reliable fields exist (route_switch,
       perturb substitutions).
    2) Run an LCS alignment fallback to ensure robust behavior for detours and
       unknown perturbation types.

    Returns:
        AlignmentResult with node+segment labels.
    """

    clean = list(clean_road_ids)
    dirty = list(dirty_road_ids)

    clean_labels: List[CleanNodeLabel] = ["missing"] * len(clean)
    dirty_labels: List[DirtyNodeLabel] = ["perturbed"] * len(dirty)

    debug: Dict[str, Any] = {
        "clean_len": len(clean),
        "dirty_len": len(dirty),
    }

    # Metadata-first heuristics.
    if abnormality_info:
        ab_type = str(abnormality_info.get("type") or "").strip().lower()

        if ab_type == "route_switch":
            _apply_route_switch_hint(clean_labels, dirty_labels, abnormality_info)
            debug["hint"] = "route_switch"

        elif ab_type == "perturb":
            _apply_perturb_hint(clean_labels, dirty_labels, abnormality_info)
            debug["hint"] = "perturb"

        elif ab_type == "detour":
            # Detours often change length; LCS is the reliable source.
            debug["hint"] = "detour"

    # LCS fallback (also repairs/overrides weak hints).
    clean_shared_idx, dirty_shared_idx = _lcs_shared_indices(clean, dirty)
    for idx in clean_shared_idx:
        clean_labels[idx] = "shared"
    for idx in dirty_shared_idx:
        dirty_labels[idx] = "shared"

    clean_segment_labels = _segment_labels_from_nodes(clean_labels)
    dirty_segment_labels = _segment_labels_from_nodes(dirty_labels)

    debug.update(
        {
            "clean_shared": len(clean_shared_idx),
            "dirty_shared": len(dirty_shared_idx),
            "clean_missing": sum(1 for x in clean_labels if x == "missing"),
            "dirty_perturbed": sum(1 for x in dirty_labels if x == "perturbed"),
        }
    )

    return AlignmentResult(
        clean_node_labels=clean_labels,
        dirty_node_labels=dirty_labels,
        clean_segment_labels=clean_segment_labels,
        dirty_segment_labels=dirty_segment_labels,
        meta=debug,
    )


def _segment_labels_from_nodes(
    node_labels: Sequence[Literal["shared", "missing", "perturbed"]],
) -> List[Any]:
    """Convert per-node labels to per-segment labels.

    A segment inherits the non-shared label if either endpoint is non-shared.
    """

    if len(node_labels) < 2:
        return []

    seg_labels: List[Any] = []
    for left, right in zip(node_labels[:-1], node_labels[1:]):
        if left == "shared" and right == "shared":
            seg_labels.append("shared")
        else:
            # Prefer missing/perturbed over shared.
            # The node label set determines which non-shared value is possible.
            if left != "shared":
                seg_labels.append(left)
            else:
                seg_labels.append(right)

    return seg_labels


def _apply_route_switch_hint(
    clean_labels: List[CleanNodeLabel],
    dirty_labels: List[DirtyNodeLabel],
    info: Dict[str, Any],
) -> None:
    """Apply seg_range hint for route_switch perturbations.

    Generator contract: seg_range is (start, end) indices in the *original*
    trajectory that were replaced in the dirty trajectory.

    We conservatively mark those dirty indices as perturbed and those clean
    indices as missing. LCS alignment will later mark any truly shared roads as
    shared.
    """

    seg_range = info.get("seg_range")
    if not isinstance(seg_range, tuple) or len(seg_range) != 2:
        return

    start, end = seg_range
    if not isinstance(start, int) or not isinstance(end, int):
        return

    start = max(0, start)
    end = max(start, end)

    for idx in range(start, min(end, len(clean_labels))):
        clean_labels[idx] = "missing"

    for idx in range(start, min(end, len(dirty_labels))):
        dirty_labels[idx] = "perturbed"


def _apply_perturb_hint(
    clean_labels: List[CleanNodeLabel],
    dirty_labels: List[DirtyNodeLabel],
    info: Dict[str, Any],
) -> None:
    """Apply index-level perturb hints.

    Generator contract: perturbed is a list of tuples (idx, old, new).
    """

    perturbed = info.get("perturbed")
    if not isinstance(perturbed, list):
        return

    for entry in perturbed:
        if not isinstance(entry, (tuple, list)) or not entry:
            continue
        idx = entry[0]
        if not isinstance(idx, int):
            continue

        if 0 <= idx < len(dirty_labels):
            dirty_labels[idx] = "perturbed"
        if 0 <= idx < len(clean_labels):
            clean_labels[idx] = "missing"


def _lcs_shared_indices(
    clean: Sequence[int],
    dirty: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Return index lists of nodes that belong to an LCS alignment.

    This is O(n*m) DP which is fine for visualization samples.
    """

    n = len(clean)
    m = len(dirty)
    if n == 0 or m == 0:
        return [], []

    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        ci = clean[i]
        row = dp[i + 1]
        prev_row = dp[i]
        for j in range(m):
            if ci == dirty[j]:
                row[j + 1] = prev_row[j] + 1
            else:
                row[j + 1] = max(row[j], prev_row[j + 1])

    # Backtrack to recover one LCS alignment.
    i = n
    j = m
    clean_idx: List[int] = []
    dirty_idx: List[int] = []

    while i > 0 and j > 0:
        if clean[i - 1] == dirty[j - 1]:
            clean_idx.append(i - 1)
            dirty_idx.append(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    clean_idx.reverse()
    dirty_idx.reverse()
    return clean_idx, dirty_idx
