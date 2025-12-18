"""Tests for `tools.trajectory_diff` alignment behavior."""

from __future__ import annotations

from tools.trajectory_diff import align_trajectories


def test_align_identical_sequences_all_shared() -> None:
    res = align_trajectories([1, 2, 3], [1, 2, 3], abnormality_info=None)
    assert res.clean_node_labels == ["shared", "shared", "shared"]
    assert res.dirty_node_labels == ["shared", "shared", "shared"]
    assert res.clean_segment_labels == ["shared", "shared"]
    assert res.dirty_segment_labels == ["shared", "shared"]


def test_align_route_switch_marks_middle_missing_and_perturbed() -> None:
    clean = [1, 2, 3, 4, 5]
    dirty = [1, 2, 9, 10, 5]
    info = {
        "type": "route_switch",
        "level": "high",
        "seg_range": (2, 4),
        "real": {"rid_list": "1,2,3,4,5", "time_list": "t"},
    }
    res = align_trajectories(clean, dirty, abnormality_info=info)

    assert res.clean_node_labels == [
        "shared",
        "shared",
        "missing",
        "missing",
        "shared",
    ]
    assert res.dirty_node_labels == [
        "shared",
        "shared",
        "perturbed",
        "perturbed",
        "shared",
    ]


def test_align_perturb_substitution_marks_index() -> None:
    clean = [1, 2, 3]
    dirty = [1, 9, 3]
    info = {
        "type": "perturb",
        "level": "low",
        "perturbed": [(1, "2", "9")],
        "real": {"rid_list": "1,2,3", "time_list": "t"},
    }
    res = align_trajectories(clean, dirty, abnormality_info=info)
    assert res.clean_node_labels == ["shared", "missing", "shared"]
    assert res.dirty_node_labels == ["shared", "perturbed", "shared"]


def test_align_detour_insertion_marks_inserted_as_perturbed() -> None:
    clean = [1, 2, 3]
    dirty = [1, 2, 9, 3]
    info = {
        "type": "detour",
        "level": "high",
        "detour": ["9"],
        "real": {"rid_list": "1,2,3", "time_list": "t"},
    }
    res = align_trajectories(clean, dirty, abnormality_info=info)

    assert res.clean_node_labels == ["shared", "shared", "shared"]
    assert res.dirty_node_labels == ["shared", "shared", "perturbed", "shared"]
