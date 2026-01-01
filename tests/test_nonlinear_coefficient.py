"""Unit tests for `tools.nonlinear_coefficient`.

These tests validate the metric definition:

  coefficient = (trajectory length) / (shortest path length)

using a tiny synthetic road graph so tests stay fast and deterministic.
"""

from tools.nonlinear_coefficient import (
    _parse_rid_list,
    non_linear_coefficient,
    shortest_path_length_m,
    trajectory_length_m,
)


def test_parse_rid_list_comma_separated():
    """Parses comma-separated strings."""
    assert _parse_rid_list("1,2,3") == [1, 2, 3]
    assert _parse_rid_list(" 1, 2 , 3 ") == [1, 2, 3]


def test_parse_rid_list_python_literal_ints():
    """Parses Python list-literal of ints."""
    assert _parse_rid_list("[1, 2, 3]") == [1, 2, 3]


def test_parse_rid_list_python_literal_pairs():
    """Parses Python list-literal of pairs (rid, time)."""
    assert _parse_rid_list("[(10, 0), (11, 1), (12, 2)]") == [10, 11, 12]


def test_shortest_path_length_includes_start_and_entered_roads():
    """Shortest path length includes start road length plus entered roads."""
    road_len_m = {1: 10.0, 2: 10.0, 3: 10.0}
    outgoing = {1: [2], 2: [3]}

    # Only path is 1 -> 2 -> 3, length = 10 + 10 + 10
    assert shortest_path_length_m(
        start_road=1, end_road=3, outgoing=outgoing, road_len_m=road_len_m
    ) == 30.0


def test_non_linear_coefficient_is_one_for_shortest_path_trajectory():
    """Coefficient is 1.0 when trajectory matches the shortest path."""
    road_len_m = {1: 10.0, 2: 10.0, 3: 10.0}
    outgoing = {1: [3, 2], 2: [3]}

    # Direct 1->3 exists and is shortest: 10 + 10 = 20
    assert shortest_path_length_m(
        start_road=1, end_road=3, outgoing=outgoing, road_len_m=road_len_m
    ) == 20.0

    traj = [1, 3]
    assert trajectory_length_m(traj, road_len_m) == 20.0
    assert non_linear_coefficient(traj, outgoing=outgoing, road_len_m=road_len_m) == 1.0


def test_non_linear_coefficient_increases_with_detour():
    """Coefficient > 1.0 for a longer-than-shortest detour trajectory."""
    road_len_m = {1: 10.0, 2: 10.0, 3: 10.0}
    outgoing = {1: [3, 2], 2: [3]}

    # Shortest is 1->3 (20m) but trajectory takes 1->2->3 (30m).
    traj = [1, 2, 3]
    assert trajectory_length_m(traj, road_len_m) == 30.0
    assert non_linear_coefficient(traj, outgoing=outgoing, road_len_m=road_len_m) == 1.5


def test_non_linear_coefficient_returns_none_when_unreachable():
    """Returns None when no path exists between endpoints."""
    road_len_m = {1: 10.0, 2: 10.0, 3: 10.0}
    outgoing = {1: [2]}  # 3 is unreachable

    assert (
        non_linear_coefficient([1, 2, 3], outgoing=outgoing, road_len_m=road_len_m)
        is None
    )
