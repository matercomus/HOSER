"""Tests for Phase B perturbation correction utilities."""

from __future__ import annotations

import math

import pytest

from tools.perturbation_correction import (
    _dtw_km,
    _parse_abnormality_info,
    _parse_rid_list,
)


def test_parse_rid_list_comma_separated() -> None:
    assert _parse_rid_list("1,2,3") == [1, 2, 3]


def test_parse_rid_list_python_literal() -> None:
    assert _parse_rid_list("[1, 2, 3]") == [1, 2, 3]


def test_parse_abnormality_info_normal_returns_none() -> None:
    assert _parse_abnormality_info("normal") is None


def test_parse_abnormality_info_requires_real() -> None:
    with pytest.raises(ValueError, match="missing 'real'"):
        _parse_abnormality_info("{'type': 'detour'}")


def test_dtw_km_identical_is_zeroish() -> None:
    road_gps = {
        1: (0.0, 0.0),
        2: (0.001, 0.0),
        3: (0.002, 0.0),
    }
    dist = _dtw_km([1, 2, 3], [1, 2, 3], road_gps)
    assert math.isfinite(dist)
    assert dist == pytest.approx(0.0, abs=1e-9)
