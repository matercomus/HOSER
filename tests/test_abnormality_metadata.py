"""Tests for `tools.abnormality_metadata`.

These tests assert the parsing contract for abnormal dataset CSV rows and the
stable metadata normalization used for grouping and display.
"""

from __future__ import annotations

import pytest

from tools.abnormality_metadata import (
    AbnormalityMetadata,
    build_abnormality_metadata,
    parse_abnormality_info,
    parse_rid_list,
)


def test_parse_rid_list_comma_separated() -> None:
    assert parse_rid_list("1,2,3") == [1, 2, 3]


def test_parse_rid_list_python_literal() -> None:
    assert parse_rid_list("[1, 2, 3]") == [1, 2, 3]


def test_parse_abnormality_info_normal_returns_none() -> None:
    assert parse_abnormality_info("normal") is None


def test_parse_abnormality_info_requires_real() -> None:
    with pytest.raises(ValueError, match="missing 'real'"):
        parse_abnormality_info("{'type': 'detour'}")


def test_build_abnormality_metadata_defaults() -> None:
    meta = build_abnormality_metadata({"real": {"rid_list": "1", "time_list": "t"}})
    assert isinstance(meta, AbnormalityMetadata)
    assert meta.abnormal_type == "unknown"
    assert meta.level == "unknown"
    assert meta.strength == "weak"


def test_build_abnormality_metadata_normalizes_fields() -> None:
    info = {
        "type": "route_switch",
        "level": "high",
        "strength": "strong",
        "real": {"rid_list": "1", "time_list": "t"},
    }
    meta = build_abnormality_metadata(info)
    assert meta.abnormal_type == "route_switch"
    assert meta.level == "high"
    assert meta.strength == "strong"
    assert "Route Switch" in meta.display_name
    assert meta.group_key == "route_switch/high/strong"
