"""Abnormality metadata and parsing utilities.

This module is the dataset-side analogue of `tools/model_detection.py`.
It centralizes:
- Parsing of `abnormality_info` rows produced by `generate_hoser_abnormalities.py`
- Parsing of `rid_list` values (comma-separated or Python list literals)
- Normalized metadata for grouping, display names, and consistent styling

The goal is to keep dataset perturbation tooling decoupled from eval directories
and model-generated outputs.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Plot styling semantics used across perturbation visualizations.
#
# - "dirty_shared": parts of the abnormal/dirty trajectory that align to the
#   clean/real trajectory.
# - "dirty_perturbed": parts of the abnormal/dirty trajectory that differ.
# - "clean_missing": parts of the clean/real trajectory that are absent from
#   the abnormal/dirty trajectory (rendered with reduced opacity).
#
# These are intentionally fixed to keep a consistent visual language.
DIRTY_SHARED_COLOR = "#2ecc71"  # green
DIRTY_PERTURBED_COLOR = "#e74c3c"  # red
CLEAN_MISSING_ALPHA = 0.5


@dataclass(frozen=True)
class AbnormalityMetadata:
    """Normalized abnormality metadata extracted from an abnormal row."""

    abnormal_type: str
    level: str
    strength: str

    @property
    def display_name(self) -> str:
        """Return a human-friendly display name."""

        type_display = self.abnormal_type.replace("_", " ").title()
        level_display = self.level.replace("_", " ").title()
        strength_display = self.strength.replace("_", " ").title()
        return f"{type_display} ({level_display}, {strength_display})"

    @property
    def group_key(self) -> str:
        """Return a stable grouping key suitable for directory names."""

        return "/".join(
            _slugify(token)
            for token in (self.abnormal_type, self.level, self.strength)
            if token
        )


def parse_rid_list(value: Any) -> list[int]:
    """Parse road ID sequences from either list-literal or comma-string.

    Supports:
    - Python list literals: "[1, 2, 3]"
    - Comma-separated strings: "1,2,3"

    Args:
        value: The CSV cell value.

    Returns:
        List of integer road IDs.

    Raises:
        TypeError: if `value` is not a string/list.
        ValueError: if parsing fails.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [int(x) for x in value]

    if not isinstance(value, str):
        raise TypeError(f"rid_list must be str/list, got {type(value).__name__}")

    text = value.strip()
    if text == "":
        return []

    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError("rid_list literal did not parse to list")
        return [int(x) for x in parsed]

    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_abnormality_info(value: Any) -> Optional[Dict[str, Any]]:
    """Parse abnormality_info; returns None for normal rows.

    Contract (from `generate_hoser_abnormalities.py`):
    - Normal rows: "normal" (string)
    - Abnormal rows: Python-literal dict string containing a mandatory `real`
      field with the clean reference trajectory.

    Args:
        value: CSV cell value.

    Returns:
        Parsed abnormality dict, or None if normal.

    Raises:
        ValueError: if abnormality_info is malformed or missing required fields.
    """

    if value is None:
        return None

    text = str(value).strip()
    if text.lower() == "normal":
        return None

    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError("abnormality_info did not parse to dict")
    if "real" not in parsed:
        raise ValueError("abnormality_info missing 'real' field")
    return parsed


def build_abnormality_metadata(info: Dict[str, Any]) -> AbnormalityMetadata:
    """Normalize an abnormality info dict into stable metadata."""

    abnormal_type = str(info.get("type") or "unknown").strip().lower()
    level = str(info.get("level") or "unknown").strip().lower()

    strength_raw = str(info.get("strength") or "").strip().lower()
    strength = "strong" if strength_raw == "strong" else "weak"

    return AbnormalityMetadata(
        abnormal_type=abnormal_type or "unknown",
        level=level or "unknown",
        strength=strength,
    )


_slug_re = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    token = str(text).strip().lower()
    token = _slug_re.sub("_", token)
    return token.strip("_") or "unknown"
