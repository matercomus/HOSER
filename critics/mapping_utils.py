"""Mapping utilities for road_id -> token mappings.

Provides robust loading and normalization for mapping files which historically
have appeared in multiple formats (plain int values, nested dicts with
`target_road_id`/`token` keys, etc.). The loader returns a NumPy array of
dtype int64 where the index is the HOSER road id and the value is the LM-TAD
token id. Missing entries are set to -1 to preserve safety for consumers.

This module is designed to be small, well-tested, and dependency-light.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np


def _extract_token_from_value(value: Any) -> int:
    """Extract a token integer from a JSON value.

    Accepts:
      - int
      - str that can be parsed as int
      - dict with common keys like 'target_road_id', 'token', 'token_id'

    Raises ValueError if no integer token can be extracted.
    """
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, dict):
        # Common candidate keys
        for key in ("target_road_id", "token", "token_id", "road_token", "id"):
            if key in value:
                v = value[key]
                if isinstance(v, int):
                    return int(v)
                if isinstance(v, str) and v.isdigit():
                    return int(v)
        # Do NOT attempt to guess token from arbitrary dict values (e.g. distance_m)
        # Only accept well-known keys above. This avoids accidentally extracting
        # unrelated integers from metadata fields.
    raise ValueError(f"Cannot extract token from value: {value}")


def load_road_to_token_mapping(
    obj: Union[Path, str, Dict[str, Any], Dict[int, Any]],
) -> np.ndarray:
    """Load and normalize a road_id -> token mapping.

    Args:
        obj: Path to JSON file or an already-parsed mapping dict. Keys are
             expected to be road ids (ints or numeric strings). Values can be
             ints or nested dicts containing the token under common keys.

    Returns:
        np.ndarray of dtype int64 where index==road_id and value==token_id.
        Missing indices up to max road id are filled with -1.

    Raises:
        ValueError on malformed input.
    """
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():
            raise ValueError(f"Mapping file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
    elif isinstance(obj, dict):
        data = obj
    else:
        raise ValueError("Unsupported input type for mapping loader")

    if not isinstance(data, dict):
        raise ValueError("Mapping JSON must be an object/dictionary")

    # Convert keys to integer road ids
    entries: Dict[int, int] = {}
    for k, v in data.items():
        try:
            rid = int(k)
        except Exception:
            # Skip keys that are not numeric
            continue
        try:
            token = _extract_token_from_value(v)
        except ValueError:
            # If unable to extract token, mark as -1 (invalid)
            token = -1
        entries[rid] = token

    if not entries:
        # Empty mapping -> return empty array
        return np.array([], dtype=np.int64)

    max_rid = max(entries.keys())
    arr = np.full((max_rid + 1,), -1, dtype=np.int64)
    for rid, token in entries.items():
        arr[rid] = int(token)
    return arr
