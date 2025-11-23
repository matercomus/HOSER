"""
Grid Mapper for LM-TAD Distillation
===================================

Purpose
-------
Map HOSER road IDs to LM-TAD grid tokens using the same centroid-to-grid
formula as in the LM-TAD preprocessing (convert_HOSER_to_LMTAD.py).

This module is standalone to keep the training code clean and readable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GridConfig:
    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float
    grid_size: float
    downsample_factor: int = 1


class GridMapper:
    """Vectorized road→grid mapper using road centroids.

    Parameters
    ----------
    boundary: GridConfig
        Geographic boundaries and grid parameters.
    road_centroids: np.ndarray
        Shape (N, 2) array of (lat, lng) for each road's centroid.
    verify_hw: Optional[Tuple[int,int]]
        Optional (height, width) to assert grid dimensions match teacher.
    """

    def __init__(
        self,
        boundary: GridConfig,
        road_centroids: np.ndarray,
        verify_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.cfg = boundary
        self.road_centroids = road_centroids.astype(np.float64, copy=False)

        # Compute base grid dims
        lat_span = max(0.0, float(self.cfg.max_lat - self.cfg.min_lat))
        lng_span = max(0.0, float(self.cfg.max_lng - self.cfg.min_lng))
        lat_grid_num = int(lat_span / self.cfg.grid_size) + 1
        lng_grid_num = int(lng_span / self.cfg.grid_size) + 1

        # Apply downsampling
        if self.cfg.downsample_factor > 1:
            lat_grid_num //= self.cfg.downsample_factor
            lng_grid_num //= self.cfg.downsample_factor
            lat_grid_num = max(lat_grid_num, 1)
            lng_grid_num = max(lng_grid_num, 1)

        self.grid_h = lat_grid_num
        self.grid_w = lng_grid_num

        # If verify_hw is provided and dimensions don't match, raise error
        # Boundaries should match training exactly - don't adjust them
        if verify_hw is not None:
            vh, vw = int(verify_hw[0]), int(verify_hw[1])
            if (self.grid_h, self.grid_w) != (vh, vw):
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Grid dimension mismatch: computed {(self.grid_h, self.grid_w)} vs teacher {(vh, vw)}. "
                    f"This indicates the boundaries used don't match training. "
                    f"Please use the exact boundaries from the converted LM-TAD data."
                )
                raise ValueError(
                    f"Grid dimension mismatch: computed {(self.grid_h, self.grid_w)} vs teacher {(vh, vw)}. "
                    f"Boundaries must match training exactly. Use boundaries from converted LM-TAD data."
                )

        # Validate provided centroids are within configured boundaries.
        # If any centroid falls outside the expected boundary, raise ValueError
        # to avoid silent clipping of off-grid coordinates which may indicate
        # broken or misaligned inputs (e.g., lat/lng switched, wrong CRS).
        if self.road_centroids.size > 0:
            min_lat, max_lat = float(self.cfg.min_lat), float(self.cfg.max_lat)
            min_lng, max_lng = float(self.cfg.min_lng), float(self.cfg.max_lng)

            lat_arr = self.road_centroids[:, 0]
            lng_arr = self.road_centroids[:, 1]
            # Detect any centroids outside specified range
            out_of_lat = np.logical_or(lat_arr < min_lat, lat_arr > max_lat)
            out_of_lng = np.logical_or(lng_arr < min_lng, lng_arr > max_lng)
            if np.any(np.logical_or(out_of_lat, out_of_lng)):
                logger = logging.getLogger(__name__)
                logger.error(
                    "Provided road centroids contain points outside the specified boundary. "
                    "This usually indicates invalid coordinate input or wrong order of lat/lng."
                )
                raise ValueError(
                    "Provided road centroids contain points outside the configured boundary."
                )

    def map_all(self) -> np.ndarray:
        """Return an array of grid tokens for each road.

        Returns
        -------
        np.ndarray
            Shape (N,), dtype=int64, each entry is the grid token id.
        """
        lat = self.road_centroids[:, 0]
        lng = self.road_centroids[:, 1]

        gi = np.floor((lat - self.cfg.min_lat) / self.cfg.grid_size).astype(np.int64)
        gj = np.floor((lng - self.cfg.min_lng) / self.cfg.grid_size).astype(np.int64)

        if self.cfg.downsample_factor > 1:
            gi //= self.cfg.downsample_factor
            gj //= self.cfg.downsample_factor

        gi = np.clip(gi, 0, self.grid_h - 1)
        gj = np.clip(gj, 0, self.grid_w - 1)

        tokens = gi * self.grid_w + gj
        return tokens.astype(np.int64, copy=False)


def map_roads_to_tokens(road_ids, road_to_token: np.ndarray):
    """Map a sequence of HOSER road IDs to LM-TAD grid tokens using a precomputed
    `road_to_token` array.

    Parameters
    ----------
    road_ids: Sequence[int]
        Iterable of integer road IDs (HOSER domain).
    road_to_token: np.ndarray
        Precomputed array of shape (num_roads,) mapping road_id -> token_id.

    Returns
    -------
    Tuple[List[int], List[int]]
        - mapped_tokens: list of int where invalid entries are set to -1
        - invalid_indices: list of indices in the input that were invalid

    Notes
    -----
    This helper is intentionally simple and defensive: it does not raise on
    out-of-range inputs but returns invalid indices for callers to decide how
    to handle the cases.
    """
    mapped = []
    invalid_indices = []
    n = int(len(road_to_token))
    for idx, rid in enumerate(road_ids):
        try:
            if not isinstance(rid, int) or rid < 0 or rid >= n:
                invalid_indices.append(idx)
                mapped.append(-1)
            else:
                mapped.append(int(road_to_token[rid]))
        except Exception:
            invalid_indices.append(idx)
            mapped.append(-1)
    return mapped, invalid_indices
