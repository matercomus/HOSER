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

import json
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

        if verify_hw is not None:
            vh, vw = int(verify_hw[0]), int(verify_hw[1])
            if (self.grid_h, self.grid_w) != (vh, vw):
                raise ValueError(
                    f"Grid dimension mismatch: computed {(self.grid_h,self.grid_w)} vs teacher {(vh,vw)}"
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


