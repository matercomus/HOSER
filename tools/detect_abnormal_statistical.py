#!/usr/bin/env python3
"""
Wang et al. 2018 Statistical Abnormality Detection

This module implements the statistical abnormality detection methodology from:
Wang, Y., Qin, K., Chen, Y., & Zhao, P. (2018). Detecting Anomalous Trajectories
and Behavior Patterns Using Hierarchical Clustering from Taxi GPS Data.
ISPRS International Journal of Geo-Information, 7(1), 25.
https://doi.org/10.3390/ijgi7010025

Core Methodology:
- OD-pair-specific baselines learned from real trajectory data
- Four abnormal behavior patterns (Abp1-4)
- Hybrid threshold strategy: fixed (5km/5min) + statistical (2.5σ)
- Pattern classification: normal, temporal_delay, route_deviation, both

Usage:
    from tools.detect_abnormal_statistical import WangStatisticalDetector, WangConfig

    config = WangConfig.from_yaml("config/abnormal_detection_statistical.yaml")
    detector = WangStatisticalDetector(baselines, config, geo_df)
    results = detector.detect_abnormal_trajectories(trajectories_df)
"""

import json
import logging
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class WangConfig:
    """Configuration for Wang et al. 2018 statistical detection.

    Attributes:
        dataset: Name of the dataset being analyzed
        baseline_dataset: Dataset used for computing baselines (may differ for cross-dataset)
        L_rho_meters: Fixed threshold for route deviation (default: 5000m from paper)
        T_rho_seconds: Fixed threshold for temporal delay (default: 300s from paper)
        sigma_length: Statistical multiplier for length deviation (default: 2.5)
        sigma_time: Statistical multiplier for time deviation (default: 2.5)
        sigma_speed: Statistical multiplier for speed detection (default: 3.0)
        threshold_strategy: "fixed", "statistical", or "hybrid" (most stringent)
        min_samples_per_od: Minimum samples to use OD-specific baseline (default: 5)
        fallback_to_global: Use global stats when OD pair has insufficient samples
    """

    dataset: str
    baseline_dataset: str
    L_rho_meters: float = 5000.0
    T_rho_seconds: float = 300.0
    sigma_length: float = 2.5
    sigma_time: float = 2.5
    sigma_speed: float = 3.0
    threshold_strategy: str = "hybrid"
    min_samples_per_od: int = 5
    fallback_to_global: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "WangConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            WangConfig instance
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        wang_config = config.get("wang_statistical", {})

        return cls(
            dataset=config.get("dataset", "Unknown"),
            baseline_dataset=config.get(
                "baseline_dataset", config.get("dataset", "Unknown")
            ),
            L_rho_meters=wang_config.get("L_rho_meters", 5000.0),
            T_rho_seconds=wang_config.get("T_rho_seconds", 300.0),
            sigma_length=wang_config.get("sigma_length", 2.5),
            sigma_time=wang_config.get("sigma_time", 2.5),
            sigma_speed=wang_config.get("sigma_speed", 3.0),
            threshold_strategy=wang_config.get("threshold_strategy", "hybrid"),
            min_samples_per_od=wang_config.get("min_samples_per_od", 5),
            fallback_to_global=wang_config.get("fallback_to_global", True),
        )


# ============================================================================
# BASELINE STATISTICS LOADER
# ============================================================================


class BaselineStatistics:
    """Container for OD-pair baseline statistics.

    Loads and provides access to pre-computed baseline statistics for
    origin-destination pairs from real trajectory data.
    """

    def __init__(self, baselines_path: Path):
        """Load baseline statistics from JSON file.

        Args:
            baselines_path: Path to baselines JSON file (e.g., baselines/baselines_beijing.json)
        """
        logger.info(f"📊 Loading baseline statistics from {baselines_path}")

        with open(baselines_path, "r") as f:
            data = json.load(f)

        self.metadata = data["metadata"]
        self.coverage = data["coverage"]
        self.global_stats = data["global_statistics"]
        self.od_baselines = data["od_pair_baselines"]

        logger.info(
            f"✅ Loaded baselines for {self.coverage['total_od_pairs']:,} OD pairs"
        )
        logger.info(
            f"   OD pairs with ≥5 samples: {self.coverage['od_pairs_with_min_5_samples']:,} "
            f"({self.coverage['coverage_pct']:.1f}%)"
        )

    def get_od_baseline(
        self, origin: int, destination: int, min_samples: int = 5
    ) -> Optional[Dict]:
        """Get baseline statistics for an OD pair.

        Args:
            origin: Origin road ID
            destination: Destination road ID
            min_samples: Minimum number of samples required to return OD-specific baseline

        Returns:
            Dictionary with baseline statistics or None if insufficient samples
        """
        od_key = f"({origin}, {destination})"
        baseline = self.od_baselines.get(od_key)

        if baseline is None:
            return None

        # Check if we have enough samples for reliable statistics
        if baseline.get("n_samples", 0) < min_samples:
            return None

        return baseline

    def get_global_baseline(self) -> Dict:
        """Get global baseline statistics (fallback for sparse OD pairs).

        Returns:
            Dictionary with global statistics
        """
        return self.global_stats


# ============================================================================
# TRAJECTORY METRICS COMPUTER
# ============================================================================


class TrajectoryMetrics:
    """Compute metrics for trajectory analysis.

    Provides utility functions for calculating route length, travel time,
    speed profiles, and haversine distances from trajectory data.
    """

    def __init__(self, geo_df: pl.DataFrame):
        """Initialize metrics computer.

        Args:
            geo_df: Road network geometry DataFrame
        """
        self.geo_df = geo_df
        self.road_gps_map = self._build_road_gps_mapping()

    def _build_road_gps_mapping(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Build mapping from road_id to (origin_lat, origin_lon, dest_lat, dest_lon).

        Returns:
            Dictionary mapping road_id to coordinate tuple
        """
        mapping = {}
        # geo_df is a Pandas DataFrame from load_road_network()
        for idx, row in self.geo_df.iterrows():
            road_id = row["road_id"]
            # Parse coordinates field which is JSON string
            coords = json.loads(row["coordinates"])
            if len(coords) >= 2:
                origin_lon, origin_lat = coords[0]
                dest_lon, dest_lat = coords[-1]
                mapping[road_id] = (origin_lat, origin_lon, dest_lat, dest_lon)
        return mapping

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two GPS points in kilometers.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in kilometers

        return c * r

    def compute_route_length(self, road_ids: List[int]) -> float:
        """Compute total route length by summing road segment distances.

        Args:
            road_ids: List of road IDs in trajectory

        Returns:
            Total route length in meters
        """
        if not road_ids or len(road_ids) < 2:
            return 0.0

        total_distance_km = 0.0

        for i in range(len(road_ids) - 1):
            current_id = road_ids[i]
            next_id = road_ids[i + 1]

            # Get coordinates for current and next road
            current_coords = self.road_gps_map.get(current_id)
            next_coords = self.road_gps_map.get(next_id)

            if current_coords and next_coords:
                # Use destination of current road and origin of next road
                lat1, lon1 = current_coords[2], current_coords[3]  # dest of current
                lat2, lon2 = next_coords[0], next_coords[1]  # origin of next

                # Add haversine distance between segments
                total_distance_km += self.haversine_distance(lat1, lon1, lat2, lon2)

        # Convert to meters
        return total_distance_km * 1000

    def compute_travel_time(self, timestamps: List[int]) -> float:
        """Compute travel time from timestamp sequence.

        Args:
            timestamps: List of timestamps in seconds

        Returns:
            Travel time in seconds
        """
        if not timestamps or len(timestamps) < 2:
            return 0.0

        return float(timestamps[-1] - timestamps[0])

    def compute_speed_indicators(
        self, road_ids: List[int], timestamps: List[int]
    ) -> Tuple[float, float]:
        """Compute average and maximum speed indicators.

        Args:
            road_ids: List of road IDs in trajectory
            timestamps: List of timestamps in seconds

        Returns:
            Tuple of (average_speed_kmh, max_speed_kmh)
        """
        if len(road_ids) < 2 or len(timestamps) < 2:
            return 0.0, 0.0

        speeds = []
        for i in range(len(road_ids) - 1):
            current_coords = self.road_gps_map.get(road_ids[i])
            next_coords = self.road_gps_map.get(road_ids[i + 1])

            if current_coords and next_coords and i + 1 < len(timestamps):
                # Distance between segments
                lat1, lon1 = current_coords[2], current_coords[3]
                lat2, lon2 = next_coords[0], next_coords[1]
                distance_km = self.haversine_distance(lat1, lon1, lat2, lon2)

                # Time difference
                time_sec = timestamps[i + 1] - timestamps[i]

                if time_sec > 0:
                    # Speed in km/h
                    speed_kmh = (distance_km / time_sec) * 3600
                    speeds.append(speed_kmh)

        if not speeds:
            return 0.0, 0.0

        return float(np.mean(speeds)), float(np.max(speeds))


# ============================================================================
# WANG STATISTICAL DETECTOR
# ============================================================================


class WangStatisticalDetector:
    """Wang et al. 2018 statistical abnormality detector.

    Implements four abnormal behavior patterns (Abp1-4):
    - Abp1: Normal trajectory (within thresholds for both length and time)
    - Abp2: Temporal delay (normal length, excessive time)
    - Abp3: Route deviation (excessive length, normal time)
    - Abp4: Both deviations (excessive length AND time)
    """

    def __init__(
        self,
        baselines: BaselineStatistics,
        config: WangConfig,
        geo_df: pl.DataFrame,
    ):
        """Initialize the Wang statistical detector.

        Args:
            baselines: Baseline statistics for OD pairs
            config: Wang detection configuration
            geo_df: Road network geometry DataFrame
        """
        self.baselines = baselines
        self.config = config
        self.metrics = TrajectoryMetrics(geo_df)

        logger.info("✅ Wang Statistical Detector initialized")
        logger.info(f"   Threshold strategy: {config.threshold_strategy}")
        logger.info(
            f"   Fixed thresholds: L_ρ={config.L_rho_meters}m, T_ρ={config.T_rho_seconds}s"
        )
        logger.info(
            f"   Statistical multipliers: σ_length={config.sigma_length}, σ_time={config.sigma_time}"
        )

    def _compute_thresholds(
        self, baseline: Dict, use_global: bool = False
    ) -> Tuple[float, float]:
        """Compute length and time thresholds using hybrid strategy.

        Args:
            baseline: Baseline statistics dictionary
            use_global: Whether this is global baseline (affects logging)

        Returns:
            Tuple of (length_threshold_meters, time_threshold_seconds)
        """
        # Extract baseline mean and std
        mean_length_m = baseline.get("mean_length_m", 0)
        std_length_m = baseline.get("std_length_m", 0)
        mean_time_sec = baseline.get("mean_duration_sec", 0)
        std_time_sec = baseline.get("std_duration_sec", 0)

        if self.config.threshold_strategy == "fixed":
            # Use Wang et al. fixed thresholds only
            length_threshold = mean_length_m + self.config.L_rho_meters
            time_threshold = mean_time_sec + self.config.T_rho_seconds

        elif self.config.threshold_strategy == "statistical":
            # Use statistical multipliers only
            length_threshold = mean_length_m + (self.config.sigma_length * std_length_m)
            time_threshold = mean_time_sec + (self.config.sigma_time * std_time_sec)

        else:  # hybrid (default)
            # Use most stringent (minimum) of fixed and statistical
            fixed_length = mean_length_m + self.config.L_rho_meters
            stat_length = mean_length_m + (self.config.sigma_length * std_length_m)
            length_threshold = min(fixed_length, stat_length)

            fixed_time = mean_time_sec + self.config.T_rho_seconds
            stat_time = mean_time_sec + (self.config.sigma_time * std_time_sec)
            time_threshold = min(fixed_time, stat_time)

        return length_threshold, time_threshold

    def classify_trajectory(
        self, road_ids: List[int], timestamps: List[int]
    ) -> Tuple[str, Dict]:
        """Classify trajectory using Wang et al. 2018 methodology.

        Args:
            road_ids: List of road IDs in trajectory
            timestamps: List of timestamps in seconds

        Returns:
            Tuple of (pattern, details) where:
            - pattern: "Abp1_normal", "Abp2_temporal_delay", "Abp3_route_deviation", "Abp4_both"
            - details: Dictionary with classification details and metrics
        """
        # Extract OD pair
        if not road_ids or len(road_ids) < 2:
            return "Abp1_normal", {"reason": "insufficient_trajectory_length"}

        origin = road_ids[0]
        destination = road_ids[-1]

        # Compute actual trajectory metrics
        actual_length_m = self.metrics.compute_route_length(road_ids)
        actual_time_sec = self.metrics.compute_travel_time(timestamps)
        avg_speed_kmh, max_speed_kmh = self.metrics.compute_speed_indicators(
            road_ids, timestamps
        )

        # Get baseline for this OD pair
        od_baseline = self.baselines.get_od_baseline(
            origin, destination, self.config.min_samples_per_od
        )

        if od_baseline is None and self.config.fallback_to_global:
            # Use global baseline as fallback
            baseline = self.baselines.get_global_baseline()
            used_baseline = "global"
        elif od_baseline is not None:
            baseline = od_baseline
            used_baseline = "od_specific"
        else:
            # No baseline available and fallback disabled
            return "Abp1_normal", {
                "reason": "no_baseline_available",
                "actual_length_m": actual_length_m,
                "actual_time_sec": actual_time_sec,
            }

        # Compute thresholds
        length_threshold, time_threshold = self._compute_thresholds(baseline)

        # Wang et al. 2018 Classification Logic
        length_normal = actual_length_m <= length_threshold
        time_normal = actual_time_sec <= time_threshold

        if length_normal and time_normal:
            pattern = "Abp1_normal"
        elif length_normal and not time_normal:
            pattern = "Abp2_temporal_delay"
        elif not length_normal and time_normal:
            pattern = "Abp3_route_deviation"
        else:
            pattern = "Abp4_both_deviations"

        # Compile details
        details = {
            "origin": origin,
            "destination": destination,
            "baseline_type": used_baseline,
            "actual_length_m": actual_length_m,
            "actual_time_sec": actual_time_sec,
            "avg_speed_kmh": avg_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "baseline_mean_length_m": baseline.get("mean_length_m", 0),
            "baseline_mean_time_sec": baseline.get("mean_duration_sec", 0),
            "length_threshold_m": length_threshold,
            "time_threshold_sec": time_threshold,
            "length_deviation_m": actual_length_m - baseline.get("mean_length_m", 0),
            "time_deviation_sec": actual_time_sec
            - baseline.get("mean_duration_sec", 0),
        }

        return pattern, details

    def detect_abnormal_trajectories(self, trajectories_df: pl.DataFrame) -> Dict:
        """Detect abnormal trajectories in dataset.

        Args:
            trajectories_df: DataFrame with trajectory data (columns: traj_id, road_ids, timestamps)

        Returns:
            Dictionary with detection results including:
            - pattern_counts: Count of each Abp pattern
            - abnormal_trajectories: List of abnormal trajectory IDs and details
            - abnormal_rate: Percentage of abnormal trajectories
        """
        logger.info(f"🔍 Analyzing {len(trajectories_df):,} trajectories...")

        pattern_counts = {
            "Abp1_normal": 0,
            "Abp2_temporal_delay": 0,
            "Abp3_route_deviation": 0,
            "Abp4_both_deviations": 0,
        }

        abnormal_trajectories = []
        baseline_type_counts = {"od_specific": 0, "global": 0, "none": 0}

        # Convert to list of dictionaries for iteration
        traj_data = trajectories_df.to_dicts()

        for i, traj in enumerate(traj_data):
            if (i + 1) % 10000 == 0:
                logger.info(
                    f"   Processed {i + 1:,} / {len(traj_data):,} trajectories..."
                )

            traj_id = traj.get("traj_id")
            road_ids = traj.get("road_ids", [])
            timestamps = traj.get("timestamps", [])

            # Classify trajectory
            pattern, details = self.classify_trajectory(road_ids, timestamps)

            # Update counts
            pattern_counts[pattern] += 1
            baseline_type_counts[details.get("baseline_type", "none")] += 1

            # Record abnormal trajectories (Abp2, Abp3, Abp4)
            if pattern != "Abp1_normal":
                abnormal_trajectories.append(
                    {"traj_id": traj_id, "pattern": pattern, "details": details}
                )

        # Compute statistics
        total = len(trajectories_df)
        normal_count = pattern_counts["Abp1_normal"]
        abnormal_count = total - normal_count
        abnormal_rate = (abnormal_count / total * 100) if total > 0 else 0

        logger.info("📊 Detection Results:")
        logger.info(f"   Total trajectories: {total:,}")
        logger.info(f"   Abp1 (Normal): {pattern_counts['Abp1_normal']:,}")
        logger.info(
            f"   Abp2 (Temporal delay): {pattern_counts['Abp2_temporal_delay']:,}"
        )
        logger.info(
            f"   Abp3 (Route deviation): {pattern_counts['Abp3_route_deviation']:,}"
        )
        logger.info(
            f"   Abp4 (Both deviations): {pattern_counts['Abp4_both_deviations']:,}"
        )
        logger.info(f"   Abnormal rate: {abnormal_rate:.2f}%")
        logger.info("📊 Baseline usage:")
        logger.info(f"   OD-specific: {baseline_type_counts['od_specific']:,}")
        logger.info(f"   Global fallback: {baseline_type_counts['global']:,}")

        results = {
            "analysis_metadata": {
                "dataset": self.config.dataset,
                "detection_method": "wang_statistical",
                "baseline_dataset": self.baselines.metadata["dataset"],
                "trajectories_analyzed": total,
                "threshold_strategy": self.config.threshold_strategy,
            },
            "pattern_counts": pattern_counts,
            "abnormal_trajectories": abnormal_trajectories,
            "abnormal_rate": abnormal_rate,
            "baseline_usage": baseline_type_counts,
        }

        return results
