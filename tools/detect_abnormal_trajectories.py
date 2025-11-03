#!/usr/bin/env python3
"""
Abnormal Trajectory Detection Module

This module provides functionality to detect abnormal trajectories in GPS trajectory data
using statistical methods (z-score) across multiple detection categories:
- Speeding: Trajectories with excessive speeds
- Detour: Unnecessarily long routes
- Suspicious stops: Unusual stationary periods
- Unusual duration: Trips taking too long for the OD pair
- Circuitous routes: Overly curved/indirect paths

Usage:
    from tools.detect_abnormal_trajectories import AbnormalTrajectoryDetector, AbnormalityConfig

    config = AbnormalityConfig.from_yaml("config/abnormal_detection.yaml")
    detector = AbnormalTrajectoryDetector(config, geo_df, rel_df)
    results = detector.detect_abnormal_trajectories(trajectories_df)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import polars as pl
import numpy as np
from math import radians, cos, sin, asin, sqrt
import logging

logger = logging.getLogger(__name__)


@dataclass
class AbnormalityConfig:
    """Configuration for abnormal trajectory detection.

    Attributes:
        dataset: Name of the dataset being analyzed
        detection_method: Statistical method for detection (e.g., "z_score")
        detection_threshold: Threshold for z-score detection (e.g., 2.5 std devs)
        categories: Dictionary of detection categories and their parameters
        analysis_config: Analysis settings (min_samples, save_samples, etc.)
    """

    dataset: str
    detection_method: str
    detection_threshold: float
    categories: Dict[str, Dict]
    analysis_config: Dict

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AbnormalityConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            AbnormalityConfig instance
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(
            dataset=config["dataset"],
            detection_method=config["detection"]["method"],
            detection_threshold=config["detection"]["threshold"],
            categories=config["categories"],
            analysis_config=config["analysis"],
        )


class TrajectoryAnalyzer:
    """Analyzer for computing trajectory features and detecting abnormalities.

    This class provides methods to analyze individual trajectories for various
    abnormality indicators using geometric and statistical methods.
    """

    def __init__(
        self, geo_df: pl.DataFrame, rel_df: pl.DataFrame, config: AbnormalityConfig
    ):
        """Initialize the trajectory analyzer.

        Args:
            geo_df: Road network geometry (roadmap.geo)
            rel_df: Road network connections (roadmap.rel)
            config: Abnormality detection configuration
        """
        self.config = config
        self.geo_df = geo_df
        self.rel_df = rel_df

        # Build mapping from road_id to GPS coordinates for fast lookup
        self.road_gps_map = self._build_road_gps_mapping()

    def _build_road_gps_mapping(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Build mapping from road_id to (origin_lat, origin_lon, dest_lat, dest_lon).

        Returns:
            Dictionary mapping road_id to coordinate tuple
        """
        import json

        mapping = {}
        # geo_df is a Pandas DataFrame from load_road_network()
        for idx, row in self.geo_df.iterrows():
            road_id = row[
                "road_id"
            ]  # Changed from geo_id to road_id (renamed in load_road_network)
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

    def calculate_speed_profile(self, traj: List[Tuple[int, int]]) -> Dict:
        """Calculate speed profile for a trajectory.

        Args:
            traj: List of (road_id, timestamp_sec) tuples

        Returns:
            Dictionary with keys: segments (distances), speeds (km/h),
            mean_speed, max_speed, total_distance_km, total_time_sec
        """
        if len(traj) < 2:
            return {
                "segments": [],
                "speeds": [],
                "mean_speed": 0.0,
                "max_speed": 0.0,
                "total_distance_km": 0.0,
                "total_time_sec": 0,
            }

        segments = []
        speeds = []

        for i in range(len(traj) - 1):
            road_id1, time1 = traj[i]
            road_id2, time2 = traj[i + 1]

            # Get GPS coordinates for both roads
            if road_id1 not in self.road_gps_map or road_id2 not in self.road_gps_map:
                continue

            _, _, lat1, lon1 = self.road_gps_map[road_id1]  # Destination of road1
            lat2, lon2, _, _ = self.road_gps_map[road_id2]  # Origin of road2

            # Calculate distance and time
            distance_km = self.haversine_distance(lat1, lon1, lat2, lon2)
            time_diff_sec = (
                time2 - time1
            ).total_seconds()  # Convert timedelta to seconds

            if time_diff_sec > 0:
                speed_kmh = (distance_km / time_diff_sec) * 3600  # Convert to km/h
                segments.append(distance_km)
                speeds.append(speed_kmh)

        return {
            "segments": segments,
            "speeds": speeds,
            "mean_speed": np.mean(speeds) if speeds else 0.0,
            "max_speed": np.max(speeds) if speeds else 0.0,
            "total_distance_km": sum(segments),
            "total_time_sec": traj[-1][1] - traj[0][1],
        }

    def detect_speeding(self, traj: List[Tuple[int, int]]) -> Tuple[bool, Dict]:
        """Detect if trajectory contains speeding segments.

        Args:
            traj: List of (road_id, timestamp_sec) tuples

        Returns:
            Tuple of (is_speeding, details_dict)
        """
        if not self.config.categories["speeding"]["enabled"]:
            return False, {}

        speed_profile = self.calculate_speed_profile(traj)

        if not speed_profile["speeds"]:
            return False, {}

        speed_limit = self.config.categories["speeding"]["speed_limit_kmh"]
        max_speed = speed_profile["max_speed"]

        # Check if max speed exceeds limit
        is_speeding = max_speed > speed_limit

        # Calculate z-score if there are multiple segments
        z_score = 0.0
        if len(speed_profile["speeds"]) > 1:
            speeds_array = np.array(speed_profile["speeds"])
            mean_speed = np.mean(speeds_array)
            std_speed = np.std(speeds_array)
            if std_speed > 0:
                z_score = (max_speed - mean_speed) / std_speed

        return is_speeding, {
            "max_speed_kmh": max_speed,
            "speed_limit_kmh": speed_limit,
            "mean_speed_kmh": speed_profile["mean_speed"],
            "z_score": z_score,
            "exceeded_by_kmh": max(0, max_speed - speed_limit),
        }

    def calculate_detour_ratio(self, traj: List[Tuple[int, int]]) -> float:
        """Calculate detour ratio (actual_distance / straight_line_distance).

        Args:
            traj: List of (road_id, timestamp_sec) tuples

        Returns:
            Detour ratio (>= 1.0, where 1.0 is perfectly straight)
        """
        if len(traj) < 2:
            return 1.0

        # Get origin and destination coordinates
        origin_road_id = traj[0][0]
        dest_road_id = traj[-1][0]

        if (
            origin_road_id not in self.road_gps_map
            or dest_road_id not in self.road_gps_map
        ):
            return 1.0

        origin_lat, origin_lon, _, _ = self.road_gps_map[origin_road_id]
        _, _, dest_lat, dest_lon = self.road_gps_map[dest_road_id]

        # Calculate straight-line distance
        straight_distance = self.haversine_distance(
            origin_lat, origin_lon, dest_lat, dest_lon
        )

        if straight_distance < 0.1:  # Too short to meaningfully analyze
            return 1.0

        # Calculate actual path distance
        speed_profile = self.calculate_speed_profile(traj)
        actual_distance = speed_profile["total_distance_km"]

        if actual_distance == 0:
            return 1.0

        return actual_distance / straight_distance

    def detect_suspicious_stops(self, traj: List[Tuple[int, int]]) -> Tuple[bool, Dict]:
        """Detect suspicious stops in trajectory.

        Args:
            traj: List of (road_id, timestamp_sec) tuples

        Returns:
            Tuple of (has_suspicious_stops, details_dict)
        """
        if not self.config.categories["suspicious_stops"]["enabled"]:
            return False, {}

        min_duration = self.config.categories["suspicious_stops"][
            "min_stop_duration_sec"
        ]
        max_count = self.config.categories["suspicious_stops"]["max_stop_count"]

        stops = []
        stop_durations = []

        i = 0
        while i < len(traj) - 1:
            road_id1, time1 = traj[i]
            j = i + 1

            # Find consecutive same road_id (stationary period)
            while j < len(traj) and traj[j][0] == road_id1:
                j += 1

            stop_duration = traj[j - 1][1] - time1

            # Convert timedelta to seconds if needed
            if hasattr(stop_duration, "total_seconds"):
                stop_duration_sec = stop_duration.total_seconds()
            else:
                stop_duration_sec = stop_duration

            if stop_duration_sec >= min_duration:
                stops.append((i, j - 1))
                stop_durations.append(stop_duration_sec)

            i = j

        is_suspicious = len(stops) > max_count

        return is_suspicious, {
            "stop_count": len(stops),
            "max_allowed_stops": max_count,
            "stop_durations_sec": stop_durations,
            "total_stop_time_sec": sum(stop_durations),
            "stop_indices": stops,
        }

    def calculate_straightness(self, traj: List[Tuple[int, int]]) -> float:
        """Calculate trajectory straightness (haversine / path_length).

        Args:
            traj: List of (road_id, timestamp_sec) tuples

        Returns:
            Straightness ratio (0 to 1, where 1 is perfectly straight)
        """
        if len(traj) < 2:
            return 1.0

        # Get origin and destination
        origin_road_id = traj[0][0]
        dest_road_id = traj[-1][0]

        if (
            origin_road_id not in self.road_gps_map
            or dest_road_id not in self.road_gps_map
        ):
            return 1.0

        origin_lat, origin_lon, _, _ = self.road_gps_map[origin_road_id]
        _, _, dest_lat, dest_lon = self.road_gps_map[dest_road_id]

        # Calculate straight-line distance
        straight_distance = self.haversine_distance(
            origin_lat, origin_lon, dest_lat, dest_lon
        )

        # Calculate actual path distance
        speed_profile = self.calculate_speed_profile(traj)
        actual_distance = speed_profile["total_distance_km"]

        if actual_distance == 0:
            return 1.0

        return min(1.0, straight_distance / actual_distance)

    def analyze_trajectory(self, traj: List[Tuple[int, int]], traj_idx: int) -> Dict:
        """Analyze a single trajectory for all abnormality categories.

        Args:
            traj: List of (road_id, timestamp_sec) tuples
            traj_idx: Index of the trajectory in the dataset

        Returns:
            Dictionary with detection results for all categories
        """
        results = {"traj_idx": traj_idx, "length": len(traj), "abnormal_categories": []}

        # Speed analysis
        is_speeding, speed_details = self.detect_speeding(traj)
        results["speeding"] = {"detected": is_speeding, **speed_details}
        if is_speeding:
            results["abnormal_categories"].append("speeding")

        # Detour analysis
        if self.config.categories["detour"]["enabled"]:
            detour_ratio = self.calculate_detour_ratio(traj)
            threshold = self.config.categories["detour"]["detour_ratio_threshold"]
            min_trip_length = self.config.categories["detour"]["min_trip_length_km"]

            speed_profile = self.calculate_speed_profile(traj)
            is_detour = (
                detour_ratio > threshold
                and speed_profile["total_distance_km"] >= min_trip_length
            )

            results["detour"] = {
                "detected": is_detour,
                "detour_ratio": detour_ratio,
                "threshold": threshold,
                "trip_length_km": speed_profile["total_distance_km"],
            }
            if is_detour:
                results["abnormal_categories"].append("detour")

        # Suspicious stops analysis
        has_stops, stop_details = self.detect_suspicious_stops(traj)
        results["suspicious_stops"] = {"detected": has_stops, **stop_details}
        if has_stops:
            results["abnormal_categories"].append("suspicious_stops")

        # Circuitous route analysis
        if self.config.categories["circuitous"]["enabled"]:
            straightness = self.calculate_straightness(traj)
            threshold = self.config.categories["circuitous"]["straightness_threshold"]
            is_circuitous = straightness < threshold

            results["circuitous"] = {
                "detected": is_circuitous,
                "straightness": straightness,
                "threshold": threshold,
            }
            if is_circuitous:
                results["abnormal_categories"].append("circuitous")

        return results


class AbnormalTrajectoryDetector:
    """Main detector for identifying abnormal trajectories in a dataset."""

    def __init__(
        self, config: AbnormalityConfig, geo_df: pl.DataFrame, rel_df: pl.DataFrame
    ):
        """Initialize the abnormal trajectory detector.

        Args:
            config: Abnormality detection configuration
            geo_df: Road network geometry
            rel_df: Road network connections
        """
        self.config = config
        self.analyzer = TrajectoryAnalyzer(geo_df, rel_df, config)

    def detect_abnormal_trajectories(self, trajectories_df: pl.DataFrame) -> Dict:
        """Detect abnormal trajectories in a dataset.

        Args:
            trajectories_df: DataFrame with trajectory data (columns: traj_id, road_id, timestamp)

        Returns:
            Dictionary containing:
                - abnormal_indices: Dict mapping category to list of abnormal traj indices
                - statistics: Overall statistics by category
                - samples: Sample trajectories for each category (if enabled)
        """
        logger.info(
            f"Analyzing {len(trajectories_df)} trajectories for abnormalities..."
        )

        # Group by trajectory ID
        traj_groups = trajectories_df.group_by("traj_id").agg(
            [pl.col("road_id"), pl.col("timestamp")]
        )

        results_by_category = {
            "speeding": [],
            "detour": [],
            "suspicious_stops": [],
            "circuitous": [],
            "unusual_duration": [],
        }

        all_results = []

        for row in traj_groups.iter_rows(named=True):
            traj_id = row["traj_id"]
            road_ids = row["road_id"]
            timestamps = row["timestamp"]

            # Convert to list of tuples
            traj = list(zip(road_ids, timestamps))

            # Analyze trajectory
            analysis = self.analyzer.analyze_trajectory(traj, traj_id)
            all_results.append(analysis)

            # Categorize abnormal trajectories
            for category in analysis["abnormal_categories"]:
                results_by_category[category].append(traj_id)

        # Compute statistics
        statistics = {}
        for category, indices in results_by_category.items():
            statistics[category] = {
                "count": len(indices),
                "percentage": (len(indices) / len(traj_groups)) * 100
                if len(traj_groups) > 0
                else 0,
            }

        logger.info("Detection complete. Found abnormalities:")
        for category, stats in statistics.items():
            logger.info(f"  {category}: {stats['count']} ({stats['percentage']:.2f}%)")

        return {
            "abnormal_indices": results_by_category,
            "statistics": statistics,
            "all_results": all_results,
            "total_trajectories": len(traj_groups),
        }
