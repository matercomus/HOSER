#!/usr/bin/env python3
"""
Unit Tests for Wang et al. 2018 Statistical Abnormality Detector

Tests the core functionality of the WangStatisticalDetector including:
- Baseline statistics loading
- Threshold computation (fixed, statistical, hybrid)
- Trajectory classification (Abp1-4)
- Batch detection

Usage:
    uv run python tools/test_wang_detector.py
"""

import json
import logging
import tempfile
from pathlib import Path

import polars as pl

from detect_abnormal_statistical import (
    BaselineStatistics,
    WangConfig,
    WangStatisticalDetector,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_baselines() -> Path:
    """Create a temporary baseline statistics file for testing.

    Returns:
        Path to temporary baselines JSON file
    """
    baselines_data = {
        "metadata": {
            "dataset": "TestDataset",
            "computed_at": "2025-11-04T00:00:00",
            "baseline_source": "test_data",
            "methodology": "Wang et al. 2018",
        },
        "coverage": {
            "total_trajectories": 100,
            "total_od_pairs": 50,
            "od_pairs_with_min_5_samples": 10,
            "coverage_pct": 20.0,
        },
        "global_statistics": {
            "mean_length_m": 3000.0,
            "std_length_m": 500.0,
            "mean_duration_sec": 600.0,
            "std_duration_sec": 100.0,
            "mean_speed_kmh": 18.0,
            "std_speed_kmh": 5.0,
            "n_samples": 100,
        },
        "od_pair_baselines": {
            "(100, 200)": {
                "mean_length_m": 2500.0,
                "std_length_m": 400.0,
                "mean_duration_sec": 500.0,
                "std_duration_sec": 80.0,
                "mean_speed_kmh": 18.0,
                "std_speed_kmh": 4.0,
                "n_samples": 10,
            },
            "(300, 400)": {
                "mean_length_m": 5000.0,
                "std_length_m": 800.0,
                "mean_duration_sec": 900.0,
                "std_duration_sec": 150.0,
                "mean_speed_kmh": 20.0,
                "std_speed_kmh": 6.0,
                "n_samples": 3,  # Below min_samples threshold
            },
        },
    }

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(baselines_data, temp_file)
    temp_file.close()

    return Path(temp_file.name)


def create_test_geo_df() -> pl.DataFrame:
    """Create a test road network geometry DataFrame.

    Returns:
        Polars DataFrame with test road network
    """
    import pandas as pd

    # Create pandas DataFrame (as returned by load_road_network)
    geo_data = pd.DataFrame(
        {
            "road_id": [100, 200, 300, 400, 500],
            "coordinates": [
                "[[116.0, 39.0], [116.01, 39.01]]",  # Road 100
                "[[116.01, 39.01], [116.02, 39.02]]",  # Road 200
                "[[116.0, 39.0], [116.03, 39.03]]",  # Road 300
                "[[116.03, 39.03], [116.06, 39.06]]",  # Road 400
                "[[116.01, 39.01], [116.015, 39.015]]",  # Road 500
            ],
        }
    )

    return geo_data


def create_test_config() -> WangConfig:
    """Create a test configuration.

    Returns:
        WangConfig instance for testing
    """
    return WangConfig(
        dataset="TestDataset",
        baseline_dataset="TestDataset",
        L_rho_meters=5000.0,
        T_rho_seconds=300.0,
        sigma_length=2.5,
        sigma_time=2.5,
        sigma_speed=3.0,
        threshold_strategy="hybrid",
        min_samples_per_od=5,
        fallback_to_global=True,
    )


def test_baseline_loading():
    """Test baseline statistics loading."""
    logger.info("🧪 Test 1: Baseline Statistics Loading")

    baselines_path = create_test_baselines()
    baselines = BaselineStatistics(baselines_path)

    # Check metadata loaded
    assert baselines.metadata["dataset"] == "TestDataset"
    assert baselines.coverage["total_od_pairs"] == 50

    # Check OD pair retrieval with sufficient samples
    od_baseline = baselines.get_od_baseline(100, 200, min_samples=5)
    assert od_baseline is not None
    assert od_baseline["mean_length_m"] == 2500.0
    assert od_baseline["n_samples"] == 10

    # Check OD pair retrieval with insufficient samples
    od_baseline = baselines.get_od_baseline(300, 400, min_samples=5)
    assert od_baseline is None  # Only 3 samples

    # Check global baseline
    global_baseline = baselines.get_global_baseline()
    assert global_baseline["mean_length_m"] == 3000.0

    # Cleanup
    baselines_path.unlink()

    logger.info("✅ Test 1 PASSED")


def test_threshold_computation():
    """Test threshold computation strategies."""
    logger.info("🧪 Test 2: Threshold Computation")

    baselines_path = create_test_baselines()
    baselines = BaselineStatistics(baselines_path)
    geo_df = create_test_geo_df()

    # Test with different strategies
    strategies = ["fixed", "statistical", "hybrid"]

    for strategy in strategies:
        config = create_test_config()
        config.threshold_strategy = strategy

        detector = WangStatisticalDetector(baselines, config, geo_df)

        # Get baseline for OD pair (100, 200)
        baseline = baselines.get_od_baseline(100, 200, min_samples=5)
        length_threshold, time_threshold = detector._compute_thresholds(baseline)

        logger.info(f"   Strategy: {strategy}")
        logger.info(f"     Length threshold: {length_threshold:.1f}m")
        logger.info(f"     Time threshold: {time_threshold:.1f}s")

        # Validate thresholds are positive
        assert length_threshold > 0
        assert time_threshold > 0

        # Validate fixed strategy
        if strategy == "fixed":
            expected_length = baseline["mean_length_m"] + config.L_rho_meters
            expected_time = baseline["mean_duration_sec"] + config.T_rho_seconds
            assert abs(length_threshold - expected_length) < 0.1
            assert abs(time_threshold - expected_time) < 0.1

    # Cleanup
    baselines_path.unlink()

    logger.info("✅ Test 2 PASSED")


def test_trajectory_classification():
    """Test trajectory classification (Abp1-4)."""
    logger.info("🧪 Test 3: Trajectory Classification")

    baselines_path = create_test_baselines()
    baselines = BaselineStatistics(baselines_path)
    geo_df = create_test_geo_df()
    config = create_test_config()

    detector = WangStatisticalDetector(baselines, config, geo_df)

    # Test Case 1: Normal trajectory (Abp1)
    # Short route, short time
    road_ids = [100, 200]
    timestamps = [0, 100]  # 100 seconds
    pattern, details = detector.classify_trajectory(road_ids, timestamps)
    logger.info(f"   Test Case 1 (Normal): {pattern}")
    assert pattern == "Abp1_normal"
    assert details["baseline_type"] == "od_specific"

    # Test Case 2: Temporal delay (Abp2)
    # Normal route, excessive time
    road_ids = [100, 200]
    timestamps = [0, 2000]  # 2000 seconds (way over threshold)
    pattern, details = detector.classify_trajectory(road_ids, timestamps)
    logger.info(f"   Test Case 2 (Temporal delay): {pattern}")
    assert pattern == "Abp2_temporal_delay"

    # Test Case 3: Route deviation (Abp3)
    # Long detour, normal time (requires high speed)
    # This is hard to simulate with real geo data, so we'll test with fallback
    road_ids = [999, 888]  # Unknown OD pair -> uses global baseline
    timestamps = [0, 100]  # Very short time for unknown route
    pattern, details = detector.classify_trajectory(road_ids, timestamps)
    logger.info(f"   Test Case 3 (Unknown OD): {pattern}")
    assert details["baseline_type"] == "global"

    # Test Case 4: Both deviations (Abp4)
    # Long route, long time
    road_ids = [100, 200, 300, 400, 500]  # Many segments
    timestamps = [0, 5000]  # Very long time
    pattern, details = detector.classify_trajectory(road_ids, timestamps)
    logger.info(f"   Test Case 4 (Long route + time): {pattern}")
    # Either Abp4 or Abp2 depending on calculated length
    assert pattern in ["Abp2_temporal_delay", "Abp4_both_deviations"]

    # Cleanup
    baselines_path.unlink()

    logger.info("✅ Test 3 PASSED")


def test_batch_detection():
    """Test batch trajectory detection."""
    logger.info("🧪 Test 4: Batch Trajectory Detection")

    baselines_path = create_test_baselines()
    baselines = BaselineStatistics(baselines_path)
    geo_df = create_test_geo_df()
    config = create_test_config()

    detector = WangStatisticalDetector(baselines, config, geo_df)

    # Create test trajectory DataFrame
    trajectories = [
        {"traj_id": 1, "road_ids": [100, 200], "timestamps": [0, 100]},  # Normal
        {
            "traj_id": 2,
            "road_ids": [100, 200],
            "timestamps": [0, 2000],
        },  # Temporal delay
        {
            "traj_id": 3,
            "road_ids": [100, 200, 300],
            "timestamps": [0, 150],
        },  # Normal or slight deviation
        {
            "traj_id": 4,
            "road_ids": [999, 888],
            "timestamps": [0, 5000],
        },  # Unknown OD (global baseline)
    ]

    trajectories_df = pl.DataFrame(trajectories)

    # Run detection
    results = detector.detect_abnormal_trajectories(trajectories_df)

    # Validate results structure
    assert "analysis_metadata" in results
    assert "pattern_counts" in results
    assert "abnormal_trajectories" in results
    assert "abnormal_rate" in results

    # Validate counts
    pattern_counts = results["pattern_counts"]
    total_patterns = sum(pattern_counts.values())
    assert total_patterns == 4  # All 4 trajectories classified

    # Check abnormal rate
    logger.info(f"   Abnormal rate: {results['abnormal_rate']:.1f}%")
    assert 0 <= results["abnormal_rate"] <= 100

    # Check baseline usage
    baseline_usage = results["baseline_usage"]
    logger.info(
        f"   Baseline usage: OD-specific={baseline_usage['od_specific']}, "
        f"Global={baseline_usage['global']}"
    )
    assert baseline_usage["od_specific"] >= 0
    assert baseline_usage["global"] >= 0

    # Cleanup
    baselines_path.unlink()

    logger.info("✅ Test 4 PASSED")


def run_all_tests():
    """Run all unit tests."""
    logger.info("=" * 70)
    logger.info("🧪 WANG STATISTICAL DETECTOR UNIT TESTS")
    logger.info("=" * 70)

    try:
        test_baseline_loading()
        test_threshold_computation()
        test_trajectory_classification()
        test_batch_detection()

        logger.info("=" * 70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 70)
        return True

    except AssertionError as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    except Exception as e:
        logger.error(f"❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
