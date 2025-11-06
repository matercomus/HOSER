#!/usr/bin/env python3
"""
Unit tests for paired statistical tests integration (Issue #51).

Tests trajectory-level metrics generation, matching, and analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pytest


# Test data fixtures
@pytest.fixture
def sample_trajectory_metrics() -> List[Dict[str, Any]]:
    """Generate sample trajectory-level metrics."""
    return [
        {
            "od_pair": [100, 200],
            "real_traj_idx": 0,
            "gen_traj_idx": 0,
            "hausdorff_km": 1.234,
            "hausdorff_norm": 0.123,
            "dtw_km": 2.345,
            "dtw_norm": 0.234,
            "edr": 0.345,
            "len_real": 10,
            "len_gen": 12,
        },
        {
            "od_pair": [100, 200],
            "real_traj_idx": 1,
            "gen_traj_idx": 1,
            "hausdorff_km": 1.456,
            "hausdorff_norm": 0.145,
            "dtw_km": 2.567,
            "dtw_norm": 0.256,
            "edr": 0.367,
            "len_real": 11,
            "len_gen": 10,
        },
        {
            "od_pair": [300, 400],
            "real_traj_idx": 2,
            "gen_traj_idx": 2,
            "hausdorff_km": 0.789,
            "hausdorff_norm": 0.078,
            "dtw_km": 1.234,
            "dtw_norm": 0.123,
            "edr": 0.234,
            "len_real": 8,
            "len_gen": 11,
        },
    ]


@pytest.fixture
def sample_trajectory_metrics_file(sample_trajectory_metrics, tmp_path) -> Path:
    """Create a temporary trajectory_metrics.json file."""
    eval_dir = tmp_path / "eval" / "2025-01-01_12-00-00"
    eval_dir.mkdir(parents=True)

    metrics_file = eval_dir / "trajectory_metrics.json"
    data = {
        "metadata": {
            "generated_file": "test.csv",
            "real_data_file": "real.csv",
            "od_source": "test",
            "evaluation_timestamp": "2025-01-01T12:00:00",
            "num_trajectory_comparisons": len(sample_trajectory_metrics),
            "grid_size": 0.001,
            "edr_eps": 100.0,
        },
        "trajectory_metrics": sample_trajectory_metrics,
    }

    with open(metrics_file, "w") as f:
        json.dump(data, f, indent=2)

    return metrics_file


@pytest.fixture
def two_model_metrics(tmp_path) -> tuple[Path, Path]:
    """Create trajectory metrics for two different models with matching OD pairs."""
    # Model 1 metrics
    model1_dir = tmp_path / "eval" / "model1" / "2025-01-01_12-00-00"
    model1_dir.mkdir(parents=True)

    model1_metrics = [
        {
            "od_pair": [100, 200],
            "real_traj_idx": 0,
            "gen_traj_idx": 0,
            "hausdorff_km": 1.5,
            "hausdorff_norm": 0.15,
            "dtw_km": 2.5,
            "dtw_norm": 0.25,
            "edr": 0.35,
            "len_real": 10,
            "len_gen": 10,
        },
        {
            "od_pair": [300, 400],
            "real_traj_idx": 1,
            "gen_traj_idx": 1,
            "hausdorff_km": 0.8,
            "hausdorff_norm": 0.08,
            "dtw_km": 1.2,
            "dtw_norm": 0.12,
            "edr": 0.22,
            "len_real": 10,
            "len_gen": 10,
        },
    ]

    model1_data = {
        "metadata": {
            "generated_file": "model1.csv",
            "od_source": "test",
            "num_trajectory_comparisons": 2,
        },
        "trajectory_metrics": model1_metrics,
    }

    model1_file = model1_dir / "trajectory_metrics.json"
    with open(model1_file, "w") as f:
        json.dump(model1_data, f)

    # Model 2 metrics (same OD pairs, different values)
    model2_dir = tmp_path / "eval" / "model2" / "2025-01-01_13-00-00"
    model2_dir.mkdir(parents=True)

    model2_metrics = [
        {
            "od_pair": [100, 200],
            "real_traj_idx": 0,
            "gen_traj_idx": 0,
            "hausdorff_km": 1.2,
            "hausdorff_norm": 0.12,
            "dtw_km": 2.0,
            "dtw_norm": 0.20,
            "edr": 0.30,
            "len_real": 10,
            "len_gen": 10,
        },
        {
            "od_pair": [300, 400],
            "real_traj_idx": 1,
            "gen_traj_idx": 1,
            "hausdorff_km": 0.6,
            "hausdorff_norm": 0.06,
            "dtw_km": 1.0,
            "dtw_norm": 0.10,
            "edr": 0.18,
            "len_real": 10,
            "len_gen": 10,
        },
    ]

    model2_data = {
        "metadata": {
            "generated_file": "model2.csv",
            "od_source": "test",
            "num_trajectory_comparisons": 2,
        },
        "trajectory_metrics": model2_metrics,
    }

    model2_file = model2_dir / "trajectory_metrics.json"
    with open(model2_file, "w") as f:
        json.dump(model2_data, f)

    return model1_dir, model2_dir


# Test trajectory metrics structure
class TestTrajectoryMetricsStructure:
    """Test the structure and format of trajectory_metrics.json."""

    def test_metrics_file_structure(self, sample_trajectory_metrics_file):
        """Test that trajectory_metrics.json has correct structure."""
        with open(sample_trajectory_metrics_file) as f:
            data = json.load(f)

        # Check top-level keys
        assert "metadata" in data
        assert "trajectory_metrics" in data

        # Check metadata fields
        metadata = data["metadata"]
        assert "generated_file" in metadata
        assert "od_source" in metadata
        assert "num_trajectory_comparisons" in metadata
        assert "grid_size" in metadata
        assert "edr_eps" in metadata

        # Check trajectory metrics structure
        metrics = data["trajectory_metrics"]
        assert isinstance(metrics, list)
        assert len(metrics) > 0

        # Check first metric record
        first_metric = metrics[0]
        required_fields = [
            "od_pair",
            "real_traj_idx",
            "gen_traj_idx",
            "hausdorff_km",
            "hausdorff_norm",
            "dtw_km",
            "dtw_norm",
            "edr",
            "len_real",
            "len_gen",
        ]
        for field in required_fields:
            assert field in first_metric, f"Missing required field: {field}"

    def test_od_pair_format(self, sample_trajectory_metrics_file):
        """Test that OD pairs are stored as lists (for JSON serialization)."""
        with open(sample_trajectory_metrics_file) as f:
            data = json.load(f)

        for metric in data["trajectory_metrics"]:
            od_pair = metric["od_pair"]
            assert isinstance(od_pair, list)
            assert len(od_pair) == 2
            assert all(isinstance(x, int) for x in od_pair)

    def test_metric_value_types(self, sample_trajectory_metrics_file):
        """Test that metric values have correct types."""
        with open(sample_trajectory_metrics_file) as f:
            data = json.load(f)

        for metric in data["trajectory_metrics"]:
            # Check integer fields
            assert isinstance(metric["real_traj_idx"], int)
            assert isinstance(metric["gen_traj_idx"], int)
            assert isinstance(metric["len_real"], int)
            assert isinstance(metric["len_gen"], int)

            # Check float fields
            assert isinstance(metric["hausdorff_km"], (int, float))
            assert isinstance(metric["hausdorff_norm"], (int, float))
            assert isinstance(metric["dtw_km"], (int, float))
            assert isinstance(metric["dtw_norm"], (int, float))
            assert isinstance(metric["edr"], (int, float))

            # Check value ranges
            assert metric["hausdorff_km"] >= 0
            assert metric["hausdorff_norm"] >= 0
            assert metric["dtw_km"] >= 0
            assert metric["dtw_norm"] >= 0
            assert 0 <= metric["edr"] <= 1  # EDR is normalized 0-1


# Test OD pair matching
class TestODPairMatching:
    """Test matching trajectory pairs across models by OD pair."""

    def test_match_by_od_pair(self, sample_trajectory_metrics):
        """Test matching trajectories by OD pair."""
        from tools.compare_models_paired_analysis import match_trajectory_pairs

        # Create two sets with overlapping OD pairs
        metrics1 = sample_trajectory_metrics[:2]  # OD pairs [100,200], [100,200]
        metrics2 = [
            sample_trajectory_metrics[1],  # OD pair [100,200]
            sample_trajectory_metrics[2],  # OD pair [300,400]
        ]

        matched1, matched2 = match_trajectory_pairs(metrics1, metrics2)

        # Should match 1 trajectory with OD pair [100,200]
        assert len(matched1) == 1
        assert len(matched2) == 1
        assert matched1[0]["od_pair"] == [100, 200]
        assert matched2[0]["od_pair"] == [100, 200]

    def test_no_matches(self):
        """Test when there are no matching OD pairs."""
        from tools.compare_models_paired_analysis import match_trajectory_pairs

        metrics1 = [{"od_pair": [100, 200], "hausdorff_norm": 0.1}]
        metrics2 = [{"od_pair": [300, 400], "hausdorff_norm": 0.2}]

        matched1, matched2 = match_trajectory_pairs(metrics1, metrics2)

        assert len(matched1) == 0
        assert len(matched2) == 0

    def test_multiple_matches_same_od(self):
        """Test matching multiple trajectories with same OD pair."""
        from tools.compare_models_paired_analysis import match_trajectory_pairs

        # Multiple trajectories with same OD pair
        metrics1 = [
            {"od_pair": [100, 200], "hausdorff_norm": 0.1},
            {"od_pair": [100, 200], "hausdorff_norm": 0.15},
            {"od_pair": [100, 200], "hausdorff_norm": 0.2},
        ]
        metrics2 = [
            {"od_pair": [100, 200], "hausdorff_norm": 0.12},
            {"od_pair": [100, 200], "hausdorff_norm": 0.18},
        ]

        matched1, matched2 = match_trajectory_pairs(metrics1, metrics2)

        # Should match min(3, 2) = 2 pairs
        assert len(matched1) == 2
        assert len(matched2) == 2


# Test metric extraction
class TestMetricExtraction:
    """Test extracting specific metrics from matched pairs."""

    def test_extract_metric_values(self, sample_trajectory_metrics):
        """Test extracting values for a specific metric."""
        from tools.compare_models_paired_analysis import extract_metric_values

        values = extract_metric_values(sample_trajectory_metrics, "hausdorff_norm")

        assert len(values) == 3
        assert values[0] == 0.123
        assert values[1] == 0.145
        assert values[2] == 0.078

    def test_extract_different_metrics(self, sample_trajectory_metrics):
        """Test extracting different metric types."""
        from tools.compare_models_paired_analysis import extract_metric_values

        # Test normalized metrics
        hausdorff_norm = extract_metric_values(
            sample_trajectory_metrics, "hausdorff_norm"
        )
        dtw_norm = extract_metric_values(sample_trajectory_metrics, "dtw_norm")
        edr = extract_metric_values(sample_trajectory_metrics, "edr")

        assert len(hausdorff_norm) == 3
        assert len(dtw_norm) == 3
        assert len(edr) == 3

        # Check all values are numeric
        assert all(isinstance(v, (int, float)) for v in hausdorff_norm)
        assert all(isinstance(v, (int, float)) for v in dtw_norm)
        assert all(isinstance(v, (int, float)) for v in edr)


# Test loading from files
class TestLoadingTrajectoryMetrics:
    """Test loading trajectory metrics from files."""

    def test_load_trajectory_metrics(self, sample_trajectory_metrics_file):
        """Test loading trajectory_metrics.json from directory."""
        from tools.compare_models_paired_analysis import load_trajectory_metrics

        eval_dir = sample_trajectory_metrics_file.parent
        data = load_trajectory_metrics(eval_dir)

        assert "metadata" in data
        assert "trajectory_metrics" in data
        assert len(data["trajectory_metrics"]) == 3

    def test_load_missing_file(self, tmp_path):
        """Test error when trajectory_metrics.json is missing."""
        from tools.compare_models_paired_analysis import load_trajectory_metrics

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(
            FileNotFoundError, match="trajectory_metrics.json not found"
        ):
            load_trajectory_metrics(empty_dir)


# Test paired comparison
class TestPairedComparison:
    """Test paired statistical comparison functionality."""

    def test_paired_comparison_integration(self, two_model_metrics):
        """Test complete paired comparison workflow."""
        from tools.compare_models_paired_analysis import (
            load_trajectory_metrics,
            match_trajectory_pairs,
            extract_metric_values,
        )
        from tools.paired_statistical_tests import compare_models_paired

        model1_dir, model2_dir = two_model_metrics

        # Load metrics
        data1 = load_trajectory_metrics(model1_dir)
        data2 = load_trajectory_metrics(model2_dir)

        # Match pairs
        matched1, matched2 = match_trajectory_pairs(
            data1["trajectory_metrics"], data2["trajectory_metrics"]
        )

        assert len(matched1) == 2
        assert len(matched2) == 2

        # Extract metric values
        values1 = extract_metric_values(matched1, "hausdorff_norm")
        values2 = extract_metric_values(matched2, "hausdorff_norm")

        assert len(values1) == 2
        assert len(values2) == 2

        # Perform paired comparison
        result = compare_models_paired(
            model1_values=values1,
            model2_values=values2,
            model1_name="model1",
            model2_name="model2",
            metric_name="hausdorff_norm",
            alpha=0.05,
        )

        # Check result structure
        assert result.test_name in ["Paired t-test", "Wilcoxon signed-rank test"]
        assert result.n_pairs == 2
        assert isinstance(result.p_value, (float, np.floating))
        assert isinstance(result.cohens_d, (float, np.floating))
        assert isinstance(result.significant, (bool, np.bool_))

    def test_comparison_with_insufficient_data(self):
        """Test that comparison fails gracefully with too few pairs."""
        from tools.paired_statistical_tests import compare_models_paired

        # Only 1 pair - should fail
        with pytest.raises(ValueError, match="Need at least 2 pairs"):
            compare_models_paired(
                model1_values=[0.1],
                model2_values=[0.2],
                model1_name="model1",
                model2_name="model2",
                metric_name="test",
            )

    def test_comparison_with_different_lengths(self):
        """Test that comparison fails when arrays have different lengths."""
        from tools.paired_statistical_tests import compare_models_paired

        with pytest.raises(ValueError, match="Arrays must have same length"):
            compare_models_paired(
                model1_values=[0.1, 0.2, 0.3],
                model2_values=[0.1, 0.2],
                model1_name="model1",
                model2_name="model2",
                metric_name="test",
            )


# Test data validation
class TestDataValidation:
    """Test validation of trajectory metrics data."""

    def test_valid_normalized_metrics(self, sample_trajectory_metrics):
        """Test that normalized metrics are properly computed."""
        for metric in sample_trajectory_metrics:
            # Normalized metrics should be smaller than raw metrics
            assert metric["hausdorff_norm"] < metric["hausdorff_km"]
            assert metric["dtw_norm"] < metric["dtw_km"]

            # EDR should be in [0, 1]
            assert 0 <= metric["edr"] <= 1

    def test_trajectory_lengths_positive(self, sample_trajectory_metrics):
        """Test that trajectory lengths are positive."""
        for metric in sample_trajectory_metrics:
            assert metric["len_real"] > 0
            assert metric["len_gen"] > 0

    def test_metrics_non_negative(self, sample_trajectory_metrics):
        """Test that distance metrics are non-negative."""
        for metric in sample_trajectory_metrics:
            assert metric["hausdorff_km"] >= 0
            assert metric["hausdorff_norm"] >= 0
            assert metric["dtw_km"] >= 0
            assert metric["dtw_norm"] >= 0


# Test edge cases
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metrics_list(self):
        """Test handling of empty metrics list."""
        from tools.compare_models_paired_analysis import match_trajectory_pairs

        matched1, matched2 = match_trajectory_pairs([], [])
        assert len(matched1) == 0
        assert len(matched2) == 0

    def test_nan_values_filtered(self):
        """Test that NaN/inf values are filtered out."""
        from tools.paired_statistical_tests import compare_models_paired

        values1 = [0.1, 0.2, np.nan, 0.4, 0.5]
        values2 = [0.15, 0.25, 0.3, np.inf, 0.55]

        result = compare_models_paired(
            model1_values=values1,
            model2_values=values2,
            model1_name="model1",
            model2_name="model2",
            metric_name="test",
        )

        # Should have filtered to 3 valid pairs
        assert result.n_pairs == 3

    def test_identical_values(self):
        """Test comparison when all values are identical."""
        from tools.paired_statistical_tests import compare_models_paired

        values = [0.5, 0.5, 0.5, 0.5, 0.5]

        result = compare_models_paired(
            model1_values=values,
            model2_values=values,
            model1_name="model1",
            model2_name="model2",
            metric_name="test",
        )

        # Should not be significant (no difference)
        assert not result.significant
        assert result.mean_difference == 0.0
        assert result.cohens_d == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
