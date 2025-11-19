"""Tests for cross-model OD comparison functionality.

This module contains comprehensive tests for the _build_cross_model_od_comparison
function and related utilities that process multiple evaluation results to compare
model performance across OD pairs.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from tools.evaluate_lmtad_spatial_abnormal import (
    _build_cross_model_od_comparison,
    _compute_log_perplexity_stats,
    _compute_segment_stats,
)


class TestComputeLogPerplexityStats:
    """Tests for _compute_log_perplexity_stats helper function."""

    def test_compute_stats_with_normal_values(self):
        """Test statistics computation with valid finite values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _compute_log_perplexity_stats(values)

        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats

        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_compute_stats_with_empty_list(self):
        """Test statistics computation with empty list."""
        stats = _compute_log_perplexity_stats([])
        assert stats == {}

    def test_compute_stats_with_single_value(self):
        """Test statistics computation with a single value."""
        values = [2.5]
        stats = _compute_log_perplexity_stats(values)

        assert stats["mean"] == 2.5
        assert stats["median"] == 2.5
        assert stats["min"] == 2.5
        assert stats["max"] == 2.5
        assert stats["std"] == 0.0

    def test_compute_stats_with_large_values(self):
        """Test statistics computation with large perplexity values."""
        values = [10.0, 20.0, 30.0]
        stats = _compute_log_perplexity_stats(values)

        assert stats["mean"] == pytest.approx(20.0)
        assert stats["median"] == pytest.approx(20.0)

    def test_compute_stats_with_nan_handling(self):
        """Test that NaN values are handled correctly."""
        values = [1.0, 2.0, np.nan, 3.0]
        stats = _compute_log_perplexity_stats(values)

        # NaN propagation - NumPy will include NaN in calculations
        assert np.isnan(stats["mean"])

    def test_compute_stats_with_inf_handling(self):
        """Test that Infinity values are handled correctly."""
        values = [1.0, 2.0, np.inf, 3.0]
        stats = _compute_log_perplexity_stats(values)

        assert np.isinf(stats["max"])
        assert stats["min"] == 1.0


class TestComputeSegmentStats:
    """Tests for _compute_segment_stats helper function."""

    def test_compute_segment_stats_normal_case(self):
        """Test segment statistics computation with valid data."""
        segment_lists = [
            [0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.8],
            [0.3, 0.4, 0.5],
        ]

        stats = _compute_segment_stats(segment_lists)

        assert stats["max_segment_length"] == 4
        assert len(stats["per_index"]) == 4

        # Check first index (index 0)
        assert stats["per_index"][0]["index"] == 0
        assert stats["per_index"][0]["count"] == 3
        assert stats["per_index"][0]["mean"] == pytest.approx(0.4)

    def test_compute_segment_stats_empty_input(self):
        """Test segment statistics computation with empty input."""
        stats = _compute_segment_stats([])
        assert stats == {"max_segment_length": 0, "per_index": []}

    def test_compute_segment_stats_single_segment(self):
        """Test segment statistics with single trajectory."""
        segment_lists = [[0.1, 0.2, 0.3]]
        stats = _compute_segment_stats(segment_lists)

        assert stats["max_segment_length"] == 3
        assert len(stats["per_index"]) == 3

    def test_compute_segment_stats_uneven_lengths(self):
        """Test segment statistics with uneven segment lengths."""
        segment_lists = [
            [0.5, 0.6],
            [0.4, 0.5, 0.7, 0.9],
            [0.3, 0.4, 0.5, 0.6, 0.8],
        ]

        stats = _compute_segment_stats(segment_lists)

        assert stats["max_segment_length"] == 5

        # Index 0 has 3 values
        assert stats["per_index"][0]["count"] == 3

        # Index 2 has 2 values (from segments 1 and 2)
        assert stats["per_index"][2]["count"] == 2

        # Index 4 has 1 value (from segment 2 only)
        assert stats["per_index"][4]["count"] == 1

    def test_compute_segment_stats_with_empty_lists(self):
        """Test segment statistics with empty segment lists."""
        segment_lists = [[], [], []]
        stats = _compute_segment_stats(segment_lists)

        assert stats["max_segment_length"] == 0
        assert stats["per_index"] == []


class TestBuildCrossModelODComparisonBasic:
    """Basic tests for _build_cross_model_od_comparison function."""

    def test_build_comparison_two_models(self):
        """Test building comparison with two models."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5, 0.6],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 6.0,
                        "segment_log_perplexities": [0.6, 0.7],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 4.5,
                        "segment_log_perplexities": [0.4, 0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 7.0,
                        "segment_log_perplexities": [0.7, 0.8],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Check metadata
        assert result["metadata"]["model_count"] == 2
        assert result["metadata"]["model_names"] == ["model_a", "model_b"]

        # Check models
        assert len(result["models"]) == 2

        # Check OD pairs
        assert len(result["od_pairs"]) == 2

        # Check OD pair (1, 2) - model_b should be better
        od_pair_1_2 = result["od_pairs"][(1, 2)]
        assert od_pair_1_2["best_model"] == "model_b"
        assert od_pair_1_2["best_model_mean_log_perplexity"] == 4.5
        assert od_pair_1_2["worst_model"] == "model_a"
        assert od_pair_1_2["performance_delta"] == 0.5

    def test_build_comparison_three_models(self):
        """Test building comparison with three models."""
        evaluation_results = [
            {
                "model": "baseline",
                "trajectories": [
                    {
                        "origin": 10,
                        "destination": 20,
                        "log_perplexity": 8.0,
                        "segment_log_perplexities": [0.8],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "improved",
                "trajectories": [
                    {
                        "origin": 10,
                        "destination": 20,
                        "log_perplexity": 6.5,
                        "segment_log_perplexities": [0.65],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "best",
                "trajectories": [
                    {
                        "origin": 10,
                        "destination": 20,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Verify ranking (best to worst)
        od_pair = result["od_pairs"][(10, 20)]
        assert od_pair["ranking"][0]["model"] == "best"
        assert od_pair["ranking"][0]["rank"] == 1
        assert od_pair["ranking"][1]["model"] == "improved"
        assert od_pair["ranking"][1]["rank"] == 2
        assert od_pair["ranking"][2]["model"] == "baseline"
        assert od_pair["ranking"][2]["rank"] == 3

        # Verify best/worst models
        assert od_pair["best_model"] == "best"
        assert od_pair["worst_model"] == "baseline"

    def test_build_comparison_empty_results(self):
        """Test error handling with empty evaluation results."""
        with pytest.raises(ValueError, match="Empty evaluation results list"):
            _build_cross_model_od_comparison([])

    def test_build_comparison_invalid_results_format(self):
        """Test error handling with invalid results format."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": "not_a_list",  # Invalid format
            }
        ]

        with pytest.raises(
            ValueError, match="All results must contain 'trajectories' list"
        ):
            _build_cross_model_od_comparison(evaluation_results)


class TestBuildCrossModelODComparisonFailedEvaluations:
    """Tests for handling failed evaluations in cross-model comparison."""

    def test_build_comparison_with_failed_evaluations(self):
        """Test that failed evaluations are properly skipped."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": np.inf,
                        "segment_log_perplexities": [],
                        "status": "evaluation_failed",
                        "source_label": None,
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 4.0,
                        "segment_log_perplexities": [0.4],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": np.inf,
                        "segment_log_perplexities": [],
                        "status": "evaluation_failed",
                        "source_label": None,
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Verify only OD pair (3, 4) is in results (valid data from model_a)
        assert "(1, 2)" not in result["od_pairs"]

        # OD pair (3, 4) should have only model_a (model_b failed)
        assert (3, 4) in result["od_pairs"]
        od_pair_3_4 = result["od_pairs"][(3, 4)]
        assert len(od_pair_3_4["per_model_stats"]) == 1
        assert "model_a" in od_pair_3_4["per_model_stats"]

    def test_build_comparison_all_failed_for_od_pair(self):
        """Test OD pair is skipped when all models fail."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": np.inf,
                        "segment_log_perplexities": [],
                        "status": "evaluation_failed",
                        "source_label": None,
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": np.inf,
                        "segment_log_perplexities": [],
                        "status": "evaluation_failed",
                        "source_label": None,
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # OD pair should not appear in results
        assert (1, 2) not in result["od_pairs"]

    def test_build_comparison_with_nan_values(self):
        """Test handling of NaN values in log_perplexity."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": np.nan,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # NaN should be skipped, only model_b should be included
        od_pair = result["od_pairs"][(1, 2)]
        assert len(od_pair["per_model_stats"]) == 1
        assert "model_a" not in od_pair["per_model_stats"]
        assert "model_b" in od_pair["per_model_stats"]


class TestBuildCrossModelODComparisonEdgeCases:
    """Tests for edge cases in cross-model comparison."""

    def test_build_comparison_single_model(self):
        """Test comparison with only one model."""
        evaluation_results = [
            {
                "model": "only_model",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Metadata should reflect single model
        assert result["metadata"]["model_count"] == 1

        # OD pair should still be created
        assert (1, 2) in result["od_pairs"]

        # Best and worst should be the same
        od_pair = result["od_pairs"][(1, 2)]
        assert od_pair["best_model"] == "only_model"
        assert od_pair["worst_model"] == "only_model"
        assert od_pair["performance_delta"] == 0.0

        # Summary should have empty best_performing_models
        assert result["od_summary"]["best_performing_models"]["only_model"]["best"] == 1

    def test_build_comparison_multiple_trajectories_per_od_pair(self):
        """Test multiple trajectories per OD pair."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 6.0,
                        "segment_log_perplexities": [0.6],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 4.5,
                        "segment_log_perplexities": [0.45],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.5,
                        "segment_log_perplexities": [0.55],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        od_pair = result["od_pairs"][(1, 2)]

        # Check per-model stats use mean of trajectories
        assert od_pair["per_model_stats"]["model_a"]["count"] == 2
        assert od_pair["per_model_stats"]["model_a"][
            "mean_log_perplexity"
        ] == pytest.approx(5.5)
        assert od_pair["per_model_stats"]["model_b"][
            "mean_log_perplexity"
        ] == pytest.approx(5.0)

    def test_build_comparison_missing_origin_destination(self):
        """Test handling of missing origin or destination."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": None,  # Missing destination
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": None,  # Missing origin
                        "destination": 2,
                        "log_perplexity": 4.0,
                        "segment_log_perplexities": [0.4],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Both trajectories should be skipped
        assert len(result["od_pairs"]) == 0

    def test_build_comparison_different_source_labels(self):
        """Test handling of different source labels for same OD pair."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 4.0,
                        "segment_log_perplexities": [0.4],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Should use first available label (route_switch from model_a)
        od_pair = result["od_pairs"][(1, 2)]
        assert od_pair["source_label"] == "route_switch"


class TestBuildCrossModelODComparisonStatistics:
    """Tests for statistical calculations in cross-model comparison."""

    def test_build_computation_of_means_and_deltas(self):
        """Test correct computation of mean log perplexity and performance deltas."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 7.0,
                        "segment_log_perplexities": [0.7],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 6.0,
                        "segment_log_perplexities": [0.6],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Check model_a stats
        model_a_stats = result["models"][0]
        assert model_a_stats["name"] == "model_a"
        assert model_a_stats["trajectory_count"] == 2
        assert model_a_stats["failed_count"] == 0
        assert model_a_stats["failed_rate"] == 0.0

        # Check overall statistics
        avg_delta = result["od_summary"]["average_performance_delta"]
        assert avg_delta > 0.0  # Should have some difference

    def test_build_comparison_source_label_statistics(self):
        """Test statistics aggregation by source label."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 8.0,
                        "segment_log_perplexities": [0.8],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 6.0,
                        "segment_log_perplexities": [0.6],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 7.0,
                        "segment_log_perplexities": [0.7],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Check source label distribution
        source_dist = result["od_summary"]["source_label_distribution"]
        assert source_dist["route_switch"] == 1
        assert source_dist["detour"] == 1

        # Check statistics by source label
        stats_by_label = result["od_summary"]["statistics_by_source_label"]
        assert "route_switch" in stats_by_label
        assert "detour" in stats_by_label

        assert stats_by_label["route_switch"]["count"] == 1
        assert stats_by_label["detour"]["count"] == 1

        # Check best models per label
        assert "best_models" in stats_by_label["route_switch"]

    def test_build_comparison_with_varying_deltas(self):
        """Test computation with varying performance deltas."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 10.0,  # Worst
                        "segment_log_perplexities": [1.0],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,  # Best
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        od_pair = result["od_pairs"][(1, 2)]
        assert od_pair["performance_delta"] == 5.0  # 10.0 - 5.0

        # Check summary statistics
        assert result["od_summary"]["min_performance_delta"] == 5.0
        assert result["od_summary"]["max_performance_delta"] == 5.0
        assert result["od_summary"]["std_performance_delta"] == 0.0


class TestBuildCrossModelODComparisonJSONSchema:
    """Tests for JSON schema validation of output structure."""

    def test_metadata_structure(self):
        """Test that metadata has correct structure."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        metadata = result["metadata"]

        # Check required metadata fields
        assert "timestamp" in metadata
        assert "output_path" in metadata
        assert "model_count" in metadata
        assert "model_names" in metadata
        assert "total_trajectories" in metadata
        assert "comparison_type" in metadata
        assert "version" in metadata

        assert metadata["model_count"] == 1
        assert metadata["model_names"] == ["model_a"]
        assert metadata["total_trajectories"] == 1
        assert metadata["comparison_type"] == "cross_model_od_comparison"
        assert metadata["version"] == "1.0"

    def test_per_model_structure(self):
        """Test that per-model structure is correct."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        models = result["models"]
        assert len(models) == 1

        model = models[0]
        assert "name" in model
        assert "trajectory_count" in model
        assert "valid_trajectory_count" in model
        assert "failed_count" in model
        assert "failed_rate" in model
        assert "log_perplexity_stats" in model
        assert "segment_stats" in model
        assert "od_pair_label_counts" in model

    def test_per_od_pair_structure(self):
        """Test that per-OD-pair structure is correct."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        od_pairs = result["od_pairs"]
        assert (1, 2) in od_pairs

        od_pair = od_pairs[(1, 2)]

        # Check required fields
        assert "origin" in od_pair
        assert "destination" in od_pair
        assert "trajectory_count" in od_pair
        assert "source_label" in od_pair
        assert "trajectories" in od_pair
        assert "per_model_stats" in od_pair
        assert "best_model" in od_pair
        assert "best_model_mean_log_perplexity" in od_pair
        assert "worst_model" in od_pair
        assert "worst_model_mean_log_perplexity" in od_pair
        assert "performance_delta" in od_pair
        assert "ranking" in od_pair

        # Check trajectory structure
        assert len(od_pair["trajectories"]) == 1
        trajectory = od_pair["trajectories"][0]
        assert "model" in trajectory
        assert "trajectory_index" in trajectory
        assert "log_perplexity" in trajectory
        assert "segment_log_perplexities" in trajectory
        assert "status" in trajectory

    def test_per_model_stats_structure(self):
        """Test that per-model stats structure is correct."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        od_pair = result["od_pairs"][(1, 2)]
        per_model_stats = od_pair["per_model_stats"]["model_a"]

        # Check required stats fields
        assert "mean_log_perplexity" in per_model_stats
        assert "median_log_perplexity" in per_model_stats
        assert "count" in per_model_stats
        assert "std_log_perplexity" in per_model_stats
        assert "min_log_perplexity" in per_model_stats
        assert "max_log_perplexity" in per_model_stats

    def test_ranking_structure(self):
        """Test that ranking structure is correct."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 6.0,
                        "segment_log_perplexities": [0.6],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 4.0,
                        "segment_log_perplexities": [0.4],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        od_pair = result["od_pairs"][(1, 2)]
        ranking = od_pair["ranking"]

        # Check ranking structure
        assert len(ranking) == 2
        assert ranking[0]["model"] == "model_b"  # Best first
        assert ranking[0]["rank"] == 1
        assert ranking[0]["mean_log_perplexity"] == 4.0
        assert ranking[1]["model"] == "model_a"  # Worst second
        assert ranking[1]["rank"] == 2
        assert ranking[1]["mean_log_perplexity"] == 6.0

    def test_od_summary_structure(self):
        """Test that OD summary structure is correct."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        od_summary = result["od_summary"]

        # Check required summary fields
        assert "total_unique_od_pairs" in od_summary
        assert "od_pairs_with_all_models" in od_summary
        assert "average_performance_delta" in od_summary
        assert "std_performance_delta" in od_summary
        assert "min_performance_delta" in od_summary
        assert "max_performance_delta" in od_summary
        assert "best_performing_models" in od_summary
        assert "source_label_distribution" in od_summary
        assert "statistics_by_source_label" in od_summary

    def test_log_perplexity_stats_structure(self):
        """Test structure of log_perplexity_stats."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 6.0,
                        "segment_log_perplexities": [0.6],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        model = result["models"][0]
        stats = model["log_perplexity_stats"]

        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats

        assert stats["mean"] == pytest.approx(5.5)

    def test_segment_stats_structure(self):
        """Test structure of segment_stats."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5, 0.6],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        model = result["models"][0]
        segment_stats = model["segment_stats"]

        assert "max_segment_length" in segment_stats
        assert "per_index" in segment_stats

        assert segment_stats["max_segment_length"] == 2
        assert len(segment_stats["per_index"]) == 2


class TestBuildCrossModelODComparisonOutputPath:
    """Tests for file output functionality."""

    @patch("builtins.open")
    @patch("tools.evaluate_lmtad_spatial_abnormal.json.dump")
    def test_save_to_file(self, mock_json_dump, mock_open, tmp_path):
        """Test that results are saved to file when output_path is provided."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        output_file = tmp_path / "comparison.json"

        # Create parent directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a mock file object
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = _build_cross_model_od_comparison(
            evaluation_results, output_path=output_file
        )

        # Verify file was opened for writing
        mock_open.assert_called_once_with(output_file, "w")

        # Verify json.dump was called
        mock_json_dump.assert_called_once()

        # Verify output_path in metadata
        assert result["metadata"]["output_path"] == str(output_file)

    @patch("builtins.open")
    @patch("tools.evaluate_lmtad_spatial_abnormal.json.dump")
    @patch("pathlib.Path.mkdir")
    def test_save_to_file_creates_parent_directories(
        self, mock_mkdir, mock_json_dump, mock_open, tmp_path
    ):
        """Test that parent directories are created when saving."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        output_file = tmp_path / "deep" / "nested" / "path" / "comparison.json"

        # Create a mock file object
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = _build_cross_model_od_comparison(
            evaluation_results, output_path=output_file
        )

        # Verify mkdir was called with parents=True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_no_output_path_metadata(self):
        """Test that output_path is None in metadata when not provided."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        assert result["metadata"]["output_path"] is None


class TestBuildCrossModelODComparisonBestWorstTracking:
    """Tests for tracking best and worst performing models."""

    def test_best_worst_model_counts(self):
        """Test tracking of how often each model is best/worst."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 10.0,  # Worst for this OD
                        "segment_log_perplexities": [1.0],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 5.0,  # Best for this OD
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,  # Best for this OD
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 10.0,  # Worst for this OD
                        "segment_log_perplexities": [1.0],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        best_models = result["od_summary"]["best_performing_models"]

        # Each model should be best once and worst once
        assert best_models["model_a"]["best"] == 1
        assert best_models["model_a"]["worst"] == 1
        assert best_models["model_b"]["best"] == 1
        assert best_models["model_b"]["worst"] == 1

    def test_consistently_best_model(self):
        """Test identification of a consistently best performing model."""
        evaluation_results = [
            {
                "model": "good_model",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 3.0,
                        "segment_log_perplexities": [0.3],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 4.0,
                        "segment_log_perplexities": [0.4],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "bad_model",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 10.0,
                        "segment_log_perplexities": [1.0],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 11.0,
                        "segment_log_perplexities": [1.1],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        best_models = result["od_summary"]["best_performing_models"]

        # good_model should be best for both OD pairs
        assert best_models["good_model"]["best"] == 2
        assert best_models["good_model"]["worst"] == 0

        # bad_model should be worst for both OD pairs
        assert best_models["bad_model"]["best"] == 0
        assert best_models["bad_model"]["worst"] == 2


class TestBuildCrossModelODComparisonOdSummary:
    """Tests for OD summary statistics."""

    def test_od_pairs_with_all_models(self):
        """Test counting of OD pairs evaluated by all models."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 4.0,
                        "segment_log_perplexities": [0.4],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        summary = result["od_summary"]
        assert summary["od_pairs_with_all_models"] == 1
        assert summary["total_unique_od_pairs"] == 1

    def test_performance_delta_statistics(self):
        """Test computation of performance delta statistics."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 10.0,
                        "segment_log_perplexities": [1.0],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        summary = result["od_summary"]
        assert summary["average_performance_delta"] == 5.0
        assert summary["std_performance_delta"] == 0.0
        assert summary["min_performance_delta"] == 5.0
        assert summary["max_performance_delta"] == 5.0


class TestBuildCrossModelODComparisonBackwardCompatibility:
    """Tests for backward compatibility and data format flexibility."""

    def test_handles_missing_source_label(self):
        """Test that missing source_label is handled gracefully."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        # No source_label
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Should work without source_label
        assert (1, 2) in result["od_pairs"]
        od_pair = result["od_pairs"][(1, 2)]
        assert od_pair["source_label"] is None

        # Summary should show None or unknown
        summary = result["od_summary"]
        assert (
            "None" in summary["source_label_distribution"]
            or "unknown" in summary["source_label_distribution"]
        )

    def test_handles_missing_trajectory_index(self):
        """Test that missing trajectory_index is handled."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        # No trajectory_index
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Should still work
        od_pair = result["od_pairs"][(1, 2)]
        trajectory = od_pair["trajectories"][0]
        assert trajectory["trajectory_index"] is None

    def test_handles_empty_segment_log_perplexities(self):
        """Test that empty segment_log_perplexities are handled."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [],  # Empty
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    }
                ],
            }
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Should work with empty segments
        assert (1, 2) in result["od_pairs"]

        model = result["models"][0]
        assert model["segment_stats"]["max_segment_length"] == 0

    def test_mixed_success_failure_scenarios(self):
        """Test complex scenario with mixed success/failure across multiple OD pairs."""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 5.0,
                        "segment_log_perplexities": [0.5],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": np.inf,
                        "segment_log_perplexities": [],
                        "status": "evaluation_failed",
                        "source_label": None,
                        "trajectory_index": 1,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories": [
                    {
                        "origin": 1,
                        "destination": 2,
                        "log_perplexity": 6.0,
                        "segment_log_perplexities": [0.6],
                        "status": "ok",
                        "source_label": "route_switch",
                        "trajectory_index": 0,
                    },
                    {
                        "origin": 3,
                        "destination": 4,
                        "log_perplexity": 7.0,
                        "segment_log_perplexities": [0.7],
                        "status": "ok",
                        "source_label": "detour",
                        "trajectory_index": 1,
                    },
                ],
            },
        ]

        result = _build_cross_model_od_comparison(evaluation_results)

        # Only OD pair (1, 2) should be in results (both models have valid data)
        assert (1, 2) in result["od_pairs"]
        assert (3, 4) in result["od_pairs"]  # Actually model_b has valid data

        # Check model_a stats
        model_a = next(m for m in result["models"] if m["name"] == "model_a")
        assert model_a["trajectory_count"] == 2
        assert model_a["failed_count"] == 1
        assert model_a["failed_rate"] == 50.0
