"""Tests for updated LM-TAD Spatial Results Analysis Module

This module tests the updated analyze_lmtad_spatial_results.py which has been
refactored to use perplexity-based metrics instead of classification-based metrics.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pytest

# Import functions to test
from tools.analyze_lmtad_spatial_results import (
    # Backward compatibility wrappers
    aggregate_lmtad_spatial_results,
    load_source_real_rates,
    compute_statistical_test,
    # New perplexity-based functions
    build_od_pair_data,
    compute_per_od_pair_statistics,
    compare_perplexity_distributions,
    paired_perplexity_test,
    aggregate_lmtad_perplexity_results,
    load_source_perplexity_rates,
    load_evaluation_result,
    ensure_json_serializable,
    PerplexityEvaluationMetrics,
    PerplexityStatisticalComparison,
)


class TestBackwardCompatibilityWrappers:
    """Test backward compatibility wrappers for old API"""

    def test_aggregate_lmtad_spatial_results_deprecation_warning(self, caplog):
        """Test that deprecated function shows warning"""
        with patch(
            "tools.analyze_lmtad_spatial_results.aggregate_lmtad_perplexity_results"
        ) as mock_aggregate:
            mock_aggregate.return_value = {
                "statistical_analysis": {
                    "distribution_tests": [
                        {
                            "model": "test_model",
                            "mean_perplexity": 7.5,
                            "ci_lower": 7.0,
                            "ci_upper": 8.0,
                            "effect_size": "large",
                            "cohens_h": 0.8,
                            "significant": True,
                        }
                    ]
                }
            }

            result = aggregate_lmtad_spatial_results(
                eval_dir=Path("/fake"),
                dataset="test",
                source_eval_dir=Path("/fake"),
            )

            # Check deprecation warning was logged
            assert any("deprecated" in record.message for record in caplog.records)
            assert result is not None
            assert "statistical_analysis" in result
            assert "statistical_tests" in result["statistical_analysis"]

    def test_load_source_real_rates_deprecation_warning(self, caplog):
        """Test that deprecated function shows warning"""
        with patch(
            "tools.analyze_lmtad_spatial_results.load_source_perplexity_rates"
        ) as mock_load:
            mock_load.return_value = {"log_perplexity_stats": {"mean": 5.0}}

            result = load_source_real_rates(Path("/fake"))

            # Check deprecation warning was logged
            assert any("deprecated" in record.message for record in caplog.records)
            assert result is not None
            assert "spatial_abnormality_rate" in result

    def test_compute_statistical_test_deprecation_warning(self, caplog):
        """Test that deprecated function shows warning"""
        with patch(
            "tools.analyze_lmtad_spatial_results.compare_perplexity_distributions"
        ) as mock_compare:
            mock_compare.return_value = {
                "ks_statistic": 0.5,
                "ks_p_value": 0.01,
                "pvalue": 0.01,  # Note: function returns 'pvalue' not 'p_value'
                "statistic": 0.5,
            }

            result = compute_statistical_test(
                results_1={},
                results_2={},
                real_count=100,
                generated_count=100,
            )

            # Check deprecation warning was logged
            assert any("deprecated" in record.message for record in caplog.records)
            assert result is not None
            assert "chi2" in result
            assert "p_value" in result

    def test_backward_compatibility_conversions(self):
        """Test proper conversion from new to old format"""
        mock_result = {
            "statistical_analysis": {
                "distribution_tests": [
                    {
                        "model": "test_model",
                        "mean_perplexity": 7.5,
                        "ci_lower": 7.0,
                        "ci_upper": 8.0,
                        "effect_size": "large",
                        "cohens_h": 0.8,
                        "significant": True,
                    }
                ]
            }
        }

        with patch(
            "tools.analyze_lmtad_spatial_results.aggregate_lmtad_perplexity_results",
            return_value=mock_result,
        ):
            result = aggregate_lmtad_spatial_results(
                eval_dir=Path("/fake"),
                dataset="test",
                source_eval_dir=Path("/fake"),
            )

            # Verify old format keys are present
            assert "statistical_analysis" in result
            assert "statistical_tests" in result["statistical_analysis"]
            assert len(result["statistical_analysis"]["statistical_tests"]) == 1

            # Verify conversion from perplexity to percentage
            old_test = result["statistical_analysis"]["statistical_tests"][0]
            assert old_test["generated_rate"] == 75.0  # 7.5 * 10
            assert old_test["model"] == "test_model"


class TestPerplexityEvaluationMetrics:
    """Test PerplexityEvaluationMetrics dataclass"""

    def test_perplexity_evaluation_metrics_creation(self):
        """Test creating PerplexityEvaluationMetrics instance"""
        metrics = PerplexityEvaluationMetrics(
            dataset="porto",
            model="test_model",
            is_real=False,
            total_trajectories=100,
            log_perplexity_stats={"mean": 7.5, "std": 1.2},
            segment_log_perplexities=[[1.0, 2.0], [1.5, 2.5]],
            od_pair_data={"od1": {"test": 123}},
        )

        assert metrics.dataset == "porto"
        assert metrics.model == "test_model"
        assert metrics.is_real is False
        assert metrics.total_trajectories == 100
        assert metrics.log_perplexity_stats["mean"] == 7.5
        assert len(metrics.segment_log_perplexities) == 2
        assert "od1" in metrics.od_pair_data

    def test_perplexity_evaluation_metrics_real_data(self):
        """Test metrics for real data (no model)"""
        metrics = PerplexityEvaluationMetrics(
            dataset="porto",
            model=None,
            is_real=True,
            total_trajectories=1000,
            log_perplexity_stats={"mean": 5.0, "std": 0.8},
        )

        assert metrics.model is None
        assert metrics.is_real is True


class TestEnsureJsonSerializable:
    """Test JSON serialization helper function"""

    def test_ensure_json_serializable_dict(self):
        """Test converting dict with numpy types"""
        data = {
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(3.14),
            "numpy_bool": np.bool_(True),
            "regular_int": 10,
            "regular_float": 2.5,
            "regular_bool": False,
            "nested": {
                "numpy_int": np.int32(100),
                "list": [1, 2, np.int64(3)],
            },
            "none_value": None,
            "string": "test",
        }

        result = ensure_json_serializable(data)

        assert isinstance(result["numpy_int"], int)
        assert result["numpy_int"] == 42
        assert isinstance(result["numpy_float"], float)
        assert result["numpy_float"] == 3.14
        assert isinstance(result["numpy_bool"], bool)
        assert result["numpy_bool"] is True
        assert isinstance(result["nested"]["numpy_int"], int)
        assert result["nested"]["numpy_int"] == 100
        assert isinstance(result["nested"]["list"][2], int)
        assert result["nested"]["list"][2] == 3
        assert result["none_value"] is None
        assert result["string"] == "test"

    def test_ensure_json_serializable_list(self):
        """Test converting list with numpy types"""
        data = [np.int64(1), np.float64(2.5), np.bool_(True), "string", None]

        result = ensure_json_serializable(data)

        assert isinstance(result[0], int)
        assert result[0] == 1
        assert isinstance(result[1], float)
        assert result[1] == 2.5
        assert isinstance(result[2], bool)
        assert result[2] is True
        assert result[3] == "string"
        assert result[4] is None

    def test_ensure_json_serializable_tuple(self):
        """Test converting tuple to list"""
        data = (1, 2, 3)

        result = ensure_json_serializable(data)

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_ensure_json_serializable_unknown_type(self):
        """Test converting unknown type to string"""
        data = {"value": Path("/test/path")}

        result = ensure_json_serializable(data)

        assert isinstance(result["value"], str)
        assert result["value"] == "/test/path"


class TestBuildOdPairData:
    """Test build_od_pair_data function"""

    def test_build_od_pair_data_basic(self):
        """Test building OD pair data from evaluation results"""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories_with_perplexity": [
                    {
                        "road_sequence": [1, 2, 3, 4],
                        "log_perplexity": 7.5,
                        "segment_log_perplexities": [1.0, 1.5, 2.0, 2.0],
                    },
                    {
                        "road_sequence": [5, 6, 7],
                        "log_perplexity": 8.0,
                    },
                ],
            },
            {
                "model": "model_b",
                "trajectories_with_perplexity": [
                    {
                        "road_sequence": [1, 2, 3, 4],
                        "log_perplexity": 7.0,
                        "segment_log_perplexities": [0.9, 1.4, 1.9, 1.9],
                    },
                ],
            },
        ]

        result = build_od_pair_data(evaluation_results)

        # Check OD pair keys
        assert "1-4" in result
        assert "5-7" in result

        # Check model data for OD pair 1-4
        assert "model_a" in result["1-4"]
        assert "model_b" in result["1-4"]
        assert result["1-4"]["model_a"]["log_perplexity"] == 7.5
        assert result["1-4"]["model_b"]["log_perplexity"] == 7.0
        assert len(result["1-4"]["model_a"]["segment_log_perplexities"]) == 4

        # Check OD pair 5-7 (only model_a)
        assert "model_a" in result["5-7"]
        assert "model_b" not in result["5-7"]
        assert result["5-7"]["model_a"]["log_perplexity"] == 8.0

    def test_build_od_pair_data_insufficient_roads(self):
        """Test handling of trajectories with insufficient roads"""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories_with_perplexity": [
                    {"road_sequence": [1], "log_perplexity": 7.5},  # Only one road
                    {"road_sequence": [], "log_perplexity": 8.0},  # Empty sequence
                ],
            }
        ]

        result = build_od_pair_data(evaluation_results)

        # Should not create any OD pairs
        assert len(result) == 0

    def test_build_od_pair_data_missing_log_perplexity(self):
        """Test handling of trajectories without log_perplexity"""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories_with_perplexity": [
                    {
                        "road_sequence": [1, 2, 3, 4],
                        "log_perplexity": None,  # Explicitly None
                    },
                ],
            }
        ]

        result = build_od_pair_data(evaluation_results)

        # OD pair key is created but has no model data when log_perplexity is None
        assert len(result) == 1
        assert "1-4" in result
        assert len(result["1-4"]) == 0  # Empty because no valid perplexity data

    def test_build_od_pair_data_with_trajectory_metadata(self):
        """Test that trajectory metadata is preserved"""
        evaluation_results = [
            {
                "model": "model_a",
                "trajectories_with_perplexity": [
                    {
                        "road_sequence": [1, 2, 3, 4],
                        "log_perplexity": 7.5,
                        "custom_field": "test_value",
                    },
                ],
            }
        ]

        result = build_od_pair_data(evaluation_results)

        assert "1-4" in result
        assert "model_a" in result["1-4"]
        assert "trajectory" in result["1-4"]["model_a"]
        assert result["1-4"]["model_a"]["trajectory"]["custom_field"] == "test_value"


class TestComputePerOdPairStatistics:
    """Test compute_per_od_pair_statistics function"""

    def test_compute_per_od_pair_statistics_basic(self):
        """Test computing statistics for multiple models"""
        od_pair_data = {
            "1-4": {
                "model_a": {"log_perplexity": 7.5},
                "model_b": {"log_perplexity": 7.0},
                "model_c": {"log_perplexity": 8.0},
            },
            "5-7": {
                "model_a": {"log_perplexity": 6.5},
                "model_b": {"log_perplexity": 6.0},
            },
            "8-10": {
                "model_a": {"log_perplexity": 9.0},
            },
        }
        models = ["model_a", "model_b", "model_c"]

        result = compute_per_od_pair_statistics(od_pair_data, models)

        # Only OD pairs with 2+ models should be included
        assert "1-4" in result
        assert "5-7" in result
        assert "8-10" not in result  # Only 1 model

        # Check statistics for OD pair 1-4
        stats_1_4 = result["1-4"]
        assert stats_1_4["od_key"] == "1-4"
        # models should contain only models in the od_pair for this specific pair
        assert len(stats_1_4["models"]) == 3
        assert set(stats_1_4["models"]) == {"model_a", "model_b", "model_c"}
        assert stats_1_4["perplexities"]["model_a"] == 7.5
        assert stats_1_4["perplexities"]["model_b"] == 7.0
        assert stats_1_4["perplexities"]["model_c"] == 8.0
        assert stats_1_4["mean_log_perplexity"] == pytest.approx(7.5)
        # std of [7.5, 7.0, 8.0] = sqrt(((7.5-7.5)^2 + (7.0-7.5)^2 + (8.0-7.5)^2)/3) = sqrt(0.166...) = 0.408
        assert stats_1_4["std_log_perplexity"] == pytest.approx(0.408, abs=0.01)
        assert stats_1_4["min_log_perplexity"] == 7.0
        assert stats_1_4["max_log_perplexity"] == 8.0

    def test_compute_per_od_pair_statistics_insufficient_models(self):
        """Test with only one model available"""
        od_pair_data = {
            "1-4": {
                "model_a": {"log_perplexity": 7.5},
            },
        }
        models = ["model_a", "model_b"]

        result = compute_per_od_pair_statistics(od_pair_data, models)

        # Should return empty dict (need 2+ models for comparison)
        assert len(result) == 0

    def test_compute_per_od_pair_statistics_missing_perplexity(self):
        """Test handling of OD pairs with missing perplexity values"""
        od_pair_data = {
            "1-4": {
                "model_a": {"log_perplexity": 7.5},
                "model_b": {"log_perplexity": None},  # Missing value
            },
        }
        models = ["model_a", "model_b"]

        result = compute_per_od_pair_statistics(od_pair_data, models)

        # Should still work but only include models with valid perplexities
        # Since model_b has None, only model_a is available, so OD pair should not be included
        assert "1-4" not in result


class TestComparePerplexityDistributions:
    """Test compare_perplexity_distributions function"""

    def test_compare_perplexity_distributions_basic(self):
        """Test basic distribution comparison"""
        perplexities_1 = np.array([7.0, 7.5, 8.0, 8.5, 9.0])
        perplexities_2 = np.array([6.0, 6.5, 7.0, 7.5, 8.0])

        result = compare_perplexity_distributions(
            perplexities_1, perplexities_2, "model_a", "model_b"
        )

        assert "ks_statistic" in result
        assert "ks_p_value" in result
        assert "mannwhitney_u_statistic" in result
        assert "mannwhitney_u_p_value" in result
        assert "significant_ks" in result
        assert "significant_mw" in result

        # Verify result types
        assert isinstance(result["ks_statistic"], float)
        assert isinstance(result["ks_p_value"], float)
        assert isinstance(result["significant_ks"], bool)

    def test_compare_perplexity_distributions_empty_arrays(self):
        """Test handling of empty arrays"""
        result = compare_perplexity_distributions(
            np.array([]), np.array([1, 2, 3]), "model_a", "model_b"
        )

        # Should return NaN values for empty input
        assert np.isnan(result["ks_statistic"])
        assert np.isnan(result["ks_p_value"])
        assert result["significant_ks"] is False

    def test_compare_perplexity_distributions_single_value(self):
        """Test with single value arrays"""
        perplexities_1 = np.array([7.5])
        perplexities_2 = np.array([6.5])

        result = compare_perplexity_distributions(
            perplexities_1, perplexities_2, "model_a", "model_b"
        )

        # Should handle single values gracefully or return NaN
        assert "ks_statistic" in result
        assert "ks_p_value" in result
        # With only one value each, statistical tests may not be meaningful
        assert isinstance(result["ks_statistic"], (float, int))
        assert isinstance(result["ks_p_value"], (float, int))

    def test_compare_perplexity_distributions_with_nan(self):
        """Test handling of arrays containing NaN"""
        perplexities_1 = np.array([7.0, 7.5, np.nan, 8.0])
        perplexities_2 = np.array([6.0, 6.5, 7.0, np.nan])

        # scipy.stats functions should handle NaN appropriately
        result = compare_perplexity_distributions(
            perplexities_1, perplexities_2, "model_a", "model_b"
        )

        # Results should be valid (NaN handling depends on scipy version)
        assert "ks_statistic" in result
        assert "ks_p_value" in result


class TestPairedPerplexityTest:
    """Test paired_perplexity_test function"""

    def test_paired_perplexity_test_basic(self):
        """Test basic paired t-test on OD pairs"""
        # Using OrderedDict or explicit order to ensure consistent iteration
        # Python 3.7+ maintains insertion order, so this should work
        from collections import OrderedDict
        od_pair_data = OrderedDict([
            ("od1", {
                "model_a": {"log_perplexity": 7.5},
                "model_b": {"log_perplexity": 7.0},
            }),
            ("od2", {
                "model_a": {"log_perplexity": 8.0},
                "model_b": {"log_perplexity": 7.5},
            }),
            ("od3", {
                "model_a": {"log_perplexity": 6.5},
                "model_b": {"log_perplexity": 6.0},
            }),
        ])
        models = ["model_a", "model_b"]

        result = paired_perplexity_test(od_pair_data, models, min_pairs=3)

        # Should have one comparison (model_a vs model_b)
        assert len(result) == 1

        comparison = result[0]
        assert comparison["model_1"] == "model_a"
        assert comparison["model_2"] == "model_b"
        assert comparison["shared_od_pairs"] == 3
        # Mean difference: (7.5-7.0) + (8.0-7.5) + (6.5-6.0) = 0.5 + 0.5 + 0.5 = 1.5
        # Mean diff = 1.5 / 3 = 0.5
        assert comparison["mean_diff"] == pytest.approx(0.5)
        assert "cohens_d" in comparison
        assert "t_statistic" in comparison
        assert "p_value" in comparison
        assert "significant" in comparison

    def test_paired_perplexity_test_insufficient_pairs(self):
        """Test with insufficient paired OD pairs"""
        od_pair_data = {
            "od1": {
                "model_a": {"log_perplexity": 7.5},
                "model_b": {"log_perplexity": 7.0},
            },
        }
        models = ["model_a", "model_b"]

        result = paired_perplexity_test(od_pair_data, models, min_pairs=5)

        # Should return empty list (insufficient pairs)
        assert len(result) == 0

    def test_paired_perplexity_test_multiple_models(self):
        """Test with multiple model pairs"""
        # Using OrderedDict to ensure consistent iteration order
        from collections import OrderedDict
        od_pair_data = OrderedDict([
            ("od1", {
                "model_a": {"log_perplexity": 7.5},
                "model_b": {"log_perplexity": 7.0},
                "model_c": {"log_perplexity": 8.0},
            }),
            ("od2", {
                "model_a": {"log_perplexity": 8.0},
                "model_b": {"log_perplexity": 7.5},
                "model_c": {"log_perplexity": 8.5},
            }),
            ("od3", {
                "model_a": {"log_perplexity": 6.0},
                "model_b": {"log_perplexity": 5.5},
                "model_c": {"log_perplexity": 7.0},
            }),
        ])
        models = ["model_a", "model_b", "model_c"]

        result = paired_perplexity_test(od_pair_data, models, min_pairs=3)

        # Should have 3 comparisons: (a,b), (a,c), (b,c)
        assert len(result) == 3
        
        model_pairs = {(r["model_1"], r["model_2"]) for r in result}
        assert ("model_a", "model_b") in model_pairs
        assert ("model_a", "model_c") in model_pairs
        assert ("model_b", "model_c") in model_pairs
        
        # Verify each comparison has correct number of shared OD pairs
        for comparison in result:
            assert comparison["shared_od_pairs"] == 3

    def test_paired_perplexity_test_missing_values(self):
        """Test handling of missing perplexity values"""
        # Using OrderedDict to ensure consistent iteration
        from collections import OrderedDict
        od_pair_data = OrderedDict([
            ("od1", {
                "model_a": {"log_perplexity": 7.5},
                "model_b": {"log_perplexity": None},  # Missing
            }),
            ("od2", {
                "model_a": {"log_perplexity": 8.0},
                "model_b": {"log_perplexity": 7.5},
            }),
            ("od3", {
                "model_a": {"log_perplexity": None},  # Missing
                "model_b": {"log_perplexity": 6.0},
            }),
        ])
        models = ["model_a", "model_b"]

        result = paired_perplexity_test(od_pair_data, models, min_pairs=1)

        # Should only use OD pairs with both values (od2 only)
        assert len(result) == 1
        assert result[0]["shared_od_pairs"] == 1
        assert result[0]["model_1"] == "model_a"
        assert result[0]["model_2"] == "model_b"

    def test_paired_perplexity_test_zero_std(self):
        """Test with zero standard deviation (identical values)"""
        od_pair_data = {
            f"od{i}": {
                "model_a": {"log_perplexity": 7.5},
                "model_b": {"log_perplexity": 7.5},
            }
            for i in range(10)
        }
        models = ["model_a", "model_b"]

        result = paired_perplexity_test(od_pair_data, models)

        # Should handle zero std gracefully
        assert len(result) == 1
        assert result[0]["cohens_d"] == 0  # No difference


class TestLoadEvaluationResult:
    """Test load_evaluation_result function"""

    def test_load_evaluation_result_success(self, tmp_path):
        """Test successful loading of evaluation result"""
        result_file = tmp_path / "result.json"
        test_data = {"model": "test_model", "perplexity": 7.5}

        with open(result_file, "w") as f:
            json.dump(test_data, f)

        result = load_evaluation_result(result_file)

        assert result == test_data
        assert result["model"] == "test_model"

    def test_load_evaluation_result_file_not_found(self):
        """Test handling of missing file"""
        result = load_evaluation_result(Path("/nonexistent/file.json"))

        # Should return None for missing file
        assert result is None

    def test_load_evaluation_result_invalid_json(self, tmp_path):
        """Test handling of invalid JSON"""
        result_file = tmp_path / "invalid.json"
        result_file.write_text("{ invalid json ")

        result = load_evaluation_result(result_file)

        # Should return None for invalid JSON
        assert result is None


class TestLoadSourcePerplexityRates:
    """Test load_source_perplexity_rates function"""

    def test_load_source_perplexity_rates_currently_returns_none(self, tmp_path):
        """Test that function currently returns None (not implemented)"""
        result = load_source_perplexity_rates(tmp_path)

        # Currently returns None (placeholder implementation)
        assert result is None


class TestAggregateLmtadPerplexityResults:
    """Test aggregate_lmtad_perplexity_results main function"""

    @patch("tools.analyze_lmtad_spatial_results.load_source_perplexity_rates")
    def test_aggregate_lmtad_perplexity_results_basic(
        self, mock_load_source, tmp_path
    ):
        """Test basic aggregation of perplexity results"""
        # Setup mock data
        mock_load_source.return_value = None

        # Create evaluation directory structure
        eval_dir = tmp_path / "eval"
        dataset_dir = eval_dir / "eval_lmtad_spatial" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        # Create mock evaluation result files
        result_file_1 = dataset_dir / "model_a_spatial_evaluation.json"
        result_file_1.write_text(
            json.dumps(
                {
                    "model": "model_a",
                    "log_perplexity_stats": {
                        "mean": 7.5,
                        "std": 1.2,
                        "count": 100,
                    },
                    "trajectories_with_perplexity": [
                        {
                            "road_sequence": [1, 2, 3],
                            "log_perplexity": 7.5,
                        }
                    ],
                }
            )
        )

        result_file_2 = dataset_dir / "model_b_spatial_evaluation.json"
        result_file_2.write_text(
            json.dumps(
                {
                    "model": "model_b",
                    "log_perplexity_stats": {
                        "mean": 7.0,
                        "std": 1.0,
                        "count": 100,
                    },
                    "trajectories_with_perplexity": [
                        {
                            "road_sequence": [1, 2, 3],
                            "log_perplexity": 7.0,
                        }
                    ],
                }
            )
        )

        # Run aggregation
        result = aggregate_lmtad_perplexity_results(
            eval_dir=eval_dir,
            dataset="test_dataset",
            source_eval_dir=tmp_path / "source",
        )

        # Verify result structure
        assert "summary_statistics" in result
        assert "generated_data" in result
        assert "od_pair_data" in result
        assert "per_od_pair_statistics" in result
        assert "statistical_analysis" in result

        # Verify dataset in results
        assert "test_dataset" in result["generated_data"]
        assert "model_a" in result["generated_data"]["test_dataset"]
        assert "model_b" in result["generated_data"]["test_dataset"]

        # Verify summary statistics
        summary = result["summary_statistics"]["test_dataset"]
        assert summary["total_models"] == 2
        assert "model_a" in summary["model_names"]
        assert "model_b" in summary["model_names"]

    @patch("tools.analyze_lmtad_spatial_results.load_source_perplexity_rates")
    def test_aggregate_lmtad_perplexity_results_no_eval_dir(self, mock_load_source):
        """Test handling of missing evaluation directory"""
        result = aggregate_lmtad_perplexity_results(
            eval_dir=Path("/nonexistent"),
            dataset="test",
            source_eval_dir=Path("/fake"),
        )

        # Should return empty dict
        assert result == {}

    @patch("tools.analyze_lmtad_spatial_results.load_source_perplexity_rates")
    def test_aggregate_lmtad_perplexity_results_no_result_files(self, mock_load_source, tmp_path):
        """Test handling when no result files are found"""
        eval_dir = tmp_path / "eval"
        dataset_dir = eval_dir / "eval_lmtad_spatial" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        # No result files created
        mock_load_source.return_value = None

        result = aggregate_lmtad_perplexity_results(
            eval_dir=eval_dir,
            dataset="test_dataset",
            source_eval_dir=tmp_path / "source",
        )

        # Should return empty dict
        assert result == {}

    @patch("tools.analyze_lmtad_spatial_results.load_source_perplexity_rates")
    def test_aggregate_lmtad_perplexity_results_fdr_correction(self, mock_load_source, tmp_path):
        """Test FDR correction is applied correctly"""
        eval_dir = tmp_path / "eval"
        dataset_dir = eval_dir / "eval_lmtad_spatial" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        # Create multiple result files to trigger multiple comparisons
        for i in range(5):
            result_file = dataset_dir / f"model_{i}_spatial_evaluation.json"
            result_file.write_text(
                json.dumps(
                    {
                        "model": f"model_{i}",
                        "log_perplexity_stats": {
                            "mean": 7.0 + i * 0.1,
                            "std": 1.0,
                            "count": 100,
                        },
                        "trajectories_with_perplexity": [
                            {
                                "road_sequence": [1, 2, 3],
                                "log_perplexity": 7.0 + i * 0.1,
                            }
                        ],
                    }
                )
            )

        mock_load_source.return_value = None

        result = aggregate_lmtad_perplexity_results(
            eval_dir=eval_dir,
            dataset="test_dataset",
            source_eval_dir=tmp_path / "source",
        )

        # Check that distribution tests include FDR correction
        dist_tests = result["statistical_analysis"]["distribution_tests"]
        if dist_tests:
            # Check that corrected p-values are present
            for test in dist_tests:
                if "ks_p_value" in test and not np.isnan(test["ks_p_value"]):
                    assert "ks_p_value_corrected" in test
                    assert "ks_significant_corrected" in test

        # Check that paired tests include FDR correction
        paired_tests = result["statistical_analysis"]["paired_tests"]
        if paired_tests:
            for test in paired_tests:
                if "p_value" in test and not np.isnan(test["p_value"]):
                    assert "p_value_corrected" in test
                    assert "significant_corrected" in test

    @patch("tools.analyze_lmtad_spatial_results.load_source_perplexity_rates")
    def test_aggregate_lmtad_perplexity_results_json_serializable(self, mock_load_source, tmp_path):
        """Test that result is JSON serializable"""
        eval_dir = tmp_path / "eval"
        dataset_dir = eval_dir / "eval_lmtad_spatial" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        result_file = dataset_dir / "model_a_spatial_evaluation.json"
        # Use regular Python types, not numpy types
        result_file.write_text(
            json.dumps(
                {
                    "model": "model_a",
                    "log_perplexity_stats": {
                        "mean": 7.5,  # Regular float
                        "std": 1.2,  # Regular float
                        "count": 100,
                    },
                }
            )
        )

        mock_load_source.return_value = None

        result = aggregate_lmtad_perplexity_results(
            eval_dir=eval_dir,
            dataset="test_dataset",
            source_eval_dir=tmp_path / "source",
        )

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Verify it can be loaded back
        loaded = json.loads(json_str)
        assert loaded == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
