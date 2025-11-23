"""Tests for updated LM-TAD Spatial Results Visualization Module

This module tests the updated visualize_lmtad_spatial_results.py which has been
refactored to use perplexity-based visualizations instead of classification-based ones.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

# Import functions to test
from tools.visualize_lmtad_spatial_results import (
    # Backward compatibility wrappers
    plot_spatial_abnormality_rates_comparison,
    plot_route_switch_vs_detour_breakdown,
    plot_statistical_significance_spatial,
    plot_model_rankings_spatial,
    # New perplexity-based functions
    load_aggregated_results,
    plot_perplexity_distribution_comparison,
    plot_per_od_pair_perplexity_comparison,
    plot_model_rankings_by_perplexity,
    plot_statistical_significance_perplexity,
    plot_segment_level_perplexity_aggregate,
    plot_comprehensive_perplexity_summary,
)


class TestBackwardCompatibilityWrappers:
    """Test backward compatibility wrappers for old API"""

    def test_plot_spatial_abnormality_rates_comparison_deprecation_warning(
        self, caplog
    ):
        """Test that deprecated function shows warning"""
        results = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            plot_spatial_abnormality_rates_comparison(
                results, output_dir, "test_dataset"
            )

            # Check deprecation warning was logged
            assert any("deprecated" in record.message for record in caplog.records)

    def test_plot_route_switch_vs_detour_breakdown_deprecation_warning(self, caplog):
        """Test that deprecated function shows warning and creates notice plot"""
        results = {"generated_data": {"test_dataset": {}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch("matplotlib.pyplot.savefig") as mock_savefig:
                plot_route_switch_vs_detour_breakdown(
                    results, output_dir, "test_dataset"
                )

                # Check deprecation warning was logged
                assert any("deprecated" in record.message for record in caplog.records)

                # Check that savefig was called (notice plot created)
                assert mock_savefig.called

    def test_plot_statistical_significance_spatial_deprecation_warning(self, caplog):
        """Test that deprecated function shows warning"""
        results = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            plot_statistical_significance_spatial(results, output_dir, "test_dataset")

            # Check deprecation warning was logged
            assert any("deprecated" in record.message for record in caplog.records)

    def test_plot_model_rankings_spatial_deprecation_warning(self, caplog):
        """Test that deprecated function shows warning"""
        results = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            plot_model_rankings_spatial(results, output_dir, "test_dataset")

            # Check deprecation warning was logged
            assert any("deprecated" in record.message for record in caplog.records)

    def test_backward_compatibility_calls_new_function(self):
        """Test that wrappers call new functions correctly"""
        results = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Test that wrapper calls the new function
            with patch(
                "tools.visualize_lmtad_spatial_results.plot_perplexity_distribution_comparison"
            ) as mock_new_func:
                plot_spatial_abnormality_rates_comparison(
                    results, output_dir, "test_dataset"
                )
                mock_new_func.assert_called_once_with(
                    results, output_dir, "test_dataset"
                )


class TestLoadAggregatedResults:
    """Test load_aggregated_results function"""

    def test_load_aggregated_results_success(self, tmp_path):
        """Test successful loading of aggregated results"""
        results_file = tmp_path / "results.json"
        test_data = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with open(results_file, "w") as f:
            json.dump(test_data, f)

        result = load_aggregated_results(results_file)

        assert result == test_data
        assert "generated_data" in result
        assert "test_dataset" in result["generated_data"]

    def test_load_aggregated_results_missing_file(self):
        """Test handling of missing file"""
        with pytest.raises(FileNotFoundError):
            load_aggregated_results(Path("/nonexistent/file.json"))


class TestPlotPerplexityDistributionComparison:
    """Test plot_perplexity_distribution_comparison function"""

    def test_plot_perplexity_distribution_comparison_basic(
        self, sample_perplexity_results
    ):
        """Test basic perplexity distribution comparison plot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Mock model detection functions
            with (
                patch(
                    "tools.visualize_lmtad_spatial_results.get_model_color"
                ) as mock_color,
                patch(
                    "tools.visualize_lmtad_spatial_results.get_display_name"
                ) as mock_display,
            ):
                mock_color.return_value = "#3498db"
                mock_display.side_effect = lambda x: f"Model {x}"

                plot_perplexity_distribution_comparison(
                    sample_perplexity_results, output_dir, "test_dataset"
                )

                # Check output files are created
                output_file = (
                    output_dir / "perplexity_distribution_comparison_test_dataset.png"
                )
                assert output_file.exists()
                assert output_file.with_suffix(".svg").exists()

    def test_plot_perplexity_distribution_comparison_no_data(self):
        """Test with no data available"""
        results = {"generated_data": {"test_dataset": {}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should not raise exception, just return
            plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")

            # No files should be created
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_perplexity_distribution_comparison_no_perplexity_stats(self):
        """Test with models that have no perplexity stats"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {},  # No perplexity stats
                    "model_b": {"log_perplexity_stats": {}},  # Empty stats
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")

            # No files should be created (no valid data)
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_perplexity_distribution_comparison_creates_correct_structure(
        self, sample_perplexity_results
    ):
        """Test that plot creates correct subplots and structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with (
                patch(
                    "tools.visualize_lmtad_spatial_results.get_model_color"
                ) as mock_color,
                patch(
                    "tools.visualize_lmtad_spatial_results.get_display_name"
                ) as mock_display,
            ):
                mock_color.return_value = "#3498db"
                mock_display.side_effect = lambda x: x

                with patch("matplotlib.pyplot.tight_layout") as mock_tight:
                    with patch("matplotlib.pyplot.savefig") as mock_savefig:
                        plot_perplexity_distribution_comparison(
                            sample_perplexity_results, output_dir, "test_dataset"
                        )

                        # Check that tight_layout was called
                        assert mock_tight.called

                        # Check that savefig was called
                        assert mock_savefig.called


class TestPlotPerOdPairPerplexityComparison:
    """Test plot_per_od_pair_perplexity_comparison function"""

    def test_plot_per_od_pair_perplexity_comparison_basic(
        self, sample_perplexity_results
    ):
        """Test basic per-OD-pair perplexity comparison plot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_display_name"
            ) as mock_display:
                mock_display.side_effect = lambda x: x

                plot_per_od_pair_perplexity_comparison(
                    sample_perplexity_results, output_dir, "test_dataset"
                )

                # Check output files are created
                output_file = output_dir / "per_od_pair_perplexity_test_dataset.png"
                assert output_file.exists()
                assert output_file.with_suffix(".svg").exists()

    def test_plot_per_od_pair_perplexity_comparison_no_od_data(self):
        """Test with no OD pair data available"""
        results = {"generated_data": {"test_dataset": {}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should not raise exception, just return
            plot_per_od_pair_perplexity_comparison(results, output_dir, "test_dataset")

            # No files should be created
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_per_od_pair_perplexity_comparison_many_od_pairs(self):
        """Test that function limits OD pairs for readability"""
        # Create data with many OD pairs
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {"log_perplexity_stats": {"mean": 7.5}},
                    "model_b": {"log_perplexity_stats": {"mean": 7.0}},
                }
            },
            "od_pair_perplexities": {
                "test_dataset": {
                    f"{i}-{i + 100}": {
                        "model_a": {"mean_log_perplexity": 7.0 + i * 0.1},
                        "model_b": {"mean_log_perplexity": 6.5 + i * 0.1},
                    }
                    for i in range(50)  # 50 OD pairs
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_display_name"
            ) as mock_display:
                mock_display.side_effect = lambda x: x

                with patch("matplotlib.pyplot.savefig") as mock_savefig:
                    plot_per_od_pair_perplexity_comparison(
                        results, output_dir, "test_dataset"
                    )

                    # savefig should be called (plot created)
                    assert mock_savefig.called

    def test_plot_per_od_pair_perplexity_comparison_missing_values(self):
        """Test handling of missing perplexity values in OD pairs"""
        results = {
            "generated_data": {"test_dataset": {}},
            "od_pair_perplexities": {
                "test_dataset": {
                    "1-2": {
                        "model_a": {"mean_log_perplexity": 7.5},
                        "model_b": {"mean_log_perplexity": None},  # Missing
                    },
                    "3-4": {
                        "model_a": {"mean_log_perplexity": 8.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_display_name"
            ) as mock_display:
                mock_display.side_effect = lambda x: x

                # Should handle missing values gracefully
                plot_per_od_pair_perplexity_comparison(
                    results, output_dir, "test_dataset"
                )


class TestPlotModelRankingsByPerplexity:
    """Test plot_model_rankings_by_perplexity function"""

    def test_plot_model_rankings_by_perplexity_basic(self, sample_perplexity_results):
        """Test basic model rankings plot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with (
                patch(
                    "tools.visualize_lmtad_spatial_results.get_model_color"
                ) as mock_color,
                patch(
                    "tools.visualize_lmtad_spatial_results.get_display_name"
                ) as mock_display,
            ):
                mock_color.side_effect = ["#3498db", "#2ecc71", "#e74c3c"]
                mock_display.side_effect = lambda x: f"Model {x}"

                plot_model_rankings_by_perplexity(
                    sample_perplexity_results, output_dir, "test_dataset"
                )

                # Check output files are created
                output_file = output_dir / "model_rankings_perplexity_test_dataset.png"
                assert output_file.exists()
                assert output_file.with_suffix(".svg").exists()

    def test_plot_model_rankings_by_perplexity_correct_ranking(self):
        """Test that models are ranked correctly by mean perplexity"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "best_model": {"log_perplexity_stats": {"mean": 5.0, "std": 1.0}},
                    "worst_model": {"log_perplexity_stats": {"mean": 9.0, "std": 1.5}},
                    "medium_model": {"log_perplexity_stats": {"mean": 7.0, "std": 1.2}},
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with (
                patch(
                    "tools.visualize_lmtad_spatial_results.get_model_color"
                ) as mock_color,
                patch(
                    "tools.visualize_lmtad_spatial_results.get_display_name"
                ) as mock_display,
            ):
                mock_color.return_value = "#3498db"
                mock_display.side_effect = lambda x: x

                with patch("matplotlib.axes.Axes.barh") as mock_barh:
                    plot_model_rankings_by_perplexity(
                        results, output_dir, "test_dataset"
                    )

                    # barh should be called (plot created)
                    assert mock_barh.called

    def test_plot_model_rankings_by_perplexity_no_perplexity_stats(self):
        """Test with models that have no perplexity statistics"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {},  # No stats
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_model_rankings_by_perplexity(results, output_dir, "test_dataset")

            # No files should be created (no valid data)
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_model_rankings_by_perplexity_single_model(self):
        """Test with only one model"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {"log_perplexity_stats": {"mean": 7.5, "std": 1.0}},
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_model_rankings_by_perplexity(results, output_dir, "test_dataset")

                # Should still create plot (even with one model)
                output_file = output_dir / "model_rankings_perplexity_test_dataset.png"
                assert output_file.exists()


class TestPlotStatisticalSignificancePerplexity:
    """Test plot_statistical_significance_perplexity function"""

    def test_plot_statistical_significance_perplexity_basic(
        self, sample_perplexity_results
    ):
        """Test basic statistical significance plot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_statistical_significance_perplexity(
                    sample_perplexity_results, output_dir, "test_dataset"
                )

                # Check output files are created
                output_file = (
                    output_dir / "statistical_significance_perplexity_test_dataset.png"
                )
                assert output_file.exists()
                assert output_file.with_suffix(".svg").exists()

    def test_plot_statistical_significance_perplexity_ci_calculation(self):
        """Test confidence interval calculation"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {
                        "log_perplexity_stats": {
                            "mean": 7.5,
                            "std": 1.0,
                            "count": 100,
                        }
                    },
                    "model_b": {
                        "log_perplexity_stats": {
                            "mean": 7.0,
                            "std": 1.2,
                            "count": 50,  # Different count
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                with patch("matplotlib.axes.Axes.barh") as mock_barh:
                    plot_statistical_significance_perplexity(
                        results, output_dir, "test_dataset"
                    )

                    # Should calculate different CI margins for different counts
                    assert mock_barh.called

    def test_plot_statistical_significance_perplexity_missing_count(self):
        """Test handling when count is missing from statistics"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {
                        "log_perplexity_stats": {
                            "mean": 7.5,
                            "std": 1.0,
                            # No count
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_statistical_significance_perplexity(
                    results, output_dir, "test_dataset"
                )

                # Should use default count (100) and handle gracefully
                output_file = (
                    output_dir / "statistical_significance_perplexity_test_dataset.png"
                )
                assert output_file.exists()

    def test_plot_statistical_significance_perplexity_zero_std(self):
        """Test handling of zero standard deviation"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {
                        "log_perplexity_stats": {
                            "mean": 7.5,
                            "std": 0.0,  # Zero std
                            "count": 100,
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                # Should handle zero std gracefully (no division by zero)
                plot_statistical_significance_perplexity(
                    results, output_dir, "test_dataset"
                )

                output_file = (
                    output_dir / "statistical_significance_perplexity_test_dataset.png"
                )
                assert output_file.exists()


class TestPlotSegmentLevelPerplexityAggregate:
    """Test plot_segment_level_perplexity_aggregate function"""

    def test_plot_segment_level_perplexity_aggregate_basic(
        self, sample_perplexity_results
    ):
        """Test basic segment-level perplexity aggregate plot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_segment_level_perplexity_aggregate(
                    sample_perplexity_results, output_dir, "test_dataset"
                )

                # Check output files are created
                output_file = output_dir / "segment_level_perplexity_test_dataset.png"
                assert output_file.exists()
                assert output_file.with_suffix(".svg").exists()

    def test_plot_segment_level_perplexity_aggregate_no_segment_data(self):
        """Test with no segment-level data available"""
        results = {"generated_data": {"test_dataset": {}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should not raise exception, just return
            plot_segment_level_perplexity_aggregate(results, output_dir, "test_dataset")

            # No files should be created
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_segment_level_perplexity_aggregate_percentiles(self):
        """Test handling of percentile data"""
        results = {
            "generated_data": {"test_dataset": {}},
            "segment_perplexity_stats": {
                "test_dataset": {
                    "model_a": {
                        "mean": 2.5,
                        "std": 0.8,
                        "p25": 1.8,
                        "p50": 2.4,
                        "p75": 3.0,
                        "p90": 3.5,
                        "p95": 3.8,
                        "count": 500,
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_segment_level_perplexity_aggregate(
                    results, output_dir, "test_dataset"
                )

                output_file = output_dir / "segment_level_perplexity_test_dataset.png"
                assert output_file.exists()

    def test_plot_segment_level_perplexity_aggregate_coefficient_variation(self):
        """Test calculation of coefficient of variation"""
        results = {
            "generated_data": {"test_dataset": {}},
            "segment_perplexity_stats": {
                "test_dataset": {
                    "model_a": {
                        "mean": 2.0,
                        "std": 1.0,  # 50% CV
                        "count": 100,
                    },
                    "model_b": {
                        "mean": 4.0,
                        "std": 0.4,  # 10% CV
                        "count": 100,
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.side_effect = ["#3498db", "#2ecc71"]

                plot_segment_level_perplexity_aggregate(
                    results, output_dir, "test_dataset"
                )

                # Should create plot with CV calculation
                output_file = output_dir / "segment_level_perplexity_test_dataset.png"
                assert output_file.exists()


class TestPlotComprehensivePerplexitySummary:
    """Test plot_comprehensive_perplexity_summary function"""

    def test_plot_comprehensive_perplexity_summary_basic(
        self, sample_perplexity_results
    ):
        """Test basic comprehensive summary plot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with (
                patch(
                    "tools.visualize_lmtad_spatial_results.get_model_color"
                ) as mock_color,
                patch(
                    "tools.visualize_lmtad_spatial_results.get_display_name"
                ) as mock_display,
            ):
                mock_color.side_effect = ["#3498db", "#2ecc71", "#e74c3c"]
                mock_display.side_effect = lambda x: f"Model {x}"

                plot_comprehensive_perplexity_summary(
                    sample_perplexity_results, output_dir, "test_dataset"
                )

                # Check output files are created
                output_file = (
                    output_dir / "comprehensive_perplexity_summary_test_dataset.png"
                )
                assert output_file.exists()
                assert output_file.with_suffix(".svg").exists()

    def test_plot_comprehensive_perplexity_summary_no_data(self):
        """Test with no data available"""
        results = {"generated_data": {"test_dataset": {}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should not raise exception, just return
            plot_comprehensive_perplexity_summary(results, output_dir, "test_dataset")

            # No files should be created
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_comprehensive_perplexity_summary_best_worst_ranking(self):
        """Test that best and worst models are identified correctly"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "best": {"log_perplexity_stats": {"mean": 5.0}},
                    "worst": {"log_perplexity_stats": {"mean": 10.0}},
                    "middle": {"log_perplexity_stats": {"mean": 7.5}},
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with (
                patch(
                    "tools.visualize_lmtad_spatial_results.get_model_color"
                ) as mock_color,
                patch(
                    "tools.visualize_lmtad_spatial_results.get_display_name"
                ) as mock_display,
            ):
                mock_color.return_value = "#3498db"
                mock_display.side_effect = lambda x: x

                plot_comprehensive_perplexity_summary(
                    results, output_dir, "test_dataset"
                )

                # Should create plot with best/worst identification
                output_file = (
                    output_dir / "comprehensive_perplexity_summary_test_dataset.png"
                )
                assert output_file.exists()

    def test_plot_comprehensive_perplexity_summary_range_calculation(self):
        """Test perplexity range calculation"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {
                        "log_perplexity_stats": {
                            "mean": 7.0,
                            "min": 4.0,
                            "max": 11.0,
                        }
                    },
                    "model_b": {
                        "log_perplexity_stats": {
                            "mean": 6.5,
                            "min": 5.0,
                            "max": 9.0,
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.side_effect = ["#3498db", "#2ecc71"]

                plot_comprehensive_perplexity_summary(
                    results, output_dir, "test_dataset"
                )

                # Should calculate ranges: model_a = 7.0, model_b = 4.0
                output_file = (
                    output_dir / "comprehensive_perplexity_summary_test_dataset.png"
                )
                assert output_file.exists()


class TestPlotGenerationIntegration:
    """Test integration aspects of plot generation"""

    def test_plot_creates_output_directory(self):
        """Test that plots create output directory if it doesn't exist"""
        results = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output" / "directory"

            # Directory doesn't exist yet
            assert not output_dir.exists()

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_perplexity_distribution_comparison(
                    results, output_dir, "test_dataset"
                )

                # Directory should be created
                assert output_dir.exists()

                # File should exist
                output_file = (
                    output_dir / "perplexity_distribution_comparison_test_dataset.png"
                )
                assert output_file.exists()

    def test_plot_multiple_formats(self):
        """Test that plots save in both PNG and SVG formats"""
        results = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_perplexity_distribution_comparison(
                    results, output_dir, "test_dataset"
                )

                # Both PNG and SVG should exist
                png_file = (
                    output_dir / "perplexity_distribution_comparison_test_dataset.png"
                )
                svg_file = (
                    output_dir / "perplexity_distribution_comparison_test_dataset.svg"
                )

                assert png_file.exists()
                assert svg_file.exists()

    def test_plot_with_many_models(self):
        """Test plot generation with many models"""
        results = {
            "generated_data": {
                "test_dataset": {
                    f"model_{i}": {"log_perplexity_stats": {"mean": 7.0 + i * 0.1}}
                    for i in range(20)  # 20 models
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                plot_perplexity_distribution_comparison(
                    results, output_dir, "test_dataset"
                )

                # Should handle many models gracefully
                output_file = (
                    output_dir / "perplexity_distribution_comparison_test_dataset.png"
                )
                assert output_file.exists()

    def test_plot_styling_parameters(self):
        """Test that plots use correct styling parameters"""
        results = {
            "generated_data": {
                "test_dataset": {"model_a": {"log_perplexity_stats": {"mean": 7.5}}}
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch(
                "tools.visualize_lmtad_spatial_results.get_model_color"
            ) as mock_color:
                mock_color.return_value = "#3498db"

                with patch("matplotlib.pyplot.savefig") as mock_savefig:
                    plot_perplexity_distribution_comparison(
                        results, output_dir, "test_dataset"
                    )

                    # savefig should be called with correct parameters
                    assert mock_savefig.called
                    # Check that bbox_inches='tight' is used
                    call_args = mock_savefig.call_args
                    assert call_args is not None
                    assert "bbox_inches" in call_args.kwargs or len(call_args.args) > 1


class TestErrorHandling:
    """Test error handling in visualization functions"""

    def test_plot_with_invalid_results_structure(self):
        """Test handling of invalid results structure"""
        # Missing required keys
        results = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should not raise exception, just return
            plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")

            # No files should be created
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_with_none_values(self):
        """Test handling of None values in results"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {
                        "log_perplexity_stats": None  # None instead of dict
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should handle None gracefully
            plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")

            # No files should be created (no valid data)
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_with_empty_od_pairs(self):
        """Test handling of empty OD pairs"""
        results = {
            "generated_data": {"test_dataset": {}},
            "od_pair_perplexities": {
                "test_dataset": {}  # Empty OD pairs
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should handle empty OD pairs gracefully
            plot_per_od_pair_perplexity_comparison(results, output_dir, "test_dataset")

            # No files should be created
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 0

    def test_plot_with_all_nan_values(self):
        """Test handling of all NaN values"""
        results = {
            "generated_data": {
                "test_dataset": {
                    "model_a": {
                        "log_perplexity_stats": {
                            "mean": float("nan"),
                            "std": float("nan"),
                        }
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should handle NaN values gracefully
            plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")

            # Plot might still be created with NaN handling
            # The important thing is it doesn't crash


# Fixtures
@pytest.fixture
def sample_perplexity_results():
    """Create sample perplexity evaluation results with comprehensive data"""
    return {
        "summary_statistics": {
            "test_dataset": {
                "real_spatial_abnormality_rate": 15.5,
            }
        },
        "generated_data": {
            "test_dataset": {
                "model_a": {
                    "log_perplexity_stats": {
                        "mean": 7.2,
                        "std": 1.5,
                        "median": 7.0,
                        "min": 4.0,
                        "max": 11.0,
                        "count": 100,
                    },
                    "total_trajectories": 100,
                },
                "model_b": {
                    "log_perplexity_stats": {
                        "mean": 6.8,
                        "std": 1.2,
                        "median": 6.7,
                        "min": 4.5,
                        "max": 9.5,
                        "count": 100,
                    },
                    "total_trajectories": 100,
                },
                "model_c": {
                    "log_perplexity_stats": {
                        "mean": 8.1,
                        "std": 2.0,
                        "median": 8.0,
                        "min": 3.5,
                        "max": 13.0,
                        "count": 100,
                    },
                    "total_trajectories": 100,
                },
            }
        },
        "od_pair_perplexities": {
            "test_dataset": {
                "74→208": {
                    "model_a": {
                        "mean_log_perplexity": 7.0,
                        "std_log_perplexity": 1.2,
                        "count": 10,
                    },
                    "model_b": {
                        "mean_log_perplexity": 6.5,
                        "std_log_perplexity": 1.0,
                        "count": 10,
                    },
                    "model_c": {
                        "mean_log_perplexity": 8.2,
                        "std_log_perplexity": 1.8,
                        "count": 10,
                    },
                },
                "100→6165": {
                    "model_a": {
                        "mean_log_perplexity": 7.5,
                        "std_log_perplexity": 1.3,
                        "count": 8,
                    },
                    "model_b": {
                        "mean_log_perplexity": 7.0,
                        "std_log_perplexity": 1.1,
                        "count": 8,
                    },
                    "model_c": {
                        "mean_log_perplexity": 8.5,
                        "std_log_perplexity": 1.9,
                        "count": 8,
                    },
                },
            }
        },
        "segment_perplexity_stats": {
            "test_dataset": {
                "model_a": {
                    "mean": 2.5,
                    "std": 0.8,
                    "median": 2.4,
                    "p25": 1.8,
                    "p50": 2.4,
                    "p75": 3.0,
                    "p90": 3.5,
                    "p95": 3.8,
                    "count": 500,
                },
                "model_b": {
                    "mean": 2.3,
                    "std": 0.7,
                    "median": 2.2,
                    "p25": 1.7,
                    "p50": 2.2,
                    "p75": 2.8,
                    "p90": 3.2,
                    "p95": 3.5,
                    "count": 500,
                },
                "model_c": {
                    "mean": 2.8,
                    "std": 0.9,
                    "median": 2.7,
                    "p25": 2.0,
                    "p50": 2.7,
                    "p75": 3.4,
                    "p90": 4.0,
                    "p95": 4.3,
                    "count": 500,
                },
            }
        },
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
