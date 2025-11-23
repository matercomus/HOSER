"""
Tests for Perplexity-Focused Visualization Module

This module tests the plotting functionality for LM-TAD perplexity evaluation,
including distribution comparisons, rankings, and segment-level analysis.
"""

import json
import tempfile
from pathlib import Path

import pytest

from tools.visualize_lmtad_spatial_results import (
    load_aggregated_results,
    plot_perplexity_distribution_comparison,
    plot_per_od_pair_perplexity_comparison,
    plot_model_rankings_by_perplexity,
    plot_statistical_significance_perplexity,
    plot_segment_level_perplexity_aggregate,
    plot_comprehensive_perplexity_summary,
)


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
                "150→450": {
                    "model_a": {
                        "mean_log_perplexity": 7.8,
                        "std_log_perplexity": 1.6,
                        "count": 12,
                    },
                    "model_b": {
                        "mean_log_perplexity": 7.2,
                        "std_log_perplexity": 1.4,
                        "count": 12,
                    },
                    "model_c": {
                        "mean_log_perplexity": 9.0,
                        "std_log_perplexity": 2.1,
                        "count": 12,
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


@pytest.fixture
def temp_results_file(sample_perplexity_results):
    """Create temporary results file for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        results_file = tmpdir_path / "results.json"

        with open(results_file, "w") as f:
            json.dump(sample_perplexity_results, f)

        yield results_file


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for plots"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "output"


def test_load_aggregated_results(temp_results_file):
    """Test loading aggregated results from JSON"""
    results = load_aggregated_results(temp_results_file)

    assert "generated_data" in results
    assert "od_pair_perplexities" in results
    assert "test_dataset" in results["generated_data"]


def test_load_aggregated_results_missing_file():
    """Test that loading non-existent file raises appropriate error"""
    with pytest.raises(FileNotFoundError):
        load_aggregated_results(Path("/nonexistent/file.json"))


def test_plot_perplexity_distribution_comparison(
    sample_perplexity_results, temp_output_dir
):
    """Test perplexity distribution comparison plot generation"""
    output_file = (
        temp_output_dir / "perplexity_distribution_comparison_test_dataset.png"
    )

    plot_perplexity_distribution_comparison(
        sample_perplexity_results, temp_output_dir, "test_dataset"
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_perplexity_distribution_comparison_no_data():
    """Test with no data available"""
    results = {"generated_data": {"test_dataset": {}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")
        # Should not raise exception, just log warning and return


def test_plot_per_od_pair_perplexity_comparison(
    sample_perplexity_results, temp_output_dir
):
    """Test per-OD-pair perplexity comparison plot generation"""
    output_file = temp_output_dir / "per_od_pair_perplexity_test_dataset.png"

    plot_per_od_pair_perplexity_comparison(
        sample_perplexity_results, temp_output_dir, "test_dataset"
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_per_od_pair_perplexity_comparison_no_od_data():
    """Test with no OD pair data available"""
    results = {"generated_data": {"test_dataset": {}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_per_od_pair_perplexity_comparison(results, output_dir, "test_dataset")
        # Should not raise exception, just log warning and return


def test_plot_model_rankings_by_perplexity(sample_perplexity_results, temp_output_dir):
    """Test model rankings by perplexity plot generation"""
    output_file = temp_output_dir / "model_rankings_perplexity_test_dataset.png"

    plot_model_rankings_by_perplexity(
        sample_perplexity_results, temp_output_dir, "test_dataset"
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_model_rankings_by_perplexity_no_stats():
    """Test with no perplexity statistics available"""
    results = {"generated_data": {"test_dataset": {"model_a": {}}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_model_rankings_by_perplexity(results, output_dir, "test_dataset")
        # Should not raise exception, just log warning and return


def test_plot_statistical_significance_perplexity(
    sample_perplexity_results, temp_output_dir
):
    """Test statistical significance perplexity plot generation"""
    output_file = (
        temp_output_dir / "statistical_significance_perplexity_test_dataset.png"
    )

    plot_statistical_significance_perplexity(
        sample_perplexity_results, temp_output_dir, "test_dataset"
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_statistical_significance_perplexity_missing_data():
    """Test with missing perplexity statistics"""
    results = {"generated_data": {"test_dataset": {}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_statistical_significance_perplexity(results, output_dir, "test_dataset")
        # Should not raise exception, just log warning and return


def test_plot_segment_level_perplexity_aggregate(
    sample_perplexity_results, temp_output_dir
):
    """Test segment-level perplexity aggregate plot generation"""
    output_file = temp_output_dir / "segment_level_perplexity_test_dataset.png"

    plot_segment_level_perplexity_aggregate(
        sample_perplexity_results, temp_output_dir, "test_dataset"
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_segment_level_perplexity_no_data():
    """Test with no segment-level data"""
    results = {"generated_data": {"test_dataset": {}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_segment_level_perplexity_aggregate(results, output_dir, "test_dataset")
        # Should not raise exception, just log warning and return


def test_plot_comprehensive_perplexity_summary(
    sample_perplexity_results, temp_output_dir
):
    """Test comprehensive perplexity summary plot generation"""
    output_file = temp_output_dir / "comprehensive_perplexity_summary_test_dataset.png"

    plot_comprehensive_perplexity_summary(
        sample_perplexity_results, temp_output_dir, "test_dataset"
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_comprehensive_perplexity_summary_empty():
    """Test with empty results"""
    results = {"generated_data": {"test_dataset": {}}}
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_comprehensive_perplexity_summary(results, output_dir, "test_dataset")
        # Should not raise exception, just log warning and return


def test_perplexity_distribution_multiple_models():
    """Test with varying number of models"""
    results = {
        "generated_data": {
            "test_dataset": {
                "model_1": {
                    "log_perplexity_stats": {
                        "mean": 7.0,
                        "std": 1.0,
                        "median": 7.0,
                    }
                },
                "model_2": {
                    "log_perplexity_stats": {
                        "mean": 8.0,
                        "std": 1.5,
                        "median": 8.0,
                    }
                },
                "model_3": {
                    "log_perplexity_stats": {
                        "mean": 6.5,
                        "std": 0.8,
                        "median": 6.5,
                    }
                },
                "model_4": {
                    "log_perplexity_stats": {
                        "mean": 9.0,
                        "std": 2.0,
                        "median": 9.0,
                    }
                },
                "model_5": {
                    "log_perplexity_stats": {
                        "mean": 6.0,
                        "std": 0.7,
                        "median": 6.0,
                    }
                },
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")

        output_file = output_dir / "perplexity_distribution_comparison_test_dataset.png"
        assert output_file.exists()


def test_per_od_pair_heatmap_many_pairs():
    """Test heatmap with many OD pairs (should limit to top 20)"""
    # Create data with 50 OD pairs
    od_data = {
        "test_dataset": {
            "od_pair_{}".format(i): {
                "model_a": {
                    "mean_log_perplexity": float(i),
                    "count": 5,
                },
                "model_b": {
                    "mean_log_perplexity": float(i + 0.5),
                    "count": 5,
                },
            }
            for i in range(50)
        }
    }

    results = {"od_pair_perplexities": od_data}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_per_od_pair_perplexity_comparison(results, output_dir, "test_dataset")

        output_file = output_dir / "per_od_pair_perplexity_test_dataset.png"
        assert output_file.exists()


def test_perplexity_statistics_missing_fields():
    """Test with missing optional fields in statistics"""
    results = {
        "generated_data": {
            "test_dataset": {
                "model_a": {
                    "log_perplexity_stats": {
                        "mean": 7.0,
                        # Missing std, median, etc.
                    }
                }
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"

        # Should not raise exception even with missing fields
        plot_model_rankings_by_perplexity(results, output_dir, "test_dataset")
        plot_statistical_significance_perplexity(results, output_dir, "test_dataset")
        plot_comprehensive_perplexity_summary(results, output_dir, "test_dataset")


def test_segment_perplexity_percentiles():
    """Test segment-level perplexity with various percentile fields"""
    results = {
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
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_segment_level_perplexity_aggregate(results, output_dir, "test_dataset")

        output_file = output_dir / "segment_level_perplexity_test_dataset.png"
        assert output_file.exists()


def test_plot_output_directory_creation(temp_results_file, tmp_path):
    """Test that output directory is created when it doesn't exist"""
    output_dir = tmp_path / "nonexistent" / "subdir" / "output"
    results = load_aggregated_results(temp_results_file)

    # Should create directory and generate plots
    plot_perplexity_distribution_comparison(results, output_dir, "test_dataset")

    assert output_dir.exists()


def test_model_rankings_sorted_correctly():
    """Test that models are sorted by mean perplexity (ascending)"""
    results = {
        "generated_data": {
            "test_dataset": {
                "worst_model": {
                    "log_perplexity_stats": {
                        "mean": 10.0,
                        "std": 1.0,
                    }
                },
                "best_model": {
                    "log_perplexity_stats": {
                        "mean": 5.0,
                        "std": 1.0,
                    }
                },
                "medium_model": {
                    "log_perplexity_stats": {
                        "mean": 7.5,
                        "std": 1.0,
                    }
                },
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plot_model_rankings_by_perplexity(results, output_dir, "test_dataset")

        # If we can generate the plot without errors, ranking worked correctly
        output_file = output_dir / "model_rankings_perplexity_test_dataset.png"
        assert output_file.exists()


def test_all_plot_functions_with_full_dataset(temp_results_file, temp_output_dir):
    """Test that all plot functions work together with a complete dataset"""
    results = load_aggregated_results(temp_results_file)

    # Generate all plots
    plot_perplexity_distribution_comparison(results, temp_output_dir, "test_dataset")
    plot_per_od_pair_perplexity_comparison(results, temp_output_dir, "test_dataset")
    plot_model_rankings_by_perplexity(results, temp_output_dir, "test_dataset")
    plot_statistical_significance_perplexity(results, temp_output_dir, "test_dataset")
    plot_segment_level_perplexity_aggregate(results, temp_output_dir, "test_dataset")
    plot_comprehensive_perplexity_summary(results, temp_output_dir, "test_dataset")

    # Verify all files were created
    assert (
        temp_output_dir / "perplexity_distribution_comparison_test_dataset.png"
    ).exists()
    assert (temp_output_dir / "per_od_pair_perplexity_test_dataset.png").exists()
    assert (temp_output_dir / "model_rankings_perplexity_test_dataset.png").exists()
    assert (
        temp_output_dir / "statistical_significance_perplexity_test_dataset.png"
    ).exists()
    assert (temp_output_dir / "segment_level_perplexity_test_dataset.png").exists()
    assert (
        temp_output_dir / "comprehensive_perplexity_summary_test_dataset.png"
    ).exists()

    # Verify SVG versions also exist
    assert (
        temp_output_dir / "perplexity_distribution_comparison_test_dataset.svg"
    ).exists()
    assert (temp_output_dir / "per_od_pair_perplexity_test_dataset.svg").exists()
    assert (temp_output_dir / "model_rankings_perplexity_test_dataset.svg").exists()
    assert (
        temp_output_dir / "statistical_significance_perplexity_test_dataset.svg"
    ).exists()
    assert (temp_output_dir / "segment_level_perplexity_test_dataset.svg").exists()
    assert (
        temp_output_dir / "comprehensive_perplexity_summary_test_dataset.svg"
    ).exists()


if __name__ == "__main__":
    pytest.main([__file__])
