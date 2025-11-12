#!/usr/bin/env python3
"""
Test suite for plot_abnormal_evaluation module.

This test suite follows TDD principles - tests are written before implementation.
Tests cover all functions in the plot_abnormal_evaluation module with comprehensive
coverage including edge cases and error conditions.
"""

import json
import tempfile
from pathlib import Path
import pytest

# Import functions to be tested (will be implemented)
from tools.plot_abnormal_evaluation import (
    load_comparison_report,
    plot_abnormality_reproduction_rates,
    plot_similarity_metrics,
    plot_abnormality_by_category,
    plot_metrics_comparison_heatmap,
    plot_evaluation_from_files,
)


class TestLoadComparisonReport:
    """Test suite for load_comparison_report function"""

    def test_load_valid_comparison_report(self):
        """Test loading valid comparison report JSON"""
        report_data = {
            "model_results": {
                "model1": {
                    "total_trajectories": 100,
                    "abnormality_detection": {
                        "wang_temporal_delay": {"count": 5},
                    },
                    "similarity_metrics": {"edr": 0.5, "dtw": 0.3, "hausdorff": 0.2},
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            temp_file = Path(f.name)

        try:
            result = load_comparison_report(temp_file)

            assert isinstance(result, dict)
            assert "model_results" in result
            assert len(result["model_results"]) == 1
        finally:
            temp_file.unlink()

    def test_load_missing_file(self):
        """Test fail-fast assertion for missing file"""
        missing_file = Path("/nonexistent/path/comparison.json")

        with pytest.raises(AssertionError, match="not found"):
            load_comparison_report(missing_file)

    def test_load_invalid_json(self):
        """Test handling of invalid JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            temp_file = Path(f.name)

        try:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                load_comparison_report(temp_file)
        finally:
            temp_file.unlink()

    def test_load_missing_required_key(self):
        """Test fail-fast assertion for missing required key"""
        report_data = {"summary": {"total": 100}}  # Missing model_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            temp_file = Path(f.name)

        try:
            with pytest.raises(AssertionError, match="model_results"):
                load_comparison_report(temp_file)
        finally:
            temp_file.unlink()


class TestPlotAbnormalityReproductionRates:
    """Test suite for plot_abnormality_reproduction_rates function"""

    def test_plot_valid_model_results(self, tmp_path):
        """Test plotting with valid model results"""
        model_results = {
            "model1": {
                "total_trajectories": 100,
                "abnormality_detection": {
                    "wang_temporal_delay": {"count": 5},
                },
            },
            "model2": {
                "total_trajectories": 100,
                "abnormality_detection": {
                    "wang_temporal_delay": {"count": 10},
                },
            },
        }
        output_file = tmp_path / "test_rates.png"

        plot_abnormality_reproduction_rates(model_results, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_empty_model_results(self, tmp_path):
        """Test fail-fast assertion for empty model results"""
        output_file = tmp_path / "test_rates.png"

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_abnormality_reproduction_rates({}, output_file, "test_dataset")


class TestPlotSimilarityMetrics:
    """Test suite for plot_similarity_metrics function"""

    def test_plot_valid_metrics(self, tmp_path):
        """Test plotting with valid similarity metrics"""
        model_results = {
            "model1": {
                "similarity_metrics": {"edr": 0.5, "dtw": 0.3, "hausdorff": 0.2},
            },
            "model2": {
                "similarity_metrics": {"edr": 0.6, "dtw": 0.4, "hausdorff": 0.3},
            },
        }
        output_file = tmp_path / "test_metrics.png"

        plot_similarity_metrics(model_results, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_empty_model_results(self, tmp_path):
        """Test fail-fast assertion for empty model results"""
        output_file = tmp_path / "test_metrics.png"

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_similarity_metrics({}, output_file, "test_dataset")


class TestPlotAbnormalityByCategory:
    """Test suite for plot_abnormality_by_category function"""

    def test_plot_valid_categories(self, tmp_path):
        """Test plotting with valid categories"""
        model_results = {
            "model1": {
                "abnormality_detection": {
                    "wang_temporal_delay": {"count": 5},
                    "wang_route_deviation": {"count": 3},
                },
            },
            "model2": {
                "abnormality_detection": {
                    "wang_temporal_delay": {"count": 10},
                },
            },
        }
        output_file = tmp_path / "test_categories.png"

        plot_abnormality_by_category(model_results, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_empty_model_results(self, tmp_path):
        """Test fail-fast assertion for empty model results"""
        output_file = tmp_path / "test_categories.png"

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_abnormality_by_category({}, output_file, "test_dataset")


class TestPlotMetricsComparisonHeatmap:
    """Test suite for plot_metrics_comparison_heatmap function"""

    def test_plot_valid_metrics_heatmap(self, tmp_path):
        """Test plotting with valid metrics data"""
        model_results = {
            "model1": {
                "total_trajectories": 100,
                "abnormality_detection": {
                    "wang_temporal_delay": {"count": 5},
                },
                "similarity_metrics": {"edr": 0.5, "dtw": 0.3, "hausdorff": 0.2},
            },
            "model2": {
                "total_trajectories": 100,
                "abnormality_detection": {
                    "wang_temporal_delay": {"count": 10},
                },
                "similarity_metrics": {"edr": 0.6, "dtw": 0.4, "hausdorff": 0.3},
            },
        }
        output_file = tmp_path / "test_heatmap.png"

        plot_metrics_comparison_heatmap(model_results, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_empty_model_results(self, tmp_path):
        """Test fail-fast assertion for empty model results"""
        output_file = tmp_path / "test_heatmap.png"

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_metrics_comparison_heatmap({}, output_file, "test_dataset")


class TestPlotEvaluationFromFiles:
    """Test suite for plot_evaluation_from_files function"""

    def test_plot_from_valid_file(self, tmp_path):
        """Test end-to-end plotting from comparison report file"""
        comparison_file = tmp_path / "comparison.json"
        report_data = {
            "model_results": {
                "model1": {
                    "total_trajectories": 100,
                    "abnormality_detection": {
                        "wang_temporal_delay": {"count": 5},
                    },
                    "similarity_metrics": {"edr": 0.5, "dtw": 0.3, "hausdorff": 0.2},
                }
            }
        }
        with open(comparison_file, "w") as f:
            json.dump(report_data, f)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = plot_evaluation_from_files(comparison_file, output_dir, "test_dataset")

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_plot_missing_comparison_file(self, tmp_path):
        """Test fail-fast assertion for missing comparison file"""
        missing_file = tmp_path / "missing.json"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(AssertionError, match="not found"):
            plot_evaluation_from_files(missing_file, output_dir, "test_dataset")

    def test_plot_output_dir_parent_missing(self, tmp_path):
        """Test fail-fast assertion when output directory parent doesn't exist"""
        comparison_file = tmp_path / "comparison.json"
        comparison_file.write_text('{"model_results": {}}')

        output_dir = tmp_path / "nonexistent" / "output"

        with pytest.raises(AssertionError, match="does not exist"):
            plot_evaluation_from_files(comparison_file, output_dir, "test_dataset")
