"""Tests for Wang analysis tools dual interface."""

import json
import pytest
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from tools.analyze_wang_results import (
    analyze_wang_results,
    WangResultsCollector,
    DetectionMetrics,
    StatisticalComparison,
)
from tools.visualize_wang_results import generate_wang_visualizations


class TestAnalyzeWangResultsProgrammaticInterface:
    """Tests for analyze_wang_results programmatic interface."""

    def test_function_signature(self):
        """Test that analyze_wang_results has correct signature."""
        import inspect
        sig = inspect.signature(analyze_wang_results)
        
        # Check parameters
        assert 'eval_dirs' in sig.parameters
        assert 'output_file' in sig.parameters
        
        # Check defaults are None (optional parameters)
        assert sig.parameters['eval_dirs'].default is None
        assert sig.parameters['output_file'].default is None
        
        # Check return type annotation
        assert sig.return_annotation == Path or str(sig.return_annotation) == 'Path'

    def test_function_docstring(self):
        """Test that analyze_wang_results has comprehensive docstring."""
        doc = analyze_wang_results.__doc__
        assert doc is not None
        assert 'programmatically' in doc.lower() or 'programmatic' in doc.lower()
        assert 'eval_dirs' in doc
        assert 'output_file' in doc
        assert 'Example' in doc or 'example' in doc

    def test_analyze_with_mock_data(self, tmp_path):
        """Test analyze_wang_results with mock evaluation directory."""
        # Create mock evaluation directory structure
        eval_dir = tmp_path / "test_eval"
        abnormal_dir = eval_dir / "abnormal" / "Beijing" / "test" / "real_data"
        abnormal_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock detection results
        detection_results = {
            "dataset": "Beijing",
            "total_trajectories": 100,
            "abnormal_indices": {
                "wang_temporal_delay": [1, 2, 3],
                "wang_route_deviation": [4, 5],
                "wang_both_deviations": [6]
            },
            "pattern_counts": {
                "Abp1_normal": 94,
                "Abp2_temporal_delay": 3,
                "Abp3_route_deviation": 2,
                "Abp4_both_deviations": 1
            },
            "wang_metadata": {
                "baseline_usage": {
                    "od_specific": 50,
                    "global": 40,
                    "none": 10
                }
            }
        }
        
        with open(abnormal_dir / "detection_results.json", "w") as f:
            json.dump(detection_results, f)
        
        # Create mock comparison report
        comparison_report = {
            "dataset": "Beijing",
            "od_source": "test",
            "real_data": {
                "total_trajectories": 100,
                "abnormal_count": 6,
                "abnormal_rate": 6.0,
                "abnormal_by_category": {}
            },
            "generated_data": {}
        }
        
        # Create comparison report directory
        comparison_dir = eval_dir / "abnormal" / "Beijing" / "test"
        with open(comparison_dir / "comparison_report.json", "w") as f:
            json.dump(comparison_report, f)
        
        # Test the function
        output_file = tmp_path / "test_output.json"
        result = analyze_wang_results(
            eval_dirs=[eval_dir],
            output_file=output_file
        )
        
        # Verify output
        assert result == output_file
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file, "r") as f:
            data = json.load(f)
        
        assert "summary_statistics" in data
        assert "real_data" in data
        assert "generated_data" in data
        assert "statistical_analysis" in data

    def test_analyze_with_none_defaults(self, tmp_path):
        """Test that default parameters work (None values)."""
        # This should not raise an error even with no eval dirs
        with patch('tools.analyze_wang_results.WangResultsCollector') as mock_collector:
            mock_instance = MagicMock()
            mock_collector.return_value = mock_instance
            mock_instance.results = []
            mock_instance.comparisons = []
            mock_instance.aggregate_results.return_value = {
                "summary_statistics": {},
                "real_data": {},
                "generated_data": {},
                "comparisons": []
            }
            mock_instance.perform_statistical_analysis.return_value = {
                "model_rankings": {},
                "pattern_distributions": {},
                "baseline_usage_analysis": {},
                "cross_dataset_comparison": {},
                "statistical_tests": {}
            }
            
            # Call with defaults (should auto-discover)
            output = analyze_wang_results()
            
            # Should return a Path object
            assert isinstance(output, Path)

    def test_return_type(self, tmp_path):
        """Test that function returns Path object."""
        output_file = tmp_path / "output.json"
        
        with patch('tools.analyze_wang_results.WangResultsCollector') as mock_collector:
            mock_instance = MagicMock()
            mock_collector.return_value = mock_instance
            mock_instance.results = []
            mock_instance.comparisons = []
            mock_instance.aggregate_results.return_value = {
                "summary_statistics": {},
                "real_data": {},
                "generated_data": {},
                "comparisons": []
            }
            mock_instance.perform_statistical_analysis.return_value = {
                "model_rankings": {},
                "pattern_distributions": {},
                "baseline_usage_analysis": {},
                "cross_dataset_comparison": {},
                "statistical_tests": {}
            }
            
            result = analyze_wang_results(eval_dirs=[], output_file=output_file)
            assert isinstance(result, Path)


class TestAnalyzeWangResultsCLI:
    """Tests for analyze_wang_results CLI interface."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.analyze_wang_results", "--help"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "--eval-dir" in result.stdout
        assert "--output" in result.stdout

    def test_cli_with_eval_dir(self, tmp_path):
        """Test CLI with --eval-dir argument."""
        # Create minimal mock structure
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        
        output_file = tmp_path / "output.json"
        
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.analyze_wang_results",
                "--eval-dir", str(eval_dir),
                "--output", str(output_file)
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        # Should complete without error (even if no data found)
        # The tool warns but doesn't error on missing data
        assert result.returncode == 0

    def test_cli_multiple_eval_dirs(self, tmp_path):
        """Test CLI with multiple --eval-dir arguments."""
        eval_dir1 = tmp_path / "eval1"
        eval_dir2 = tmp_path / "eval2"
        eval_dir1.mkdir()
        eval_dir2.mkdir()
        
        output_file = tmp_path / "output.json"
        
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.analyze_wang_results",
                "--eval-dir", str(eval_dir1),
                "--eval-dir", str(eval_dir2),
                "--output", str(output_file)
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0


class TestGenerateWangVisualizationsProgrammaticInterface:
    """Tests for generate_wang_visualizations programmatic interface."""

    def test_function_signature(self):
        """Test that generate_wang_visualizations has correct signature."""
        import inspect
        sig = inspect.signature(generate_wang_visualizations)
        
        # Check parameters
        assert 'results_file' in sig.parameters
        assert 'output_dir' in sig.parameters
        
        # Check defaults are None (optional parameters)
        assert sig.parameters['results_file'].default is None
        assert sig.parameters['output_dir'].default is None

    def test_function_docstring(self):
        """Test that generate_wang_visualizations has comprehensive docstring."""
        doc = generate_wang_visualizations.__doc__
        assert doc is not None
        assert 'programmatically' in doc.lower() or 'programmatic' in doc.lower()
        assert 'results_file' in doc
        assert 'output_dir' in doc
        assert 'Example' in doc or 'example' in doc

    def test_visualize_with_mock_results(self, tmp_path):
        """Test generate_wang_visualizations with mock results file."""
        # Create mock results file
        results_file = tmp_path / "results.json"
        mock_results = {
            "summary_statistics": {
                "Beijing": {
                    "total_evaluations": 2,
                    "real_data_evaluations": 1,
                    "generated_evaluations": 1,
                    "models_evaluated": 1,
                    "mean_real_rate": 5.0,
                    "mean_generated_rate": 4.5
                }
            },
            "real_data": {},
            "generated_data": {},
            "comparisons": [],
            "statistical_analysis": {
                "model_rankings": {
                    "Beijing": []
                },
                "pattern_distributions": {},
                "baseline_usage_analysis": {},
                "cross_dataset_comparison": {},
                "statistical_tests": {
                    "Beijing": [],
                    "_metadata": {
                        "num_comparisons": 0,
                        "correction_method": "None",
                        "alpha": 0.05
                    }
                }
            }
        }
        
        with open(results_file, "w") as f:
            json.dump(mock_results, f)
        
        output_dir = tmp_path / "figures"
        
        # Test the function (should not crash regardless of matplotlib availability)
        generate_wang_visualizations(
            results_file=results_file,
            output_dir=output_dir
        )
        
        # Check matplotlib availability and verify output accordingly
        try:
            import matplotlib
            # If matplotlib is available, output_dir should be created
            assert output_dir.exists()
        except ImportError:
            # matplotlib not available - function should complete without creating output
            # (it logs an error but doesn't crash)
            pass

    def test_visualize_with_none_defaults(self):
        """Test that default parameters work (None values)."""
        # Should handle missing file gracefully
        with patch('tools.visualize_wang_results.HAS_MATPLOTLIB', False):
            # This should log an error but not crash
            generate_wang_visualizations()

    def test_handles_missing_results_file(self, tmp_path):
        """Test that function handles missing results file gracefully."""
        nonexistent_file = tmp_path / "nonexistent.json"
        output_dir = tmp_path / "figures"
        
        # Should not crash, just log error
        generate_wang_visualizations(
            results_file=nonexistent_file,
            output_dir=output_dir
        )
        # No assertion needed - just checking it doesn't crash


class TestGenerateWangVisualizationsCLI:
    """Tests for generate_wang_visualizations CLI interface."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            [sys.executable, "-m", "tools.visualize_wang_results", "--help"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output-dir" in result.stdout

    def test_cli_with_custom_paths(self, tmp_path):
        """Test CLI with custom input and output paths."""
        # Create mock results file
        results_file = tmp_path / "results.json"
        mock_results = {
            "summary_statistics": {},
            "real_data": {},
            "generated_data": {},
            "statistical_analysis": {
                "model_rankings": {},
                "pattern_distributions": {},
                "statistical_tests": {"_metadata": {"num_comparisons": 0}}
            }
        }
        
        with open(results_file, "w") as f:
            json.dump(mock_results, f)
        
        output_dir = tmp_path / "figures"
        
        result = subprocess.run(
            [
                sys.executable, "-m", "tools.visualize_wang_results",
                "--input", str(results_file),
                "--output-dir", str(output_dir)
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        # Should complete (will create output_dir if matplotlib available)
        assert result.returncode == 0


class TestWangResultsCollectorClass:
    """Tests for WangResultsCollector class functionality."""

    def test_collector_initialization(self, tmp_path):
        """Test WangResultsCollector initialization."""
        collector = WangResultsCollector(tmp_path)
        assert collector.project_root == tmp_path
        assert collector.results == []
        assert collector.comparisons == []

    def test_collector_with_empty_dir(self, tmp_path):
        """Test collector handles empty directory gracefully."""
        eval_dir = tmp_path / "empty_eval"
        eval_dir.mkdir()
        
        collector = WangResultsCollector(tmp_path)
        collector.collect_from_eval_dir(eval_dir)
        
        # Should not crash, just find no results
        assert len(collector.results) == 0
        assert len(collector.comparisons) == 0

    def test_aggregate_results_structure(self, tmp_path):
        """Test that aggregate_results returns correct structure."""
        collector = WangResultsCollector(tmp_path)
        
        aggregated = collector.aggregate_results()
        
        # Check required keys
        assert "summary_statistics" in aggregated
        assert "real_data" in aggregated
        assert "generated_data" in aggregated
        assert "comparisons" in aggregated

    def test_statistical_analysis_structure(self, tmp_path):
        """Test that perform_statistical_analysis returns correct structure."""
        collector = WangResultsCollector(tmp_path)
        
        analysis = collector.perform_statistical_analysis()
        
        # Check required keys
        assert "model_rankings" in analysis
        assert "pattern_distributions" in analysis
        assert "baseline_usage_analysis" in analysis
        assert "cross_dataset_comparison" in analysis
        assert "statistical_tests" in analysis


class TestDataClassesStructure:
    """Tests for data classes used in Wang analysis."""

    def test_detection_metrics_structure(self):
        """Test DetectionMetrics dataclass structure."""
        metrics = DetectionMetrics(
            dataset="Beijing",
            od_source="test",
            model=None,
            is_real=True,
            total_trajectories=100,
            abnormal_count=5,
            abnormal_rate=5.0,
            pattern_counts={"Abp1_normal": 95},
            abnormal_by_category={}
        )
        
        assert metrics.dataset == "Beijing"
        assert metrics.is_real is True
        assert metrics.total_trajectories == 100

    def test_statistical_comparison_structure(self):
        """Test StatisticalComparison dataclass structure."""
        comparison = StatisticalComparison(
            dataset="Beijing",
            od_source="test",
            model="distilled",
            real_rate=5.0,
            generated_rate=4.5,
            difference=-0.5,
            relative_difference_pct=-10.0,
            trajectory_count_real=100,
            trajectory_count_generated=100
        )
        
        assert comparison.dataset == "Beijing"
        assert comparison.model == "distilled"
        assert comparison.difference == -0.5


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_programmatic_workflow(self, tmp_path):
        """Test complete programmatic workflow: analyze -> visualize."""
        # Step 1: Create mock evaluation directory
        eval_dir = tmp_path / "eval"
        abnormal_dir = eval_dir / "abnormal" / "Beijing" / "test" / "real_data"
        abnormal_dir.mkdir(parents=True, exist_ok=True)
        
        detection_data = {
            "dataset": "Beijing",
            "total_trajectories": 100,
            "pattern_counts": {
                "Abp1_normal": 95,
                "Abp2_temporal_delay": 3,
                "Abp3_route_deviation": 1,
                "Abp4_both_deviations": 1
            }
        }
        
        with open(abnormal_dir / "detection_results.json", "w") as f:
            json.dump(detection_data, f)
        
        comparison_data = {
            "dataset": "Beijing",
            "od_source": "test",
            "real_data": {
                "total_trajectories": 100,
                "abnormal_count": 5,
                "abnormal_rate": 5.0
            },
            "generated_data": {}
        }
        
        with open(eval_dir / "abnormal" / "Beijing" / "test" / "comparison_report.json", "w") as f:
            json.dump(comparison_data, f)
        
        # Step 2: Run analysis
        results_file = tmp_path / "results.json"
        analyze_wang_results(
            eval_dirs=[eval_dir],
            output_file=results_file
        )
        
        assert results_file.exists()
        
        # Step 3: Generate visualizations
        output_dir = tmp_path / "figures"
        generate_wang_visualizations(
            results_file=results_file,
            output_dir=output_dir
        )
        
        # If matplotlib available, figures should be created
        # Otherwise, function should complete without error

    def test_cli_workflow(self, tmp_path):
        """Test complete CLI workflow: analyze -> visualize."""
        # Step 1: Create minimal mock data
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        
        results_file = tmp_path / "results.json"
        output_dir = tmp_path / "figures"
        
        # Step 2: Run analysis via CLI
        result1 = subprocess.run(
            [
                sys.executable, "-m", "tools.analyze_wang_results",
                "--eval-dir", str(eval_dir),
                "--output", str(results_file)
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result1.returncode == 0
        
        # Step 3: Run visualization via CLI (if results exist)
        if results_file.exists():
            result2 = subprocess.run(
                [
                    sys.executable, "-m", "tools.visualize_wang_results",
                    "--input", str(results_file),
                    "--output-dir", str(output_dir)
                ],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True
            )
            
            assert result2.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
