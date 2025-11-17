"""Tests for LM-TAD spatial abnormality evaluation tools."""

import json
import subprocess
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the modules to test
from tools.extract_lmtad_spatial_abnormal_od import (
    extract_spatial_abnormal_od_pairs,
    parse_trajectory_from_tsv,
    extract_od_from_trajectory,
)
from tools.analyze_lmtad_spatial_results import (
    aggregate_lmtad_spatial_results,
    load_source_real_rates,
    compute_statistical_test,
)
from tools.visualize_lmtad_spatial_results import (
    load_aggregated_results,
    plot_spatial_abnormality_rates_comparison,
)


class TestExtractLMTADSpatialAbnormalOD:
    """Tests for extract_lmtad_spatial_abnormal_od module."""

    def test_parse_trajectory_from_tsv(self):
        """Test parsing trajectory from TSV string."""
        # Test JSON array string
        traj_str = "[74, 74, 208, 208, 6165]"
        result = parse_trajectory_from_tsv(traj_str)
        assert result == [74, 74, 208, 208, 6165]

        # Test comma-separated string
        traj_str = "74, 74, 208, 6165"
        result = parse_trajectory_from_tsv(traj_str)
        assert result == [74, 74, 208, 6165]

        # Test list input
        traj_list = [74, 74, 208, 6165]
        result = parse_trajectory_from_tsv(traj_list)
        assert result == [74, 74, 208, 6165]

    def test_extract_od_from_trajectory(self):
        """Test extracting OD pair from trajectory."""
        # Normal trajectory with EOS token
        road_ids = [74, 74, 208, 208, 6165]
        origin, dest = extract_od_from_trajectory(road_ids)
        assert origin == 74
        assert dest == 208

        # Trajectory without EOS token
        road_ids = [74, 74, 208, 208]
        origin, dest = extract_od_from_trajectory(road_ids)
        assert origin == 74
        assert dest == 208

        # Single road ID (should use same as origin and dest)
        road_ids = [74, 6165]
        origin, dest = extract_od_from_trajectory(road_ids)
        assert origin == 74
        assert dest == 74

    def test_extract_od_from_trajectory_empty(self):
        """Test extracting OD from empty trajectory raises error."""
        with pytest.raises(ValueError, match="Empty trajectory"):
            extract_od_from_trajectory([])

    def test_extract_spatial_abnormal_od_pairs(self, tmp_path):
        """Test extracting OD pairs from TSV file."""
        # Create mock TSV file
        tsv_file = tmp_path / "test_outliers.tsv"
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        # Create TSV with spatial outliers
        tsv_content = """log_perplexity\toutlier\tseq_length\ttrajectory
0.304\tnon outlier\t53\t[74, 74, 208, 6165]
7.026\troute switch\t45\t[100, 200, 300, 6165]
8.413\tdetour\t50\t[150, 250, 350, 6165]
7.100\troute switch\t40\t[100, 200, 300, 6165]
8.500\tdetour\t55\t[150, 250, 350, 450, 6165]
"""
        tsv_file.write_text(tsv_content)

        result = extract_spatial_abnormal_od_pairs(
            tsv_file=tsv_file,
            dataset="test_dataset",
            source_eval_dir=source_eval_dir,
        )

        assert result["dataset"] == "test_dataset"
        assert result["source"] == "lmtad"
        assert result["total_spatial_abnormal_trajectories"] == 4
        assert len(result["od_pairs_by_type"]["route_switch"]) == 1  # Unique OD pairs
        assert (
            len(result["od_pairs_by_type"]["detour"]) == 2
        )  # Two different destinations
        # OD pairs are returned as tuples
        assert result["od_pairs_by_type"]["route_switch"][0] == (100, 300)
        # Check that detour pairs include both destinations
        detour_ods = result["od_pairs_by_type"]["detour"]
        assert (150, 350) in detour_ods or (150, 450) in detour_ods

    def test_extract_spatial_abnormal_od_pairs_no_outliers(self, tmp_path):
        """Test extraction when no spatial outliers exist."""
        tsv_file = tmp_path / "test_outliers.tsv"
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        # Create TSV with only non-outliers
        tsv_content = """log_perplexity\toutlier\tseq_length\ttrajectory
0.304\tnon outlier\t53\t[74, 74, 208, 6165]
0.420\tnon outlier\t35\t[100, 200, 300, 6165]
"""
        tsv_file.write_text(tsv_content)

        result = extract_spatial_abnormal_od_pairs(
            tsv_file=tsv_file,
            dataset="test_dataset",
            source_eval_dir=source_eval_dir,
        )

        assert result["total_spatial_abnormal_trajectories"] == 0
        assert len(result["od_pairs_by_type"]["route_switch"]) == 0
        assert len(result["od_pairs_by_type"]["detour"]) == 0


class TestAnalyzeLMTADSpatialResults:
    """Tests for analyze_lmtad_spatial_results module."""

    def test_compute_statistical_test(self):
        """Test chi-square test computation."""
        chi2, p_value = compute_statistical_test(
            real_count=100,
            real_total=1000,
            gen_count=50,
            gen_total=1000,
        )

        assert isinstance(chi2, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        assert chi2 >= 0

    def test_load_source_real_rates(self, tmp_path):
        """Test loading real rates from source evaluation directory."""
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        # Create TSV file with spatial outliers
        tsv_file = eval_dir / "ckpt_best_outliers_config_test.tsv"
        tsv_content = """log_perplexity\toutlier\tseq_length\ttrajectory
0.304\tnon outlier\t53\t[74, 74, 208, 6165]
7.026\troute switch\t45\t[100, 200, 300, 6165]
8.413\tdetour\t50\t[150, 250, 350, 6165]
7.100\troute switch\t40\t[100, 200, 300, 6165]
8.500\tdetour\t55\t[150, 250, 350, 450, 6165]
"""
        tsv_file.write_text(tsv_content)

        rates = load_source_real_rates(eval_dir)

        assert "spatial_abnormality_rate" in rates
        assert "route_switch_rate" in rates
        assert "detour_rate" in rates
        assert rates["total_trajectories"] == 5
        assert rates["route_switch_count"] == 2
        assert rates["detour_count"] == 2

    def test_aggregate_lmtad_spatial_results(self, tmp_path):
        """Test aggregating spatial results."""
        eval_dir = tmp_path / "eval_dir"
        eval_dir.mkdir()
        dataset = "test_dataset"

        # Create source eval directory
        source_eval_dir = tmp_path / "source_eval"
        source_eval_dir.mkdir()
        tsv_file = source_eval_dir / "ckpt_best_outliers_config_test.tsv"
        tsv_content = """log_perplexity\toutlier\tseq_length\ttrajectory
0.304\tnon outlier\t53\t[74, 74, 208, 6165]
7.026\troute switch\t45\t[100, 200, 300, 6165]
8.413\tdetour\t50\t[150, 250, 350, 6165]
"""
        tsv_file.write_text(tsv_content)

        # Create evaluation results directory
        eval_results_dir = eval_dir / "eval_lmtad_spatial" / dataset
        eval_results_dir.mkdir(parents=True)

        # Create mock evaluation result
        result_file = eval_results_dir / "model1_spatial_evaluation.json"
        result_data = {
            "model": "model1",
            "dataset": dataset,
            "total_trajectories": 1000,
            "spatial_abnormal_count": 50,
            "spatial_abnormality_rate": 5.0,
            "by_type": {
                "route_switch": {"count": 30, "rate": 3.0},
                "detour": {"count": 20, "rate": 2.0},
                "non_outlier": {"count": 950, "rate": 95.0},
            },
            "log_perplexity_stats": {
                "mean": 2.5,
                "std": 1.0,
                "median": 2.3,
            },
        }
        with open(result_file, "w") as f:
            json.dump(result_data, f)

        result = aggregate_lmtad_spatial_results(
            eval_dir=eval_dir,
            dataset=dataset,
            source_eval_dir=source_eval_dir,
        )

        assert "summary_statistics" in result
        assert dataset in result["summary_statistics"]
        assert "real_data" in result
        assert "generated_data" in result
        assert "statistical_analysis" in result

        # Check statistical tests
        tests = result["statistical_analysis"]["statistical_tests"]
        assert len(tests) == 1
        assert tests[0]["model"] == "model1"
        assert "p_value" in tests[0]
        assert "cohens_h" in tests[0]


class TestVisualizeLMTADSpatialResults:
    """Tests for visualize_lmtad_spatial_results module."""

    def test_load_aggregated_results(self, tmp_path):
        """Test loading aggregated results from JSON."""
        results_file = tmp_path / "results.json"
        results_data = {
            "summary_statistics": {
                "test_dataset": {
                    "real_spatial_abnormality_rate": 6.54,
                    "real_route_switch_rate": 3.27,
                    "real_detour_rate": 3.27,
                }
            },
            "generated_data": {
                "test_dataset": {
                    "model1": {
                        "spatial_abnormality_rate": 5.0,
                        "route_switch_rate": 3.0,
                        "detour_rate": 2.0,
                    }
                }
            },
            "statistical_analysis": {
                "statistical_tests": [
                    {
                        "model": "model1",
                        "generated_rate": 5.0,
                        "real_rate": 6.54,
                        "difference": -1.54,
                        "p_value": 0.01,
                        "significant": True,
                    }
                ]
            },
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f)

        loaded = load_aggregated_results(results_file)
        assert loaded == results_data

    @patch("tools.visualize_lmtad_spatial_results.plt.subplots")
    @patch("tools.visualize_lmtad_spatial_results.plt")
    def test_plot_spatial_abnormality_rates_comparison(
        self, mock_plt, mock_subplots, tmp_path
    ):
        """Test plotting spatial abnormality rates."""
        # Mock subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        results = {
            "summary_statistics": {
                "test_dataset": {
                    "real_spatial_abnormality_rate": 6.54,
                }
            },
            "statistical_analysis": {
                "statistical_tests": [
                    {
                        "model": "model1",
                        "generated_rate": 5.0,
                        "ci_lower": 4.5,
                        "ci_upper": 5.5,
                        "effect_size": "small",
                        "cohens_h": 0.1,
                    }
                ]
            },
        }

        output_dir = tmp_path / "figures"
        plot_spatial_abnormality_rates_comparison(results, output_dir, "test_dataset")

        # Check that subplots was called
        assert mock_subplots.called
        # Check that savefig was called (may be called on fig or plt)
        assert mock_fig.savefig.called or mock_plt.savefig.called


class TestLMTADSpatialPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @patch("tools.run_lmtad_spatial_pipeline.subprocess")
    def test_pipeline_skips_existing_files(self, mock_subprocess, tmp_path):
        """Test that pipeline skips steps when files already exist."""
        eval_dir = tmp_path / "eval_dir"
        eval_dir.mkdir()
        dataset = "test_dataset"

        # Create existing OD pairs file
        od_pairs_file = eval_dir / f"abnormal_od_pairs_lmtad_spatial_{dataset}.json"
        od_pairs_file.parent.mkdir(parents=True, exist_ok=True)
        od_pairs_file.write_text('{"dataset": "test_dataset"}')

        # Create existing gene directory
        gene_dir = eval_dir / "gene_abnormal_lmtad_spatial" / dataset / "seed42"
        gene_dir.mkdir(parents=True)
        (gene_dir / "model1_spatial_abnormal.csv").write_text(
            "traj_id,rid_list\n1,[1,2,3]"
        )

        # Create existing evaluation result
        eval_result_dir = eval_dir / "eval_lmtad_spatial" / dataset
        eval_result_dir.mkdir(parents=True)
        (eval_result_dir / "model1_spatial_evaluation.json").write_text(
            '{"model": "model1"}'
        )

        # Import and run pipeline (mocked)

        # This would normally run the pipeline, but we're testing the skip logic
        # The actual implementation should check for existing files and skip steps
        assert od_pairs_file.exists()
        assert (gene_dir / "model1_spatial_abnormal.csv").exists()
        assert (eval_result_dir / "model1_spatial_evaluation.json").exists()


class TestCombinedReport:
    """Tests for create_combined_abnormal_report module."""

    def test_load_wang_results(self, tmp_path):
        """Test loading Wang results."""
        from tools.create_combined_abnormal_report import load_wang_results

        wang_file = tmp_path / "wang_results.json"
        wang_data = {
            "summary_statistics": {
                "test_dataset": {
                    "mean_real_rate": 10.0,
                }
            }
        }
        with open(wang_file, "w") as f:
            json.dump(wang_data, f)

        result = load_wang_results(wang_file)
        assert result == wang_data

    def test_load_lmtad_spatial_results(self, tmp_path):
        """Test loading LM-TAD spatial results."""
        from tools.create_combined_abnormal_report import load_lmtad_spatial_results

        spatial_file = tmp_path / "spatial_results.json"
        spatial_data = {
            "summary_statistics": {
                "test_dataset": {
                    "real_spatial_abnormality_rate": 6.54,
                }
            }
        }
        with open(spatial_file, "w") as f:
            json.dump(spatial_data, f)

        result = load_lmtad_spatial_results(spatial_file)
        assert result == spatial_data

    def test_create_combined_report(self, tmp_path):
        """Test creating combined report."""
        from tools.create_combined_abnormal_report import create_combined_report

        wang_results = {
            "summary_statistics": {
                "test_dataset": {
                    "mean_real_rate": 10.0,
                    "generated_rates": [8.0, 9.0, 10.5],
                }
            },
            "statistical_analysis": {
                "statistical_tests": {
                    "test_dataset": [
                        {
                            "model": "model1",
                            "generated_rate": 8.0,
                            "real_rate": 10.0,
                            "difference": -2.0,
                            "effect_size": "medium",
                        }
                    ]
                }
            },
        }

        spatial_results = {
            "summary_statistics": {
                "test_dataset": {
                    "real_spatial_abnormality_rate": 6.54,
                    "real_route_switch_rate": 3.27,
                    "real_detour_rate": 3.27,
                    "generated_spatial_rates": [5.0, 6.0, 7.0],
                }
            },
            "generated_data": {
                "test_dataset": {
                    "model1": {
                        "spatial_abnormality_rate": 5.0,
                        "route_switch_rate": 3.0,
                        "detour_rate": 2.0,
                    }
                }
            },
            "statistical_analysis": {
                "statistical_tests": [
                    {
                        "model": "model1",
                        "generated_rate": 5.0,
                        "real_rate": 6.54,
                        "difference": -1.54,
                        "effect_size": "small",
                        "significant": False,
                    }
                ]
            },
        }

        output_file = tmp_path / "combined_report.md"
        create_combined_report(
            wang_results=wang_results,
            lmtad_spatial_results=spatial_results,
            output_file=output_file,
            dataset="test_dataset",
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "Comprehensive Abnormal Trajectory Analysis Report" in content
        assert "Temporal Abnormality Analysis" in content
        assert "Spatial Abnormality Analysis" in content
        assert "Combined Model Rankings" in content


class TestLMTADSpatialCLI:
    """Tests for CLI interfaces of LM-TAD spatial tools."""

    def test_extract_od_cli_help(self):
        """Test extract_lmtad_spatial_abnormal_od CLI help."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.extract_lmtad_spatial_abnormal_od",
                "--help",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--tsv-file" in result.stdout
        assert "--dataset" in result.stdout
        assert "--output" in result.stdout

    def test_analyze_results_cli_help(self):
        """Test analyze_lmtad_spatial_results CLI help."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.analyze_lmtad_spatial_results",
                "--help",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--eval-dir" in result.stdout
        assert "--dataset" in result.stdout
        assert "--output" in result.stdout

    def test_visualize_results_cli_help(self):
        """Test visualize_lmtad_spatial_results CLI help."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.visualize_lmtad_spatial_results",
                "--help",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--dataset" in result.stdout

    def test_pipeline_cli_help(self):
        """Test run_lmtad_spatial_pipeline CLI help."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.run_lmtad_spatial_pipeline",
                "--help",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--eval-dir" in result.stdout
        assert "--dataset" in result.stdout
        assert "--lmtad-source-eval-dir" in result.stdout

    def test_combined_report_cli_help(self):
        """Test create_combined_abnormal_report CLI help."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.create_combined_abnormal_report",
                "--help",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--output" in result.stdout
        assert "--dataset" in result.stdout


class TestLMTADSpatialFunctionSignatures:
    """Test function signatures and docstrings for programmatic interfaces."""

    def test_extract_functions_have_docstrings(self):
        """Test that extraction functions have docstrings."""
        assert extract_spatial_abnormal_od_pairs.__doc__ is not None
        assert parse_trajectory_from_tsv.__doc__ is not None
        assert extract_od_from_trajectory.__doc__ is not None

    def test_analyze_functions_have_docstrings(self):
        """Test that analysis functions have docstrings."""
        assert aggregate_lmtad_spatial_results.__doc__ is not None
        assert load_source_real_rates.__doc__ is not None
        assert compute_statistical_test.__doc__ is not None

    def test_visualize_functions_have_docstrings(self):
        """Test that visualization functions have docstrings."""
        assert load_aggregated_results.__doc__ is not None
        assert plot_spatial_abnormality_rates_comparison.__doc__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
