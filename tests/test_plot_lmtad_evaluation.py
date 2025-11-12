"""
Tests for LM-TAD Evaluation Plotting Module

This module tests the plotting functionality for LM-TAD teacher model evaluation,
including outlier detection rates, perplexity distributions, and summary tables.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.plot_lmtad_evaluation import (
    LMTADPlotConfig,
    create_lmtad_summary_table,
    load_evaluation_results,
    plot_lmtad_evaluation_from_files,
    plot_normal_trajectory_rates,
    plot_outlier_rate_comparison,
    plot_perplexity_distributions,
    plot_perplexity_scatter,
)


@pytest.fixture
def sample_real_results():
    """Create sample real trajectory evaluation results"""
    return {
        "total_trajectories": 1000,
        "outlier_rate": 0.15,
        "normal_trajectory_rate": 0.85,
        "mean_perplexity": 2.5,
        "std_perplexity": 0.8,
        "perplexity_values": [float(x) for x in np.random.lognormal(0.9, 0.3, 1000)],
        "trajectory_lengths": [int(x) for x in np.random.randint(5, 50, 1000)],
    }


@pytest.fixture
def sample_generated_results():
    """Create sample generated trajectory evaluation results"""
    return {
        "total_trajectories": 500,
        "outlier_rate": 0.18,
        "normal_trajectory_rate": 0.82,
        "mean_perplexity": 2.7,
        "std_perplexity": 0.9,
        "perplexity_values": [float(x) for x in np.random.lognormal(1.0, 0.35, 500)],
        "trajectory_lengths": [int(x) for x in np.random.randint(5, 50, 500)],
    }


@pytest.fixture
def temp_results_files(sample_real_results, sample_generated_results):
    """Create temporary result files for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        real_file = tmpdir_path / "real_results.json"
        gen_file = tmpdir_path / "generated_results.json"

        with open(real_file, "w") as f:
            json.dump(sample_real_results, f)

        with open(gen_file, "w") as f:
            json.dump(sample_generated_results, f)

        yield real_file, gen_file


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for plots"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "output"


def test_load_evaluation_results(temp_results_files):
    """Test loading evaluation results from JSON"""
    real_file, _ = temp_results_files

    results = load_evaluation_results(real_file)

    assert "total_trajectories" in results
    assert "outlier_rate" in results
    assert "normal_trajectory_rate" in results
    assert results["total_trajectories"] == 1000
    assert 0 <= results["outlier_rate"] <= 1


def test_load_evaluation_results_missing_file():
    """Test that loading non-existent file raises assertion"""
    with pytest.raises(AssertionError, match="Results file not found"):
        load_evaluation_results(Path("/nonexistent/file.json"))


def test_load_evaluation_results_missing_keys(tmp_path):
    """Test that loading file with missing keys raises assertion"""
    results_file = tmp_path / "incomplete_results.json"
    incomplete_results = {"total_trajectories": 100}  # Missing required keys
    with open(results_file, "w") as f:
        json.dump(incomplete_results, f)

    with pytest.raises(AssertionError, match="Results file must contain"):
        load_evaluation_results(results_file)


def test_plot_outlier_rate_comparison(
    sample_real_results, sample_generated_results, temp_output_dir
):
    """Test outlier rate comparison plot generation"""
    output_file = temp_output_dir / "outlier_comparison.png"

    plot_outlier_rate_comparison(
        sample_real_results,
        sample_generated_results,
        output_file,
        "test_dataset",
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_outlier_rate_comparison_missing_keys():
    """Test outlier rate comparison with missing data"""
    real_results = {"outlier_rate": 0.15}
    gen_results = {}  # Missing outlier_rate

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.png"
        with pytest.raises(AssertionError, match="Generated results missing"):
            plot_outlier_rate_comparison(real_results, gen_results, output_file, "test")


def test_plot_perplexity_distributions(
    sample_real_results, sample_generated_results, temp_output_dir
):
    """Test perplexity distributions plot generation"""
    output_file = temp_output_dir / "perplexity_dist.png"

    plot_perplexity_distributions(
        sample_real_results,
        sample_generated_results,
        output_file,
        "test_dataset",
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_perplexity_distributions_missing_data():
    """Test perplexity distributions with missing data"""
    real_results = {}  # Missing perplexity data
    gen_results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.png"
        with pytest.raises(
            AssertionError, match="Real results missing perplexity data"
        ):
            plot_perplexity_distributions(
                real_results, gen_results, output_file, "test"
            )


def test_plot_perplexity_distributions_summary_stats_only():
    """Test perplexity distributions with summary stats only (no raw values)"""
    real_results = {"mean_perplexity": 2.5, "std_perplexity": 0.8}
    gen_results = {"mean_perplexity": 2.7, "std_perplexity": 0.9}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.png"
        # Should not raise exception with summary stats only
        plot_perplexity_distributions(real_results, gen_results, output_file, "test")
        assert output_file.exists()


def test_plot_normal_trajectory_rates(
    sample_real_results, sample_generated_results, temp_output_dir
):
    """Test normal trajectory rates plot generation"""
    output_file = temp_output_dir / "normal_rates.png"

    plot_normal_trajectory_rates(
        sample_real_results,
        sample_generated_results,
        output_file,
        "test_dataset",
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_normal_trajectory_rates_missing_keys():
    """Test normal trajectory rates with missing data"""
    real_results = {"normal_trajectory_rate": 0.85}
    gen_results = {}  # Missing normal_trajectory_rate

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.png"
        with pytest.raises(AssertionError, match="Generated results missing"):
            plot_normal_trajectory_rates(real_results, gen_results, output_file, "test")


def test_plot_perplexity_scatter(
    sample_real_results, sample_generated_results, temp_output_dir
):
    """Test perplexity scatter plot generation"""
    output_file = temp_output_dir / "perplexity_scatter.png"

    plot_perplexity_scatter(
        sample_real_results,
        sample_generated_results,
        output_file,
        "test_dataset",
    )

    assert output_file.exists()
    assert output_file.with_suffix(".svg").exists()


def test_plot_perplexity_scatter_mismatched_lengths():
    """Test perplexity scatter with mismatched array lengths"""
    real_results = {
        "perplexity_values": [1.0, 2.0, 3.0],
        "trajectory_lengths": [10, 20],  # Different length
    }
    gen_results = {
        "perplexity_values": [1.5, 2.5],
        "trajectory_lengths": [15, 25],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test.png"
        with pytest.raises(AssertionError, match="Mismatched lengths"):
            plot_perplexity_scatter(real_results, gen_results, output_file, "test")


def test_plot_perplexity_scatter_missing_data(temp_output_dir):
    """Test perplexity scatter with missing length data"""
    real_results = {"perplexity_values": [1.0, 2.0, 3.0]}  # No trajectory_lengths
    gen_results = {"perplexity_values": [1.5, 2.5, 3.5]}

    output_file = temp_output_dir / "perplexity_scatter.png"
    # Should not raise exception when length data is missing
    plot_perplexity_scatter(real_results, gen_results, output_file, "test")
    # Plot should not be generated when data is missing
    assert not output_file.exists() or output_file.stat().st_size == 0


def test_create_lmtad_summary_table(
    sample_real_results, sample_generated_results, temp_output_dir
):
    """Test summary table generation"""
    output_file = temp_output_dir / "summary_table"

    create_lmtad_summary_table(
        sample_real_results,
        sample_generated_results,
        output_file,
        "test_dataset",
    )

    assert output_file.with_suffix(".tex").exists()
    assert output_file.with_suffix(".md").exists()

    # Verify LaTeX content
    with open(output_file.with_suffix(".tex")) as f:
        latex_content = f.read()
        assert "\\begin{table}" in latex_content
        assert "test_dataset" in latex_content
        assert "Mean Perplexity" in latex_content

    # Verify Markdown content
    with open(output_file.with_suffix(".md")) as f:
        md_content = f.read()
        assert "# LM-TAD Evaluation Summary" in md_content
        assert "test_dataset" in md_content
        assert "| Metric |" in md_content


def test_plot_lmtad_evaluation_from_files(temp_results_files, temp_output_dir):
    """Test complete plot generation from files"""
    real_file, gen_file = temp_results_files

    plot_files = plot_lmtad_evaluation_from_files(
        real_results_file=real_file,
        generated_results_file=gen_file,
        output_dir=temp_output_dir,
        dataset="test_dataset",
    )

    # Verify all expected plots were created
    assert "outlier_rate_comparison" in plot_files
    assert "perplexity_distributions" in plot_files
    assert "normal_trajectory_rates" in plot_files
    assert "perplexity_scatter" in plot_files
    assert "summary_table" in plot_files

    # Verify files exist
    for plot_name, plot_path in plot_files.items():
        if plot_name == "summary_table":
            assert plot_path.with_suffix(".tex").exists()
            assert plot_path.with_suffix(".md").exists()
        else:
            assert plot_path.exists()
            assert plot_path.with_suffix(".svg").exists()


def test_plot_lmtad_evaluation_custom_config(temp_results_files, temp_output_dir):
    """Test plot generation with custom configuration"""
    real_file, gen_file = temp_results_files

    config = LMTADPlotConfig(
        figure_size=(10, 6),
        dpi=150,
        perplexity_bins=30,
    )

    plot_files = plot_lmtad_evaluation_from_files(
        real_results_file=real_file,
        generated_results_file=gen_file,
        output_dir=temp_output_dir,
        dataset="test_dataset",
        config=config,
    )

    assert len(plot_files) > 0
    assert temp_output_dir.exists()


def test_plot_missing_trajectory_lengths(temp_output_dir):
    """Test handling of missing trajectory length data"""
    real_results = {
        "total_trajectories": 100,
        "outlier_rate": 0.1,
        "normal_trajectory_rate": 0.9,
        "mean_perplexity": 2.0,
        "std_perplexity": 0.5,
        "perplexity_values": [2.0, 2.5, 1.8],
        # No trajectory_lengths field
    }

    gen_results = {
        "total_trajectories": 50,
        "outlier_rate": 0.12,
        "normal_trajectory_rate": 0.88,
        "mean_perplexity": 2.1,
        "std_perplexity": 0.6,
        "perplexity_values": [2.1, 2.6, 1.9],
        # No trajectory_lengths field
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        real_file = tmpdir_path / "real.json"
        gen_file = tmpdir_path / "gen.json"

        with open(real_file, "w") as f:
            json.dump(real_results, f)
        with open(gen_file, "w") as f:
            json.dump(gen_results, f)

        plot_files = plot_lmtad_evaluation_from_files(
            real_results_file=real_file,
            generated_results_file=gen_file,
            output_dir=temp_output_dir,
            dataset="test_dataset",
        )

        # Should not include perplexity_scatter
        assert "perplexity_scatter" not in plot_files
        # Should still have other plots
        assert "outlier_rate_comparison" in plot_files
        assert "perplexity_distributions" in plot_files


def test_plot_file_assertions(temp_output_dir):
    """Test that missing files raise appropriate assertions"""
    nonexistent_file = Path("/nonexistent/results.json")

    with pytest.raises(AssertionError, match="Real results file not found"):
        plot_lmtad_evaluation_from_files(
            real_results_file=nonexistent_file,
            generated_results_file=nonexistent_file,
            output_dir=temp_output_dir,
            dataset="test",
        )


def test_plot_output_directory_creation(temp_results_files, tmp_path):
    """Test that output directory is created when it doesn't exist"""
    real_file, gen_file = temp_results_files
    output_dir = tmp_path / "nonexistent" / "subdir" / "output"

    plot_files = plot_lmtad_evaluation_from_files(
        real_results_file=real_file,
        generated_results_file=gen_file,
        output_dir=output_dir,
        dataset="test_dataset",
    )

    # Directory should be created
    assert output_dir.exists()
    # Files should be created
    assert len(plot_files) > 0


if __name__ == "__main__":
    pytest.main([__file__])
