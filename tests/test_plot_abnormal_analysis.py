#!/usr/bin/env python3
"""
Test suite for plot_abnormal_analysis module.

This test suite follows TDD principles - tests are written before implementation.
Tests cover all functions in the plot_abnormal_analysis module with comprehensive
coverage including edge cases and error conditions.
"""

import json
import tempfile
from pathlib import Path
import pytest
import numpy as np

# Import functions to be tested (will be implemented)
from tools.plot_abnormal_analysis import (
    load_abnormal_od_pairs,
    extract_normal_od_pairs_from_data,
    compute_od_heatmap_matrix,
    plot_abnormal_od_distribution,
    plot_abnormal_categories_summary,
    plot_temporal_delay_analysis,
    plot_abnormal_od_heatmap,
    plot_normal_od_heatmap,
    plot_od_heatmap_comparison,
    plot_analysis_from_files,
)


class TestLoadAbnormalOdPairs:
    """Test suite for load_abnormal_od_pairs function"""

    def test_load_valid_od_pairs_file(self):
        """Test loading valid OD pairs JSON file"""
        od_data = {
            "od_pairs_by_category": {
                "wang_temporal_delay": [(1, 2), (2, 3), (1, 3)],
                "wang_route_deviation": [(4, 5)],
            },
            "summary": {"total_pairs": 4},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(od_data, f)
            temp_file = Path(f.name)

        try:
            result = load_abnormal_od_pairs(temp_file)

            assert isinstance(result, dict)
            assert "od_pairs_by_category" in result
            assert len(result["od_pairs_by_category"]) == 2
            assert result["od_pairs_by_category"]["wang_temporal_delay"] == [
                (1, 2),
                (2, 3),
                (1, 3),
            ]
        finally:
            temp_file.unlink()

    def test_load_missing_file(self):
        """Test fail-fast assertion for missing file"""
        missing_file = Path("/nonexistent/path/od_pairs.json")

        with pytest.raises(AssertionError, match="not found"):
            load_abnormal_od_pairs(missing_file)

    def test_load_invalid_json(self):
        """Test handling of invalid JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            temp_file = Path(f.name)

        try:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                load_abnormal_od_pairs(temp_file)
        finally:
            temp_file.unlink()

    def test_load_missing_required_key(self):
        """Test fail-fast assertion for missing required key"""
        od_data = {"summary": {"total_pairs": 4}}  # Missing od_pairs_by_category

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(od_data, f)
            temp_file = Path(f.name)

        try:
            with pytest.raises(AssertionError, match="od_pairs_by_category"):
                load_abnormal_od_pairs(temp_file)
        finally:
            temp_file.unlink()


class TestExtractNormalOdPairs:
    """Test suite for extract_normal_od_pairs_from_data function"""

    def test_extract_from_rid_list_format(self):
        """Test extracting normal OD pairs from rid_list format"""
        # Create mock CSV file with rid_list format (quoted to handle commas)
        csv_content = 'traj_id,rid_list\n0,"[1,2,3]"\n1,"[2,3,4]"\n2,"[1,4,5]"\n'
        abnormal_traj_ids = {1}  # traj_id 1 is abnormal

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_file = Path(f.name)

        try:
            result = extract_normal_od_pairs_from_data(
                [temp_file], abnormal_traj_ids, max_trajectories=None
            )

            assert isinstance(result, list)
            # Should have OD pairs from traj_id 0 and 2 (normal), not 1 (abnormal)
            assert (1, 3) in result  # From traj_id 0
            assert (1, 5) in result  # From traj_id 2
            assert (2, 4) not in result  # From traj_id 1 (abnormal)
        finally:
            temp_file.unlink()

    def test_extract_empty_abnormal_list(self):
        """Test extracting when no abnormal trajectories (all are normal)"""
        csv_content = 'traj_id,rid_list\n0,"[1,2,3]"\n1,"[2,3,4]"\n'
        abnormal_traj_ids = set()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_file = Path(f.name)

        try:
            result = extract_normal_od_pairs_from_data(
                [temp_file], abnormal_traj_ids, max_trajectories=None
            )

            assert len(result) == 2  # Both trajectories are normal
            assert (1, 3) in result
            assert (2, 4) in result
        finally:
            temp_file.unlink()

    def test_extract_with_max_trajectories(self):
        """Test sampling with max_trajectories limit"""
        csv_content = "traj_id,rid_list\n" + "\n".join(
            [f'{i},"[{i},{i + 1},{i + 2}]"' for i in range(10)]
        )
        abnormal_traj_ids = set()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_file = Path(f.name)

        try:
            result = extract_normal_od_pairs_from_data(
                [temp_file], abnormal_traj_ids, max_trajectories=5
            )

            assert len(result) <= 5  # Should be limited
        finally:
            temp_file.unlink()

    def test_extract_missing_file(self):
        """Test fail-fast assertion for missing file"""
        missing_file = Path("/nonexistent/path/data.csv")
        abnormal_traj_ids = set()

        with pytest.raises(AssertionError, match="not found"):
            extract_normal_od_pairs_from_data([missing_file], abnormal_traj_ids)

    def test_extract_empty_file_list(self):
        """Test fail-fast assertion for empty file list"""
        abnormal_traj_ids = set()

        with pytest.raises(AssertionError, match="cannot be empty"):
            extract_normal_od_pairs_from_data([], abnormal_traj_ids)


class TestComputeOdHeatmapMatrix:
    """Test suite for compute_od_heatmap_matrix function"""

    def test_compute_matrix_basic(self):
        """Test basic matrix computation"""
        od_pairs = [(1, 2), (1, 3), (2, 3), (1, 2), (1, 2)]  # (1,2) appears 3 times

        matrix, origins, dests = compute_od_heatmap_matrix(od_pairs, top_n=10)

        assert isinstance(matrix, np.ndarray)
        assert len(origins) > 0
        assert len(dests) > 0
        assert matrix.shape == (len(origins), len(dests))

        # Find indices for origin=1, dest=2
        if 1 in origins and 2 in dests:
            origin_idx = origins.index(1)
            dest_idx = dests.index(2)
            assert matrix[origin_idx, dest_idx] == 3  # Count of (1,2)

    def test_compute_matrix_with_top_origins_dests(self):
        """Test matrix computation with predefined top origins/destinations"""
        od_pairs = [(1, 2), (1, 3), (2, 3), (4, 5)]
        top_origins = [1, 2]
        top_dests = [2, 3]

        matrix, origins, dests = compute_od_heatmap_matrix(
            od_pairs, top_origins=top_origins, top_dests=top_dests
        )

        assert origins == top_origins
        assert dests == top_dests
        assert matrix.shape == (2, 2)

    def test_compute_matrix_empty_pairs(self):
        """Test fail-fast assertion for empty OD pairs"""
        with pytest.raises(AssertionError, match="cannot be empty"):
            compute_od_heatmap_matrix([])

    def test_compute_matrix_invalid_top_n(self):
        """Test fail-fast assertion for invalid top_n"""
        od_pairs = [(1, 2), (2, 3)]

        with pytest.raises(AssertionError, match="must be positive"):
            compute_od_heatmap_matrix(od_pairs, top_n=0)


class TestPlotAbnormalOdDistribution:
    """Test suite for plot_abnormal_od_distribution function"""

    def test_plot_valid_data(self, tmp_path):
        """Test plotting with valid OD data"""
        od_data = {
            "od_pairs_by_category": {
                "wang_temporal_delay": [(1, 2), (1, 3), (2, 3), (1, 2)],
            }
        }
        output_file = tmp_path / "test_distribution.png"

        plot_abnormal_od_distribution(od_data, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_missing_category_key(self, tmp_path):
        """Test fail-fast assertion for missing required key"""
        od_data = {"summary": {"total": 4}}  # Missing od_pairs_by_category
        output_file = tmp_path / "test_distribution.png"

        with pytest.raises(AssertionError, match="od_pairs_by_category"):
            plot_abnormal_od_distribution(od_data, output_file, "test_dataset")


class TestPlotAbnormalCategoriesSummary:
    """Test suite for plot_abnormal_categories_summary function"""

    def test_plot_valid_categories(self, tmp_path):
        """Test plotting with valid categories"""
        od_data = {
            "od_pairs_by_category": {
                "wang_temporal_delay": [(1, 2), (2, 3)],
                "wang_route_deviation": [(4, 5)],
            }
        }
        output_file = tmp_path / "test_categories.png"

        plot_abnormal_categories_summary(od_data, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_empty_categories(self, tmp_path):
        """Test fail-fast assertion for empty categories"""
        od_data = {"od_pairs_by_category": {}}
        output_file = tmp_path / "test_categories.png"

        with pytest.raises(AssertionError, match="No abnormal categories"):
            plot_abnormal_categories_summary(od_data, output_file, "test_dataset")


class TestPlotTemporalDelayAnalysis:
    """Test suite for plot_temporal_delay_analysis function"""

    def test_plot_valid_samples(self, tmp_path):
        """Test plotting with valid sample files"""
        samples_data = [
            {
                "traj_id": 1,
                "pattern": "Abp2_temporal_delay",
                "details": {
                    "time_deviation_sec": 100,
                    "length_deviation_m": 50,
                    "baseline_type": "od_specific",
                },
            }
        ]

        samples_file = tmp_path / "samples.json"
        with open(samples_file, "w") as f:
            json.dump(samples_data, f)

        output_file = tmp_path / "test_temporal.png"

        plot_temporal_delay_analysis([samples_file], output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_no_samples_files(self):
        """Test fail-fast assertion when no sample files exist"""
        missing_files = [Path("/nonexistent/samples.json")]
        output_file = Path("/tmp/test.png")

        with pytest.raises(AssertionError, match="No samples files found"):
            plot_temporal_delay_analysis(missing_files, output_file, "test_dataset")

    def test_plot_empty_file_list(self):
        """Test fail-fast assertion for empty file list"""
        output_file = Path("/tmp/test.png")

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_temporal_delay_analysis([], output_file, "test_dataset")


class TestPlotAbnormalOdHeatmap:
    """Test suite for plot_abnormal_od_heatmap function"""

    def test_plot_valid_od_data(self, tmp_path):
        """Test plotting with valid OD data"""
        od_data = {
            "od_pairs_by_category": {
                "wang_temporal_delay": [(1, 2), (1, 3), (2, 3), (1, 2)],
            }
        }
        output_file = tmp_path / "test_heatmap.png"

        plot_abnormal_od_heatmap(od_data, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_missing_category_key(self, tmp_path):
        """Test fail-fast assertion for missing required key"""
        od_data = {"summary": {"total": 4}}
        output_file = tmp_path / "test_heatmap.png"

        with pytest.raises(AssertionError, match="od_pairs_by_category"):
            plot_abnormal_od_heatmap(od_data, output_file, "test_dataset")


class TestPlotNormalOdHeatmap:
    """Test suite for plot_normal_od_heatmap function"""

    def test_plot_valid_od_pairs(self, tmp_path):
        """Test plotting with valid normal OD pairs"""
        normal_od_pairs = [(1, 2), (1, 3), (2, 3), (1, 2)]
        output_file = tmp_path / "test_normal_heatmap.png"

        plot_normal_od_heatmap(normal_od_pairs, output_file, "test_dataset")

        assert output_file.exists()

    def test_plot_empty_od_pairs(self, tmp_path):
        """Test fail-fast assertion for empty OD pairs"""
        output_file = tmp_path / "test_normal_heatmap.png"

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_normal_od_heatmap([], output_file, "test_dataset")


class TestPlotOdHeatmapComparison:
    """Test suite for plot_od_heatmap_comparison function"""

    def test_plot_comparison_valid_data(self, tmp_path):
        """Test plotting comparison with valid data"""
        abnormal_od_pairs = [(1, 2), (1, 3), (2, 3)]
        normal_od_pairs = [(1, 2), (4, 5), (6, 7)]
        output_file = tmp_path / "test_comparison.png"

        plot_od_heatmap_comparison(
            abnormal_od_pairs, normal_od_pairs, output_file, "test_dataset"
        )

        assert output_file.exists()

    def test_plot_comparison_empty_abnormal(self, tmp_path):
        """Test fail-fast assertion for empty abnormal pairs"""
        output_file = tmp_path / "test_comparison.png"

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_od_heatmap_comparison([], [(1, 2)], output_file, "test_dataset")

    def test_plot_comparison_empty_normal(self, tmp_path):
        """Test fail-fast assertion for empty normal pairs"""
        output_file = tmp_path / "test_comparison.png"

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_od_heatmap_comparison([(1, 2)], [], output_file, "test_dataset")


class TestPlotAnalysisFromFiles:
    """Test suite for plot_analysis_from_files function"""

    def test_plot_from_valid_files(self, tmp_path):
        """Test end-to-end plotting from files"""
        # Create mock files
        od_pairs_file = tmp_path / "od_pairs.json"
        od_data = {"od_pairs_by_category": {"wang_temporal_delay": [(1, 2), (2, 3)]}}
        with open(od_pairs_file, "w") as f:
            json.dump(od_data, f)

        train_csv = tmp_path / "train.csv"
        train_csv.write_text("traj_id,rid_list\n0,[1,2,3]\n")

        test_csv = tmp_path / "test.csv"
        test_csv.write_text("traj_id,rid_list\n1,[2,3,4]\n")

        detection_train = tmp_path / "detection_train.json"
        detection_train.write_text('{"abnormal_indices": {"wang_temporal_delay": []}}')

        detection_test = tmp_path / "detection_test.json"
        detection_test.write_text('{"abnormal_indices": {"wang_temporal_delay": []}}')

        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = plot_analysis_from_files(
            abnormal_od_pairs_file=od_pairs_file,
            real_data_files=[train_csv, test_csv],
            detection_results_files=[detection_train, detection_test],
            samples_dir=samples_dir,
            output_dir=output_dir,
            dataset="test_dataset",
        )

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_plot_missing_od_pairs_file(self, tmp_path):
        """Test fail-fast assertion for missing OD pairs file"""
        missing_file = tmp_path / "missing.json"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(AssertionError, match="not found"):
            plot_analysis_from_files(
                abnormal_od_pairs_file=missing_file,
                real_data_files=[],
                detection_results_files=[],
                samples_dir=tmp_path,
                output_dir=output_dir,
                dataset="test",
            )

    def test_plot_empty_real_data_files(self, tmp_path):
        """Test fail-fast assertion for empty real data files list"""
        od_pairs_file = tmp_path / "od_pairs.json"
        od_pairs_file.write_text('{"od_pairs_by_category": {}}')

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(AssertionError, match="cannot be empty"):
            plot_analysis_from_files(
                abnormal_od_pairs_file=od_pairs_file,
                real_data_files=[],
                detection_results_files=[],
                samples_dir=tmp_path,
                output_dir=output_dir,
                dataset="test",
            )
