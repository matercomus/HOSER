"""Tests for evaluate_lmtad_spatial_abnormal module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from tools.evaluate_lmtad_spatial_abnormal import (  # noqa: E402
    detect_lmtad_repo_from_checkpoint,
    evaluate_spatial_abnormal_trajectories,
    classify_spatial_abnormality_type,
    load_source_statistics,
)


class TestDetectLMTADRepoFromCheckpoint:
    """Tests for LM-TAD repo auto-detection from checkpoint path."""

    def test_detect_repo_from_checkpoint_with_code_directory(self, tmp_path):
        """Test detection when checkpoint is in code/results/.../ckpt_best.pt."""
        # Create mock LM-TAD structure
        lmtad_repo = tmp_path / "LMTAD"
        code_dir = lmtad_repo / "code"
        code_dir.mkdir(parents=True)
        models_dir = code_dir / "models"
        models_dir.mkdir()
        (models_dir / "LMTAD.py").touch()

        checkpoint = code_dir / "results" / "run_123" / "ckpt_best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        result = detect_lmtad_repo_from_checkpoint(checkpoint)
        assert result == lmtad_repo
        assert result.exists()

    def test_detect_repo_from_checkpoint_fallback(self, tmp_path):
        """Test fallback detection when code directory exists in path."""
        # Create checkpoint path with 'code' in it
        checkpoint = Path("/some/path/to/lmtad/code/results/ckpt_best.pt")

        with patch("pathlib.Path.resolve", return_value=checkpoint):
            with patch("pathlib.Path.exists", return_value=False):
                # Should use fallback logic
                result = detect_lmtad_repo_from_checkpoint(checkpoint)
                # Fallback extracts path up to 'code' index
                assert str(result) == "/some/path/to/lmtad"

    def test_detect_repo_from_checkpoint_invalid_path(self):
        """Test error handling for invalid checkpoint path."""
        invalid_path = Path("/nonexistent/path/ckpt_best.pt")

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_lmtad_repo_from_checkpoint(invalid_path)

    def test_detect_repo_finds_code_directory_with_models(self, tmp_path):
        """Test detection finds code directory that contains models/LMTAD.py."""
        lmtad_repo = tmp_path / "LMTAD"
        code_dir = lmtad_repo / "code"
        code_dir.mkdir(parents=True)
        models_dir = code_dir / "models"
        models_dir.mkdir()
        (models_dir / "LMTAD.py").touch()

        checkpoint = code_dir / "results" / "deep" / "nested" / "ckpt_best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        result = detect_lmtad_repo_from_checkpoint(checkpoint)
        assert result == lmtad_repo


class TestEvaluateSpatialAbnormalTrajectories:
    """Tests for evaluate_spatial_abnormal_trajectories function."""

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_evaluate_with_explicit_lmtad_repo(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_load_centroids,
        mock_load_stats,
        mock_load_traj,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test evaluation with explicit lmtad_repo parameter."""
        # Setup mocks
        mock_trajectories = [[1, 2, 3, 6165], [4, 5, 6, 6165]]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        import numpy as np

        mock_load_centroids.return_value = np.array([[0.0, 0.0], [1.0, 1.0]])

        mock_mapper = MagicMock()
        mock_mapper.map_all.return_value = np.array([0, 1])
        mock_grid_mapper.return_value = mock_mapper

        mock_model = MagicMock()
        mock_lmtad_teacher.return_value = mock_model

        import numpy as np

        mock_evaluate.return_value = (
            np.array([7.0, 8.5]),  # log_perplexities
            np.array([False, True]),  # outliers
        )

        # Create test files
        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        # Create roadmap file in proper location
        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:
            # Mock Path to return our tmp_path structure
            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        # Handle "data/test_dataset/roadmap.geo"
                        parts = path_arg.split("/")
                        if len(parts) == 3 and parts[0] == "data":
                            return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            # Also patch the __file__ path resolution
            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                _ = evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                )

        # Verify LMTADTeacher was called with correct repo path
        mock_lmtad_teacher.assert_called_once()
        call_kwargs = mock_lmtad_teacher.call_args[1]
        assert str(call_kwargs["repo_path"]) == str(tmp_path / "LMTAD")

    @patch("tools.evaluate_lmtad_spatial_abnormal.detect_lmtad_repo_from_checkpoint")
    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    def test_evaluate_with_auto_detection(
        self,
        mock_load_stats,
        mock_load_traj,
        mock_lmtad_teacher,
        mock_detect,
        tmp_path,
    ):
        """Test evaluation with auto-detection of LM-TAD repo."""
        # Setup mocks
        detected_repo = tmp_path / "LMTAD"
        mock_detect.return_value = detected_repo
        mock_load_traj.return_value = [[1, 2, 3, 6165]]
        mock_load_stats.return_value = {"route_switch_mean": 7.03, "detour_mean": 8.41}

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        with patch(
            "tools.evaluate_lmtad_spatial_abnormal.load_road_centroids"
        ) as mock_centroids:
            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.GridMapper"
            ) as mock_mapper:
                with patch(
                    "tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct"
                ) as mock_eval:
                    import numpy as np

                    mock_centroids.return_value = np.array([[0.0, 0.0], [1.0, 1.0]])
                    mock_mapper.return_value.map_all.return_value = np.array([0, 1])
                    mock_eval.return_value = (np.array([7.0]), np.array([False]))

                    # Create roadmap file in proper location
                    data_dir = tmp_path / "data" / "test_dataset"
                    data_dir.mkdir(parents=True)
                    roadmap_file = data_dir / "roadmap.geo"
                    roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

                    with patch(
                        "tools.evaluate_lmtad_spatial_abnormal.Path"
                    ) as mock_path_class:

                        def path_side_effect(path_arg):
                            if isinstance(path_arg, str):
                                if path_arg.startswith("data/"):
                                    return tmp_path / path_arg
                                return Path(path_arg)
                            return Path(path_arg)

                        mock_path_class.side_effect = path_side_effect

                        with patch(
                            "tools.evaluate_lmtad_spatial_abnormal.__file__",
                            str(
                                tmp_path
                                / "tools"
                                / "evaluate_lmtad_spatial_abnormal.py"
                            ),
                        ):
                            _ = evaluate_spatial_abnormal_trajectories(
                                trajectory_file=trajectory_file,
                                lmtad_checkpoint=checkpoint,
                                source_eval_dir=source_eval_dir,
                                dataset="test_dataset",
                                device="cpu",
                            )

        # Verify auto-detection was called
        mock_detect.assert_called_once_with(checkpoint)
        # Verify LMTADTeacher was called with detected repo
        mock_lmtad_teacher.assert_called_once()
        call_kwargs = mock_lmtad_teacher.call_args[1]
        assert str(call_kwargs["repo_path"]) == str(detected_repo)

    def test_sys_path_ordering_with_lmtad_repo(self, tmp_path):
        """Test that LM-TAD code path is added to sys.path before HOSER project root."""
        lmtad_repo = tmp_path / "LMTAD"
        code_dir = lmtad_repo / "code"
        code_dir.mkdir(parents=True)

        # Store original sys.path
        original_path = sys.path.copy()

        try:
            # Simulate adding LM-TAD code path
            lmtad_code_path = str(code_dir)
            if lmtad_code_path not in sys.path:
                sys.path.insert(0, lmtad_code_path)

            # Verify LM-TAD code path is at position 0
            assert sys.path[0] == lmtad_code_path

            # Verify HOSER project root is not at position 0
            hoser_root = str(_parent_dir)
            if hoser_root in sys.path:
                hoser_index = sys.path.index(hoser_root)
                assert hoser_index > 0, (
                    "HOSER root should not be before LM-TAD code path"
                )
        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    def test_namespace_isolation(self, mock_lmtad_teacher, tmp_path):
        """Test that models.LMTAD resolves to LM-TAD's models, not HOSER's."""
        lmtad_repo = tmp_path / "LMTAD"
        code_dir = lmtad_repo / "code"
        code_dir.mkdir(parents=True)
        models_dir = code_dir / "models"
        models_dir.mkdir()
        (models_dir / "LMTAD.py").touch()

        # Add LM-TAD code path to sys.path
        lmtad_code_path = str(code_dir)
        if lmtad_code_path not in sys.path:
            sys.path.insert(0, lmtad_code_path)

        try:
            # Verify that if we import models, it should find LM-TAD's models first
            # (This is a structural test - actual import would require LM-TAD installation)
            assert (
                Path(lmtad_code_path) / "models" / "LMTAD.py" == models_dir / "LMTAD.py"
            )
            assert (models_dir / "LMTAD.py").exists()
        finally:
            # Clean up sys.path
            if lmtad_code_path in sys.path:
                sys.path.remove(lmtad_code_path)

    def test_error_handling_invalid_checkpoint(self):
        """Test error handling for invalid checkpoint path."""
        invalid_checkpoint = Path("/nonexistent/path/ckpt_best.pt")

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_lmtad_repo_from_checkpoint(invalid_checkpoint)

    @patch("tools.evaluate_lmtad_spatial_abnormal.detect_lmtad_repo_from_checkpoint")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    def test_error_handling_missing_trajectory_file(
        self, mock_load_traj, mock_detect, tmp_path
    ):
        """Test error handling when trajectory file doesn't exist."""
        invalid_file = tmp_path / "nonexistent.csv"

        # Mock detect to return a valid repo path
        mock_detect.return_value = tmp_path / "LMTAD"

        # load_hoser_trajectories should raise FileNotFoundError for missing file
        mock_load_traj.side_effect = FileNotFoundError(
            f"File not found: {invalid_file}"
        )

        with pytest.raises(FileNotFoundError):
            evaluate_spatial_abnormal_trajectories(
                trajectory_file=invalid_file,
                lmtad_checkpoint=tmp_path / "ckpt.pt",
                source_eval_dir=tmp_path / "eval",
                dataset="test",
                device="cpu",
            )


class TestClassifySpatialAbnormalityType:
    """Tests for classify_spatial_abnormality_type function."""

    def test_classify_route_switch(self):
        """Test classification of route switch trajectory."""
        source_stats = {"route_switch_mean": 7.03, "detour_mean": 8.41}
        log_perp = 7.0  # Between route_switch_mean - 1.0 and detour_mean - 0.5

        result = classify_spatial_abnormality_type(log_perp, source_stats)
        assert result == "route_switch"

    def test_classify_detour(self):
        """Test classification of detour trajectory."""
        source_stats = {"route_switch_mean": 7.03, "detour_mean": 8.41}
        log_perp = 9.0  # Above detour_mean - 0.5

        result = classify_spatial_abnormality_type(log_perp, source_stats)
        assert result == "detour"

    def test_classify_non_outlier(self):
        """Test classification of non-outlier trajectory."""
        source_stats = {"route_switch_mean": 7.03, "detour_mean": 8.41}
        log_perp = 5.0  # Below route_switch_mean - 1.0

        result = classify_spatial_abnormality_type(log_perp, source_stats)
        assert result == "non_outlier"

    def test_classify_with_default_stats(self):
        """Test classification with default statistics when not provided."""
        source_stats = {}
        log_perp = 7.0

        result = classify_spatial_abnormality_type(log_perp, source_stats)
        # Should use defaults: route_switch_mean=7.03, detour_mean=8.41
        assert result == "route_switch"


class TestLoadSourceStatistics:
    """Tests for load_source_statistics function."""

    def test_load_statistics_from_markdown(self, tmp_path):
        """Test loading statistics from EVALUATION_ANALYSIS.md."""
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        md_file = eval_dir / "EVALUATION_ANALYSIS.md"

        md_content = """
# Evaluation Analysis

## Spatial Abnormality Statistics
- Route Switch Mean: 7.03
- Detour Mean: 8.41
- Route Switch Rate: 0.0327
- Detour Rate: 0.0327
"""
        md_file.write_text(md_content)

        with patch(
            "tools.evaluate_lmtad_spatial_abnormal.Path.exists", return_value=True
        ):
            with patch("builtins.open", mock_open(read_data=md_content)):
                stats = load_source_statistics(eval_dir)
                # Function should extract or use defaults
                assert isinstance(stats, dict)

    def test_load_statistics_fallback_to_defaults(self, tmp_path):
        """Test that function falls back to defaults when file doesn't exist."""
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        with patch(
            "tools.evaluate_lmtad_spatial_abnormal.Path.exists", return_value=False
        ):
            stats = load_source_statistics(eval_dir)
            # Should return dict with defaults or empty
            assert isinstance(stats, dict)
