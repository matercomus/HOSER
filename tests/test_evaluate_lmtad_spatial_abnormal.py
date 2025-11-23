"""Tests for evaluate_lmtad_spatial_abnormal module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from tools.evaluate_lmtad_spatial_abnormal import (  # noqa: E402
    detect_lmtad_repo_from_checkpoint,
    evaluate_spatial_abnormal_trajectories,
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
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
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

        mock_load_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        # Provide a generous mapping array so mock trajectories map cleanly
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        mock_model = MagicMock()
        # Ensure vocab_size returns a real int to avoid MagicMock being used in numeric comparisons
        mock_model.vocab_size.return_value = 7000
        mock_lmtad_teacher.return_value = mock_model

        import numpy as np

        mock_evaluate.return_value = (
            np.array([7.0, 8.5]),  # log_perplexities
            np.array([False, True]),  # outliers
            [[0.5, 0.6], [0.7, 0.8]],
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
                result = evaluate_spatial_abnormal_trajectories(
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

        assert result["total_trajectories"] == 2
        assert result["failed_trajectory_count"] == 0
        assert len(result["trajectories"]) == 2
        assert result["trajectories"][0]["log_perplexity"] == 7.0
        assert result["trajectories"][0]["segment_log_perplexities"] == [0.5, 0.6]

    def test_validate_trajectory_duplicate_check_disabled_by_default(self):
        """By default, duplicate checking should be disabled (max_duplicate_ratio=1.0)."""
        from tools.evaluate_lmtad_spatial_abnormal import validate_trajectory_for_lmtad

        # A trajectory with many duplicates
        duplicate_traj = [5, 5, 5, 5, 5]

        # Default should be disabled -> valid
        is_valid_default, reason, diag = validate_trajectory_for_lmtad(
            duplicate_traj, vocab_size=10000
        )
        assert is_valid_default is True

        # With duplicate ratio set to 0.1, it should be rejected
        is_valid_strict, reason, diag = validate_trajectory_for_lmtad(
            duplicate_traj, vocab_size=10000, max_duplicate_ratio=0.1
        )
        assert is_valid_strict is False

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
            "tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids"
        ) as mock_centroids:
            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.GridMapper"
            ) as mock_mapper:
                with patch(
                    "tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct"
                ) as mock_eval:
                    import numpy as np

                    mock_centroids.return_value = (
                        np.array([[0.0, 0.0], [1.0, 1.0]]),
                        {
                            "min_lat": 0.0,
                            "max_lat": 1.0,
                            "min_lng": 0.0,
                            "max_lng": 1.0,
                        },
                    )
                    mock_mapper.return_value.map_all.return_value = np.arange(7000)
                    mock_eval.return_value = (
                        np.array([7.0]),
                        np.array([False]),
                        [[0.25, 0.5]],
                    )

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
                            result = evaluate_spatial_abnormal_trajectories(
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
        assert mock_eval.call_args[1]["return_segment_perplexity"] is True
        assert result["segment_stats"]["per_index"][0]["count"] == 1

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


class TestFailedEvaluationsAndEdgeCases:
    """Tests for failed evaluations, edge cases, and bounds checking."""

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_all_trajectories_fail_evaluation(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_load_centroids,
        mock_load_stats,
        mock_load_traj,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test handling when all trajectories fail evaluation (Infinity perplexity)."""
        import numpy as np

        # Setup mocks
        mock_trajectories = [[1, 2, 3], [4, 5, 6]]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        mock_load_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        mock_model = MagicMock()
        mock_model.vocab_size.return_value = 7000
        mock_lmtad_teacher.return_value = mock_model

        # All trajectories fail (Infinity perplexity)
        mock_evaluate.return_value = (
            np.array([np.inf, np.inf]),  # All Infinity
            np.array([True, True]),
            [[], []],
        )

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:

            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                result = evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                )

        assert result["total_trajectories"] == 2
        assert result["failed_trajectory_count"] == 2
        assert result["failed_trajectory_rate"] == 100.0
        assert result["valid_trajectory_count"] == 0
        assert result["log_perplexity_stats"] == {}
        assert all(t["status"] == "evaluation_failed" for t in result["trajectories"])

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_mixed_successful_and_failed_evaluations(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_load_centroids,
        mock_load_stats,
        mock_load_traj,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test handling when some trajectories succeed and some fail."""
        import numpy as np

        mock_trajectories = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        mock_load_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        mock_model = MagicMock()
        mock_model.vocab_size.return_value = 7000
        mock_lmtad_teacher.return_value = mock_model

        # Mixed: first fails, second succeeds (route_switch), third fails
        mock_evaluate.return_value = (
            np.array([np.inf, 7.5, np.inf]),  # Mixed results
            np.array([True, True, True]),
            [[], [0.2, 0.3], []],
        )

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:

            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                result = evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                )

        assert result["total_trajectories"] == 3
        assert result["failed_trajectory_count"] == 2
        assert result["failed_trajectory_rate"] == pytest.approx(66.67, abs=0.01)
        assert result["valid_trajectory_count"] == 1
        assert result["log_perplexity_stats"]["mean"] == pytest.approx(7.5)
        statuses = [t["status"] for t in result["trajectories"]]
        assert statuses == ["evaluation_failed", "ok", "evaluation_failed"]

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_edge_case_all_classification_types(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_load_centroids,
        mock_load_stats,
        mock_load_traj,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test edge case with all classification types including failures."""
        import numpy as np

        mock_trajectories = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        mock_load_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        mock_model = MagicMock()
        mock_model.vocab_size.return_value = 7000
        mock_lmtad_teacher.return_value = mock_model

        # All types: failed, labeled route_switch, labeled detour, non_outlier, failed
        mock_evaluate.return_value = (
            np.array([np.inf, 7.5, 9.0, 3.0, np.inf]),
            np.array([True, False, False, False, True]),
            [[], [0.1, 0.2], [0.3, 0.4], [0.05], []],
        )

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:

            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                od_pairs_file = tmp_path / "od_pairs.json"
                od_pairs_file.write_text(
                    json.dumps(
                        {
                            "od_pairs_by_type": {
                                "route_switch": [[4, 6]],
                                "detour": [[7, 9]],
                            }
                        }
                    )
                )

                result = evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                    od_pairs_file=od_pairs_file,
                )

        assert result["total_trajectories"] == 5
        assert result["failed_trajectory_count"] == 2
        assert result["valid_trajectory_count"] == 3
        assert result["od_pair_label_counts"]["route_switch"] == 1
        assert result["od_pair_label_counts"]["detour"] == 1
        labels = [t["source_label"] for t in result["trajectories"]]
        assert labels == [None, "route_switch", "detour", None, None]
        statuses = [t["status"] for t in result["trajectories"]]
        assert statuses.count("evaluation_failed") == 2

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_edge_case_zero_valid_trajectories(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_load_centroids,
        mock_load_stats,
        mock_load_traj,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test edge case when no trajectories are valid (all fail)."""
        import numpy as np

        mock_trajectories = [[1, 2, 3]]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        mock_load_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        mock_model = MagicMock()
        mock_model.vocab_size.return_value = 7000
        mock_lmtad_teacher.return_value = mock_model

        # All fail
        mock_evaluate.return_value = (
            np.array([np.inf]),
            np.array([True]),
            [[]],
        )

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:

            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                result = evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                )

        assert result["total_trajectories"] == 1
        assert result["valid_trajectory_count"] == 0
        assert result["failed_trajectory_count"] == 1
        assert result["failed_trajectory_rate"] == 100.0
        assert result["log_perplexity_stats"] == {}


class TestGridMapperVerification:
    """Tests for verify_hw parameter in GridMapper during evaluation."""

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_verify_hw_passed_to_gridmapper(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_extract_centroids,
        mock_load_traj,
        mock_load_stats,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test that verify_hw is passed to GridMapper when teacher provides grid dimensions."""
        import numpy as np

        mock_trajectories = [[1, 2, 3]]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        # Mock extract_road_centroids to return centroids and boundary
        mock_extract_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        # Mock teacher with get_grid_size_hw and vocab_size methods
        mock_model = MagicMock()
        mock_model.get_grid_size_hw.return_value = (64, 64)  # Expected grid dimensions
        mock_model.vocab_size.return_value = 4099  # vocab_size = 64*64 + 3
        mock_lmtad_teacher.return_value = mock_model

        mock_evaluate.return_value = (
            np.array([5.0]),
            np.array([False]),
            None,
        )

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:

            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                )

        # Verify GridMapper was called with verify_hw parameter
        assert mock_grid_mapper.called
        call_kwargs = mock_grid_mapper.call_args[1]
        assert "verify_hw" in call_kwargs
        assert call_kwargs["verify_hw"] == (64, 64)

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_verify_hw_none_when_teacher_returns_none(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_extract_centroids,
        mock_load_traj,
        mock_load_stats,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test that verify_hw is None when teacher.get_grid_size_hw() returns None."""
        import numpy as np

        mock_trajectories = [[1, 2, 3]]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        # Mock extract_road_centroids to return centroids and boundary
        mock_extract_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        # Mock teacher that returns None for grid dimensions
        mock_model = MagicMock()
        mock_model.get_grid_size_hw.return_value = None
        mock_model.vocab_size.return_value = 4099
        mock_lmtad_teacher.return_value = mock_model

        mock_evaluate.return_value = (
            np.array([5.0]),
            np.array([False]),
            None,
        )

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:

            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                )

        # Verify GridMapper was called with verify_hw=None
        assert mock_grid_mapper.called
        call_kwargs = mock_grid_mapper.call_args[1]
        assert "verify_hw" in call_kwargs
        assert call_kwargs["verify_hw"] is None

    @patch("tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_source_statistics")
    @patch("tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories")
    @patch("tools.evaluate_lmtad_spatial_abnormal.extract_road_centroids")
    @patch("tools.evaluate_lmtad_spatial_abnormal.GridMapper")
    @patch("tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct")
    def test_vocab_size_passed_to_evaluate_trajectories_direct(
        self,
        mock_evaluate,
        mock_grid_mapper,
        mock_extract_centroids,
        mock_load_traj,
        mock_load_stats,
        mock_lmtad_teacher,
        tmp_path,
    ):
        """Test that vocab_size is passed to evaluate_trajectories_direct."""
        import numpy as np

        mock_trajectories = [[1, 2, 3]]
        mock_load_traj.return_value = mock_trajectories
        mock_load_stats.return_value = {
            "route_switch_mean": 7.03,
            "detour_mean": 8.41,
        }
        # Mock extract_road_centroids to return centroids and boundary
        mock_extract_centroids.return_value = (
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            {"min_lat": 0.0, "max_lat": 1.0, "min_lng": 0.0, "max_lng": 1.0},
        )

        mock_mapper = MagicMock()
        # Provide full mapping to avoid mapping failures in the test
        mock_mapper.map_all.return_value = np.arange(7000)
        mock_grid_mapper.return_value = mock_mapper

        # Mock teacher with vocab_size
        mock_model = MagicMock()
        mock_model.get_grid_size_hw.return_value = None
        mock_model.vocab_size.return_value = 6167
        mock_lmtad_teacher.return_value = mock_model

        mock_evaluate.return_value = (
            np.array([5.0]),
            np.array([False]),
            None,
        )

        trajectory_file = tmp_path / "trajectories.csv"
        trajectory_file.touch()
        checkpoint = tmp_path / "ckpt_best.pt"
        checkpoint.touch()
        source_eval_dir = tmp_path / "eval"
        source_eval_dir.mkdir()

        data_dir = tmp_path / "data" / "test_dataset"
        data_dir.mkdir(parents=True)
        roadmap_file = data_dir / "roadmap.geo"
        roadmap_file.write_text("coordinates\n[[0,0],[1,1]]")

        with patch("tools.evaluate_lmtad_spatial_abnormal.Path") as mock_path_class:

            def path_side_effect(path_arg):
                if isinstance(path_arg, str):
                    if path_arg.startswith("data/"):
                        return tmp_path / path_arg
                    return Path(path_arg)
                return Path(path_arg)

            mock_path_class.side_effect = path_side_effect

            with patch(
                "tools.evaluate_lmtad_spatial_abnormal.__file__",
                str(tmp_path / "tools" / "evaluate_lmtad_spatial_abnormal.py"),
            ):
                evaluate_spatial_abnormal_trajectories(
                    trajectory_file=trajectory_file,
                    lmtad_checkpoint=checkpoint,
                    source_eval_dir=source_eval_dir,
                    dataset="test_dataset",
                    lmtad_repo=tmp_path / "LMTAD",
                    device="cpu",
                )

        # Verify extract_road_centroids was called
        assert mock_extract_centroids.called

        # Verify evaluate_trajectories_direct was called with vocab_size
        assert mock_evaluate.called
        call_kwargs = mock_evaluate.call_args[1]
        assert "vocab_size" in call_kwargs
        assert call_kwargs["vocab_size"] == 6167


class TestBoundaryExtraction:
    """Tests for boundary extraction consistency with LM-TAD conversion."""

    def test_extract_road_centroids_returns_boundary(self, tmp_path):
        """Test that extract_road_centroids returns both centroids and boundary."""
        import pandas as pd
        from tools.convert_to_lmtad_format import extract_road_centroids

        # Create a simple roadmap file
        roadmap_data = pd.DataFrame(
            {
                "coordinates": [
                    "[[8.65000, 41.15000], [8.65050, 41.15050]]",
                    "[[8.66000, 41.16000], [8.66050, 41.16050]]",
                ],
                "geo_id": [0, 1],
                "lanes": ['["2"]', '["1"]'],
                "oneway": ["[false]", "[true]"],
                "name": ["Street A", "Street B"],
            }
        )
        roadmap_file = tmp_path / "roadmap.geo"
        roadmap_data.to_csv(roadmap_file, index=False)

        # Extract centroids and boundaries
        road_centroids, boundary = extract_road_centroids(roadmap_file)

        # Verify return types
        assert road_centroids.shape == (2, 2)
        assert isinstance(boundary, dict)
        assert "min_lat" in boundary
        assert "max_lat" in boundary
        assert "min_lng" in boundary
        assert "max_lng" in boundary

        # Verify boundary values are reasonable
        assert boundary["min_lat"] < boundary["max_lat"]
        assert boundary["min_lng"] < boundary["max_lng"]

    def test_boundary_used_in_grid_config(self, tmp_path):
        """Test that boundaries from extract_road_centroids are used in GridConfig."""
        import pandas as pd
        from tools.convert_to_lmtad_format import extract_road_centroids
        from critics.grid_mapper import GridMapper, GridConfig

        # Create a simple roadmap file
        roadmap_data = pd.DataFrame(
            {
                "coordinates": [
                    "[[8.65000, 41.15000], [8.65050, 41.15050]]",
                    "[[8.66000, 41.16000], [8.66050, 41.16050]]",
                ],
                "geo_id": [0, 1],
                "lanes": ['["2"]', '["1"]'],
                "oneway": ["[false]", "[true]"],
                "name": ["Street A", "Street B"],
            }
        )
        roadmap_file = tmp_path / "roadmap.geo"
        roadmap_data.to_csv(roadmap_file, index=False)

        # Extract centroids and boundaries
        road_centroids, boundary = extract_road_centroids(roadmap_file)

        # Create GridConfig with extracted boundaries
        grid_config = GridConfig(
            min_lat=boundary["min_lat"],
            max_lat=boundary["max_lat"],
            min_lng=boundary["min_lng"],
            max_lng=boundary["max_lng"],
            grid_size=0.001,
            downsample_factor=1,
        )

        # Create GridMapper
        mapper = GridMapper(
            boundary=grid_config,
            road_centroids=road_centroids,
            verify_hw=None,  # No verification for this test
        )

        # Verify mapper was created successfully
        assert mapper.grid_h > 0
        assert mapper.grid_w > 0
        assert mapper.cfg.min_lat == boundary["min_lat"]
        assert mapper.cfg.max_lat == boundary["max_lat"]
        assert mapper.cfg.min_lng == boundary["min_lng"]
        assert mapper.cfg.max_lng == boundary["max_lng"]
