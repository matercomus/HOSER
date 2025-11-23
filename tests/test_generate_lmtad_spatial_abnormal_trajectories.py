"""Tests for generate_lmtad_spatial_abnormal_trajectories module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
import sys

_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from tools.generate_lmtad_spatial_abnormal_trajectories import (  # noqa: E402
    generate_spatial_abnormal_trajectories,
)


class TestGenerateLMTADSpatialAbnormalTrajectories:
    """Tests for generate_lmtad_spatial_abnormal_trajectories module."""

    def test_working_directory_change(self, tmp_path):
        """Test that working directory is changed to project root's parent before calling generate_trajectories_programmatic."""
        # Setup
        od_pairs_file = tmp_path / "od_pairs.json"
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()

        # Create mock OD pairs file
        od_pairs_data = {
            "od_pairs_by_type": {
                "route_switch": [[1, 2], [3, 4]],
                "detour": [[5, 6]],
            }
        }
        with open(od_pairs_file, "w") as f:
            json.dump(od_pairs_data, f)

        # Mock generate_trajectories_programmatic to track working directory
        original_cwd = os.getcwd()
        captured_cwd = None

        def mock_generate(*args, **kwargs):
            nonlocal captured_cwd
            captured_cwd = os.getcwd()
            return {"output_file": str(tmp_path / "output.csv"), "num_generated": 10}

        with patch(
            "tools.generate_lmtad_spatial_abnormal_trajectories.generate_trajectories_programmatic",
            side_effect=mock_generate,
        ):
            with patch(
                "tools.generate_lmtad_spatial_abnormal_trajectories.find_models"
            ) as mock_find:
                mock_find.return_value = [("test_model", tmp_path / "model.pth")]

                # Call the function
                generate_spatial_abnormal_trajectories(
                    od_pairs_file=od_pairs_file,
                    eval_dir=eval_dir,
                    dataset="test_dataset",
                    models=[],
                    seed=42,
                    num_traj_per_od=1,
                )

        # Verify working directory was changed to tools directory
        # From tools/, ../data/ resolves to project_root/data/
        project_root = Path(__file__).parent.parent
        expected_cwd = project_root / "tools"
        assert captured_cwd == str(expected_cwd), (
            f"Expected CWD to be {expected_cwd}, got {captured_cwd}"
        )

        # Verify original working directory was restored
        assert os.getcwd() == original_cwd, (
            "Original working directory should be restored"
        )

    def test_data_path_resolution(self, tmp_path):
        """Test that data paths resolve correctly when working directory is changed."""
        # Setup
        od_pairs_file = tmp_path / "od_pairs.json"
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()

        # Create mock OD pairs file
        od_pairs_data = {
            "od_pairs_by_type": {
                "route_switch": [[1, 2]],
            }
        }
        with open(od_pairs_file, "w") as f:
            json.dump(od_pairs_data, f)

        # Track the working directory when generate_trajectories_programmatic is called
        captured_cwd = None
        project_root = Path(__file__).parent.parent

        def mock_generate(*args, **kwargs):
            nonlocal captured_cwd
            captured_cwd = os.getcwd()
            # Verify that ../data/ would resolve correctly from tools/ directory
            data_path = Path("../data/test_dataset/roadmap.geo")
            # From tools/, ../data/ should resolve to project_root/data/
            resolved = data_path.resolve()
            expected = project_root / "data" / "test_dataset" / "roadmap.geo"
            assert resolved == expected, (
                f"Data path should resolve to {expected}, got {resolved}"
            )
            return {"output_file": str(tmp_path / "output.csv"), "num_generated": 10}

        with patch(
            "tools.generate_lmtad_spatial_abnormal_trajectories.generate_trajectories_programmatic",
            side_effect=mock_generate,
        ):
            with patch(
                "tools.generate_lmtad_spatial_abnormal_trajectories.find_models"
            ) as mock_find:
                mock_find.return_value = [("test_model", tmp_path / "model.pth")]

                generate_spatial_abnormal_trajectories(
                    od_pairs_file=od_pairs_file,
                    eval_dir=eval_dir,
                    dataset="test_dataset",
                    models=[],
                    seed=42,
                    num_traj_per_od=1,
                )

        # Verify working directory was tools directory
        assert captured_cwd == str(project_root / "tools")

    def test_paths_resolved_before_directory_change(self, tmp_path):
        """Test that output_file and model_path are resolved to absolute paths before changing directory."""
        # Setup
        od_pairs_file = tmp_path / "od_pairs.json"
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()
        model_file = eval_dir / "models" / "test_model.pth"
        model_file.touch()

        # Create mock OD pairs file
        od_pairs_data = {
            "od_pairs_by_type": {
                "route_switch": [[1, 2]],
            }
        }
        with open(od_pairs_file, "w") as f:
            json.dump(od_pairs_data, f)

        # Track the arguments passed to generate_trajectories_programmatic
        captured_args = {}

        def mock_generate(*args, **kwargs):
            captured_args.update(kwargs)
            return {"output_file": str(tmp_path / "output.csv"), "num_generated": 10}

        with patch(
            "tools.generate_lmtad_spatial_abnormal_trajectories.generate_trajectories_programmatic",
            side_effect=mock_generate,
        ):
            with patch(
                "tools.generate_lmtad_spatial_abnormal_trajectories.detect_model_files"
            ) as mock_detect:
                from tools.model_detection import ModelFile

                mock_detect.return_value = [
                    ModelFile(
                        path=model_file,
                        model_name="test_model",
                        seed=None,
                        base_model=None,
                        filename="test_model.pth",
                    )
                ]

                generate_spatial_abnormal_trajectories(
                    od_pairs_file=od_pairs_file,
                    eval_dir=eval_dir,
                    dataset="test_dataset",
                    models=["test_model"],
                    seed=42,
                    num_traj_per_od=1,
                )

        # Verify paths are absolute
        assert Path(captured_args["model_path"]).is_absolute(), (
            "model_path should be absolute"
        )
        assert Path(captured_args["output_file"]).is_absolute(), (
            "output_file should be absolute"
        )

        # Verify paths point to correct files
        assert Path(captured_args["model_path"]) == model_file.resolve()
        assert (
            Path(captured_args["output_file"]).name == "test_model_spatial_abnormal.csv"
        )

    def test_working_directory_restored_on_error(self, tmp_path):
        """Test that working directory is restored even if generate_trajectories_programmatic raises an error."""
        # Setup
        od_pairs_file = tmp_path / "od_pairs.json"
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()

        # Create mock OD pairs file
        od_pairs_data = {
            "od_pairs_by_type": {
                "route_switch": [[1, 2]],
            }
        }
        with open(od_pairs_file, "w") as f:
            json.dump(od_pairs_data, f)

        original_cwd = os.getcwd()

        def mock_generate(*args, **kwargs):
            raise ValueError("Test error")

        with patch(
            "tools.generate_lmtad_spatial_abnormal_trajectories.generate_trajectories_programmatic",
            side_effect=mock_generate,
        ):
            with patch(
                "tools.generate_lmtad_spatial_abnormal_trajectories.find_models"
            ) as mock_find:
                mock_find.return_value = [("test_model", tmp_path / "model.pth")]

                # Error is caught and logged, not re-raised
                generate_spatial_abnormal_trajectories(
                    od_pairs_file=od_pairs_file,
                    eval_dir=eval_dir,
                    dataset="test_dataset",
                    models=[],
                    seed=42,
                    num_traj_per_od=1,
                )

        # Verify working directory was restored even though error was caught
        assert os.getcwd() == original_cwd, (
            "Working directory should be restored even on error"
        )
