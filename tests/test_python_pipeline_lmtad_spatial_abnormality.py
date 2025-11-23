"""Tests for run_lmtad_spatial_abnormality phase programmatic interface."""

import pytest
from pathlib import Path
from unittest.mock import patch

# Import the pipeline class
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from python_pipeline import PipelineConfig, EvaluationPipeline  # noqa: E402


class TestRunLMTADSpatialAbnormalityPhase:
    """Tests for run_lmtad_spatial_abnormality phase programmatic interface."""

    def test_phase_calls_programmatic_interface(self, tmp_path):
        """Test that run_lmtad_spatial_abnormality calls programmatic function."""
        # Create mock eval directory structure
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()  # Required by EvaluationPipeline

        # Create mock LM-TAD source eval directory
        lmtad_eval_dir = tmp_path / "lmtad_eval"
        lmtad_eval_dir.mkdir()
        checkpoint = lmtad_eval_dir.parent / "ckpt_best.pt"
        checkpoint.write_text("mock checkpoint")

        # Create config
        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = True
        config.lmtad_source_eval_dir = None
        config.seed = 42
        config.lmtad_num_trajectories_per_od = 20
        config.lmtad_max_od_pairs = 250
        config.force = False

        # Mock validation before creating pipeline
        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            # Create pipeline instance
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock the helper methods and programmatic function together
        with (
            patch.object(
                pipeline, "_detect_lmtad_source_eval_dir", return_value=lmtad_eval_dir
            ),
            patch.object(pipeline, "_find_lmtad_checkpoint", return_value=checkpoint),
            patch(
                "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
            ) as mock_run,
        ):
            mock_run.return_value = True

            # Run the phase
            pipeline.run_lmtad_spatial_abnormality()

            # Verify function was called
            assert mock_run.called
            call_args = mock_run.call_args

            # Verify parameters
            assert call_args.kwargs["eval_dir"] == eval_dir
            assert call_args.kwargs["dataset"] == "porto_hoser"
            assert call_args.kwargs["lmtad_source_eval_dir"] == lmtad_eval_dir
            assert call_args.kwargs["lmtad_checkpoint"] == checkpoint
            assert call_args.kwargs["skip_extraction"] is False
            assert call_args.kwargs["skip_generation"] is False
            assert call_args.kwargs["skip_evaluation"] is False
            assert call_args.kwargs["skip_aggregation"] is False
            assert call_args.kwargs["skip_visualization"] is False
            assert call_args.kwargs["seed"] == 42
            assert call_args.kwargs["num_traj_per_od"] == 20
            assert call_args.kwargs["max_od_pairs"] == 250
            assert call_args.kwargs["force"] is False

    def test_phase_skips_when_not_configured(self, tmp_path):
        """Test that phase skips when run_lmtad_spatial_detection is False."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = False

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        with patch(
            "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
        ) as mock_run:
            pipeline.run_lmtad_spatial_abnormality()

            # Function should not be called
            assert not mock_run.called

    def test_phase_skips_when_source_eval_dir_missing(self, tmp_path):
        """Test that phase skips when source eval directory cannot be detected."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = True
        config.lmtad_source_eval_dir = None

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock detection to return None and ensure function is not called
        with (
            patch.object(pipeline, "_detect_lmtad_source_eval_dir", return_value=None),
            patch(
                "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
            ) as mock_run,
        ):
            pipeline.run_lmtad_spatial_abnormality()

            # Function should not be called
            assert not mock_run.called

    def test_phase_skips_when_checkpoint_missing(self, tmp_path):
        """Test that phase skips when checkpoint cannot be found."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        lmtad_eval_dir = tmp_path / "lmtad_eval"
        lmtad_eval_dir.mkdir()

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = True
        config.lmtad_source_eval_dir = None

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock detection to return eval dir but checkpoint to return None
        with (
            patch.object(
                pipeline, "_detect_lmtad_source_eval_dir", return_value=lmtad_eval_dir
            ),
            patch.object(pipeline, "_find_lmtad_checkpoint", return_value=None),
            patch(
                "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
            ) as mock_run,
        ):
            pipeline.run_lmtad_spatial_abnormality()

            # Function should not be called
            assert not mock_run.called

    def test_phase_handles_function_failure(self, tmp_path):
        """Test that phase handles function failure correctly."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        lmtad_eval_dir = tmp_path / "lmtad_eval"
        lmtad_eval_dir.mkdir()
        checkpoint = lmtad_eval_dir.parent / "ckpt_best.pt"
        checkpoint.write_text("mock checkpoint")

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = True
        config.lmtad_source_eval_dir = None

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock helper methods
        with patch.object(pipeline, "_detect_lmtad_source_eval_dir") as mock_detect:
            mock_detect.return_value = lmtad_eval_dir
        with patch.object(pipeline, "_find_lmtad_checkpoint") as mock_find:
            mock_find.return_value = checkpoint

        # Mock function to return False (failure)
        with patch(
            "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
        ) as mock_run:
            mock_run.return_value = False

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="LM-TAD spatial pipeline failed"):
                pipeline.run_lmtad_spatial_abnormality()

    def test_phase_handles_exception(self, tmp_path):
        """Test that phase handles exceptions correctly."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        lmtad_eval_dir = tmp_path / "lmtad_eval"
        lmtad_eval_dir.mkdir()
        checkpoint = lmtad_eval_dir.parent / "ckpt_best.pt"
        checkpoint.write_text("mock checkpoint")

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = True
        config.lmtad_source_eval_dir = None

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock helper methods
        with patch.object(pipeline, "_detect_lmtad_source_eval_dir") as mock_detect:
            mock_detect.return_value = lmtad_eval_dir
        with patch.object(pipeline, "_find_lmtad_checkpoint") as mock_find:
            mock_find.return_value = checkpoint

        # Mock function to raise exception
        with patch(
            "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
        ) as mock_run:
            mock_run.side_effect = ValueError("Test error")

            # Should raise the exception
            with pytest.raises(ValueError, match="Test error"):
                pipeline.run_lmtad_spatial_abnormality()

    def test_phase_passes_force_flag(self, tmp_path):
        """Test that phase passes force flag correctly."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        lmtad_eval_dir = tmp_path / "lmtad_eval"
        lmtad_eval_dir.mkdir()
        checkpoint = lmtad_eval_dir.parent / "ckpt_best.pt"
        checkpoint.write_text("mock checkpoint")

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = True
        config.lmtad_source_eval_dir = None
        config.force = True

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock helper methods
        with patch.object(pipeline, "_detect_lmtad_source_eval_dir") as mock_detect:
            mock_detect.return_value = lmtad_eval_dir
        with patch.object(pipeline, "_find_lmtad_checkpoint") as mock_find:
            mock_find.return_value = checkpoint

        # Mock the programmatic function
        with patch(
            "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
        ) as mock_run:
            mock_run.return_value = True
            pipeline.run_lmtad_spatial_abnormality()

            # Verify force flag is passed
            call_args = mock_run.call_args
            assert call_args.kwargs["force"] is True

    def test_phase_passes_custom_parameters(self, tmp_path):
        """Test that phase passes custom trajectory generation parameters."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        lmtad_eval_dir = tmp_path / "lmtad_eval"
        lmtad_eval_dir.mkdir()
        checkpoint = lmtad_eval_dir.parent / "ckpt_best.pt"
        checkpoint.write_text("mock checkpoint")

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_lmtad_spatial_detection = True
        config.lmtad_source_eval_dir = None
        config.seed = 100
        config.lmtad_num_trajectories_per_od = 10
        config.lmtad_max_od_pairs = 50

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock helper methods
        with patch.object(pipeline, "_detect_lmtad_source_eval_dir") as mock_detect:
            mock_detect.return_value = lmtad_eval_dir
        with patch.object(pipeline, "_find_lmtad_checkpoint") as mock_find:
            mock_find.return_value = checkpoint

        # Mock the programmatic function
        with patch(
            "tools.run_lmtad_spatial_pipeline.run_lmtad_spatial_pipeline"
        ) as mock_run:
            mock_run.return_value = True
            pipeline.run_lmtad_spatial_abnormality()

            # Verify custom parameters are passed
            call_args = mock_run.call_args
            assert call_args.kwargs["seed"] == 100
            assert call_args.kwargs["num_traj_per_od"] == 10
            assert call_args.kwargs["max_od_pairs"] == 50
