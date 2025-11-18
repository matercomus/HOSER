"""Tests for run_wang_abnormality phase programmatic interface."""

import pytest
from pathlib import Path
from unittest.mock import patch

# Import the pipeline class
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from python_pipeline import PipelineConfig, EvaluationPipeline  # noqa: E402


class TestRunWangAbnormalityPhase:
    """Tests for run_wang_abnormality phase programmatic interface."""

    def test_phase_calls_programmatic_interface(self, tmp_path):
        """Test that run_wang_abnormality calls programmatic function."""
        # Create mock eval directory structure
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()  # Required by EvaluationPipeline
        config_dir = eval_dir / "config"
        config_dir.mkdir(parents=True)

        # Create config file
        config_file = config_dir / "abnormal_detection_statistical.yaml"
        config_file.write_text("test: config")

        # Create config
        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_wang_detection = True
        config.wang_config = None

        # Mock validation before creating pipeline
        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            # Create pipeline instance
            pipeline = EvaluationPipeline(config, eval_dir)

            # Mock the programmatic function
        with patch(
            "tools.run_wang_detection_pipeline.run_wang_detection_pipeline"
        ) as mock_run:
            mock_run.return_value = True

            # Run the phase
            pipeline.run_wang_abnormality()

            # Verify function was called
            assert mock_run.called
            call_args = mock_run.call_args

            # Verify parameters
            assert call_args.kwargs["eval_dir"] == eval_dir
            assert call_args.kwargs["dataset"] == "porto_hoser"
            assert call_args.kwargs["skip_real"] is False
            assert call_args.kwargs["skip_generated"] is False
            assert call_args.kwargs["skip_aggregation"] is False
            assert call_args.kwargs["skip_visualization"] is False

    def test_phase_skips_when_not_configured(self, tmp_path):
        """Test that phase skips when run_wang_detection is False."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()  # Required by EvaluationPipeline

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_wang_detection = False

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        with patch(
            "tools.run_wang_detection_pipeline.run_wang_detection_pipeline"
        ) as mock_run:
            pipeline.run_wang_abnormality()

            # Function should not be called
            assert not mock_run.called

    def test_phase_skips_when_config_missing(self, tmp_path):
        """Test that phase skips when config file doesn't exist."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        (eval_dir / "models").mkdir()  # Required by EvaluationPipeline

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_wang_detection = True
        config.wang_config = None

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        with patch(
            "tools.run_wang_detection_pipeline.run_wang_detection_pipeline"
        ) as mock_run:
            pipeline.run_wang_abnormality()

            # Function should not be called when config is missing
            assert not mock_run.called

    def test_phase_handles_function_failure(self, tmp_path):
        """Test that phase handles function failure correctly."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        config_dir = eval_dir / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "abnormal_detection_statistical.yaml"
        config_file.write_text("test: config")

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_wang_detection = True
        config.wang_config = None

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock function to return False (failure)
        with patch(
            "tools.run_wang_detection_pipeline.run_wang_detection_pipeline"
        ) as mock_run:
            mock_run.return_value = False

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Wang detection pipeline failed"):
                pipeline.run_wang_abnormality()

    def test_phase_handles_exception(self, tmp_path):
        """Test that phase handles exceptions correctly."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        config_dir = eval_dir / "config"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "abnormal_detection_statistical.yaml"
        config_file.write_text("test: config")

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_wang_detection = True
        config.wang_config = None

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        # Mock function to raise exception
        with patch(
            "tools.run_wang_detection_pipeline.run_wang_detection_pipeline"
        ) as mock_run:
            mock_run.side_effect = ValueError("Test error")

            # Should raise the exception
            with pytest.raises(ValueError, match="Test error"):
                pipeline.run_wang_abnormality()

    def test_phase_uses_custom_config_path(self, tmp_path):
        """Test that phase uses custom config path when provided."""
        eval_dir = tmp_path / "test_eval"
        eval_dir.mkdir()
        custom_config = tmp_path / "custom_config.yaml"
        custom_config.write_text("test: custom")

        config = PipelineConfig(eval_dir=eval_dir)
        config.dataset = "porto_hoser"
        config.run_wang_detection = True
        config.wang_config = str(custom_config)

        with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
            pipeline = EvaluationPipeline(config, eval_dir)

        with patch(
            "tools.run_wang_detection_pipeline.run_wang_detection_pipeline"
        ) as mock_run:
            mock_run.return_value = True
            pipeline.run_wang_abnormality()

            # Function should be called
            assert mock_run.called
