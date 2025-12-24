"""Tests for per-model seed inference in generation."""

from pathlib import Path
from unittest.mock import patch

import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from python_pipeline import PipelineConfig, TrajectoryGenerator  # noqa: E402
from python_pipeline import EvaluationPipeline  # noqa: E402


@pytest.mark.parametrize(
    ("model_type", "expected_seed"),
    [
        ("seed43", 43),
        ("distilled_l0p001_seed44", 44),
        ("vanilla_seed7", 7),
    ],
)
def test_generator_uses_seed_from_model_type(tmp_path: Path, model_type: str, expected_seed: int) -> None:
    """Generator passes model-derived seed to programmatic interface."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    config = PipelineConfig(eval_dir=eval_dir)
    config.dataset = "Beijing"
    config.seed = 42
    config.num_gene = 1
    config.cuda_device = 0
    config.beam_search = False
    config.beam_width = 4

    generator = TrajectoryGenerator(config)

    with patch("python_pipeline.generate_trajectories_programmatic") as mock_generate:
        mock_generate.return_value = {
            "output_file": str(tmp_path / "out.csv"),
            "performance": {},
        }

        generator.generate_trajectories(
            model_path=tmp_path / "dummy.pth",
            model_type=model_type,
            od_source="train",
        )

        assert mock_generate.called
        assert mock_generate.call_args.kwargs["seed"] == expected_seed


def test_generator_falls_back_to_config_seed_when_missing(tmp_path: Path) -> None:
    """Generator falls back to PipelineConfig.seed when model type has no seed."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    config = PipelineConfig(eval_dir=eval_dir)
    config.dataset = "Beijing"
    config.seed = 42
    config.num_gene = 1
    config.cuda_device = 0
    config.beam_search = False
    config.beam_width = 4

    generator = TrajectoryGenerator(config)

    with patch("python_pipeline.generate_trajectories_programmatic") as mock_generate:
        mock_generate.return_value = {
            "output_file": str(tmp_path / "out.csv"),
            "performance": {},
        }

        generator.generate_trajectories(
            model_path=tmp_path / "dummy.pth",
            model_type="distilled_l0p001",
            od_source="train",
        )

        assert mock_generate.called
        assert mock_generate.call_args.kwargs["seed"] == 42


def test_check_existing_results_uses_model_seed_dir(tmp_path: Path, monkeypatch) -> None:
    """Existing-file lookup should search in seed parsed from model_type."""

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "models").mkdir()

    config = PipelineConfig(eval_dir=eval_dir)
    config.dataset = "Beijing"
    config.seed = 42
    config.force = False
    config.num_gene = 1

    with patch.object(EvaluationPipeline, "_validate_config", return_value=None):
        pipeline = EvaluationPipeline(config, eval_dir)

    monkeypatch.chdir(eval_dir)

    gene_dir = eval_dir / "gene" / "Beijing" / "seed43"
    gene_dir.mkdir(parents=True)
    csv_path = gene_dir / "2025-01-01_00-00-00_seed43_train.csv"
    csv_path.write_text("col\nrow\n")
    perf_path = gene_dir / "2025-01-01_00-00-00_seed43_train_perf.json"
    perf_path.write_text('{"beam_search_enabled": false, "num_trajectories": 1}')

    found = pipeline._check_existing_results("seed43", "train")
    assert found is not None
    assert found.resolve() == csv_path.resolve()
