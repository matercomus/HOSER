"""
Tests for Configuration Loader Module

This module tests the loading and management of evaluation configuration
from YAML files, including LM-TAD teacher baseline evaluation settings.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tools.config_loader import EvaluationConfig, load_evaluation_config


@pytest.fixture
def sample_hoser_config():
    """Create sample HOSER evaluation configuration"""
    return {
        "dataset": "Beijing",
        "data_dir": "/fake/data/Beijing",
        "seed": 42,
        "cuda_device": 0,
        "num_gene": 5000,
        "beam_width": 4,
        "beam_search": True,
        "grid_size": 0.001,
        "edr_eps": 100.0,
    }


@pytest.fixture
def sample_lmtad_config(sample_hoser_config):
    """Create sample LM-TAD teacher evaluation configuration"""
    config = sample_hoser_config.copy()
    config.update(
        {
            "lmtad_evaluation": True,
            "lmtad_repo": "/fake/lmtad/repo",
            "lmtad_checkpoint": "code/results/checkpoint.pt",
            "lmtad_real_data_dir": "/fake/lmtad/data/beijing_hoser_reference",
        }
    )
    return config


@pytest.fixture
def sample_translation_config(sample_hoser_config):
    """Create sample cross-dataset translation configuration"""
    config = sample_hoser_config.copy()
    config.update(
        {
            "source_dataset": "porto_hoser",
            "target_dataset": "BJUT_Beijing",
            "translation_max_distance": 25.0,
            "translation_mapping_file": "road_mapping_porto_to_beijing.json",
        }
    )
    return config


def test_evaluation_config_defaults():
    """Test EvaluationConfig with default values"""
    eval_dir = Path("/fake/eval_dir")
    config = EvaluationConfig(eval_dir=eval_dir)

    # Test default values
    assert config.dataset == "Beijing"
    assert config.seed == 42
    assert config.cuda_device == 0
    assert config.num_gene == 100
    assert config.beam_width == 4
    assert config.grid_size == 0.001
    assert config.edr_eps == 100.0
    assert config.lmtad_evaluation is False
    assert config.skip_generation is False


def test_evaluation_config_custom_values(sample_hoser_config):
    """Test EvaluationConfig with custom values"""
    eval_dir = Path("/fake/eval_dir")
    config = EvaluationConfig(eval_dir=eval_dir, **sample_hoser_config)

    # Test custom values
    assert config.dataset == "Beijing"
    assert config.seed == 42
    assert config.num_gene == 5000
    assert config.beam_width == 4
    assert config.grid_size == 0.001
    assert config.edr_eps == 100.0


def test_evaluation_config_get_data_dir_relative():
    """Test data directory resolution with relative paths"""
    eval_dir = Path("/fake/project/eval_xyz")
    config = EvaluationConfig(
        eval_dir=eval_dir, data_dir=Path("data/Beijing"), dataset="Beijing"
    )

    data_dir = config.get_data_dir()
    expected = Path("/fake/project/eval_xyz/data/Beijing").resolve()
    assert data_dir == expected


def test_evaluation_config_get_data_dir_absolute():
    """Test data directory resolution with absolute paths"""
    eval_dir = Path("/fake/project/eval_xyz")
    config = EvaluationConfig(
        eval_dir=eval_dir, data_dir=Path("/absolute/data/Beijing"), dataset="Beijing"
    )

    data_dir = config.get_data_dir()
    assert data_dir == Path("/absolute/data/Beijing").resolve()


def test_evaluation_config_get_data_dir_none():
    """Test data directory resolution when data_dir is None"""
    eval_dir = Path("/fake/project/eval_xyz")
    config = EvaluationConfig(eval_dir=eval_dir, data_dir=None, dataset="Beijing")

    data_dir = config.get_data_dir()
    expected = Path("/fake/project/data/Beijing").resolve()
    assert data_dir == expected


def test_evaluation_config_get_methods():
    """Test various getter methods"""
    eval_dir = Path("/fake/eval_dir")
    config = EvaluationConfig(eval_dir=eval_dir, dataset="Beijing")

    # Test get method
    assert config.get("nonexistent", "default") == "default"
    assert config.get("dataset", "default") == "Beijing"

    # Test translation mapping file auto-detection
    config.source_dataset = "porto_hoser"
    config.target_dataset = "BJUT_Beijing"
    mapping_file = config.get_translation_mapping_file()
    expected = Path("/fake/eval_dir/road_mapping_porto_hoser_to_BJUT_Beijing.json")
    assert mapping_file == expected

    # Test explicit mapping file
    config.translation_mapping_file = Path("/explicit/mapping.json")
    mapping_file = config.get_translation_mapping_file()
    assert mapping_file == Path("/explicit/mapping.json")


def test_evaluation_config_lmtad_checkpoint(sample_lmtad_config):
    """Test LM-TAD checkpoint path resolution"""
    eval_dir = Path("/fake/project/eval_xyz")
    config = EvaluationConfig(eval_dir=eval_dir, **sample_lmtad_config)

    # Test that checkpoint returns None when lmtad_evaluation is False
    config.lmtad_evaluation = False
    assert config.get_lmtad_checkpoint() is None

    # Test error when lmtad_evaluation=True but checkpoint not specified
    config.lmtad_evaluation = True
    config.lmtad_checkpoint = None
    with pytest.raises(
        ValueError, match="lmtad_evaluation=True but lmtad_checkpoint not specified"
    ):
        config.get_lmtad_checkpoint()


def test_evaluation_config_lmtad_checkpoint_resolution(tmp_path):
    """Test LM-TAD checkpoint path resolution with real files"""
    # Create temporary directory structure
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    lmtad_repo = tmp_path / "lmtad"
    lmtad_repo.mkdir()

    checkpoint_dir = lmtad_repo / "code" / "results"
    checkpoint_dir.mkdir(parents=True)

    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    checkpoint_file.touch()

    # Test with relative path resolved to repo
    config = EvaluationConfig(
        eval_dir=eval_dir,
        dataset="Beijing",
        lmtad_evaluation=True,
        lmtad_repo=lmtad_repo,
        lmtad_checkpoint=Path("code/results/checkpoint.pt"),
    )

    checkpoint_path = config.get_lmtad_checkpoint()
    assert checkpoint_path == checkpoint_file.resolve()

    # Test with absolute path
    config_absolute = EvaluationConfig(
        eval_dir=eval_dir,
        dataset="Beijing",
        lmtad_evaluation=True,
        lmtad_checkpoint=checkpoint_file,
    )

    checkpoint_path = config_absolute.get_lmtad_checkpoint()
    assert checkpoint_path == checkpoint_file.resolve()


@patch("tools.config_loader.logger")
def test_evaluation_config_lmtad_real_data_path(mock_logger, sample_lmtad_config):
    """Test LM-TAD real data path resolution"""
    eval_dir = Path("/fake/project/eval_xyz")
    config = EvaluationConfig(eval_dir=eval_dir, **sample_lmtad_config)

    # Test that returns None when lmtad_evaluation is False
    config.lmtad_evaluation = False
    assert config.get_lmtad_real_data_path() is None

    # Test with explicit real data directory
    config.lmtad_evaluation = True
    real_data_path = config.get_lmtad_real_data_path()
    expected = Path("/fake/lmtad/data/beijing_hoser_reference").resolve()
    assert real_data_path == expected


def test_evaluation_config_lmtad_real_data_path_auto_detection(tmp_path):
    """Test LM-TAD real data path auto-detection"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    lmtad_repo = tmp_path / "lmtad"
    lmtad_repo.mkdir()

    data_dir = lmtad_repo / "data" / "beijing_hoser_reference"
    data_dir.mkdir(parents=True)

    config = EvaluationConfig(
        eval_dir=eval_dir,
        dataset="Beijing",
        lmtad_evaluation=True,
        lmtad_repo=lmtad_repo,
    )

    real_data_path = config.get_lmtad_real_data_path()
    assert real_data_path == data_dir.resolve()


def test_load_evaluation_config_defaults(tmp_path):
    """Test loading configuration with defaults (no config file)"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    config = load_evaluation_config(eval_dir=eval_dir)

    # Should have default values
    assert config.dataset == "Beijing"
    assert config.seed == 42
    assert config.lmtad_evaluation is False


def test_load_evaluation_config_from_file(tmp_path, sample_hoser_config):
    """Test loading configuration from YAML file"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    config_dir = eval_dir / "config"
    config_dir.mkdir()

    config_file = config_dir / "evaluation.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_hoser_config, f)

    config = load_evaluation_config(eval_dir=eval_dir, config_path=config_file)

    # Should have values from file
    assert config.dataset == "Beijing"
    assert config.seed == 42
    assert config.num_gene == 5000
    assert config.grid_size == 0.001


def test_load_evaluation_config_from_file_with_lmtad(tmp_path, sample_lmtad_config):
    """Test loading LM-TAD configuration from YAML file"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    config_dir = eval_dir / "config"
    config_dir.mkdir()

    config_file = config_dir / "evaluation.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_lmtad_config, f)

    config = load_evaluation_config(eval_dir=eval_dir, config_path=config_file)

    # Should have LM-TAD values
    assert config.lmtad_evaluation is True
    assert config.lmtad_repo == Path("/fake/lmtad/repo")
    assert config.lmtad_checkpoint == Path("code/results/checkpoint.pt")


def test_load_evaluation_config_with_translation(tmp_path, sample_translation_config):
    """Test loading configuration with translation settings"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    config_dir = eval_dir / "config"
    config_dir.mkdir()

    config_file = config_dir / "evaluation.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_translation_config, f)

    config = load_evaluation_config(eval_dir=eval_dir, config_path=config_file)

    # Should have translation values
    assert config.source_dataset == "porto_hoser"
    assert config.target_dataset == "BJUT_Beijing"
    assert config.translation_max_distance == 25.0


def test_load_evaluation_config_with_legacy_fields(tmp_path):
    """Test loading configuration with legacy field names"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    config_dir = eval_dir / "config"
    config_dir.mkdir()

    legacy_config = {
        "dataset": "Beijing",
        "cross_dataset_name": "porto_hoser",  # Legacy field
        "target_dataset": "BJUT_Beijing",
        "skip_generation": True,
    }

    config_file = config_dir / "evaluation.yaml"
    with open(config_file, "w") as f:
        yaml.dump(legacy_config, f)

    config = load_evaluation_config(eval_dir=eval_dir, config_path=config_file)

    # Should map legacy cross_dataset_name to source_dataset
    assert config.source_dataset == "porto_hoser"
    assert config.skip_generation is True


def test_load_evaluation_config_file_not_found(tmp_path):
    """Test loading configuration when file doesn't exist"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    config_path = eval_dir / "config" / "nonexistent.yaml"

    # Should not raise exception and use defaults
    config = load_evaluation_config(eval_dir=eval_dir, config_path=config_path)

    assert config.dataset == "Beijing"  # Default value
    assert config.seed == 42  # Default value


def test_load_evaluation_config_invalid_yaml(tmp_path):
    """Test loading configuration with invalid YAML"""
    eval_dir = tmp_path / "eval_xyz"
    eval_dir.mkdir()

    config_dir = eval_dir / "config"
    config_dir.mkdir()

    config_file = config_dir / "evaluation.yaml"
    with open(config_file, "w") as f:
        f.write("invalid: yaml: content:")

    # Should not raise exception and use defaults
    config = load_evaluation_config(eval_dir=eval_dir, config_path=config_file)

    assert config.dataset == "Beijing"  # Default value


if __name__ == "__main__":
    pytest.main([__file__])
