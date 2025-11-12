"""
Integration tests for the LM-TAD workflow.

Tests the complete LM-TAD integration pipeline including:
- End-to-end workflow execution
- Error handling and validation
- Performance under load
"""

import pytest
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile

from critics.lmtad_teacher import LMTADTeacher
from critics.grid_mapper import GridMapper, GridConfig
from critics.distill_hook import DistillationManager, DistillConfig
from evaluation import GlobalMetrics, evaluate_trajectories_programmatic


# Test Fixtures
@pytest.fixture
def lmtad_config():
    """Basic LM-TAD configuration for testing."""
    return {
        "repo_path": "/home/matt/Dev/LMTAD",
        "ckpt_path": "/home/matt/Dev/LMTAD/code/results/LMTAD/beijing_hoser_reference/run_20250928_202718/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/weights_only.pt",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": "float16",
        "window": 64,
    }


@pytest.fixture
def grid_config():
    """Grid mapper configuration for testing."""
    return GridConfig(
        min_lat=39.8,
        max_lat=40.0,
        min_lng=116.2,
        max_lng=116.5,
        grid_size=0.001,
        downsample_factor=1,
    )


@pytest.fixture
def sample_road_network():
    """Create a minimal road network for testing."""
    coordinates = [
        [[116.3, 39.9], [116.31, 39.91]],
        [[116.31, 39.91], [116.32, 39.92]],
        [[116.32, 39.92], [116.33, 39.93]],
    ]

    geo_data = {
        "road_id": list(range(len(coordinates))),
        "coordinates": [str(coord) for coord in coordinates],
        "length": [1000.0] * len(coordinates),
        "highway": ["primary"] * len(coordinates),
    }
    return pd.DataFrame(geo_data)


@pytest.fixture
def sample_trajectories():
    """Create sample trajectories for testing."""
    # 3 trajectories using the sample road network
    trajectories = [
        [(0, datetime.now()), (1, datetime.now()), (2, datetime.now())],
        [(2, datetime.now()), (1, datetime.now()), (0, datetime.now())],
        [(0, datetime.now()), (2, datetime.now())],
    ]
    return trajectories


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with required data files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        yield tmp_dir


# End-to-End Workflow Tests
def test_complete_pipeline(
    temp_data_dir, lmtad_config, grid_config, sample_road_network, sample_trajectories
):
    """Test the complete LM-TAD pipeline execution."""

    # Write test data
    geo_file = temp_data_dir / "roadmap.geo"
    sample_road_network.to_csv(geo_file, index=False)

    test_file = temp_data_dir / "test.csv"
    pd.DataFrame(
        {
            "rid_list": [
                ",".join(str(road_id) for road_id, _ in traj)
                for traj in sample_trajectories
            ],
            "time_list": [
                ",".join(t.strftime("%Y-%m-%dT%H:%M:%SZ") for _, t in traj)
                for traj in sample_trajectories
            ],
        }
    ).to_csv(test_file, index=False)

    # Initialize LM-TAD teacher
    teacher = LMTADTeacher(**lmtad_config)
    assert teacher is not None
    assert hasattr(teacher, "model")

    # Initialize grid mapper with sample road centroids
    road_centroids = np.array([[39.905, 116.305], [39.915, 116.315], [39.925, 116.325]])
    mapper = GridMapper(grid_config, road_centroids)
    assert mapper is not None

    # Test grid mapping
    road_tokens = mapper.map_all()
    assert len(road_tokens) == len(road_centroids)
    assert not np.any(np.isnan(road_tokens))

    # Test distillation manager
    distill_config = DistillConfig(
        enabled=True,
        repo_path=lmtad_config["repo_path"],
        ckpt_path=lmtad_config["ckpt_path"],
        dtype=lmtad_config["dtype"],
        window=lmtad_config["window"],
        grid_size=grid_config.grid_size,
    )

    manager = DistillationManager(
        distill_config,
        lmtad_config["device"],
        grid_config.min_lat,
        grid_config.max_lat,
        grid_config.min_lng,
        grid_config.max_lng,
        road_centroids[:, 0],
        road_centroids[:, 1],
    )

    assert manager is not None
    assert hasattr(manager, "teacher")
    assert hasattr(manager, "mapper")

    # Validate evaluation metrics can be computed
    metrics = GlobalMetrics(
        sample_trajectories, sample_trajectories, sample_road_network
    )
    results = metrics.evaluate()

    assert results is not None
    assert "Distance_JSD" in results
    assert "Duration_JSD" in results
    assert "Radius_JSD" in results


# Error Handling Tests
def test_invalid_teacher_checkpoint(lmtad_config):
    """Test error handling for invalid teacher checkpoint."""
    invalid_config = lmtad_config.copy()
    invalid_config["ckpt_path"] = "nonexistent.pt"

    with pytest.raises(FileNotFoundError):
        LMTADTeacher(**invalid_config)


def test_invalid_grid_dimensions(grid_config, sample_road_network):
    """Test error handling for invalid grid dimensions."""
    invalid_road_centroids = np.array([[100.0, 200.0]])  # Invalid coordinates

    with pytest.raises(ValueError):
        mapper = GridMapper(grid_config, invalid_road_centroids)
        mapper.map_all()


def test_missing_data_files(temp_data_dir):
    """Test error handling for missing data files."""
    with pytest.raises(FileNotFoundError):
        evaluate_trajectories_programmatic(
            "nonexistent.csv", dataset="test", od_source="test"
        )


# Performance Tests
@pytest.mark.slow
def test_large_scale_mapping(grid_config):
    """Test performance with large number of roads."""
    num_roads = 10000
    road_centroids = np.random.uniform(
        low=[grid_config.min_lat, grid_config.min_lng],
        high=[grid_config.max_lat, grid_config.max_lng],
        size=(num_roads, 2),
    )

    mapper = GridMapper(grid_config, road_centroids)
    tokens = mapper.map_all()

    assert len(tokens) == num_roads
    assert not np.any(np.isnan(tokens))


@pytest.mark.slow
def test_trajectory_batch_processing(lmtad_config):
    """Test teacher model performance with large batch of trajectories."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    teacher = LMTADTeacher(**lmtad_config)
    batch_size = 128
    seq_length = 32

    # Create random trajectory tokens
    history_tokens = torch.randint(
        0,
        1000,  # Assuming vocab size of 1000
        (batch_size, seq_length),
        device=lmtad_config["device"],
    )

    with torch.no_grad():
        probs = teacher.predict_next_distribution(history_tokens)

    assert probs.shape[0] == batch_size
    assert not torch.isnan(probs).any()
