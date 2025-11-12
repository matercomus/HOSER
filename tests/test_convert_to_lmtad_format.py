"""Tests for trajectory format conversion."""

import json
from pprint import pprint

import numpy as np
import pandas as pd
import pytest

from tools.convert_to_lmtad_format import (
    DATASET_CONFIGS,
    convert_hoser_to_lmtad_format,
    create_grid_mapper,
    extract_road_centroids,
)


@pytest.fixture
def mock_roadmap_file(tmp_path, monkeypatch):
    """Create a mock roadmap.geo file."""
    # Create coordinates for a 2x2 grid with 0.02 degree spacing
    # Each cell is 0.01 degrees, so total span is 0.02
    roadmap_data = pd.DataFrame(
        {
            "coordinates": [
                # Bottom row (y=0)
                "[[8.65000, 41.15000], [8.65050, 41.15050]]",  # Cell (0,0)
                "[[8.66000, 41.15000], [8.66050, 41.15050]]",  # Cell (0,1)
                # Top row (y=1)
                "[[8.65000, 41.16000], [8.65050, 41.16050]]",  # Cell (1,0)
                "[[8.66000, 41.16000], [8.66050, 41.16050]]",  # Cell (1,1)
            ],
            "geo_id": [0, 1, 2, 3],  # Use 0-based indices for road IDs
            "lanes": ['["2"]', '["1"]', '["2"]', '["1"]'],
            "oneway": ["[false]", "[true]", "[false]", "[true]"],
            "name": ["Street A", "Street B", "Street C", "Street D"],
        }
    )
    roadmap_file = tmp_path / "roadmap.geo"
    roadmap_data.to_csv(roadmap_file, index=False)

    # Force grid size to be 0.01 for test data (creates 2x2 grid)
    monkeypatch.setitem(DATASET_CONFIGS["porto_hoser"], "grid_size", 0.01)

    return roadmap_file


@pytest.fixture
def mock_trajectory_file(tmp_path):
    """Create a mock trajectory file."""
    trajectory_data = pd.DataFrame(
        {
            "mm_id": [1, 2],
            "entity_id": [100, 200],
            "traj_id": [1001, 1002],
            "rid_list": [
                "[0, 1, 2]",  # Valid road IDs (0, 1, 2)
                "[1, 2, 0]",  # Valid road IDs reversed
            ],
            "time_list": [
                "2024-01-01T00:00:00Z,2024-01-01T00:01:00Z,2024-01-01T00:02:00Z",
                "2024-01-01T01:00:00Z,2024-01-01T01:01:00Z,2024-01-01T01:02:00Z",
            ],
        }
    )
    trajectory_file = tmp_path / "trajectories.csv"
    trajectory_data.to_csv(trajectory_file, index=False)
    return trajectory_file


def test_extract_road_centroids(mock_roadmap_file):
    """Test extraction of road centroids."""
    road_centroids, boundary = extract_road_centroids(mock_roadmap_file)

    # Check centroid shape
    assert road_centroids.shape == (4, 2)  # 4 roads, 2 coordinates each
    assert road_centroids.dtype == np.float64

    # Check boundary values
    assert boundary["min_lat"] == pytest.approx(41.15000)
    assert boundary["max_lat"] == pytest.approx(41.16050)
    assert boundary["min_lng"] == pytest.approx(8.65000)
    assert boundary["max_lng"] == pytest.approx(8.66050)

    print("\nExtracting road centroids:")
    print(f"  Centroids shape: {road_centroids.shape}")
    print("  Centroids:")
    print(road_centroids)
    print("\n  Boundaries:")
    pprint(boundary)


def test_create_grid_mapper(mock_roadmap_file):
    """Test grid mapper creation."""
    # Extract centroids first
    road_centroids, boundary = extract_road_centroids(mock_roadmap_file)

    # Create mapper with debug output
    config = DATASET_CONFIGS["porto_hoser"]
    print("\nCreating grid mapper with:")
    print(f"  Grid size: {config['grid_size']}")
    print(f"  Boundary: {boundary}")
    lat_span = boundary["max_lat"] - boundary["min_lat"]
    lng_span = boundary["max_lng"] - boundary["min_lng"]
    print(f"  Spans: lat={lat_span:.6f}, lng={lng_span:.6f}")
    grid_h = int(lat_span / config["grid_size"]) + 1
    grid_w = int(lng_span / config["grid_size"]) + 1
    print(f"  Raw grid dimensions: h={grid_h}, w={grid_w}")

    mapper, vocab = create_grid_mapper("porto_hoser", road_centroids, boundary)

    print("\nMapper properties:")
    print(f"  grid_h: {mapper.grid_h}")
    print(f"  grid_w: {mapper.grid_w}")
    print("  Vocabulary:")
    pprint(vocab)

    # Check mapper properties (should be 2x2 grid for test data)
    assert mapper.grid_h == 2  # 2x2 grid with grid_size=0.01
    assert mapper.grid_w == 2
    assert mapper.grid_h * mapper.grid_w == 4  # 2x2 grid for test data

    # Check vocabulary
    assert "PAD" in vocab
    assert "EOT" in vocab
    assert "SOT" in vocab
    assert all(str(i) in vocab for i in range(4))  # Grid tokens 0-3
    assert vocab["PAD"] == 4
    assert vocab["EOT"] == 5
    assert vocab["SOT"] == 6


def test_convert_hoser_to_lmtad_format(
    tmp_path, mock_roadmap_file, mock_trajectory_file
):
    """Test end-to-end conversion."""
    # Setup output paths
    output_file = tmp_path / "converted.csv"
    vocab_file = tmp_path / "vocab.json"

    print("\nStarting HOSER to LM-TAD conversion:")
    print(f"  Input: {mock_trajectory_file}")
    print(f"  Roadmap: {mock_roadmap_file}")

    # Run conversion
    result = convert_hoser_to_lmtad_format(
        trajectory_file=mock_trajectory_file,
        roadmap_file=mock_roadmap_file,
        output_file=output_file,
        vocab_file=vocab_file,
        dataset="porto_hoser",
    )

    print("\nConversion complete:")
    # Read input
    with open(mock_trajectory_file) as f:
        print("\nInput trajectories:")
        print(f.read())

    # Read output
    with open(output_file) as f:
        print("\nConverted trajectories:")
        print(f.read())

    with open(vocab_file) as f:
        print("\nVocabulary:")
        print(f.read())

    # Check outputs exist
    assert output_file.exists()
    assert vocab_file.exists()

    # Check vocabulary format
    with open(vocab_file) as f:
        vocab = json.loads(f.read())  # Use json.loads for safety
        assert "PAD" in vocab
        assert "EOT" in vocab
        assert "SOT" in vocab
        max_token = max(int(k) for k in vocab.keys() if k.isdigit())
        assert max_token == 3  # 2x2 grid = 4 cells (0-3)
        assert vocab["PAD"] == 4
        assert vocab["EOT"] == 5
        assert vocab["SOT"] == 6

    # Check trajectory format
    with open(output_file) as f:
        lines = f.readlines()
        assert len(lines) == 2  # Two trajectories
        # Each line should be a list of grid tokens
        grid_tokens = [eval(line.strip()) for line in lines]
        assert all(isinstance(tokens, list) for tokens in grid_tokens)
        assert all(len(tokens) == 3 for tokens in grid_tokens)  # Each has 3 points
        assert all(isinstance(t, int) for tokens in grid_tokens for t in tokens)
        assert all(0 <= t < 4 for tokens in grid_tokens for t in tokens)
        # Verify tokens are in the correct range (0-3 for 2x2 grid)
        for tokens in grid_tokens:
            for t in tokens:
                assert 0 <= t <= 3, f"Token {t} out of range (0-3)"

    # Return path matches
    assert result == output_file


def test_invalid_dataset():
    """Test error handling for invalid dataset."""
    with pytest.raises(ValueError, match=r"Unknown dataset"):
        create_grid_mapper(
            "invalid_dataset",
            np.array([[0, 0]]),
            {"min_lat": 0, "max_lat": 1, "min_lng": 0, "max_lng": 1},
        )


def test_missing_files(tmp_path):
    """Test error handling for missing files."""
    with pytest.raises(FileNotFoundError, match=r"Roadmap file not found"):
        convert_hoser_to_lmtad_format(
            trajectory_file=tmp_path / "trajectories.csv",
            roadmap_file=tmp_path / "nonexistent.geo",
            output_file=tmp_path / "out.csv",
            vocab_file=tmp_path / "vocab.json",
            dataset="porto_hoser",
        )
