"""
Tests for LM-TAD Trajectory Format Conversion Module

This module tests the conversion functionality from HOSER-format trajectories
(road IDs) to LM-TAD grid token format for teacher model evaluation.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from tools.convert_to_lmtad_format import (
    convert_hoser_to_lmtad_format,
    convert_trajectory_batch,
    create_grid_mapper,
    extract_road_centroids,
    save_lmtad_format,
)


@pytest.fixture
def sample_geo_data():
    """Create sample roadmap.geo data for testing"""
    return pd.DataFrame(
        {
            "road_id": [0, 1, 2, 3],
            "coordinates": [
                "[[116.397, 39.908], [116.398, 39.909]]",  # Beijing coordinates
                "[[116.400, 39.910], [116.401, 39.911]]",
                "[[116.402, 39.912], [116.403, 39.913]]",
                "[[116.404, 39.914], [116.405, 39.915]]",
            ],
        }
    )


@pytest.fixture
def sample_trajectory_data():
    """Create sample HOSER trajectory data for testing"""
    return pd.DataFrame(
        {
            "mm_id": [1, 2, 3],
            "entity_id": [100, 101, 102],
            "traj_id": [1000, 1001, 1002],
            "rid_list": ["0,1,2", "1,2,3", "0,3"],
            "time_list": [
                "2015-11-05T13:14:32Z,2015-11-05T13:15:16Z,2015-11-05T13:16:20Z",
                "2015-11-05T14:20:00Z,2015-11-05T14:21:30Z,2015-11-05T14:22:45Z",
                "2015-11-05T15:30:00Z,2015-11-05T15:32:15Z",
            ],
        }
    )


@pytest.fixture
def sample_road_centroids():
    """Create sample road centroids for testing"""
    # (lat, lng) coordinates matching the sample geo data
    return np.array(
        [
            [39.9085, 116.3975],  # Road 0 centroid
            [39.9105, 116.4005],  # Road 1 centroid
            [39.9125, 116.4025],  # Road 2 centroid
            [39.9145, 116.4045],  # Road 3 centroid
        ],
        dtype=np.float32,
    )


def test_extract_road_centroids(sample_geo_data, tmp_path):
    """Test extraction of road centroids from roadmap.geo"""
    # Create temporary geo file
    geo_file = tmp_path / "roadmap.geo"
    sample_geo_data.to_csv(geo_file, index=False)

    # Test successful extraction
    road_centroids, geo_df = extract_road_centroids(geo_file)

    assert isinstance(road_centroids, np.ndarray)
    assert road_centroids.shape == (4, 2)  # 4 roads, 2 coordinates each
    assert road_centroids.dtype == np.float32
    assert len(geo_df) == 4

    # Test that centroids are computed correctly (mean of coordinates)
    # Road 0: [[116.397, 39.908], [116.398, 39.909]] -> [39.9085, 116.3975]
    np.testing.assert_array_almost_equal(
        road_centroids[0], [39.9085, 116.3975], decimal=4
    )

    # Test missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        extract_road_centroids(tmp_path / "nonexistent.geo")


def test_create_grid_mapper(sample_geo_data, sample_road_centroids):
    """Test creation of grid mapper and road-to-token mapping"""
    with patch("tools.convert_to_lmtad_format.GridMapper") as mock_mapper_class:
        mock_mapper = Mock()
        mock_mapper.grid_h = 100
        mock_mapper.grid_w = 100
        mock_mapper.map_all.return_value = np.array([10, 20, 30, 40])
        mock_mapper_class.return_value = mock_mapper

        mapper, road_to_token = create_grid_mapper(
            geo_df=sample_geo_data,
            road_centroids=sample_road_centroids,
            grid_size=0.001,
            downsample_factor=1,
        )

        # Verify the mapper was created with correct parameters
        assert mock_mapper_class.called
        assert isinstance(road_to_token, np.ndarray)
        assert road_to_token.dtype == np.int64
        assert len(road_to_token) == 4
        np.testing.assert_array_equal(road_to_token, [10, 20, 30, 40])


def test_convert_trajectory_batch(sample_trajectory_data, tmp_path):
    """Test conversion of trajectory batch from road IDs to grid tokens"""
    # Create road-to-token mapping
    road_to_token = np.array([100, 101, 102, 103])  # Map road IDs 0,1,2,3 to tokens

    # Convert trajectories
    converted_df = convert_trajectory_batch(sample_trajectory_data, road_to_token)

    # Verify output structure
    assert len(converted_df) == 3
    assert set(converted_df.columns) == {
        "mm_id",
        "entity_id",
        "traj_id",
        "grid_token_list",
        "time_list",
    }

    # Verify grid token conversion
    assert converted_df.iloc[0]["grid_token_list"] == "100,101,102"  # Roads 0,1,2
    assert converted_df.iloc[1]["grid_token_list"] == "101,102,103"  # Roads 1,2,3
    assert converted_df.iloc[2]["grid_token_list"] == "100,103"  # Roads 0,3

    # Verify metadata preservation
    assert converted_df.iloc[0]["mm_id"] == 1
    assert converted_df.iloc[0]["entity_id"] == 100
    assert converted_df.iloc[0]["traj_id"] == 1000
    assert (
        converted_df.iloc[0]["time_list"] == sample_trajectory_data.iloc[0]["time_list"]
    )


def test_convert_trajectory_batch_invalid_road_ids(sample_trajectory_data):
    """Test handling of invalid road IDs in trajectory conversion"""
    # Create road-to-token mapping with fewer entries than needed
    road_to_token = np.array([100, 101, 102])  # Only 3 entries, but need 4

    # Convert trajectories - should skip trajectories with invalid road IDs
    converted_df = convert_trajectory_batch(sample_trajectory_data, road_to_token)

    # Should have fewer trajectories due to skipping
    assert len(converted_df) < len(sample_trajectory_data)


def test_save_lmtad_format(tmp_path):
    """Test saving converted trajectories to LM-TAD format"""
    # Create sample converted data
    converted_df = pd.DataFrame(
        {
            "mm_id": [1, 2],
            "entity_id": [100, 101],
            "traj_id": [1000, 1001],
            "grid_token_list": ["100,101,102", "101,102,103"],
            "time_list": ["t1,t2,t3", "t4,t5,t6"],
        }
    )

    output_file = tmp_path / "converted_trajectories.csv"
    save_lmtad_format(converted_df, output_file)

    # Verify file was created
    assert output_file.exists()

    # Verify content
    loaded_df = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(loaded_df, converted_df)


@patch("tools.convert_to_lmtad_format.extract_road_centroids")
@patch("tools.convert_to_lmtad_format.create_grid_mapper")
@patch("tools.convert_to_lmtad_format.pd.read_csv")
@patch("tools.convert_to_lmtad_format.convert_trajectory_batch")
@patch("tools.convert_to_lmtad_format.save_lmtad_format")
def test_convert_hoser_to_lmtad_format(
    mock_save,
    mock_convert,
    mock_read_csv,
    mock_create_mapper,
    mock_extract_centroids,
    sample_trajectory_data,
    sample_road_centroids,
    sample_geo_data,
    tmp_path,
):
    """Test the main conversion function"""
    # Set up mocks
    mock_extract_centroids.return_value = (sample_road_centroids, sample_geo_data)
    mock_create_mapper.return_value = (Mock(), np.array([100, 101, 102, 103]))
    mock_read_csv.return_value = sample_trajectory_data
    mock_convert.return_value = sample_trajectory_data.copy()
    mock_convert.return_value["grid_token_list"] = [
        "100,101,102",
        "101,102,103",
        "100,103",
    ]

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    data_dir = tmp_path / "data"

    # Create data directory and files
    data_dir.mkdir()
    geo_file = data_dir / "roadmap.geo"
    sample_geo_data.to_csv(geo_file, index=False)

    # Test successful conversion
    convert_hoser_to_lmtad_format(
        input_csv=input_file,
        output_csv=output_file,
        dataset="Beijing",
        data_dir=data_dir,
    )

    # Verify all steps were called
    mock_extract_centroids.assert_called_once()
    mock_create_mapper.assert_called_once()
    mock_read_csv.assert_called_once_with(input_file)
    mock_convert.assert_called_once()
    mock_save.assert_called_once()

    # Test missing input file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        convert_hoser_to_lmtad_format(
            input_csv=tmp_path / "nonexistent.csv",
            output_csv=output_file,
            dataset="Beijing",
        )


def test_convert_hoser_to_lmtad_format_with_config(sample_trajectory_data, tmp_path):
    """Test conversion with custom configuration"""
    with (
        patch("tools.convert_to_lmtad_format.extract_road_centroids") as mock_extract,
        patch("tools.convert_to_lmtad_format.create_grid_mapper") as mock_create,
        patch("tools.convert_to_lmtad_format.pd.read_csv") as mock_read,
        patch("tools.convert_to_lmtad_format.convert_trajectory_batch") as mock_convert,
        patch("tools.convert_to_lmtad_format.save_lmtad_format"),
    ):
        # Set up mocks
        mock_extract.return_value = (
            np.array([[39.9, 116.4], [39.91, 116.41]], dtype=np.float32),
            pd.DataFrame(
                {
                    "road_id": [0, 1],
                    "coordinates": ["[[116.4, 39.9]]", "[[116.41, 39.91]]"],
                }
            ),
        )
        mock_create.return_value = (Mock(), np.array([200, 201]))
        mock_read.return_value = sample_trajectory_data
        mock_convert.return_value = pd.DataFrame(
            {
                "mm_id": [1],
                "entity_id": [100],
                "traj_id": [1000],
                "grid_token_list": ["200,201"],
                "time_list": ["t1,t2"],
            }
        )

        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        # Test with custom config
        config = {"distill": {"grid_size": 0.002, "downsample": 2}}

        convert_hoser_to_lmtad_format(
            input_csv=input_file,
            output_csv=output_file,
            dataset="Beijing",
            config=config,
        )

        # Verify config was used
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        # The grid_size and downsample_factor should be passed to create_grid_mapper


if __name__ == "__main__":
    pytest.main([__file__])
