import numpy as np
from tools.evaluate_lmtad_spatial_abnormal import filter_valid_trajectories


def test_filter_with_mapping_tokenizes_before_validation():
    # road_to_token maps raw road ids -> token ids
    # tokens: [0,1,2,3,4,5]
    road_to_token = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    # vocab_size = 5 (tokens 0-4 valid)
    vocab_size = 5

    # Trajectories: first valid (raw 0 -> token 0), second invalid (raw 5 -> token 5)
    trajectories = [[0, 1, 2], [5, 4, 3]]
    od_labels = {}

    valid, reasons, _ = filter_valid_trajectories(
        trajectories, od_labels, vocab_size=vocab_size, road_to_token=road_to_token
    )

    assert len(valid) == 1
    assert any(0 in t for t in valid)
    assert len(reasons) == 1


def test_filter_without_mapping_uses_raw_ids():
    # When no mapping provided, raw IDs are compared against vocab_size
    trajectories = [[0, 1, 2], [6, 7]]
    od_labels = {}

    # vocab_size small so raw id 6 is invalid
    valid, reasons, _ = filter_valid_trajectories(trajectories, od_labels, vocab_size=5)
    assert len(valid) == 1
    assert len(reasons) == 1


def test_valid_trajectory_mapped_and_accepted():
    road_to_token = np.array([10, 2, 3, 4, 0], dtype=np.int64)
    # vocab covers tokens up to 10
    vocab_size = 11
    trajectories = [[0, 1, 2]]  # raw [0,1,2] -> tokens [10,2,3]
    valid, reasons, _ = filter_valid_trajectories(
        trajectories, {}, vocab_size=vocab_size, road_to_token=road_to_token
    )
    assert len(valid) == 1
    assert reasons == []
