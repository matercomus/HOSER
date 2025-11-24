import json

import numpy as np

from critics.mapping_utils import load_road_to_token_mapping


def test_load_plain_mapping(tmp_path):
    data = {"0": 100, "1": 101, "2": 102}
    p = tmp_path / "plain.json"
    p.write_text(json.dumps(data))

    arr = load_road_to_token_mapping(p)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int64
    assert arr.tolist() == [100, 101, 102]


def test_load_nested_mapping(tmp_path):
    data = {"0": {"target_road_id": 200}, "1": {"token": "201"}, "3": {"distance_m": 5}}
    p = tmp_path / "nested.json"
    p.write_text(json.dumps(data))

    arr = load_road_to_token_mapping(p)
    # max key is 3 -> length 4
    assert arr.shape[0] == 4
    # index 0 -> 200, index 1 -> 201, index 2 -> -1 (missing), index 3 -> -1 (no token)
    assert arr[0] == 200
    assert arr[1] == 201
    assert int(arr[2]) == -1
    assert int(arr[3]) == -1


def test_load_from_dict_direct():
    data = {0: 5, 2: {"token_id": 7}}
    arr = load_road_to_token_mapping(data)
    assert arr.tolist() == [5, -1, 7]
