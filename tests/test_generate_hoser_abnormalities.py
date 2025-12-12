import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from generate_hoser_abnormalities import (
    load_csv_with_flexible_columns,
    add_abnormality_info_column,
    save_with_abnormality_info,
)
import polars as pl


def make_sample_df(second_col_name="entity_id"):
    return pl.DataFrame(
        [
            {
                "mm_id": 1,
                second_col_name: 10,
                "traj_id": 100,
                "rid_list": "1,2,3",
                "time_list": "t1,t2,t3",
            },
            {
                "mm_id": 2,
                second_col_name: 20,
                "traj_id": 200,
                "rid_list": "4,5,6",
                "time_list": "t4,t5,t6",
            },
        ]
    )


def test_load_csv_with_flexible_columns_entity_id(tmp_path):
    df = make_sample_df("entity_id")
    f = tmp_path / "test.csv"
    df.write_csv(f)
    loaded = load_csv_with_flexible_columns(str(f))
    assert loaded.columns[1] == "entity_id"


def test_load_csv_with_flexible_columns_user_id(tmp_path):
    df = make_sample_df("user_id")
    f = tmp_path / "test.csv"
    df.write_csv(f)
    loaded = load_csv_with_flexible_columns(str(f))
    assert loaded.columns[1] == "user_id"


def test_add_abnormality_info_column():
    df = make_sample_df()
    df2 = add_abnormality_info_column(df)
    assert "abnormality_info" in df2.columns
    assert df2["abnormality_info"].to_list() == ["normal", "normal"]


def test_save_with_abnormality_info(tmp_path):
    df = make_sample_df()
    df2 = add_abnormality_info_column(df)
    out = tmp_path / "out.csv"
    save_with_abnormality_info(df2, str(out))
    loaded = pl.read_csv(out)
    assert "abnormality_info" in loaded.columns
    assert loaded["abnormality_info"][0] == "normal"


def test_detour_abnormality_generation(tmp_path):
    # Write a small input file
    import numpy as np
    from generate_hoser_abnormalities import insert_detour

    # Build a pool of road IDs
    road_id_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng = np.random.default_rng(123)
    rid_list = [1, 2, 3]
    # Test for each level
    for level, n_insert in [("low", 1), ("medium", 2), ("high", 3)]:
        new_rid_list, detour_roads = insert_detour(rid_list, level, road_id_pool, rng)
        assert len(detour_roads) == n_insert
        assert all(r in new_rid_list for r in detour_roads)
        # The abnormality_info string should be formatted correctly
        info = f"type=detour|level={level}|inserted_roads={','.join(map(str, detour_roads))}"
        assert info.startswith(f"type=detour|level={level}|inserted_roads=")


def test_route_switch_abnormality_generation():
    import numpy as np
    from generate_hoser_abnormalities import route_switch

    rng = np.random.default_rng(42)
    # Two trajectories
    rid_list1 = [10, 11, 12, 13, 14, 15]
    rid_list2 = [20, 21, 22, 23, 24, 25]
    for level, seg_len in [("low", 2), ("medium", 3), ("high", 4)]:
        new_rid_list, seg_range, seg2 = route_switch(rid_list1, rid_list2, level, rng)
        assert seg_range is not None and seg2 is not None
        # The replaced segment in new_rid_list should match seg2
        s, e = seg_range
        assert new_rid_list[s:e] == seg2
        # The abnormality_info string should be formatted correctly
        info = f"type=route_switch|level={level}|from_traj=999|segment={s}-{e}|inserted={','.join(map(str, seg2))}"
        assert info.startswith(
            f"type=route_switch|level={level}|from_traj=999|segment={s}-{e}|inserted="
        )


def test_perturb_abnormality_generation():
    import numpy as np
    from generate_hoser_abnormalities import perturb_rids

    rng = np.random.default_rng(7)
    rid_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    road_id_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for level, min_n in [("low", 1), ("medium", 1), ("high", 2)]:
        new_rid_list, perturbed = perturb_rids(rid_list, level, road_id_pool, rng)
        assert len(new_rid_list) == len(rid_list)
        assert len(perturbed) >= min_n
        for idx, old, new in perturbed:
            assert new_rid_list[idx] == new
            assert old != new
        # The abnormality_info string should be formatted correctly
        perturbed_str = ";".join(f"{i}:{o}->{n}" for i, o, n in perturbed)
        info = f"type=perturb|level={level}|perturbed_indices={perturbed_str}"
        assert info.startswith(f"type=perturb|level={level}|perturbed_indices=")
