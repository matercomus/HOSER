import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from generate_hoser_abnormalities import (
    load_csv_with_flexible_columns,
    add_abnormality_info_column,
    save_with_abnormality_info,
)
import polars as pl


def test_abnormality_info_contains_real_metadata(tmp_path):
    import ast
    from generate_hoser_abnormalities import process_split_streaming

    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"

    in_csv.write_text(
        "mm_id,entity_id,traj_id,rid_list,time_list\n1,10,100,1,2020-01-01T00:00:00Z\n"
    )

    process_split_streaming(
        input_path=str(in_csv),
        output_path=str(out_csv),
        seed=123,
        level="low",
        abnormal_types=["detour"],
        abnormality_rate=None,
        abnormality_weights=None,
        ensure_change=False,
        progress_interval=10000,
        strong_prob=0.0,
    )

    loaded = pl.read_csv(out_csv)
    assert loaded.shape[0] == 2
    assert loaded["abnormality_info"][0] == "normal"

    info = ast.literal_eval(loaded["abnormality_info"][1])
    assert "real" in info
    assert info["real"]["rid_list"] == "1"
    assert info["real"]["time_list"] == "2020-01-01T00:00:00Z"


def test_real_metadata_matches_original_row(tmp_path):
    import ast
    from generate_hoser_abnormalities import process_split_streaming

    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"

    in_csv.write_text(
        "mm_id,entity_id,traj_id,rid_list,time_list\n1,10,100,1,2020-01-01T00:00:00Z\n"
    )

    process_split_streaming(
        input_path=str(in_csv),
        output_path=str(out_csv),
        seed=123,
        level="low",
        abnormal_types=["detour"],
        abnormality_rate=None,
        abnormality_weights=None,
        ensure_change=False,
        progress_interval=10000,
        strong_prob=0.0,
    )

    loaded = pl.read_csv(out_csv)
    assert loaded.shape[0] == 2

    original_rid_list = loaded["rid_list"][0]
    original_time_list = loaded["time_list"][0]

    info = ast.literal_eval(loaded["abnormality_info"][1])
    assert info["real"]["rid_list"] == original_rid_list
    assert info["real"]["time_list"] == original_time_list

    # When detour succeeds, the abnormal row should differ from the original,
    # but `real` must still point to the original.
    assert loaded["rid_list"][1] != original_rid_list


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


def test_generated_abnormal_rows_are_valid_walks_when_rel_present(tmp_path):
    from generate_hoser_abnormalities import process_split_streaming

    # Create a small directed graph with alternate valid routes.
    # Original trajectory: 1->2->3->4->5->6
    # Alternate middle node between 2 and 4: 2->8->4
    # Alternate route for switching between 1 and 4: 1->10->11->4
    rel = tmp_path / "roadmap.rel"
    rel.write_text(
        "origin_id,destination_id\n"
        "1,2\n"
        "2,3\n"
        "3,4\n"
        "4,5\n"
        "5,6\n"
        "2,8\n"
        "8,4\n"
        "1,10\n"
        "10,11\n"
        "11,4\n"
    )

    in_csv = tmp_path / "train.csv"
    out_csv = tmp_path / "out.csv"

    # Write multiple identical rows so deterministic RNG across indices yields
    # at least some successful changes.
    header = "mm_id,entity_id,traj_id,rid_list,time_list\n"
    base_rids = "1,2,3,4,5,6"
    base_times = ",".join(
        [
            "2020-01-01T00:00:00Z",
            "2020-01-01T00:01:00Z",
            "2020-01-01T00:02:00Z",
            "2020-01-01T00:03:00Z",
            "2020-01-01T00:04:00Z",
            "2020-01-01T00:05:00Z",
        ]
    )
    rows = []
    for i in range(20):
        rows.append(f'{i + 1},10,{1000 + i},"{base_rids}","{base_times}"\n')
    in_csv.write_text(header + "".join(rows))

    process_split_streaming(
        input_path=str(in_csv),
        output_path=str(out_csv),
        seed=123,
        level="low",
        abnormal_types=["detour", "perturb", "route_switch"],
        abnormality_rate=None,
        abnormality_weights=None,
        ensure_change=True,
        progress_interval=10000,
        strong_prob=0.0,
    )

    # Validate every abnormal row is a valid walk in roadmap.rel.
    import polars as pl

    loaded = pl.read_csv(out_csv)
    abnormal = loaded.filter(pl.col("abnormality_info") != "normal")
    assert abnormal.shape[0] > 0

    edge_set = set(
        (line.split(",")[0], line.split(",")[1])
        for line in rel.read_text().splitlines()[1:]
        if line.strip()
    )

    for rid_list in abnormal["rid_list"].to_list():
        rids = [x for x in str(rid_list).split(",") if x]
        assert len(rids) >= 2
        for a, b in zip(rids[:-1], rids[1:]):
            assert (a, b) in edge_set

    # And ensure timestamps stay aligned with road IDs after edits.
    for rid_list, time_list in zip(
        abnormal["rid_list"].to_list(), abnormal["time_list"].to_list()
    ):
        rids = [x for x in str(rid_list).split(",") if x]
        times = [x for x in str(time_list).split(",") if x]
        assert len(rids) == len(times)
