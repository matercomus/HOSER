import json
from pathlib import Path


from tools.evaluate_lmtad_spatial_abnormal import (
    evaluate_spatial_abnormal_trajectories,
)


def test_pipeline_mapping_autoload(tmp_path, patch_lmtad_teacher, monkeypatch):
    """Integration-style test: ensure evaluator auto-loads road_to_token.json.

    The test creates a tiny eval workspace with a nested-format mapping JSON and
    a trivial trajectory. It monkeypatches internal heavy functions so the test
    runs quickly and asserts that mapping is auto-loaded (GridMapper.map_all
    should NOT be called).
    """
    # Apply the patch that swaps real LMTADTeacher with a fake
    patch_lmtad_teacher()

    # Create mapping file in the same directory as the (fake) trajectory
    mapping = {"0": {"token": 200}, "1": {"token": "201"}, "3": {"distance_m": 5}}
    mapping_file = tmp_path / "road_to_token.json"
    mapping_file.write_text(json.dumps(mapping))

    # Create a dummy trajectory file path (content is not used because we patch loader)
    traj_file = tmp_path / "dummy_traj.csv"
    traj_file.write_text("id,seq\n1,0 1 0")

    # Patch the trajectory loader to return a simple trajectory list
    monkeypatch.setattr(
        "tools.evaluate_lmtad_spatial_abnormal.load_hoser_trajectories",
        lambda p: [[0, 1, 0]],
    )

    # Prevent GridMapper.map_all from being called; if it's invoked the test fails
    from critics import grid_mapper

    def _fail_map_all(self):
        raise AssertionError(
            "GridMapper.map_all should not be called when mapping auto-loaded"
        )

    monkeypatch.setattr(grid_mapper.GridMapper, "map_all", _fail_map_all)

    # Call evaluator: pass lmtad_repo as project root to avoid auto-detection
    result = evaluate_spatial_abnormal_trajectories(
        trajectory_file=traj_file,
        lmtad_checkpoint=Path("/tmp/fake_ckpt.pt"),
        source_eval_dir=tmp_path,
        dataset="porto_hoser",
        device="cpu",
        batch_size=1,
        lmtad_repo=Path("."),
        od_pairs_file=None,
        eval_config=None,
        max_duplicate_ratio=1.0,
        road_to_token_override=None,
    )

    assert isinstance(result, dict)
