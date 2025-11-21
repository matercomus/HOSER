import csv
from pathlib import Path

import numpy as np


def make_csv(path: Path, trajectories):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "origin_road_id",
        "destination_road_id",
        "source_index",
        "source_origin_time",
        "gene_trace_road_id",
        "gene_trace_datetime",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for traj in trajectories:
            origin = traj[0]
            destination = traj[-1]
            w.writerow(
                [origin, destination, -1, "2020-01-01T00:00:00Z", str(traj), "[]"]
            )


def test_evaluate_spatial_abnormal_e2e_monkeypatched(tmp_path, monkeypatch):
    """End-to-end smoke test for evaluate_spatial_abnormal_trajectories.

    This test runs the evaluation function end-to-end but monkeypatches the
    heavy LM-TAD teacher and the token-evaluation routine so the test stays
    fast and deterministic while exercising the real mapping and validation
    code paths.
    """
    from tools.evaluate_lmtad_spatial_abnormal import (
        evaluate_spatial_abnormal_trajectories,
    )

    # Create a tiny CSV with two short trajectories using road IDs known to exist
    # in the Porto roadmap (sample IDs from repo data). These were observed in
    # real generated files and should map successfully.
    trajs = [[1152, 1676, 1801], [1915, 543, 5194]]
    csv_path = tmp_path / "gene.csv"
    make_csv(csv_path, trajs)

    # Monkeypatch LMTADTeacher used inside the evaluate function to a light stub
    class FakeTeacher:
        def __init__(self, *args, **kwargs):
            pass

        def get_grid_size_hw(self):
            # Return None to avoid strict grid-dimension verification in GridMapper
            return None

        def vocab_size(self):
            # Return a large vocab so mapped tokens will be considered valid
            return 100000

        def sot_token(self):
            return None

    monkeypatch.setattr(
        "tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher", FakeTeacher
    )

    # Monkeypatch the heavy evaluation function to return deterministic perplexities
    def fake_evaluate_trajectories_direct(
        trajectories,
        model,
        road_to_token,
        device,
        batch_size,
        vocab_size,
        return_segment_perplexity,
    ):
        # Return a finite log-perplexity per trajectory and empty segment lists
        log_perp = [float(np.log(1.0 + i + 1.0)) for i in range(len(trajectories))]
        return log_perp, None, [[] for _ in trajectories]

    monkeypatch.setattr(
        "tools.evaluate_lmtad_spatial_abnormal.evaluate_trajectories_direct",
        fake_evaluate_trajectories_direct,
    )

    # Provide a fake LM-TAD repo path (function accepts it and will not auto-detect)
    fake_repo = Path(".")
    # Use a dummy checkpoint path (not used by FakeTeacher)
    fake_ckpt = Path("/tmp/ckpt.pt")

    # Run evaluation - should complete without loading real LM-TAD
    result = evaluate_spatial_abnormal_trajectories(
        trajectory_file=csv_path,
        lmtad_checkpoint=fake_ckpt,
        source_eval_dir=Path("/tmp"),
        dataset="porto_hoser",
        device="cpu",
        batch_size=1,
        lmtad_repo=fake_repo,
        od_pairs_file=None,
        eval_config=None,
    )

    # Basic assertions about structure
    assert isinstance(result, dict)
    # Result should contain dataset metadata and computed perplexities summary
    assert result.get("dataset") == "porto_hoser"
    assert "log_perplexity_stats" in result
    assert result.get("failed_trajectory_count", 0) == 0
