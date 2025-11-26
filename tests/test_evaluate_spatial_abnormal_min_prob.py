import tempfile
from pathlib import Path

from tests.conftest import make_fake_lmtad_teacher

from tools.evaluate_lmtad_spatial_abnormal import evaluate_spatial_abnormal_trajectories


def test_evaluate_spatial_abnormal_min_prob(monkeypatch):
    # Prepare fake teacher with zero distribution for certain target
    fake = make_fake_lmtad_teacher(vocab_size=10)

    def zero_pred(history):
        import numpy as np

        return np.zeros(10)

    fake.predict_next_distribution.side_effect = zero_pred

    # Monkeypatch LMTADTeacher constructor in module so it returns our fake
    monkeypatch.setattr(
        "tools.evaluate_lmtad_spatial_abnormal.LMTADTeacher", lambda **kwargs: fake
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Write a small CSV with a generated trajectory
        traj_file = tmpdir / "test_spatial_abnormal.csv"
        import pandas as pd

        df = pd.DataFrame({"gene_trace_road_id": ["[2, 1]"]})
        df.to_csv(traj_file, index=False)

        # Create a minimal lmtad checkpoint path (not used because of monkeypatch)
        fake_checkpoint = tmpdir / "ckpt_best.pt"
        fake_checkpoint.write_text("fake")

        # Provide an identity road_to_token mapping to skip expensive GridMapper
        import numpy as np

        road_to_token_override = np.arange(100, dtype=np.int64)

        result = evaluate_spatial_abnormal_trajectories(
            trajectory_file=traj_file,
            lmtad_checkpoint=fake_checkpoint,
            source_eval_dir=tmpdir,
            dataset="porto_hoser",
            device="cpu",
            batch_size=1,
            lmtad_repo=tmpdir,
            eval_config=None,
            max_duplicate_ratio=1.0,
            min_prob=1e-6,
            road_to_token_override=road_to_token_override,
        )

        assert "inf_handling" in result
        stats = result["inf_handling"]
        assert stats["min_prob"] == 1e-6
        # At least the presence of clipped / zero-nan counts should be recorded
        assert "clipped_count" in stats
        # Basic check that log_perplexity entries are finite or marked as failure
        perps = [t["log_perplexity"] for t in result["trajectories"]]
        assert all(np.isfinite(x) or x == float("inf") for x in perps)
