import numpy as np
import torch

from simple_evaluate_with_lmtad import evaluate_trajectories_direct


class FakeTeacher:
    def __init__(self, vocab_size=10):
        self._vocab = vocab_size

    def sot_token(self):
        return None

    def predict_next_distribution(self, history_tokens):
        # Always zero distribution (simulate underflow/zero prob for target)
        V = self._vocab
        return np.zeros(V, dtype=float)


def test_evaluate_trajectories_min_prob_clamp():
    model = FakeTeacher(vocab_size=10)
    # identity mapping: road ID == token
    road_to_token = torch.arange(20, dtype=torch.long)
    device = "cpu"
    trajectories = [[2, 1]]  # predict token 1 which will have 0 prob

    # Default min_prob 1e-6 ensures log_prob finite
    perps, outliers, seg_logs, stats = evaluate_trajectories_direct(
        trajectories,
        model,
        road_to_token,
        device,
        batch_size=1,
        vocab_size=10,
        return_segment_perplexity=True,
        min_prob=1e-6,
        collect_stats=True,
    )
    assert np.isfinite(perps[0]), "Perplexity should be finite after clamping"
    assert stats.get("clipped_count", 0) >= 1

    # min_prob large => lower perplexity
    perps2, outliers2, seg_logs2, stats2 = evaluate_trajectories_direct(
        trajectories,
        model,
        road_to_token,
        device,
        batch_size=1,
        vocab_size=10,
        return_segment_perplexity=True,
        min_prob=1e-2,
        collect_stats=True,
    )
    assert perps2[0] < perps[0]

    # min_prob zero (no clamping) -> log prob log(0) -> numerically we expect -inf (this may be masked by prior +1e-10)
    perps3, outliers3, seg_logs3, stats3 = evaluate_trajectories_direct(
        trajectories,
        model,
        road_to_token,
        device,
        batch_size=1,
        vocab_size=10,
        return_segment_perplexity=True,
        min_prob=0.0,
        collect_stats=True,
    )
    # It should succeed, but perps might be large. At minimum check that clipped_count is 0
    assert stats3.get("clipped_count", 0) == 0
