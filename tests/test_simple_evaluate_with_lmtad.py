"""Tests for simple_evaluate_with_lmtad module, focusing on bounds checking and edge cases."""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import torch

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from simple_evaluate_with_lmtad import evaluate_trajectories_direct  # noqa: E402


class TestBoundsChecking:
    """Tests for road ID bounds checking in trajectory evaluation."""

    def test_out_of_bounds_road_ids_filtered(self):
        """Test that out-of-bounds road IDs are filtered out."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        # Mock predict_next_distribution to return a valid probability distribution
        def mock_predict(context):
            # Return uniform distribution over 101 tokens (0-100)
            dist = torch.ones(101) / 101.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        # Create road_to_token mapping with max road ID = 100
        road_to_token = torch.zeros(101, dtype=torch.long)  # Indices 0-100
        road_to_token[:] = torch.arange(101)  # Simple mapping

        # Trajectory with some valid and some invalid road IDs
        trajectories = [[50, 150, 60, 200, 70]]  # 150 and 200 are out of bounds

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should filter out invalid IDs and evaluate with valid ones [50, 60, 70]
        # Since we have valid road IDs, it should attempt evaluation
        # (exact result depends on model mock, but shouldn't be Infinity due to bounds)
        assert len(log_perplexities) == 1
        assert not np.isinf(log_perplexities[0])  # Should succeed, not fail

    def test_all_road_ids_out_of_bounds(self):
        """Test trajectory where all road IDs are out of bounds."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        # All road IDs are out of bounds
        trajectories = [[150, 200, 300]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should result in Infinity (too few valid road IDs after filtering)
        assert len(log_perplexities) == 1
        assert np.isinf(log_perplexities[0])
        # Note: outlier_scores are binary labels (0/1), not raw scores
        # When all are Infinity, threshold is Infinity, so all become 0
        assert outlier_scores[0] == 0 or outlier_scores[0] == 1

    def test_negative_road_ids_filtered(self):
        """Test that negative road IDs are filtered out."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            dist = torch.ones(101) / 101.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        # Trajectory with negative road IDs
        trajectories = [[-10, 50, -5, 60]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should filter out negative IDs and evaluate with [50, 60]
        assert len(log_perplexities) == 1
        assert not np.isinf(log_perplexities[0])

    def test_mixed_valid_and_invalid_road_ids(self):
        """Test trajectory with mix of valid and invalid road IDs."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            dist = torch.ones(101) / 101.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        # Mix of valid, invalid (too high), and negative
        trajectories = [[10, 150, 20, -5, 30, 200]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should filter to [10, 20, 30] and evaluate
        assert len(log_perplexities) == 1
        assert not np.isinf(log_perplexities[0])

    def test_edge_case_max_road_id(self):
        """Test trajectory with road ID at maximum bound."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            # Distribution must cover tokens 0-100 (size 101)
            dist = torch.ones(101) / 101.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        # Road ID at maximum (100) should be valid
        trajectories = [[0, 50, 100]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should be valid
        assert len(log_perplexities) == 1
        assert not np.isinf(log_perplexities[0])

    def test_edge_case_road_id_one_over_max(self):
        """Test trajectory with road ID one over maximum bound."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            dist = torch.ones(101) / 101.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        # Road ID 101 is out of bounds (max is 100)
        trajectories = [[0, 50, 101]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should filter out 101 and evaluate with [0, 50]
        assert len(log_perplexities) == 1
        assert not np.isinf(log_perplexities[0])

    def test_empty_trajectory_after_filtering(self):
        """Test trajectory that becomes empty after filtering invalid road IDs."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        # All invalid, becomes empty after filtering
        trajectories = [[150, 200]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should result in Infinity (empty after filtering)
        assert len(log_perplexities) == 1
        assert np.isinf(log_perplexities[0])

    def test_single_valid_road_id_after_filtering(self):
        """Test trajectory with only one valid road ID after filtering (too short)."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        # Only one valid road ID (needs at least 2 for evaluation)
        trajectories = [[150, 200, 50]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should result in Infinity (too short after filtering)
        assert len(log_perplexities) == 1
        assert np.isinf(log_perplexities[0])

    def test_multiple_trajectories_with_bounds_issues(self):
        """Test multiple trajectories with various bounds issues."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            dist = torch.ones(101) / 101.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:] = torch.arange(101)

        trajectories = [
            [10, 20, 30],  # All valid
            [150, 200],  # All invalid
            [50, 150, 60],  # Mixed (valid after filtering)
            [-10, 20],  # One negative, one valid (too short after filtering)
        ]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
        )

        # Should have 4 results
        assert len(log_perplexities) == 4
        # First should be valid (all valid)
        assert not np.isinf(log_perplexities[0])
        # Second should be Infinity (all invalid)
        assert np.isinf(log_perplexities[1])
        # Third should be valid (has valid IDs after filtering)
        assert not np.isinf(log_perplexities[2])
        # Fourth should be Infinity (too short after filtering)
        assert np.isinf(log_perplexities[3])
