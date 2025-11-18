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


class TestVocabSizeTokenValidation:
    """Tests for vocab_size-based token validation in trajectory evaluation."""

    def test_out_of_bounds_tokens_filtered(self):
        """Test that tokens exceeding vocab_size are filtered out."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            # Return uniform distribution over vocab_size tokens (0-99)
            dist = torch.ones(100) / 100.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        # Create road_to_token mapping that produces tokens 0-150
        # But vocab_size is only 100, so tokens 100-150 are invalid
        road_to_token = torch.zeros(101, dtype=torch.long)
        road_to_token[:50] = torch.arange(50)  # Roads 0-49 map to tokens 0-49 (valid)
        road_to_token[50:] = torch.arange(50, 101)  # Roads 50-100 map to tokens 50-100
        # Make some roads map to tokens > vocab_size by extending the tensor
        # We'll create a larger mapping for testing
        road_to_token_large = torch.zeros(151, dtype=torch.long)
        road_to_token_large[:50] = torch.arange(50)  # Valid tokens 0-49
        road_to_token_large[50:] = torch.arange(
            50, 151
        )  # Tokens 50-150 (50-99 valid, 100-150 invalid)

        trajectories = [
            [10, 50, 20, 100, 30]
        ]  # Road 50 and 100 will produce tokens 50 and 100
        # Token 50 is valid, token 100 is invalid (>= vocab_size)

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token_large,
            device="cpu",
            batch_size=128,
            vocab_size=100,  # Vocab size is 100, so tokens >= 100 are invalid
        )

        # Should filter out invalid tokens and evaluate with valid ones
        assert len(log_perplexities) == 1
        # Should succeed if enough valid tokens remain (tokens 10, 20, 30, 50 are valid)
        assert not np.isinf(log_perplexities[0])

    def test_all_tokens_out_of_vocab_bounds(self):
        """Test trajectory where all tokens exceed vocab_size."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        # All roads map to tokens >= vocab_size
        road_to_token = torch.zeros(10, dtype=torch.long)
        road_to_token[:] = torch.arange(100, 110)  # All tokens are 100-109

        trajectories = [[0, 1, 2, 3, 4]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
            vocab_size=100,  # All tokens 100-109 are invalid
        )

        # Should result in Infinity (all tokens invalid)
        assert len(log_perplexities) == 1
        assert np.isinf(log_perplexities[0])

    def test_target_token_exceeds_vocab_size_during_computation(self):
        """Test that target tokens exceeding vocab_size are caught during perplexity computation."""
        mock_model = MagicMock()
        # Use SOT token that's within vocab_size for this test
        mock_model.sot_token.return_value = 99  # Valid SOT token

        def mock_predict(context):
            # Return distribution with vocab_size=100
            dist = torch.ones(100) / 100.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        # Create mapping where tokens are valid initially, but we'll manually create
        # a scenario where a token exceeds vocab_size after SOT is added
        # Actually, the validation happens before SOT is added, so let's test a different scenario:
        # Create a case where the token validation passes, but during perplexity computation
        # we encounter an issue (though this is less likely since we validate first)

        # Instead, test that tokens are properly filtered before evaluation
        road_to_token = torch.zeros(10, dtype=torch.long)
        road_to_token[:5] = torch.arange(5)  # First 5 roads -> tokens 0-4 (valid)
        road_to_token[5:] = torch.arange(
            100, 105
        )  # Next 5 roads -> tokens 100-104 (invalid)

        trajectories = [
            [0, 1, 2, 5, 3]
        ]  # Roads: 0,1,2 -> tokens 0,1,2 (valid), 5 -> token 100 (invalid), 3 -> token 3 (valid)

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
            vocab_size=100,
        )

        # Should filter out invalid token 100, leaving [0, 1, 2, 3] which is valid
        assert len(log_perplexities) == 1
        # After filtering, we have valid tokens, so it should succeed
        assert not np.isinf(log_perplexities[0])

    def test_target_token_exceeds_pred_dist_size(self):
        """Test that target tokens exceeding prediction distribution size are caught."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 0  # Valid SOT token

        def mock_predict(context):
            # Return distribution smaller than vocab_size (edge case)
            # This simulates a model that returns a smaller distribution than expected
            dist = torch.ones(50) / 50.0  # Only 50 tokens, but vocab_size is 100
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        # Create tokens that pass vocab_size validation (0-99) but exceed pred_dist size (50)
        road_to_token = torch.zeros(10, dtype=torch.long)
        road_to_token[:5] = torch.arange(5)  # Tokens 0-4 (within pred_dist size 50)
        road_to_token[5:] = torch.arange(
            50, 55
        )  # Tokens 50-54 (exceed pred_dist size 50)

        trajectories = [
            [0, 1, 2, 5, 3]
        ]  # Tokens: 0,1,2 (valid), 50 (exceeds pred_dist), 3 (valid)

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
            vocab_size=100,  # Tokens 0-99 are valid for vocab_size
        )

        # Should fail when target_token 50 exceeds pred_dist size (50)
        # Note: token 50 is >= 50, so it exceeds pred_dist[50] which has size 50 (indices 0-49)
        assert len(log_perplexities) == 1
        assert np.isinf(log_perplexities[0])

    def test_token_validation_with_vocab_size_none(self):
        """Test that token validation is skipped when vocab_size is None."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            dist = torch.ones(200) / 200.0  # Large distribution
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        # Create tokens that would be invalid if vocab_size=100
        road_to_token = torch.zeros(10, dtype=torch.long)
        road_to_token[:] = torch.arange(100, 110)  # Tokens 100-109

        trajectories = [[0, 1, 2, 3, 4]]

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
            vocab_size=None,  # No validation
        )

        # Should proceed without validation
        assert len(log_perplexities) == 1
        # Result depends on model mock, but should not fail due to vocab_size validation

    def test_mixed_valid_and_invalid_tokens(self):
        """Test trajectory with mix of valid and invalid tokens."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            dist = torch.ones(100) / 100.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        # Mix of valid and invalid tokens
        road_to_token = torch.zeros(10, dtype=torch.long)
        road_to_token[:5] = torch.arange(5)  # Valid tokens 0-4
        road_to_token[5:] = torch.arange(100, 105)  # Invalid tokens 100-104

        trajectories = [[0, 5, 1, 6, 2]]  # Mix: valid, invalid, valid, invalid, valid

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
            vocab_size=100,
        )

        # Should filter to [0, 1, 2] and evaluate
        assert len(log_perplexities) == 1
        assert not np.isinf(log_perplexities[0])

    def test_edge_case_vocab_size_boundary(self):
        """Test trajectory with tokens at vocab_size boundary."""
        mock_model = MagicMock()
        mock_model.sot_token.return_value = 6166

        def mock_predict(context):
            dist = torch.ones(100) / 100.0
            return dist

        mock_model.predict_next_distribution.side_effect = mock_predict

        road_to_token = torch.zeros(10, dtype=torch.long)
        road_to_token[:] = torch.arange(95, 105)  # Tokens 95-104

        trajectories = [[0, 1, 2, 3, 4]]  # Tokens 95-99 are valid, 100-104 are invalid

        log_perplexities, outlier_scores = evaluate_trajectories_direct(
            trajectories=trajectories,
            model=mock_model,
            road_to_token=road_to_token,
            device="cpu",
            batch_size=128,
            vocab_size=100,  # Tokens 0-99 are valid, 100+ are invalid
        )

        # Should filter to tokens 95-99 (first 5 roads) and evaluate
        assert len(log_perplexities) == 1
        assert not np.isinf(log_perplexities[0])


class TestLMTADTeacherVocabSize:
    """Tests for vocab_size() method in LMTADTeacher."""

    def test_vocab_size_from_model_config(self):
        """Test that vocab_size() returns value from model config."""
        from critics.lmtad_teacher import LMTADTeacher
        from unittest.mock import MagicMock

        # Create mock model with config
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.vocab_size = 6167
        mock_model.config = mock_config

        teacher = LMTADTeacher.__new__(LMTADTeacher)
        teacher.model = mock_model

        vocab_size = teacher.vocab_size()
        assert vocab_size == 6167

    def test_vocab_size_from_embedding_weights(self):
        """Test that vocab_size() infers from embedding weights when config unavailable."""
        from critics.lmtad_teacher import LMTADTeacher
        from unittest.mock import MagicMock

        # Create mock model without config but with transformer.wte
        mock_model = MagicMock()
        mock_model.config = None
        mock_transformer = MagicMock()
        mock_wte = MagicMock()
        mock_wte.weight.shape = (6167, 512)  # vocab_size=6167, embedding_dim=512
        mock_transformer.wte = mock_wte
        mock_model.transformer = mock_transformer

        teacher = LMTADTeacher.__new__(LMTADTeacher)
        teacher.model = mock_model

        vocab_size = teacher.vocab_size()
        assert vocab_size == 6167

    def test_vocab_size_returns_none_when_unavailable(self):
        """Test that vocab_size() returns None when cannot be determined."""
        from critics.lmtad_teacher import LMTADTeacher
        from unittest.mock import MagicMock

        # Create mock model without config or transformer
        mock_model = MagicMock()
        mock_model.config = None
        mock_model.transformer = None

        teacher = LMTADTeacher.__new__(LMTADTeacher)
        teacher.model = mock_model

        vocab_size = teacher.vocab_size()
        assert vocab_size is None
