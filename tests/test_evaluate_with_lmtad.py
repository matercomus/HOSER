"""
Tests for LM-TAD Evaluation Module

This module tests the evaluation functionality using LM-TAD teacher model,
including perplexity computation and outlier classification.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from tools.evaluate_with_lmtad import (
    LMTADDataset,
    classify_outliers,
    collate_sequences,
    compute_perplexities,
    create_lmtad_dataloader,
    evaluate_with_lmtad,
    load_lmtad_evaluator,
    save_evaluation_results,
)


@pytest.fixture
def sample_trajectory_data():
    """Create sample trajectory data for testing"""
    return pd.DataFrame(
        {
            "trajectory_id": [1, 2, 3, 4, 5],
            "trajectory_tokens": [
                "0 10 20 30 1",  # SOT=0, EOT=1
                "0 15 25 35 45 1",
                "0 12 22 1",
                "0 18 28 38 48 58 1",
                "0 11 21 31 41 1",
            ],
        }
    )


@pytest.fixture
def sample_perplexities():
    """Create sample perplexity scores for testing"""
    return torch.tensor([1.5, 2.0, 3.5, 1.8, 2.7])


def test_lmtad_dataset(sample_trajectory_data, tmp_path):
    """Test LMTADDataset class"""
    # Create temporary CSV file
    csv_file = tmp_path / "trajectories.csv"
    sample_trajectory_data.to_csv(csv_file, index=False)

    # Test dataset creation
    dataset = LMTADDataset(csv_file)

    assert len(dataset) == 5
    assert dataset.df.equals(sample_trajectory_data)

    # Test item retrieval
    tokens = dataset[0]
    assert isinstance(tokens, torch.LongTensor)
    expected_tokens = torch.LongTensor([0, 10, 20, 30, 1])
    torch.testing.assert_close(tokens, expected_tokens)

    # Test missing column raises ValueError
    bad_df = sample_trajectory_data.drop(columns=["trajectory_tokens"])
    bad_csv = tmp_path / "bad_trajectories.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="trajectory_tokens column missing"):
        LMTADDataset(bad_csv)


def test_collate_sequences():
    """Test sequence collation function"""
    # Test normal sequences
    sequences = [
        torch.LongTensor([0, 10, 20, 1]),  # Length 4
        torch.LongTensor([0, 15, 25, 35, 1]),  # Length 5
        torch.LongTensor([0, 12, 1]),  # Length 3
    ]

    batched = collate_sequences(sequences)

    assert isinstance(batched, torch.Tensor)
    assert batched.shape == (3, 5)  # 3 sequences, max length 5
    assert batched[0, 0] == 0  # SOT
    assert batched[0, 3] == 1  # EOT (padded at index 3)

    # Test missing SOT/EOT raises ValueError
    bad_sequences = [
        torch.LongTensor([10, 20, 1]),  # Missing SOT
        torch.LongTensor([0, 15, 25]),
    ]  # Missing EOT

    with pytest.raises(ValueError, match="must start with SOT"):
        collate_sequences(bad_sequences)


@patch("tools.evaluate_with_lmtad.LMTADTeacher")
def test_load_lmtad_evaluator(mock_lmtad_teacher):
    """Test loading LM-TAD evaluator"""
    mock_model = Mock()
    mock_lmtad_teacher.return_value = mock_model

    checkpoint = Path("/fake/checkpoint.pt")
    repo_path = Path("/fake/repo")
    dataset = "Beijing"
    device = "cuda:0"

    model = load_lmtad_evaluator(checkpoint, repo_path, dataset, device)

    assert model == mock_model
    mock_lmtad_teacher.assert_called_once_with(
        repo_path=str(repo_path),
        ckpt_path=str(checkpoint),
        device=device,
        dtype="float16",
        window=256,
    )

    # Test exception handling
    mock_lmtad_teacher.side_effect = Exception("Failed to load")
    with pytest.raises(RuntimeError, match="Failed to load LM-TAD model"):
        load_lmtad_evaluator(checkpoint, repo_path, dataset, device)


def test_create_lmtad_dataloader(sample_trajectory_data, tmp_path):
    """Test creation of LM-TAD dataloader"""
    # Create temporary CSV file
    csv_file = tmp_path / "trajectories.csv"
    sample_trajectory_data.to_csv(csv_file, index=False)

    # Test successful dataloader creation
    dataloader = create_lmtad_dataloader(csv_file, batch_size=2)

    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == 2
    assert len(dataloader.dataset) == 5

    # Test missing file raises RuntimeError
    with pytest.raises(RuntimeError, match="Failed to create LM-TAD dataloader"):
        create_lmtad_dataloader(tmp_path / "nonexistent.csv")


@patch("tools.evaluate_with_lmtad.torch")
def test_compute_perplexities(mock_torch, sample_trajectory_data, tmp_path):
    """Test computation of perplexity scores"""
    # Mock the model forward pass and tensor operations
    mock_logits = torch.randn(2, 4, 100)  # (batch_size, seq_len, vocab_size)
    mock_model = Mock()
    mock_model.model.return_value = (mock_logits, None)

    # Create a simple dataloader
    csv_file = tmp_path / "trajectories.csv"
    sample_trajectory_data.head(2).to_csv(csv_file, index=False)
    dataloader = create_lmtad_dataloader(csv_file, batch_size=2)

    # Mock torch operations in compute_perplexities
    mock_torch.log_softmax.return_value = torch.log_softmax(mock_logits, dim=-1)
    mock_torch.gather.return_value = torch.randn(2, 3, 1)  # (batch_size, seq_len-1, 1)
    mock_torch.stack.return_value = torch.randn(2)

    # Test perplexity computation
    perplexities = compute_perplexities(mock_model, dataloader, "cuda:0")

    assert isinstance(perplexities, torch.Tensor)
    assert len(perplexities) == 2  # Two trajectories


def test_classify_outliers_auto_threshold(sample_perplexities):
    """Test outlier classification with auto-computed threshold"""
    # Test with auto-threshold
    outlier_flags, threshold = classify_outliers(sample_perplexities)

    assert isinstance(outlier_flags, torch.Tensor)
    assert outlier_flags.dtype == torch.bool
    assert len(outlier_flags) == len(sample_perplexities)
    assert isinstance(threshold, float)

    # Verify threshold calculation (mean + 2*std)
    perplexity_np = sample_perplexities.numpy()
    expected_threshold = perplexity_np.mean() + 2 * perplexity_np.std()
    np.testing.assert_almost_equal(threshold, expected_threshold, decimal=5)


def test_classify_outliers_custom_threshold(sample_perplexities):
    """Test outlier classification with custom threshold"""
    # Test with custom threshold
    custom_threshold = 2.5
    outlier_flags, threshold = classify_outliers(
        sample_perplexities, threshold=custom_threshold
    )

    assert threshold == custom_threshold
    # Only scores > 2.5 should be outliers: [3.5, 2.7] -> 2 outliers
    assert outlier_flags.sum() == 2


def test_save_evaluation_results(sample_perplexities, tmp_path):
    """Test saving evaluation results"""
    outlier_flags = torch.tensor([False, False, True, False, True])
    threshold = 2.5

    output_dir = tmp_path / "results"
    save_evaluation_results(sample_perplexities, outlier_flags, threshold, output_dir)

    # Verify directory was created
    assert output_dir.exists()

    # Verify TSV file
    tsv_file = output_dir / "evaluation_results.tsv"
    assert tsv_file.exists()
    results_df = pd.read_csv(tsv_file, sep="\t", index_col=0)
    assert len(results_df) == 5
    assert "perplexity" in results_df.columns
    assert "is_outlier" in results_df.columns

    # Verify JSON file
    json_file = output_dir / "outlier_stats.json"
    assert json_file.exists()
    with open(json_file, "r") as f:
        stats = json.load(f)
    assert "num_trajectories" in stats
    assert "perplexity_threshold" in stats
    assert "outlier_rate" in stats
    assert stats["num_trajectories"] == 5
    assert stats["num_outliers"] == 2


@patch("tools.evaluate_with_lmtad.load_lmtad_evaluator")
@patch("tools.evaluate_with_lmtad.create_lmtad_dataloader")
@patch("tools.evaluate_with_lmtad.compute_perplexities")
@patch("tools.evaluate_with_lmtad.classify_outliers")
@patch("tools.evaluate_with_lmtad.save_evaluation_results")
def test_evaluate_with_lmtad(
    mock_save,
    mock_classify,
    mock_compute,
    mock_create_loader,
    mock_load_evaluator,
    sample_perplexities,
    tmp_path,
):
    """Test the main evaluation function"""
    # Set up mocks
    mock_model = Mock()
    mock_load_evaluator.return_value = mock_model

    mock_dataloader = Mock()
    mock_create_loader.return_value = mock_dataloader

    mock_compute.return_value = sample_perplexities
    mock_classify.return_value = (
        torch.tensor([False, False, True, False, True]),
        2.5,
    )

    # Test files
    trajectory_file = tmp_path / "trajectories.csv"
    vocab_file = tmp_path / "vocab.json"
    checkpoint = tmp_path / "checkpoint.pt"
    repo_path = tmp_path / "lmtad_repo"
    output_dir = tmp_path / "results"

    # Test successful evaluation
    results_df = evaluate_with_lmtad(
        trajectory_file=trajectory_file,
        vocab_file=vocab_file,
        lmtad_checkpoint=checkpoint,
        lmtad_repo_path=repo_path,
        dataset="Beijing",
        output_dir=output_dir,
    )

    # Verify all steps were called
    mock_load_evaluator.assert_called_once()
    mock_create_loader.assert_called_once()
    mock_compute.assert_called_once_with(
        model=mock_model, dataloader=mock_dataloader, device="cuda"
    )
    mock_classify.assert_called_once_with(
        perplexities=sample_perplexities, threshold=None
    )
    mock_save.assert_called_once()

    # Verify results DataFrame
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 5
    assert set(results_df.columns) == {"perplexity", "is_outlier"}

    # Test exception handling
    mock_compute.side_effect = Exception("Evaluation failed")
    with pytest.raises(RuntimeError, match="Failed to evaluate trajectories"):
        evaluate_with_lmtad(
            trajectory_file=trajectory_file,
            vocab_file=vocab_file,
            lmtad_checkpoint=checkpoint,
            lmtad_repo_path=repo_path,
            dataset="Beijing",
            output_dir=output_dir,
        )


def test_evaluate_with_lmtad_custom_params(sample_perplexities, tmp_path):
    """Test evaluation with custom parameters"""
    with (
        patch("tools.evaluate_with_lmtad.load_lmtad_evaluator") as mock_load,
        patch("tools.evaluate_with_lmtad.create_lmtad_dataloader") as mock_create,
        patch("tools.evaluate_with_lmtad.compute_perplexities") as mock_compute,
        patch("tools.evaluate_with_lmtad.classify_outliers") as mock_classify,
        patch("tools.evaluate_with_lmtad.save_evaluation_results"),
    ):
        # Set up mocks
        mock_load.return_value = Mock()
        mock_create.return_value = Mock()
        mock_compute.return_value = sample_perplexities
        mock_classify.return_value = (torch.tensor([True, False]), 3.0)

        trajectory_file = tmp_path / "trajectories.csv"
        vocab_file = tmp_path / "vocab.json"
        checkpoint = tmp_path / "checkpoint.pt"
        repo_path = tmp_path / "lmtad_repo"
        output_dir = tmp_path / "results"

        # Test with custom parameters
        evaluate_with_lmtad(
            trajectory_file=trajectory_file,
            vocab_file=vocab_file,
            lmtad_checkpoint=checkpoint,
            lmtad_repo_path=repo_path,
            dataset="Beijing",
            output_dir=output_dir,
            perplexity_threshold=3.0,
            device="cuda:1",
            batch_size=64,
        )

        # Verify custom parameters were used
        mock_compute.assert_called_once()
        mock_classify.assert_called_once_with(
            perplexities=sample_perplexities, threshold=3.0
        )


if __name__ == "__main__":
    pytest.main([__file__])
