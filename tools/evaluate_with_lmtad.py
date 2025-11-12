"""
LM-TAD Evaluation Module
=======================

Purpose
-------
Evaluate trajectories using LM-TAD teacher model to compute perplexity scores
and classify outliers. Supports both real and generated trajectories.

Key functionality:
1. Load LM-TAD teacher model
2. Convert trajectories to grid-token format
3. Compute perplexity scores in batches
4. Classify outliers based on threshold
5. Save evaluation results

Dependencies:
- LMTADTeacher wrapper (critics.lmtad_teacher)
- GridMapper (critics.grid_mapper)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from critics.lmtad_teacher import LMTADTeacher

logger = logging.getLogger(__name__)


class LMTADDataset(Dataset):
    """Dataset for LM-TAD trajectory evaluation.

    Parameters
    ----------
    trajectory_file : Path
        Path to CSV file with trajectory_tokens column
    """

    def __init__(self, trajectory_file: Path) -> None:
        self.df = pd.read_csv(trajectory_file)
        if "trajectory_tokens" not in self.df.columns:
            raise ValueError("trajectory_tokens column missing from CSV")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.LongTensor:
        """Return grid token sequence for a single trajectory."""
        token_str = self.df.iloc[idx]["trajectory_tokens"]
        tokens = [int(t) for t in token_str.strip().split()]
        return torch.LongTensor(tokens)


def collate_sequences(batch: list[torch.Tensor]) -> torch.Tensor:
    """Collate variable-length sequences into a padded batch.

    All sequences must start with SOT (0) and end with EOT (1).
    """
    # Verify SOT/EOT tokens
    for seq in batch:
        if seq[0] != 0 or seq[-1] != 1:
            raise ValueError(
                "All sequences must start with SOT (0) and end with EOT (1)"
            )

    # Pad to max length
    max_len = max(len(seq) for seq in batch)
    padded = []
    for seq in batch:
        if len(seq) < max_len:
            # Pad with EOT token (1)
            padding = torch.ones(max_len - len(seq), dtype=torch.long)
            padded.append(torch.cat([seq, padding]))
        else:
            padded.append(seq)

    return torch.stack(padded)


def load_lmtad_evaluator(
    checkpoint: Path, repo_path: Path, dataset: str, device: str
) -> LMTADTeacher:
    """Load LM-TAD teacher model for evaluation.

    Parameters
    ----------
    checkpoint : Path
        Path to teacher checkpoint (.pt file)
    repo_path : Path
        Path to LM-TAD repository root
    dataset : str
        Dataset name (e.g., "porto_hoser")
    device : str
        Torch device to use

    Returns
    -------
    LMTADTeacher
        Initialized teacher model wrapper
    """
    logger.info(f"Loading LM-TAD model from {checkpoint}...")
    try:
        model = LMTADTeacher(
            repo_path=str(repo_path),
            ckpt_path=str(checkpoint),
            device=device,
            dtype="float16",  # Use AMP for memory efficiency
            window=256,  # Large window for evaluation
        )
        logger.info("LM-TAD model loaded successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load LM-TAD model: {e}") from e


def create_lmtad_dataloader(trajectory_file: Path, batch_size: int = 128) -> DataLoader:
    """Create DataLoader for LM-TAD trajectory evaluation.

    Parameters
    ----------
    trajectory_file : Path
        Path to CSV file with trajectory_tokens column
    batch_size : int
        Batch size for evaluation

    Returns
    -------
    DataLoader
        DataLoader configured for LM-TAD batch evaluation
    """
    try:
        dataset = LMTADDataset(trajectory_file)
        logger.info(f"Loaded {len(dataset):,} trajectories from {trajectory_file}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep order for results
            collate_fn=collate_sequences,
            pin_memory=True,
            num_workers=4,
        )
        return loader
    except Exception as e:
        raise RuntimeError(f"Failed to create LM-TAD dataloader: {e}") from e


@torch.no_grad()
def compute_perplexities(
    model: LMTADTeacher, dataloader: DataLoader, device: str
) -> torch.Tensor:
    """Compute perplexity scores for trajectories in batches.

    Uses teacher's forward pass with temperature=1.0 since we're evaluating,
    not doing knowledge distillation.

    Parameters
    ----------
    model : LMTADTeacher
        Initialized teacher model
    dataloader : DataLoader
        DataLoader returning padded token sequences
    device : str
        Torch device string (e.g., "cuda:0")

    Returns
    -------
    torch.Tensor
        Shape (N,) array of log perplexity scores
    """
    logger.info("Computing perplexity scores...")
    all_scores = []

    for batch in tqdm(dataloader, desc="Evaluating batches"):
        batch = batch.to(device)

        # Use teacher's forward pass
        logits, _ = model.model(batch)  # (B, T, V)

        # Get target tokens (shifted left)
        targets = batch[:, 1:]  # Remove SOT
        logits = logits[:, :-1]  # Remove last prediction

        # Compute per-token negative log likelihood
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, T-1, V)
        nll = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(
            -1
        )  # (B, T-1)

        # Find sequence lengths (excluding padding)
        seq_lens = (batch != 1).sum(dim=1) - 1  # -1 for SOT

        # Average NLL over sequence length (excluding padding)
        mean_nll = []
        for i, seq_len in enumerate(seq_lens):
            mean_nll.append(nll[i, : seq_len - 1].mean())  # -1 for EOT

        # Append batch scores
        mean_nll = torch.stack(mean_nll)  # (B,)
        all_scores.append(mean_nll.cpu())

    # Combine all batches
    return torch.cat(all_scores)  # (N,)


def classify_outliers(
    perplexities: torch.Tensor, threshold: Optional[float] = None
) -> Tuple[torch.Tensor, float]:
    """Classify trajectories as normal/outlier based on perplexity scores.

    Parameters
    ----------
    perplexities : torch.Tensor
        Shape (N,) array of perplexity scores
    threshold : Optional[float]
        Perplexity threshold for outlier detection
        If None, auto-compute as mean + 2*std

    Returns
    -------
    outlier_flags : torch.Tensor
        Shape (N,) boolean array, True = outlier
    threshold : float
        Threshold value used for classification
    """
    # Convert to numpy for stats
    scores = perplexities.numpy()

    # Auto-compute threshold if not provided
    if threshold is None:
        mean = scores.mean()
        std = scores.std()
        threshold = float(mean + 2.0 * std)  # Convert to Python float
        logger.info(
            f"Auto-computed threshold {threshold:.4f} "
            f"(mean={float(mean):.4f} + 2*std={float(2 * std):.4f})"
        )

    # Classify outliers
    outlier_flags = scores > threshold
    outlier_rate = outlier_flags.mean()
    logger.info(
        f"Classified {int(outlier_flags.sum())} outliers "
        f"({float(outlier_rate):.2%}) using threshold {threshold:.4f}"
    )

    return torch.from_numpy(outlier_flags), threshold


def save_evaluation_results(
    perplexities: torch.Tensor,
    outlier_flags: torch.Tensor,
    threshold: float,
    output_dir: Path,
) -> None:
    """Save evaluation results as TSV and JSON files.

    Parameters
    ----------
    perplexities : torch.Tensor
        Shape (N,) array of perplexity scores
    outlier_flags : torch.Tensor
        Shape (N,) boolean array of outlier classifications
    threshold : float
        Perplexity threshold used
    output_dir : Path
        Directory to save output files

    Results are saved as:
    - evaluation_results.tsv: Per-trajectory perplexity and classification
    - outlier_stats.json: Summary statistics and threshold
    """
    # Create results directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-trajectory results
    results_df = pd.DataFrame(
        {
            "perplexity": perplexities.numpy(),
            "is_outlier": outlier_flags.numpy(),
        }
    )
    results_path = output_dir / "evaluation_results.tsv"
    results_df.to_csv(results_path, sep="\t", index=True)
    logger.info(f"Saved per-trajectory results to {results_path}")

    # Save summary statistics
    stats = {
        "num_trajectories": len(perplexities),
        "perplexity_threshold": float(threshold),
        "perplexity_stats": {
            "mean": float(perplexities.mean()),
            "std": float(perplexities.std()),
            "min": float(perplexities.min()),
            "max": float(perplexities.max()),
        },
        "outlier_rate": float(outlier_flags.float().mean()),
        "num_outliers": int(outlier_flags.sum()),
    }
    stats_path = output_dir / "outlier_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved summary statistics to {stats_path}")


def evaluate_with_lmtad(
    trajectory_file: Path,
    vocab_file: Path,
    lmtad_checkpoint: Path,
    lmtad_repo_path: Path,
    dataset: str,
    output_dir: Path,
    perplexity_threshold: Optional[float] = None,
    device: str = "cuda",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Evaluate trajectories using LM-TAD teacher model.

    The main evaluation pipeline that:
    1. Loads LM-TAD teacher model
    2. Creates dataloader for batch processing
    3. Computes perplexity scores
    4. Classifies outliers
    5. Saves results and returns summary DataFrame

    Parameters
    ----------
    trajectory_file : Path
        Path to CSV file with trajectory_tokens column
    vocab_file : Path
        Path to vocab.json with grid config
    lmtad_checkpoint : Path
        Path to teacher checkpoint (.pt file)
    lmtad_repo_path : Path
        Path to LM-TAD repository root
    dataset : str
        Dataset name (e.g., "porto_hoser")
    output_dir : Path
        Directory to save results
    perplexity_threshold : Optional[float]
        Override threshold for outlier detection (auto-compute if None)
    device : str
        Device to use for model (e.g., "cuda:0")
    batch_size : int
        Batch size for evaluation

    Returns
    -------
    pd.DataFrame
        Results DataFrame with perplexity scores and classifications
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load teacher model
        model = load_lmtad_evaluator(
            checkpoint=lmtad_checkpoint,
            repo_path=lmtad_repo_path,
            dataset=dataset,
            device=device,
        )

        # 2. Create dataloader
        loader = create_lmtad_dataloader(
            trajectory_file=trajectory_file,
            batch_size=batch_size,
        )

        # 3. Compute perplexity scores
        perplexities = compute_perplexities(
            model=model,
            dataloader=loader,
            device=device,
        )

        # 4. Classify outliers
        outlier_flags, threshold = classify_outliers(
            perplexities=perplexities,
            threshold=perplexity_threshold,
        )

        # 5. Save results
        save_evaluation_results(
            perplexities=perplexities,
            outlier_flags=outlier_flags,
            threshold=threshold,
            output_dir=output_dir,
        )

        # Return results DataFrame
        results = pd.DataFrame(
            {
                "perplexity": perplexities.numpy(),
                "is_outlier": outlier_flags.numpy(),
            }
        )
        logger.info(
            f"✅ Evaluation complete: {len(results):,} trajectories processed, "
            f"{outlier_flags.sum():,} outliers identified "
            f"({outlier_flags.float().mean():.2%})"
        )
        return results

    except Exception as e:
        logger.exception("LM-TAD evaluation failed")
        raise RuntimeError(f"Failed to evaluate trajectories: {e}") from e
