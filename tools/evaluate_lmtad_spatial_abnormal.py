#!/usr/bin/env python3
"""
Evaluate Generated Trajectories with LM-TAD and Classify Spatial Abnormality Types

This script evaluates generated trajectories using LM-TAD and classifies them into
spatial abnormality types (route switch, detour, non-outlier) based on perplexity thresholds.

Usage:
    uv run python tools/evaluate_lmtad_spatial_abnormal.py \\
        --trajectory-file gene_abnormal_lmtad_spatial/porto_hoser/seed42/vanilla_spatial_abnormal.csv \\
        --lmtad-checkpoint /path/to/ckpt_best.pt \\
        --source-eval-dir /path/to/lmtad/eval \\
        --dataset porto_hoser \\
        --output eval_lmtad_spatial/porto_hoser/vanilla_spatial_evaluation.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_evaluate_with_lmtad import (  # noqa: E402
    load_hoser_trajectories,
    evaluate_trajectories_direct,
    load_road_centroids,
)
from critics.lmtad_teacher import LMTADTeacher  # noqa: E402
from critics.grid_mapper import GridMapper, GridConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_source_statistics(source_eval_dir: Path) -> Dict:
    """Load source evaluation statistics from EVALUATION_ANALYSIS.md or TSV files

    Args:
        source_eval_dir: Path to LM-TAD source evaluation directory

    Returns:
        Dictionary with perplexity statistics for each type
    """
    logger.info(f"📂 Loading source statistics from {source_eval_dir}")

    # Try to load from EVALUATION_ANALYSIS.md first
    analysis_file = source_eval_dir / "EVALUATION_ANALYSIS.md"
    if analysis_file.exists():
        # Parse markdown to extract statistics
        with open(analysis_file, "r") as f:
            content = f.read()

        # Extract statistics from markdown table
        stats = {}
        for line in content.split("\n"):
            if "**Non-outlier**" in line or "Non-outlier" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    try:
                        stats["non_outlier_mean"] = float(parts[2].strip())
                        stats["non_outlier_std"] = float(parts[3].strip())
                    except (ValueError, IndexError):
                        pass
            elif "**Route Switch**" in line or "Route Switch" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    try:
                        stats["route_switch_mean"] = float(parts[2].strip())
                        stats["route_switch_std"] = float(parts[3].strip())
                    except (ValueError, IndexError):
                        pass
            elif "**Detour**" in line or "Detour" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    try:
                        stats["detour_mean"] = float(parts[2].strip())
                        stats["detour_std"] = float(parts[3].strip())
                    except (ValueError, IndexError):
                        pass

        if stats:
            logger.info("✅ Loaded statistics from EVALUATION_ANALYSIS.md")
            return stats

    # Fallback: compute from TSV files
    tsv_files = list(source_eval_dir.glob("ckpt_best_outliers_*.tsv"))
    if tsv_files:
        logger.info(f"  Computing statistics from {len(tsv_files)} TSV files")
        tsv_file = tsv_files[0]  # Use first available

        try:
            df = pd.read_csv(tsv_file, sep="\t")
            stats = {}

            # Compute statistics by outlier type
            for outlier_type in ["non outlier", "route switch", "detour"]:
                type_df = df[df["outlier"] == outlier_type]
                if len(type_df) > 0:
                    key = outlier_type.replace(" ", "_")
                    stats[f"{key}_mean"] = float(type_df["log_perplexity"].mean())
                    stats[f"{key}_std"] = float(type_df["log_perplexity"].std())

            if stats:
                logger.info("✅ Computed statistics from TSV file")
                return stats
        except Exception as e:
            logger.warning(f"Failed to compute from TSV: {e}")

    # Default values from Porto evaluation (fallback)
    logger.warning("Using default statistics (Porto dataset)")
    return {
        "non_outlier_mean": 0.3822,
        "non_outlier_std": 0.1249,
        "route_switch_mean": 7.0265,
        "route_switch_std": 1.6068,
        "detour_mean": 8.4132,
        "detour_std": 1.2098,
    }


def classify_spatial_abnormality_type(log_perplexity: float, source_stats: Dict) -> str:
    """Classify trajectory as spatial abnormality type based on log perplexity

    Args:
        log_perplexity: Log perplexity value
        source_stats: Source evaluation statistics

    Returns:
        Classification: "route_switch", "detour", or "non_outlier"
    """
    # Use thresholds based on source statistics
    # Route Switch: mean ~7.03, use range [6.0, 8.0]
    # Detour: mean ~8.41, use range [8.0, 10.0]
    # Non-outlier: mean ~0.38, use range [0.0, 5.0]

    route_switch_mean = source_stats.get("route_switch_mean", 7.03)
    detour_mean = source_stats.get("detour_mean", 8.41)

    # Use mean values to determine thresholds
    # Route switch: between route_switch_mean - 1.0 and route_switch_mean + 1.0
    # Detour: above route_switch_mean + 1.0
    # Non-outlier: below route_switch_mean - 1.0

    if log_perplexity < route_switch_mean - 1.0:
        return "non_outlier"
    elif log_perplexity < detour_mean - 0.5:
        return "route_switch"
    else:
        return "detour"


def evaluate_spatial_abnormal_trajectories(
    trajectory_file: Path,
    lmtad_checkpoint: Path,
    source_eval_dir: Path,
    dataset: str,
    device: str = "cuda:0",
    batch_size: int = 128,
) -> Dict:
    """Evaluate generated trajectories with LM-TAD and classify spatial abnormality types

    Args:
        trajectory_file: Path to generated trajectory CSV file
        lmtad_checkpoint: Path to LM-TAD checkpoint
        source_eval_dir: Path to LM-TAD source evaluation directory
        dataset: Dataset name
        device: CUDA device
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with evaluation results and classifications
    """
    logger.info(f"📂 Loading trajectories from {trajectory_file}")

    # Load trajectories
    trajectories = load_hoser_trajectories(trajectory_file)
    logger.info(f"✅ Loaded {len(trajectories)} trajectories")

    # Load source statistics
    source_stats = load_source_statistics(source_eval_dir)

    # Load LM-TAD teacher
    logger.info(f"📂 Loading LM-TAD teacher from {lmtad_checkpoint}")
    lmtad_repo = Path("/home/matt/Dev/LMTAD")
    model = LMTADTeacher(
        repo_path=str(lmtad_repo),
        ckpt_path=str(lmtad_checkpoint),
        device=device,
        dtype="float16",
        window=256,
    )
    logger.info("✅ LM-TAD teacher loaded")

    # Create grid mapper
    logger.info("📂 Creating grid mapper...")
    from pathlib import Path as PathLib

    data_dir = PathLib("data") / dataset
    roadmap_file = data_dir / "roadmap.geo"
    if not roadmap_file.exists():
        # Try relative to project root
        roadmap_file = (
            PathLib(__file__).parent.parent / "data" / dataset / "roadmap.geo"
        )

    if not roadmap_file.exists():
        raise FileNotFoundError(f"Roadmap file not found: {roadmap_file}")

    road_centroids = load_road_centroids(roadmap_file)
    grid_config = GridConfig(
        min_lat=road_centroids[:, 1].min(),
        max_lat=road_centroids[:, 1].max(),
        min_lng=road_centroids[:, 0].min(),
        max_lng=road_centroids[:, 0].max(),
        grid_size=0.001,
    )
    mapper = GridMapper(
        boundary=grid_config,
        road_centroids=road_centroids,
        verify_hw=None,
    )
    road_to_token = torch.from_numpy(mapper.map_all()).to(device)
    logger.info("✅ Grid mapper created")

    # Evaluate trajectories
    logger.info("🔍 Evaluating trajectories with LM-TAD...")
    log_perplexities, _ = evaluate_trajectories_direct(
        trajectories=trajectories,
        model=model,
        road_to_token=road_to_token,
        device=device,
        batch_size=batch_size,
    )

    # Classify each trajectory
    classifications = []
    for log_perp in log_perplexities:
        if np.isinf(log_perp):
            classifications.append("non_outlier")  # Failed evaluation = non-outlier
        else:
            classifications.append(
                classify_spatial_abnormality_type(log_perp, source_stats)
            )

    # Count by type
    from collections import Counter

    type_counts = Counter(classifications)

    total_trajectories = len(trajectories)
    route_switch_count = type_counts.get("route_switch", 0)
    detour_count = type_counts.get("detour", 0)
    non_outlier_count = type_counts.get("non_outlier", 0)
    spatial_abnormal_count = route_switch_count + detour_count

    # Compute rates
    spatial_abnormality_rate = (
        (spatial_abnormal_count / total_trajectories * 100)
        if total_trajectories > 0
        else 0
    )
    route_switch_rate = (
        (route_switch_count / total_trajectories * 100) if total_trajectories > 0 else 0
    )
    detour_rate = (
        (detour_count / total_trajectories * 100) if total_trajectories > 0 else 0
    )

    # Extract model name from filename
    model_name = trajectory_file.stem.replace("_spatial_abnormal", "")

    # Compute log perplexity statistics
    valid_perplexities = log_perplexities[~np.isinf(log_perplexities)]
    log_perplexity_stats = {}
    if len(valid_perplexities) > 0:
        log_perplexity_stats = {
            "mean": float(np.mean(valid_perplexities)),
            "std": float(np.std(valid_perplexities)),
            "median": float(np.median(valid_perplexities)),
            "min": float(np.min(valid_perplexities)),
            "max": float(np.max(valid_perplexities)),
        }

    result = {
        "model": model_name,
        "dataset": dataset,
        "total_trajectories": total_trajectories,
        "spatial_abnormal_count": spatial_abnormal_count,
        "spatial_abnormality_rate": spatial_abnormality_rate,
        "by_type": {
            "route_switch": {"count": route_switch_count, "rate": route_switch_rate},
            "detour": {"count": detour_count, "rate": detour_rate},
            "non_outlier": {
                "count": non_outlier_count,
                "rate": 100 - spatial_abnormality_rate,
            },
        },
        "log_perplexity_stats": log_perplexity_stats,
        "classifications": classifications,  # Per-trajectory classifications
        "log_perplexities": log_perplexities.tolist(),  # Per-trajectory perplexities
        "source_statistics": source_stats,
    }

    logger.info("✅ Evaluation complete:")
    logger.info(f"   Total trajectories: {total_trajectories}")
    logger.info(
        f"   Spatial abnormal: {spatial_abnormal_count} ({spatial_abnormality_rate:.2f}%)"
    )
    logger.info(f"     Route switch: {route_switch_count} ({route_switch_rate:.2f}%)")
    logger.info(f"     Detour: {detour_count} ({detour_rate:.2f}%)")
    logger.info(
        f"   Non-outlier: {non_outlier_count} ({100 - spatial_abnormality_rate:.2f}%)"
    )

    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate generated trajectories with LM-TAD and classify spatial abnormality types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  uv run python tools/evaluate_lmtad_spatial_abnormal.py \\
    --trajectory-file gene_abnormal_lmtad_spatial/porto_hoser/seed42/vanilla_spatial_abnormal.csv \\
    --lmtad-checkpoint /home/matt/Dev/LMTAD/.../ckpt_best.pt \\
    --source-eval-dir /home/matt/Dev/LMTAD/.../eval \\
    --dataset porto_hoser \\
    --output eval_lmtad_spatial/porto_hoser/vanilla_spatial_evaluation.json
        """,
    )

    parser.add_argument(
        "--trajectory-file",
        type=Path,
        required=True,
        help="Path to generated trajectory CSV file",
    )
    parser.add_argument(
        "--lmtad-checkpoint",
        type=Path,
        required=True,
        help="Path to LM-TAD checkpoint file",
    )
    parser.add_argument(
        "--source-eval-dir",
        type=Path,
        required=True,
        help="Path to LM-TAD source evaluation directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., porto_hoser, Beijing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.trajectory_file.exists():
        logger.error(f"Trajectory file not found: {args.trajectory_file}")
        return 1

    if not args.lmtad_checkpoint.exists():
        logger.error(f"LM-TAD checkpoint not found: {args.lmtad_checkpoint}")
        return 1

    if not args.source_eval_dir.exists():
        logger.error(f"Source eval directory not found: {args.source_eval_dir}")
        return 1

    # Evaluate trajectories
    try:
        result = evaluate_spatial_abnormal_trajectories(
            trajectory_file=args.trajectory_file,
            lmtad_checkpoint=args.lmtad_checkpoint,
            source_eval_dir=args.source_eval_dir,
            dataset=args.dataset,
            device=args.device,
            batch_size=args.batch_size,
        )

        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"✅ Results saved to {args.output}")
        return 0

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
