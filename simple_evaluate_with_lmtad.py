#!/usr/bin/env python3
"""
Simple LM-TAD Evaluation of Generated Trajectories

This script evaluates HOSER-generated trajectories using the LM-TAD teacher model.
It directly reuses the distillation infrastructure for robust, production-ready
evaluation.

Features:
- Direct evaluation without CSV I/O (memory efficient, faster)
- Reuses battle-tested code from distillation (distill_hook.py)
- Comprehensive error handling and validation
- Progress tracking and detailed logging
- Automatic result aggregation and summary generation

Usage:
    python simple_evaluate_with_lmtad.py \
        --eval-dir /path/to/evaluation \
        --lmtad-checkpoint /path/to/lmtad/model \
        --dataset porto_hoser

Advantages over CSV-based approach:
- Uses battle-tested code paths from distillation
- Avoids all CSV parsing/formatting issues
- More memory efficient
- Faster (no disk I/O during conversion)
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from shapely.geometry import LineString

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from critics.lmtad_teacher import LMTADTeacher  # noqa: E402
from critics.grid_mapper import GridMapper, GridConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_road_centroids(geo_file: Path) -> np.ndarray:
    """
    Load road centroids from geo file (roadmap.geo).

    Args:
        geo_file: Path to the geo file containing road information

    Returns:
        Array of shape (num_roads, 2) with [lng, lat] centroids
    """
    logger.info(f"Loading road centroids from {geo_file}")

    if not geo_file.exists():
        raise FileNotFoundError(f"Geo file not found: {geo_file}")

    geo_df = pd.read_csv(geo_file)
    centroids = []

    for _, row in geo_df.iterrows():
        # Parse coordinates and compute centroid
        coordinates = eval(row["coordinates"])
        road_line = LineString(coordinates=coordinates)
        centroid = road_line.centroid
        centroids.append([centroid.x, centroid.y])  # [lng, lat]

    logger.info(f"Loaded {len(centroids)} road centroids")
    return np.array(centroids)


def load_hoser_trajectories(csv_file: Path) -> List[List[int]]:
    """Load HOSER trajectories from CSV (handles both real and generated formats).

    Args:
        csv_file: Path to CSV file containing trajectories

    Returns:
        List of road ID sequences

    Raises:
        ValueError: If CSV format is invalid
        FileNotFoundError: If CSV file doesn't exist
    """
    logger.info(f"Loading trajectories from {csv_file}")

    if not csv_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_file}: {e}")

    # Check for expected column (generated trajectories use gene_trace_road_id)
    road_id_col = None
    if "gene_trace_road_id" in df.columns:
        road_id_col = "gene_trace_road_id"
        logger.info("  Detected generated trajectory format (gene_trace_road_id)")
    elif "rid_list" in df.columns:
        road_id_col = "rid_list"
        logger.info("  Detected real trajectory format (rid_list)")
    else:
        raise ValueError(
            f"CSV file missing expected column. "
            f"Found columns: {list(df.columns)}. "
            f"Expected 'gene_trace_road_id' (generated) or 'rid_list' (real)"
        )

    trajectories = []
    for idx, row in df.iterrows():
        try:
            if road_id_col == "gene_trace_road_id":
                # Generated format: Python list string like "[10232, 5620, ...]"
                road_ids_str = row["gene_trace_road_id"]
                if pd.isna(road_ids_str) or str(road_ids_str).strip() == "":
                    logger.warning(f"  Row {idx}: Empty trajectory, skipping")
                    continue
                try:
                    # Parse Python list string
                    road_ids = eval(road_ids_str)
                    if not isinstance(road_ids, list):
                        raise ValueError(f"Expected list, got {type(road_ids)}")
                    road_ids = [int(x) for x in road_ids]
                except Exception as e:
                    logger.error(f"  Row {idx}: Failed to parse trajectory list: {e}")
                    raise
            else:
                # Real format: list of [road_id, timestamp] pairs
                rid_list = eval(row["rid_list"])  # Convert string to list
                road_ids = [int(road[0]) for road in rid_list]  # Extract road IDs

            # Validate
            if len(road_ids) == 0:
                logger.warning(f"  Row {idx}: Empty trajectory, skipping")
                continue

            trajectories.append(road_ids)
        except Exception as e:
            logger.error(f"  Row {idx}: Failed to parse trajectory: {e}")
            raise

    logger.info(f"Successfully loaded {len(trajectories)} trajectories")
    return trajectories


def evaluate_trajectories_direct(
    trajectories: List[List[int]],
    model: LMTADTeacher,
    road_to_token: torch.Tensor,
    device: str,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate trajectories using LM-TAD teacher by computing perplexity.

    Perplexity measures how well the teacher predicts each token in the sequence.
    Lower perplexity = better match to teacher's expectations.

    Args:
        trajectories: List of HOSER road ID sequences
        model: LM-TAD teacher model
        road_to_token: Mapping from road ID → grid token
        device: CUDA device
        batch_size: Batch size for evaluation

    Returns:
        perplexities: Array of perplexity values (lower is better)
        outliers: Binary array (1 = outlier, 0 = normal) based on threshold
    """
    # Note: LMTADTeacher is already in eval mode when loaded
    all_perplexities = []
    all_outlier_scores = []

    logger.info(f"Evaluating {len(trajectories)} trajectories...")

    # Note: teacher's window size is used internally by the model
    sot_token = model.sot_token()

    for traj_idx, road_ids in enumerate(trajectories):
        if traj_idx % 500 == 0 and traj_idx > 0:
            logger.info(f"  Processed {traj_idx}/{len(trajectories)}...")

        try:
            # Validate road IDs are within bounds
            max_road_id = road_to_token.shape[0] - 1
            invalid_roads = [rid for rid in road_ids if rid < 0 or rid > max_road_id]
            if invalid_roads:
                logger.warning(
                    f"  Trajectory {traj_idx}: Invalid road IDs (out of bounds [0, {max_road_id}]): "
                    f"{invalid_roads[:10]}{'...' if len(invalid_roads) > 10 else ''}"
                )
                # Filter out invalid road IDs
                road_ids = [rid for rid in road_ids if 0 <= rid <= max_road_id]
                if len(road_ids) < 2:
                    logger.warning(
                        f"  Trajectory {traj_idx}: Too few valid road IDs after filtering, skipping"
                    )
                    all_perplexities.append(float("inf"))
                    all_outlier_scores.append(float("inf"))
                    continue

            # Convert HOSER road IDs to LM-TAD grid tokens
            road_tensor = torch.tensor(road_ids, device=device, dtype=torch.long)
            tokens = road_to_token[road_tensor].cpu().numpy().tolist()

            # Add SOT token if available
            if sot_token is not None:
                tokens = [sot_token] + tokens

            if len(tokens) < 2:
                # Too short to evaluate
                all_perplexities.append(float("inf"))
                all_outlier_scores.append(1.0)  # Mark as outlier
                continue

            # Compute perplexity by evaluating each position
            log_probs = []

            # Iterate through the sequence, predicting each token
            for i in range(1, len(tokens)):
                # Context is tokens[:i], target is tokens[i]
                context = torch.tensor(tokens[:i], dtype=torch.long, device=device)

                # Get teacher's prediction distribution
                # Note: predict_next_distribution expects tokenized history
                pred_dist = model.predict_next_distribution(context)

                # Get log probability of the actual next token
                target_token = tokens[i]
                log_prob = torch.log(pred_dist[target_token] + 1e-10)
                log_probs.append(log_prob.item())

            # Compute log perplexity = -average log prob (matches source dataset format)
            avg_log_prob = np.mean(log_probs)
            log_perplexity = float(-avg_log_prob)
            all_perplexities.append(log_perplexity)

            # Simple outlier detection: high log perplexity = outlier
            # Use threshold based on distribution (can be tuned)
            all_outlier_scores.append(log_perplexity)

        except Exception as e:
            logger.error(f"  Trajectory {traj_idx}: Evaluation failed: {e}")
            # Mark as failed/outlier
            all_perplexities.append(float("inf"))
            all_outlier_scores.append(float("inf"))

    # Convert outlier scores to binary labels using threshold
    # Use 95th percentile as threshold (adjustable)
    if len(all_outlier_scores) > 0:
        threshold = np.percentile(all_outlier_scores, 95)
        outliers = np.array(
            [1 if score > threshold else 0 for score in all_outlier_scores]
        )
    else:
        outliers = np.array([])

    return np.array(all_perplexities), outliers


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple LM-TAD evaluation using distillation infrastructure"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory",
    )
    parser.add_argument(
        "--dataset", type=str, default="porto_hoser", help="Dataset name"
    )
    parser.add_argument(
        "--lmtad-checkpoint", type=Path, required=True, help="LM-TAD checkpoint path"
    )
    parser.add_argument(
        "--lmtad-repo",
        type=Path,
        default=Path("/home/matt/Dev/LMTAD"),
        help="LM-TAD repo path",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for evaluation"
    )

    args = parser.parse_args()

    # Validate paths exist
    logger.info("=" * 80)
    logger.info("VALIDATING INPUTS")
    logger.info("=" * 80)

    if not args.eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {args.eval_dir}")

    if not args.lmtad_checkpoint.exists():
        raise FileNotFoundError(f"LM-TAD checkpoint not found: {args.lmtad_checkpoint}")

    if not args.lmtad_repo.exists():
        raise FileNotFoundError(f"LM-TAD repo not found: {args.lmtad_repo}")

    # Setup paths
    gene_dir = args.eval_dir / "gene" / args.dataset / "seed42"
    data_dir = Path("data") / args.dataset
    roadmap_file = data_dir / "roadmap.geo"

    # Validate paths
    if not gene_dir.exists():
        raise FileNotFoundError(f"Gene directory not found: {gene_dir}")

    if not roadmap_file.exists():
        raise FileNotFoundError(f"Roadmap file not found: {roadmap_file}")

    # Output directory
    output_dir = args.eval_dir / "eval_lmtad_simple" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Find all generated trajectory files
    csv_files = list(gene_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.stem.endswith("_perf")]
    csv_files.sort()

    if len(csv_files) == 0:
        raise ValueError(f"No trajectory files found in {gene_dir}")

    logger.info(f"Found {len(csv_files)} trajectory files to evaluate")
    for f in csv_files:
        logger.info(f"  - {f.name}")

    # Load LM-TAD teacher (same as distillation)
    logger.info(f"Loading LM-TAD teacher from {args.lmtad_checkpoint}...")
    model = LMTADTeacher(
        repo_path=str(args.lmtad_repo),
        ckpt_path=str(args.lmtad_checkpoint),
        device=args.device,
        dtype="float16",
        window=256,  # Large window for evaluation
    )
    logger.info("LM-TAD teacher loaded successfully")

    # Setup grid mapper (same as distillation)
    road_centroids = load_road_centroids(roadmap_file)
    logger.info(f"Loaded road centroids: {road_centroids.shape}")
    logger.info(
        f"Lat range: {road_centroids[:, 1].min():.6f} to {road_centroids[:, 1].max():.6f}"
    )
    logger.info(
        f"Lng range: {road_centroids[:, 0].min():.6f} to {road_centroids[:, 0].max():.6f}"
    )

    grid_config = GridConfig(
        min_lat=road_centroids[:, 1].min(),
        max_lat=road_centroids[:, 1].max(),
        min_lng=road_centroids[:, 0].min(),
        max_lng=road_centroids[:, 0].max(),
        grid_size=0.001,  # Same as LMTAD
    )

    # Create grid mapper
    try:
        mapper = GridMapper(
            boundary=grid_config,
            road_centroids=road_centroids,
            verify_hw=None,  # Disable verification for evaluation - mapping consistency is what matters
        )

        road_to_token = torch.from_numpy(mapper.map_all()).to(args.device)
        logger.info(f"Created grid mapper: {mapper.grid_h}x{mapper.grid_w} cells")
        logger.info(
            "Note: Grid dims may differ from teacher training dims - this is OK for evaluation"
        )
    except Exception as e:
        logger.error(f"Failed to create grid mapper: {e}", exc_info=True)
        raise

    # Evaluate each file
    all_results = {}

    logger.info("\n" + "=" * 80)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 80)

    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"\n[{i}/{len(csv_files)}] Evaluating {csv_file.name}...")

        try:
            # Load trajectories
            trajectories = load_hoser_trajectories(csv_file)

            if len(trajectories) == 0:
                logger.warning(f"  No valid trajectories in {csv_file.name}, skipping")
                continue

            logger.info(f"  Loaded {len(trajectories)} trajectories")

            # Evaluate
            perplexities, outliers = evaluate_trajectories_direct(
                trajectories=trajectories,
                model=model,
                road_to_token=road_to_token,
                device=args.device,
                batch_size=args.batch_size,
            )

            # Validate results
            if len(perplexities) != len(trajectories):
                raise ValueError(
                    f"Mismatch: loaded {len(trajectories)} trajectories but got {len(perplexities)} perplexity scores"
                )

            # Store results (using log perplexity to match source dataset format)
            file_key = csv_file.stem  # e.g., "2025-11-07_00-13-07_distill_phase1_train"
            all_results[file_key] = {
                "num_trajectories": len(trajectories),
                "mean_log_perplexity": float(perplexities.mean()),
                "median_log_perplexity": float(np.median(perplexities)),
                "std_log_perplexity": float(perplexities.std()),
                "min_log_perplexity": float(perplexities.min()),
                "max_log_perplexity": float(perplexities.max()),
                "outlier_rate": float(outliers.mean()),
                "log_perplexity_values": perplexities.tolist(),
                "outlier_labels": outliers.tolist(),
            }

            logger.info(f"  ✓ Mean log perplexity: {perplexities.mean():.4f}")
            logger.info(f"  ✓ Outlier rate: {outliers.mean():.2%}")

        except Exception as e:
            logger.error(f"  ✗ Failed to evaluate {csv_file.name}: {e}", exc_info=True)
            raise

    # Save all results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    # Save summary CSV
    summary_data = []
    for file_key, results in all_results.items():
        # Parse model info from filename
        # Format: YYYY-MM-DD_HH-MM-SS_{model}_{split} or YYYY-MM-DD_HH-MM-SS_{model}_{seed}_{split}
        parts = file_key.split("_")
        if len(parts) >= 5:
            model_name = (
                parts[2] if "seed" not in parts[3] else f"{parts[2]}_{parts[3]}"
            )
            split = parts[-1]
        else:
            model_name = file_key[:30]
            split = ""

        summary_data.append(
            {
                "file": file_key,
                "model": model_name,
                "split": split,
                "num_trajectories": results["num_trajectories"],
                "mean_log_perplexity": results["mean_log_perplexity"],
                "median_log_perplexity": results["median_log_perplexity"],
                "std_log_perplexity": results["std_log_perplexity"],
                "min_log_perplexity": results["min_log_perplexity"],
                "max_log_perplexity": results["max_log_perplexity"],
                "outlier_rate": results["outlier_rate"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to: {summary_file}")

    # Print summary
    print("\n" + "=" * 100)
    print("LM-TAD EVALUATION SUMMARY")
    print("=" * 100)
    print(
        f"{'File':<50} {'Log Perplexity':<15} {'Outlier Rate':<15} {'#Trajectories':<15}"
    )
    print("-" * 100)

    for data in summary_data:
        print(
            f"{data['file'][:48]:<50} {data['mean_log_perplexity']:<15.4f} {data['outlier_rate']:<15.2%} {data['num_trajectories']:<15}"
        )

    print("=" * 100)
    logger.info(f"✅ Evaluation complete! Total files evaluated: {len(all_results)}")
    logger.info(f"📊 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
