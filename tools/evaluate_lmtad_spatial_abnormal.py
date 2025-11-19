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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_evaluate_with_lmtad import (  # noqa: E402
    load_hoser_trajectories,
    evaluate_trajectories_direct,
)
from tools.convert_to_lmtad_format import extract_road_centroids  # noqa: E402
from critics.lmtad_teacher import LMTADTeacher  # noqa: E402
from critics.grid_mapper import GridMapper, GridConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_lmtad_repo_from_checkpoint(checkpoint_path: Path) -> Path:
    """Detect LM-TAD repo root from checkpoint path.

    Checkpoint is typically at: /path/to/lmtad/code/results/.../ckpt_best.pt
    Repo root is: /path/to/lmtad (parent of 'code' directory)

    Args:
        checkpoint_path: Path to LM-TAD checkpoint file

    Returns:
        Path to LM-TAD repository root

    Raises:
        ValueError: If cannot auto-detect repo from checkpoint path
    """
    path = Path(checkpoint_path).resolve()
    # Walk up to find 'code' directory, then return parent
    for parent in [path] + list(path.parents):
        if parent.name == "code" and (parent / "models" / "LMTAD.py").exists():
            return parent.parent
    # Fallback: assume checkpoint is in code/results/.../ckpt_best.pt
    if "code" in path.parts:
        code_idx = path.parts.index("code")
        return Path(*path.parts[:code_idx])
    raise ValueError(
        f"Cannot auto-detect LM-TAD repo from checkpoint: {checkpoint_path}"
    )


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


def _compute_log_perplexity_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for finite log perplexity values."""
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _compute_segment_stats(segment_lists: List[List[float]]) -> Dict[str, Any]:
    """Compute per-segment perplexity statistics across all trajectories."""
    if not segment_lists:
        return {"max_segment_length": 0, "per_index": []}

    max_len = max((len(seg) for seg in segment_lists), default=0)
    per_index = []

    for idx in range(max_len):
        values = [seg[idx] for seg in segment_lists if idx < len(seg)]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        per_index.append(
            {
                "index": idx,
                "count": int(len(values)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )

    return {"max_segment_length": max_len, "per_index": per_index}


def _build_cross_model_od_comparison(
    evaluation_results: List[Dict[str, Any]],
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """Build cross-model comparison data for OD pair analysis.

    This function processes multiple evaluation results to compare model performance
    across OD pairs by computing rankings and deltas for visualization.

    Args:
        evaluation_results: List of evaluation results, one per model. Each result
            contains:
            {
              "model": str,
              "trajectories": [
                {
                  "origin": int,
                  "destination": int,
                  "log_perplexity": float,
                  "segment_log_perplexities": [float, ...],
                  "status": "ok" | "evaluation_failed",
                  "source_label": "route_switch" | "detour" | None
                },
                ...
              ]
            }
        output_path: Optional path to save the comparison JSON

    Returns:
        Dictionary with structure:
        {
            "metadata": {
                "timestamp": ISO timestamp,
                "output_path": str | None,
                "model_count": int,
                "model_names": [str, ...],
                "total_trajectories": int
            },
            "models": [
                {
                    "name": str,
                    "trajectory_count": int,
                    "failed_count": int,
                    "failed_rate": float,
                    "log_perplexity_stats": {mean, std, median, min, max},
                    "segment_stats": {max_segment_length, per_index: [...]},
                    "od_pair_label_counts": {str: count, ...}
                },
                ...
            ],
            "od_pairs": {
                "(origin, destination)": {
                    "origin": int,
                    "destination": int,
                    "trajectory_count": int,
                    "source_label": "route_switch" | "detour" | None,
                    "trajectories": [
                        {
                            "model": str,
                            "trajectory_index": int,
                            "log_perplexity": float,
                            "segment_log_perplexities": [float, ...],
                            "status": str
                        },
                        ...
                    ],
                    "per_model_stats": {
                        "model_name": {
                            "mean_log_perplexity": float,
                            "median_log_perplexity": float,
                            "count": int,
                            "best_log_perplexity": float,
                            "worst_log_perplexity": float
                        },
                        ...
                    },
                    "best_model": str,
                    "best_model_mean_log_perplexity": float,
                    "worst_model": str,
                    "worst_model_mean_log_perplexity": float,
                    "performance_delta": float,
                    "ranking": [
                        {"model": str, "rank": int, "mean_log_perplexity": float},
                        ...
                    ]
                },
                ...
            },
            "od_summary": {
                "total_unique_od_pairs": int,
                "od_pairs_with_all_models": int,
                "average_performance_delta": float,
                "best_performing_models": {str: count},
                "source_label_distribution": {str: count},
                "statistics_by_source_label": {
                    "route_switch": {
                        "count": int,
                        "best_models": {str: count},
                        "avg_delta": float,
                        "std_delta": float
                    },
                    "detour": {...},
                    None: {...}
                }
            }
        }
    """
    logger.info("🚀 Building cross-model OD comparison")
    logger.info(f"   Processing {len(evaluation_results)} model results")

    # Validate inputs
    if not evaluation_results:
        raise ValueError("Empty evaluation results list")

    if not all(isinstance(r.get("trajectories"), list) for r in evaluation_results):
        raise ValueError("All results must contain 'trajectories' list")

    # Extract model names
    model_names = [r["model"] for r in evaluation_results]
    logger.info(f"   Models: {model_names}")

    # Extract trajectory data from all models
    all_trajectories_by_model: Dict[str, List[Dict[str, Any]]] = {
        r["model"]: r["trajectories"] for r in evaluation_results
    }

    # Build OD pair index across all models
    od_pair_trajectories: Dict[Tuple[int, int], Dict[str, List[Dict[str, Any]]]] = {}

    for model_name, trajectories in all_trajectories_by_model.items():
        for traj in trajectories:
            # Skip failed evaluations
            if traj.get("status") == "evaluation_failed":
                continue

            origin = traj.get("origin")
            destination = traj.get("destination")

            if origin is None or destination is None:
                logger.warning(
                    f"   Skipping trajectory {traj.get('trajectory_index')} "
                    f"from {model_name}: missing origin/destination"
                )
                continue

            od_pair = (origin, destination)

            if od_pair not in od_pair_trajectories:
                od_pair_trajectories[od_pair] = {}

            if model_name not in od_pair_trajectories[od_pair]:
                od_pair_trajectories[od_pair][model_name] = []

            od_pair_trajectories[od_pair][model_name].append(traj)

    logger.info(f"   Found {len(od_pair_trajectories)} unique OD pairs")

    # Build per-model metadata
    model_metadata = []
    for result in evaluation_results:
        model_name = result["model"]
        trajectories = result["trajectories"]
        total_count = len(trajectories)
        failed_count = sum(1 for t in trajectories if t.get("status") == "evaluation_failed")
        valid_count = total_count - failed_count

        # Extract finite perplexities for stats
        finite_perplexities = [
            t["log_perplexity"]
            for t in trajectories
            if t.get("status") == "ok" and np.isfinite(t["log_perplexity"])
        ]

        log_perplexity_stats = _compute_log_perplexity_stats(finite_perplexities)

        # Compute segment stats
        segment_lists = [
            t["segment_log_perplexities"]
            for t in trajectories
            if t.get("status") == "ok" and np.isfinite(t["log_perplexity"])
        ]
        segment_stats = _compute_segment_stats(segment_lists)

        # OD pair label counts (only for valid trajectories)
        od_label_counts: Counter[str] = Counter()
        for t in trajectories:
            if t.get("status") == "ok" and t.get("source_label"):
                od_label_counts[t["source_label"]] += 1

        model_metadata.append(
            {
                "name": model_name,
                "trajectory_count": total_count,
                "valid_trajectory_count": valid_count,
                "failed_count": failed_count,
                "failed_rate": (failed_count / total_count * 100) if total_count > 0 else 0,
                "log_perplexity_stats": log_perplexity_stats,
                "segment_stats": segment_stats,
                "od_pair_label_counts": dict(od_label_counts),
            }
        )

    # Build OD pair comparison data
    od_pair_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    source_label_stats: Dict[str, Dict[str, Any]] = {None: {"count": 0}, "route_switch": {"count": 0}, "detour": {"count": 0}}
    model_best_worst_counts: Dict[str, Dict[str, int]] = {name: {"best": 0, "worst": 0} for name in model_names}

    for od_pair, model_trajs in od_pair_trajectories.items():
        origin, destination = od_pair

        # Get source label (use first available)
        source_label = None
        for model_name, trajs in model_trajs.items():
            if trajs and trajs[0].get("source_label"):
                source_label = trajs[0]["source_label"]
                break

        # Count trajectories
        trajectory_count = sum(len(trajs) for trajs in model_trajs.values())

        # Build per-model stats
        per_model_stats = {}
        for model_name, trajs in model_trajs.items():
            perplexities = [
                t["log_perplexity"]
                for t in trajs
                if t.get("status") == "ok" and np.isfinite(t["log_perplexity"])
            ]

            if not perplexities:
                continue

            arr = np.array(perplexities, dtype=np.float64)
            per_model_stats[model_name] = {
                "mean_log_perplexity": float(np.mean(arr)),
                "median_log_perplexity": float(np.median(arr)),
                "count": len(perplexities),
                "std_log_perplexity": float(np.std(arr)),
                "min_log_perplexity": float(np.min(arr)),
                "max_log_perplexity": float(np.max(arr)),
            }

        # Skip if no valid stats
        if not per_model_stats:
            continue

        # Find best and worst performing models
        sorted_models = sorted(
            per_model_stats.items(),
            key=lambda x: x[1]["mean_log_perplexity"],
        )

        best_model, best_stats = sorted_models[0]
        worst_model, worst_stats = sorted_models[-1]

        best_model_mean = best_stats["mean_log_perplexity"]
        worst_model_mean = worst_stats["mean_log_perplexity"]
        performance_delta = worst_model_mean - best_model_mean

        # Build ranking
        ranking = [
            {
                "model": model_name,
                "rank": rank + 1,
                "mean_log_perplexity": stats["mean_log_perplexity"],
            }
            for rank, (model_name, stats) in enumerate(sorted_models)
        ]

        # Build trajectory list for this OD pair
        od_trajectories = []
        for model_name, trajs in model_trajs.items():
            for t in trajs:
                od_trajectories.append(
                    {
                        "model": model_name,
                        "trajectory_index": t.get("trajectory_index"),
                        "log_perplexity": t.get("log_perplexity"),
                        "segment_log_perplexities": t.get("segment_log_perplexities", []),
                        "status": t.get("status"),
                    }
                )

        od_pair_results[od_pair] = {
            "origin": origin,
            "destination": destination,
            "trajectory_count": trajectory_count,
            "source_label": source_label,
            "trajectories": od_trajectories,
            "per_model_stats": per_model_stats,
            "best_model": best_model,
            "best_model_mean_log_perplexity": best_model_mean,
            "worst_model": worst_model,
            "worst_model_mean_log_perplexity": worst_model_mean,
            "performance_delta": performance_delta,
            "ranking": ranking,
        }

        # Update summary stats
        if source_label:
            source_label_stats[source_label]["count"] += 1
        else:
            source_label_stats[None]["count"] += 1

        # Track best/worst counts
        model_best_worst_counts[best_model]["best"] += 1
        model_best_worst_counts[worst_model]["worst"] += 1

    # Compute summary statistics
    all_deltas = [data["performance_delta"] for data in od_pair_results.values()]
    avg_delta = float(np.mean(all_deltas)) if all_deltas else 0.0
    std_delta = float(np.std(all_deltas)) if all_deltas else 0.0

    # Build statistics by source label
    stats_by_source_label = {}
    for label in [None, "route_switch", "detour"]:
        label_pairs = {
            od: data
            for od, data in od_pair_results.items()
            if data["source_label"] == label
        }

        if label_pairs:
            deltas = [data["performance_delta"] for data in label_pairs.values()]
            best_model_counts = Counter(data["best_model"] for data in label_pairs.values())

            stats_by_source_label[str(label) if label else "unknown"] = {
                "count": len(label_pairs),
                "best_models": dict(best_model_counts),
                "avg_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
            }

    od_summary = {
        "total_unique_od_pairs": len(od_pair_results),
        "od_pairs_with_all_models": sum(
            1 for data in od_pair_results.values() if len(data["per_model_stats"]) == len(model_names)
        ),
        "average_performance_delta": avg_delta,
        "std_performance_delta": std_delta,
        "min_performance_delta": float(np.min(all_deltas)) if all_deltas else 0.0,
        "max_performance_delta": float(np.max(all_deltas)) if all_deltas else 0.0,
        "best_performing_models": model_best_worst_counts,
        "source_label_distribution": {str(k) if k else "unknown": v["count"] for k, v in source_label_stats.items()},
        "statistics_by_source_label": stats_by_source_label,
    }

    # Build final output structure
    output_data = {
        "metadata": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "output_path": str(output_path) if output_path else None,
            "model_count": len(evaluation_results),
            "model_names": model_names,
            "total_trajectories": sum(len(r["trajectories"]) for r in evaluation_results),
            "comparison_type": "cross_model_od_comparison",
            "version": "1.0",
        },
        "models": model_metadata,
        "od_pairs": od_pair_results,
        "od_summary": od_summary,
    }

    # Save to file if output_path provided
    if output_path is not None:
        logger.info(f"💾 Saving comparison results to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"✅ Results saved successfully")

    logger.info("✅ Cross-model OD comparison complete:")
    logger.info(f"   Total unique OD pairs: {len(od_pair_results)}")
    logger.info(f"   Average performance delta: {avg_delta:.4f}")
    logger.info(f"   Best performing model distribution: {model_best_worst_counts}")

    return output_data


def evaluate_spatial_abnormal_trajectories(
    trajectory_file: Path,
    lmtad_checkpoint: Path,
    source_eval_dir: Path,
    dataset: str,
    device: str = "cuda:0",
    batch_size: int = 128,
    lmtad_repo: Path | None = None,
    od_pairs_file: Path | None = None,
) -> Dict:
    """Evaluate generated trajectories with LM-TAD and classify spatial abnormality types

    Args:
        trajectory_file: Path to generated trajectory CSV file
        lmtad_checkpoint: Path to LM-TAD checkpoint
        source_eval_dir: Path to LM-TAD source evaluation directory
        dataset: Dataset name
        device: CUDA device
        batch_size: Batch size for evaluation
        lmtad_repo: Path to LM-TAD repository root (auto-detected from checkpoint if None)
        od_pairs_file: Path to OD pairs JSON file (optional, for using known labels)

    Returns:
        Dictionary with evaluation results and classifications
    """
    logger.info(f"📂 Loading trajectories from {trajectory_file}")

    # Load trajectories
    trajectories = load_hoser_trajectories(trajectory_file)
    logger.info(f"✅ Loaded {len(trajectories)} trajectories")

    # Load OD pairs file to get known labels (if available)
    od_pair_labels = {}  # Maps (origin, dest) -> "route_switch" or "detour"
    if od_pairs_file is not None and od_pairs_file.exists():
        logger.info(f"📂 Loading OD pairs labels from {od_pairs_file}")
        with open(od_pairs_file, "r") as f:
            od_pairs_data = json.load(f)
        for od_type, pairs in od_pairs_data.get("od_pairs_by_type", {}).items():
            for pair in pairs:
                # Normalize pair to tuple
                od_pair = tuple(pair) if isinstance(pair, list) else pair
                od_pair_labels[od_pair] = od_type
        logger.info(
            f"✅ Loaded {len(od_pair_labels)} OD pair labels "
            f"({sum(1 for v in od_pair_labels.values() if v == 'route_switch')} route_switch, "
            f"{sum(1 for v in od_pair_labels.values() if v == 'detour')} detour)"
        )
    else:
        logger.info(
            "⚠️  OD pairs file not provided - will use perplexity thresholds for classification"
        )

    # Load source statistics (still needed for perplexity-based fallback)
    source_stats = load_source_statistics(source_eval_dir)

    # Determine LM-TAD repo path
    if lmtad_repo is None:
        logger.info("🔍 Auto-detecting LM-TAD repo from checkpoint path...")
        lmtad_repo = detect_lmtad_repo_from_checkpoint(lmtad_checkpoint)
        logger.info(f"✅ Detected LM-TAD repo: {lmtad_repo}")
    else:
        logger.info(f"📂 Using provided LM-TAD repo: {lmtad_repo}")

    # Add LM-TAD code path to sys.path BEFORE importing LMTADTeacher
    # This ensures models.LMTAD resolves to LM-TAD's models, not HOSER's
    lmtad_code_path = str(lmtad_repo / "code")
    hoser_project_root = str(Path(__file__).parent.parent)

    # Temporarily move HOSER project root to end of sys.path to avoid namespace conflicts
    hoser_index = None
    if hoser_project_root in sys.path:
        hoser_index = sys.path.index(hoser_project_root)
        sys.path.remove(hoser_project_root)

    # Ensure LM-TAD code path is first
    if lmtad_code_path not in sys.path:
        sys.path.insert(0, lmtad_code_path)
        logger.debug(f"Added LM-TAD code path to sys.path: {lmtad_code_path}")
    elif sys.path.index(lmtad_code_path) != 0:
        # Move to front if already present
        sys.path.remove(lmtad_code_path)
        sys.path.insert(0, lmtad_code_path)

    # Clear any cached 'models' and 'utils' modules from HOSER to prevent namespace conflicts
    # LMTADTeacher will handle restoring utils, but we need to clear it first
    cached_models = sys.modules.pop("models", None)
    cached_utils = sys.modules.pop("utils", None)

    try:
        # Load LM-TAD teacher
        logger.info(f"📂 Loading LM-TAD teacher from {lmtad_checkpoint}")
        model = LMTADTeacher(
            repo_path=str(lmtad_repo),
            ckpt_path=str(lmtad_checkpoint),
            device=device,
            dtype="float16",
            window=256,
        )
        logger.info("✅ LM-TAD teacher loaded")
    finally:
        # Restore HOSER project root to its original position
        if hoser_index is not None:
            sys.path.insert(hoser_index, hoser_project_root)
        elif hoser_project_root not in sys.path:
            sys.path.append(hoser_project_root)

        # Restore cached modules if they existed
        # Note: LMTADTeacher handles utils restoration, but we restore it here too for safety
        if cached_models is not None:
            sys.modules["models"] = cached_models
        if cached_utils is not None and "utils" not in sys.modules:
            sys.modules["utils"] = cached_utils

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

    # Get expected grid dimensions from teacher (matches training process)
    raw_teacher_hw = model.get_grid_size_hw()
    teacher_hw: Optional[Tuple[int, int]] = None
    if isinstance(raw_teacher_hw, (list, tuple)) and len(raw_teacher_hw) == 2:
        teacher_hw = (int(raw_teacher_hw[0]), int(raw_teacher_hw[1]))
        logger.info(f"📐 Expected grid dimensions from teacher: {teacher_hw}")
    elif raw_teacher_hw is not None:
        logger.warning(
            "⚠️  Teacher returned unexpected grid dimension format; proceeding without verification"
        )
    else:
        logger.warning(
            "⚠️  Could not get grid dimensions from teacher, proceeding without verification"
        )

    # Get vocab_size from teacher for token validation
    vocab_size = model.vocab_size()
    if vocab_size is not None:
        logger.info(f"📚 Vocab size from teacher: {vocab_size}")
    else:
        logger.warning("⚠️  Could not determine vocab_size from teacher")

    # Extract road centroids (always needed for mapping)
    logger.info("📂 Extracting road centroids from roadmap...")
    road_centroids, boundary_from_roadmap = extract_road_centroids(roadmap_file)

    # Use same grid_size and downsample_factor as training (matches config files)
    # Default: grid_size=0.001, downsample_factor=1 (no downsampling)
    grid_size = 0.001
    downsample_factor = 1

    # If we have expected grid dimensions from teacher, compute boundaries to match exactly
    # Otherwise, use boundaries extracted from roadmap
    if teacher_hw is not None:
        vh, vw = teacher_hw
        logger.info(
            f"📐 Computing boundaries from known grid dimensions {(vh, vw)} "
            f"(from teacher/config)"
        )

        # Compute required spans from expected grid dimensions
        # Formula: span = (grid_dim - 1) * grid_size
        required_lat_span = (vh - 1) * grid_size
        required_lng_span = (vw - 1) * grid_size

        # Center boundaries around road centroids to match training
        lat_center = (
            boundary_from_roadmap["min_lat"] + boundary_from_roadmap["max_lat"]
        ) / 2.0
        lng_center = (
            boundary_from_roadmap["min_lng"] + boundary_from_roadmap["max_lng"]
        ) / 2.0

        boundary = {
            "min_lat": lat_center - required_lat_span / 2.0,
            "max_lat": lat_center + required_lat_span / 2.0,
            "min_lng": lng_center - required_lng_span / 2.0,
            "max_lng": lng_center + required_lng_span / 2.0,
        }
        logger.info(
            f"✅ Computed boundaries from grid dimensions: "
            f"lat=[{boundary['min_lat']:.6f}, {boundary['max_lat']:.6f}], "
            f"lng=[{boundary['min_lng']:.6f}, {boundary['max_lng']:.6f}]"
        )
    else:
        # Fallback: use boundaries extracted from roadmap
        boundary = boundary_from_roadmap
        logger.warning(
            "⚠️  Using boundaries from roadmap (expected grid dimensions not available). "
            "Grid dimensions may not match training exactly."
        )
        logger.info(
            f"✅ Using extracted boundaries: "
            f"lat=[{boundary['min_lat']:.6f}, {boundary['max_lat']:.6f}], "
            f"lng=[{boundary['min_lng']:.6f}, {boundary['max_lng']:.6f}]"
        )

    grid_config = GridConfig(
        min_lat=boundary["min_lat"],
        max_lat=boundary["max_lat"],
        min_lng=boundary["min_lng"],
        max_lng=boundary["max_lng"],
        grid_size=grid_size,
        downsample_factor=downsample_factor,
    )
    mapper = GridMapper(
        boundary=grid_config,
        road_centroids=road_centroids,
        verify_hw=teacher_hw,  # Ensure grid dimensions match training
    )
    road_to_token = torch.from_numpy(mapper.map_all()).to(device)
    logger.info("✅ Grid mapper created")

    # Evaluate trajectories
    logger.info("🔍 Evaluating trajectories with LM-TAD...")
    log_perplexities, _, segment_log_perplexities = evaluate_trajectories_direct(
        trajectories=trajectories,
        model=model,
        road_to_token=road_to_token,
        device=device,
        batch_size=batch_size,
        vocab_size=vocab_size,
        return_segment_perplexity=True,
    )

    if segment_log_perplexities is None:
        segment_log_perplexities = [[] for _ in trajectories]
    elif len(segment_log_perplexities) < len(trajectories):
        segment_log_perplexities.extend(
            [[] for _ in range(len(trajectories) - len(segment_log_perplexities))]
        )
    elif len(segment_log_perplexities) > len(trajectories):
        segment_log_perplexities = segment_log_perplexities[: len(trajectories)]

    total_trajectories = len(trajectories)
    trajectory_records: List[Dict[str, Any]] = []
    finite_perplexities: List[float] = []
    failed_evaluations = 0
    od_label_counts: Counter[str] = Counter()

    for idx, (trajectory, log_perp, segment_logs) in enumerate(
        zip(trajectories, log_perplexities, segment_log_perplexities)
    ):
        log_perp_value = float(log_perp)
        origin = trajectory[0] if trajectory else None
        destination = trajectory[-1] if trajectory else None
        source_label = od_pair_labels.get((origin, destination))

        status = "ok"
        if np.isinf(log_perp_value):
            status = "evaluation_failed"
            failed_evaluations += 1
        else:
            finite_perplexities.append(log_perp_value)
            if source_label:
                od_label_counts[source_label] += 1

        trajectory_records.append(
            {
                "trajectory_index": idx,
                "origin": origin,
                "destination": destination,
                "log_perplexity": log_perp_value,
                "segment_log_perplexities": segment_logs,
                "status": status,
                "source_label": source_label if status == "ok" else None,
            }
        )

    valid_trajectories = total_trajectories - failed_evaluations
    failed_rate = (
        (failed_evaluations / total_trajectories * 100) if total_trajectories > 0 else 0
    )

    log_perplexity_stats = _compute_log_perplexity_stats(finite_perplexities)
    segment_stats = _compute_segment_stats(segment_log_perplexities)

    model_name = trajectory_file.stem.replace("_spatial_abnormal", "")

    result = {
        "model": model_name,
        "dataset": dataset,
        "total_trajectories": total_trajectories,
        "valid_trajectory_count": valid_trajectories,
        "failed_trajectory_count": failed_evaluations,
        "failed_trajectory_rate": failed_rate,
        "log_perplexity_stats": log_perplexity_stats,
        "segment_stats": segment_stats,
        "trajectories": trajectory_records,
        "od_pair_label_counts": dict(od_label_counts),
        "source_statistics": source_stats,
    }

    logger.info("✅ Evaluation complete:")
    logger.info(f"   Total trajectories: {total_trajectories}")
    logger.info(f"   Failed trajectories: {failed_evaluations} ({failed_rate:.2f}%)")
    logger.info(f"   Valid trajectories: {valid_trajectories}")

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
    parser.add_argument(
        "--lmtad-repo",
        type=Path,
        default=None,
        help="Path to LM-TAD repository root (auto-detected from checkpoint if not provided)",
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
            lmtad_repo=args.lmtad_repo,
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
