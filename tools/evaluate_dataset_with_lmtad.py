#!/usr/bin/env python3
"""
Evaluate raw dataset CSV splits (train/val/test) with LM-TAD teacher.

This script is a lightweight wrapper around the project's LM-TAD evaluation
utilities and is intended for evaluating the raw CSV splits that live under
`data/<dataset>/` (for example `data/Beijing_abnormal/train.csv`) while
pointing to a roadmap file from another dataset directory (for example
`data/Beijing/roadmap.geo`).

Usage example:
    uv run python tools/evaluate_dataset_with_lmtad.py \
        --dataset Beijing_abnormal \
        --data-dir data/Beijing_abnormal \
        --roadmap data/Beijing/roadmap.geo \
        --lmtad-checkpoint /path/to/ckpt_best.pt

First run on baseline (writes a baseline file you can reuse later):
    uv run python tools/evaluate_dataset_with_lmtad.py \
        --dataset Beijing \
        --lmtad-checkpoint /path/to/ckpt_best.pt \
        --write-baseline

Baseline-calibrated outliers (recommended):
    uv run python tools/evaluate_dataset_with_lmtad.py \
        --dataset generated_beijing \
        --data-dir gene/Beijing \
        --roadmap data/Beijing/roadmap.geo \
        --lmtad-checkpoint /path/to/ckpt_best.pt \
        --baseline-eval tools_eval_lmtad/Beijing \
        --baseline-quantile 0.95

One-command baseline + target evals (recommended for perturbed-data experiments):
    # Evaluates baseline, writes baseline_eval.json, then evaluates target(s)
    # using the baseline-calibrated threshold — all in one invocation.
    uv run python tools/evaluate_dataset_with_lmtad.py \
        --baseline-dataset Beijing \
        --target-datasets Beijing_abnormal_3 \
        --lmtad-checkpoint /path/to/ckpt_best.pt \
        --device cuda:0 \
        --splits train \
        --baseline-quantile 0.95 \
        --baseline-split train \
        --sample-frac 0.01
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simple_evaluate_with_lmtad import (  # noqa: E402
    load_hoser_trajectories,
    evaluate_trajectories_direct,
)  # noqa: E402
from critics.lmtad_teacher import LMTADTeacher  # noqa: E402
from critics.grid_mapper import GridMapper, GridConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_GRID_SIZE = 0.001

ScoresBySplit = Dict[str, List[float]]


@dataclass(frozen=True)
class BaselineCalibrator:
    """Apply a fixed threshold calibrated on a baseline eval file."""

    baseline_eval: Optional[Path]
    scores_by_split: Optional[ScoresBySplit]
    quantile: float
    fallback_split: str

    @classmethod
    def from_optional_eval(
        cls,
        *,
        baseline_eval: Optional[Path],
        quantile: float,
        fallback_split: str,
    ) -> "BaselineCalibrator":
        resolved = _resolve_baseline_eval_path(baseline_eval)
        if resolved is None:
            return cls(
                baseline_eval=None,
                scores_by_split=None,
                quantile=float(quantile),
                fallback_split=str(fallback_split),
            )

        _require_exists(resolved, "Baseline eval JSON")
        return cls(
            baseline_eval=resolved,
            scores_by_split=_load_eval_scores_by_split(resolved),
            quantile=float(quantile),
            fallback_split=str(fallback_split),
        )

    def apply(
        self,
        *,
        split: str,
        perplexities: np.ndarray,
        within_split_outliers: np.ndarray,
    ) -> Tuple[np.ndarray, str, float, Optional[Dict[str, Any]]]:
        """Return outliers, method name, within-split rate, and optional metadata."""
        return _compute_outliers(
            split=split,
            perplexities=perplexities,
            within_split_outliers=within_split_outliers,
            baseline_eval=self.baseline_eval,
            baseline_scores_by_split=self.scores_by_split,
            baseline_quantile=self.quantile,
            baseline_split_fallback=self.fallback_split,
        )


@dataclass(frozen=True)
class ResultsWriter:
    out_dir: Path

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / "evaluation_results.jsonl"

    @property
    def agg_json_path(self) -> Path:
        return self.out_dir / "evaluation_results.json"

    @property
    def baseline_path(self) -> Path:
        return self.out_dir / "baseline_eval.json"

    def write_split(self, *, split: str, payload: Dict[str, Any]) -> None:
        try:
            _append_jsonl(self.jsonl_path, {"split": split, "results": payload})
        except Exception as e:
            logger.error("Failed to append JSONL for split %s: %s", split, e)

        try:
            _update_aggregated_json(self.agg_json_path, split, payload)
        except Exception as e:
            logger.error("Failed to update aggregated results file: %s", e)


def _load_eval_scores_by_split(path: Path) -> ScoresBySplit:
    """Load per-split log-perplexity scores from an evaluator JSON file."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid evaluation JSON structure: {path}")

    out: ScoresBySplit = {}
    for split, payload in data.items():
        if not isinstance(payload, dict):
            continue
        scores = payload.get("log_perplexity_values")
        if isinstance(scores, list) and scores:
            out[str(split)] = [float(x) for x in scores]

    if not out:
        raise ValueError(f"No splits with 'log_perplexity_values' found in: {path}")
    return out


def _resolve_baseline_eval_path(path: Optional[Path]) -> Optional[Path]:
    """Resolve a baseline eval path.

    Accepts:
    - None
    - a file path
    - a directory containing `baseline_eval.json` or `evaluation_results.json`
    """
    if path is None:
        return None

    p = Path(path)
    if p.is_dir():
        candidate = p / "baseline_eval.json"
        if candidate.exists():
            return candidate
        candidate = p / "evaluation_results.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            "Baseline eval directory does not contain baseline_eval.json or evaluation_results.json: "
            f"{p}"
        )

    return p


def _write_baseline_eval_json(
    *,
    path: Path,
    results_by_split: Dict[str, Dict[str, Any]],
) -> None:
    """Write a compact baseline file consumable by `--baseline-eval`.

    The baseline loader expects a dict of split -> {'log_perplexity_values': [...]}
    (it ignores extra keys).
    """
    payload: Dict[str, Dict[str, Any]] = {}
    for split, res in results_by_split.items():
        payload[str(split)] = {
            "num_trajectories": int(res.get("num_trajectories", 0)),
            "log_perplexity_values": list(res.get("log_perplexity_values", [])),
        }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _pick_baseline_split(
    *,
    split: str,
    baseline_scores_by_split: Dict[str, List[float]],
    fallback_split: str,
) -> str:
    """Pick a baseline split for calibrating the threshold.

    Preference order:
    1) Use the same split name if present in the baseline eval.
    2) Otherwise, use the provided fallback split.
    """
    if split in baseline_scores_by_split:
        return split
    if fallback_split in baseline_scores_by_split:
        return fallback_split
    available = ", ".join(sorted(baseline_scores_by_split.keys()))
    raise ValueError(
        "Baseline eval does not contain a usable split. "
        f"Requested='{split}', fallback='{fallback_split}', available=[{available}]"
    )


def _quantile_threshold(values: np.ndarray, q: float) -> float:
    """Return the q-quantile threshold for baseline calibration."""
    if not (0.0 < float(q) < 1.0):
        raise ValueError(f"baseline_quantile must be in (0,1), got {q}")
    if values.size == 0:
        raise ValueError("Cannot compute quantile threshold on empty values")
    return float(np.quantile(values, float(q)))


def _require_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def _stream_sample_csv(
    *,
    csv_file: Path,
    out_file: Path,
    sample_frac: float,
    sample_seed: int,
) -> Optional[Path]:
    """Bernoulli stream sample a CSV to `out_file`.

    Returns:
        - `out_file` if sampling succeeded and >=1 row was kept.
        - None if the input file was empty or 0 rows were kept.

    Raises:
        Any exception that occurs while reading/writing.
    """
    random.seed(sample_seed)
    with (
        open(csv_file, "r", newline="") as inf,
        open(out_file, "w", newline="") as outf,
    ):
        reader = csv.reader(inf)
        writer = csv.writer(outf)

        header = next(reader, None)
        if header is None:
            return None
        writer.writerow(header)

        kept = 0
        for row in reader:
            if random.random() < float(sample_frac):
                writer.writerow(row)
                kept += 1

    return out_file if kept > 0 else None


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object per line."""
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _update_aggregated_json(
    path: Path, split: str, split_payload: Dict[str, Any]
) -> None:
    """Update (overwrite) an aggregated JSON file with one split."""
    if path.exists():
        with open(path, "r") as f:
            agg = json.load(f)
            if not isinstance(agg, dict):
                agg = {}
    else:
        agg = {}
    agg[split] = split_payload
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)


def _abnormality_info_stats(csv_path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort stats about injected abnormality rows.

    For perturbed datasets like `Beijing_abnormal_3`, only the train split is
    typically perturbed; val/test often remain clean. Reporting this helps
    interpret baseline-calibrated outlier rates.
    """
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "abnormality_info" not in reader.fieldnames:
                return None

            total = 0
            abnormal = 0
            for row in reader:
                total += 1
                raw = row.get("abnormality_info")
                if raw is None:
                    continue
                s = str(raw).strip()
                if not s:
                    continue
                if s.lower() in {"nan", "none", "null", "normal"}:
                    continue
                abnormal += 1

        if total == 0:
            return None

        return {
            "column": "abnormality_info",
            "num_rows": int(total),
            "num_abnormal": int(abnormal),
            "abnormal_fraction": float(abnormal / total),
        }
    except Exception:
        return None


def _iter_splits(data_dir: Path, splits: List[str]) -> Iterable[Tuple[str, Path]]:
    for split in splits:
        yield split, data_dir / f"{split}.csv"


def _prepare_target_csv(
    *,
    csv_file: Path,
    out_dir: Path,
    sample_frac: Optional[float],
    sample_seed: int,
) -> Optional[Path]:
    """Return a CSV path to evaluate (sampled or original)."""
    if sample_frac is None or not (0.0 < float(sample_frac) < 1.0):
        return csv_file

    logger.info(
        "Stream-sampling %.1f%% of %s (seed=%d)",
        float(sample_frac) * 100.0,
        csv_file.name,
        int(sample_seed),
    )

    tmp_file = out_dir / f"{csv_file.stem}_sampled.csv"
    sampled = _stream_sample_csv(
        csv_file=csv_file,
        out_file=tmp_file,
        sample_frac=float(sample_frac),
        sample_seed=int(sample_seed),
    )
    return sampled


def _extract_road_centroids_and_boundary_from_roadmap(
    roadmap_file: Path,
) -> Tuple[np.ndarray, GridConfig]:
    """Extract road centroids and grid boundary using the preferred method.

    This matches the intent of `tools/convert_to_lmtad_format.py`:
    - Centroid per road is the mean of polyline points (lat/lng).
    - Boundary is computed from *all* polyline points (not from centroids).

    Returns:
        (road_centroids_latlng, grid_config)
    """
    _require_exists(roadmap_file, "Roadmap file")

    # Keep dtype overrides in sync with the converter to avoid pandas type churn.
    schema_overrides = {
        "lanes": str,
        "oneway": str,
        "coordinates": str,
        "name": str,
        "highway": str,
        "access": str,
        "maxspeed": str,
        "ref": str,
        "tunnel": str,
        "junction": str,
        "width": str,
        "bridge": str,
    }

    import pandas as pd

    roadmap = pd.read_csv(roadmap_file, dtype=schema_overrides)
    if "coordinates" not in roadmap.columns:
        raise ValueError(
            f"Required column 'coordinates' not found in {roadmap_file}. "
            f"Available columns: {roadmap.columns.tolist()}"
        )

    min_lat, max_lat = float("inf"), float("-inf")
    min_lng, max_lng = float("inf"), float("-inf")
    centroids: list[list[float]] = []
    invalid_rows: list[int] = []

    for idx, coords_str in enumerate(roadmap["coordinates"].tolist()):
        if not isinstance(coords_str, str) or not coords_str.strip():
            invalid_rows.append(idx)
            centroids.append([float("nan"), float("nan")])
            continue

        try:
            coords = json.loads(coords_str)
        except json.JSONDecodeError:
            invalid_rows.append(idx)
            centroids.append([float("nan"), float("nan")])
            continue

        if not isinstance(coords, list) or len(coords) == 0:
            invalid_rows.append(idx)
            centroids.append([float("nan"), float("nan")])
            continue

        try:
            lngs = [float(c[0]) for c in coords]
            lats = [float(c[1]) for c in coords]
        except (TypeError, ValueError, IndexError):
            invalid_rows.append(idx)
            centroids.append([float("nan"), float("nan")])
            continue

        if any(lat < -90 or lat > 90 for lat in lats) or any(
            lng < -180 or lng > 180 for lng in lngs
        ):
            invalid_rows.append(idx)
            centroids.append([float("nan"), float("nan")])
            continue

        centroid_lat = float(sum(lats) / len(lats))
        centroid_lng = float(sum(lngs) / len(lngs))
        centroids.append([centroid_lat, centroid_lng])

        # Boundary from *all* points.
        min_lat = min(min_lat, min(lats))
        max_lat = max(max_lat, max(lats))
        min_lng = min(min_lng, min(lngs))
        max_lng = max(max_lng, max(lngs))

    if invalid_rows:
        preview = ",".join(str(i) for i in invalid_rows[:10])
        raise ValueError(
            "Roadmap contains invalid 'coordinates' rows; refusing to proceed to avoid road-id/index mismatch. "
            f"Invalid rows: {len(invalid_rows)} (first: {preview}{'...' if len(invalid_rows) > 10 else ''})."
        )

    if not (min_lat < max_lat and min_lng < max_lng):
        raise ValueError(
            f"Invalid boundary computed from roadmap points: "
            f"lat=[{min_lat}, {max_lat}], lng=[{min_lng}, {max_lng}]"
        )

    road_centroids_latlng = np.asarray(centroids, dtype=np.float64)
    grid_config = GridConfig(
        min_lat=float(min_lat),
        max_lat=float(max_lat),
        min_lng=float(min_lng),
        max_lng=float(max_lng),
        grid_size=DEFAULT_GRID_SIZE,
    )
    return road_centroids_latlng, grid_config


def _create_grid_mapper(
    *,
    roadmap_file: Path,
    device: str,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    road_centroids_latlng, grid_config = (
        _extract_road_centroids_and_boundary_from_roadmap(roadmap_file)
    )
    mapper = GridMapper(
        boundary=grid_config, road_centroids=road_centroids_latlng, verify_hw=None
    )
    road_to_token = torch.from_numpy(mapper.map_all()).to(device)
    return road_to_token, (int(mapper.grid_h), int(mapper.grid_w))


def _evaluate_csv(
    *,
    csv_path: Path,
    model: LMTADTeacher,
    road_to_token: torch.Tensor,
    device: str,
    batch_size: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    logger.info("Loading trajectories from %s...", csv_path)
    trajectories = load_hoser_trajectories(csv_path)
    if len(trajectories) == 0:
        return None

    logger.info("Evaluating %d trajectories...", len(trajectories))
    perplexities, within_split_outliers, _ = evaluate_trajectories_direct(
        trajectories=trajectories,
        model=model,
        road_to_token=road_to_token,
        device=device,
        batch_size=batch_size,
    )
    return perplexities, within_split_outliers, len(trajectories)


def _build_split_result(
    *,
    num_trajectories: int,
    perplexities: np.ndarray,
    outliers: np.ndarray,
    outlier_method: str,
    within_split_outliers: np.ndarray,
    within_split_outlier_rate: float,
    baseline_calibrated: Optional[Dict[str, Any]],
    abnormality_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "num_trajectories": int(num_trajectories),
        "mean_log_perplexity": float(perplexities.mean()),
        "median_log_perplexity": float(np.median(perplexities)),
        "std_log_perplexity": float(perplexities.std()),
        "min_log_perplexity": float(perplexities.min()),
        "max_log_perplexity": float(perplexities.max()),
        "outlier_method": str(outlier_method),
        "outlier_rate": float(outliers.mean()),
        "log_perplexity_values": perplexities.tolist(),
        "outlier_labels": outliers.tolist(),
        "within_split_outlier_rate": float(within_split_outlier_rate),
        "within_split_outlier_labels": within_split_outliers.tolist(),
        "baseline_calibrated": baseline_calibrated,
        "abnormality_stats": abnormality_stats,
    }


def _compute_outliers(
    *,
    split: str,
    perplexities: np.ndarray,
    within_split_outliers: np.ndarray,
    baseline_eval: Optional[Path],
    baseline_scores_by_split: Optional[ScoresBySplit],
    baseline_quantile: float,
    baseline_split_fallback: str,
) -> Tuple[np.ndarray, str, float, Optional[Dict[str, Any]]]:
    """Return outlier labels + metadata.

    If `baseline_scores_by_split` is provided, we override the evaluator's
    within-split outlier labels with a baseline-calibrated quantile threshold.
    Otherwise we keep the evaluator output.
    """
    within_rate = float(within_split_outliers.mean())

    if baseline_scores_by_split is None:
        return within_split_outliers, "within_split_95th_percentile", within_rate, None

    chosen_baseline_split = _pick_baseline_split(
        split=split,
        baseline_scores_by_split=baseline_scores_by_split,
        fallback_split=baseline_split_fallback,
    )
    base_scores = np.asarray(
        baseline_scores_by_split[chosen_baseline_split], dtype=np.float64
    )
    threshold = _quantile_threshold(base_scores, baseline_quantile)

    baseline_outlier_rate = float((base_scores > threshold).mean())
    calibrated_outliers = (perplexities > threshold).astype(np.float32)

    logger.info(
        "Baseline-calibrated threshold (target_split=%s): baseline_split=%s, q=%.3f, threshold=%.6f, "
        "baseline_outlier_rate=%.2f%%, target_outlier_rate=%.2f%%",
        str(split),
        chosen_baseline_split,
        float(baseline_quantile),
        float(threshold),
        baseline_outlier_rate * 100.0,
        float(calibrated_outliers.mean()) * 100.0,
    )

    baseline_calibrated = {
        "method": "quantile",
        "quantile": float(baseline_quantile),
        "threshold": float(threshold),
        "baseline_eval": str(baseline_eval) if baseline_eval is not None else None,
        "baseline_split_used": str(chosen_baseline_split),
        "baseline_outlier_rate": float(baseline_outlier_rate),
        "target_outlier_rate": float(calibrated_outliers.mean()),
    }

    return calibrated_outliers, "baseline_quantile", within_rate, baseline_calibrated


def evaluate_splits(
    data_dir: Path,
    roadmap_file: Path,
    lmtad_ckpt: Path,
    lmtad_repo: Path,
    device: str = "cuda:0",
    batch_size: int = 128,
    splits: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    sample_frac: float = 0.1,
    sample_seed: int = 42,
    baseline_eval: Optional[Path] = None,
    baseline_quantile: float = 0.95,
    baseline_split: str = "train",
    write_baseline: bool = False,
    baseline_out: Optional[Path] = None,
):
    if splits is None:
        splits = ["train", "val", "test"]

    if output_dir is None:
        output_dir = Path("tools_eval_lmtad") / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    _require_exists(data_dir, "Data directory")
    _require_exists(roadmap_file, "Roadmap file")
    _require_exists(lmtad_ckpt, "LM-TAD checkpoint")
    _require_exists(lmtad_repo, "LM-TAD repo")

    calibrator = BaselineCalibrator.from_optional_eval(
        baseline_eval=baseline_eval,
        quantile=baseline_quantile,
        fallback_split=baseline_split,
    )
    if calibrator.scores_by_split is not None:
        logger.info(
            "Using baseline-calibrated outliers: method=quantile, "
            f"q={calibrator.quantile}, fallback_baseline_split={calibrator.fallback_split}"
        )

    # Load model once
    logger.info(f"Loading LM-TAD teacher from {lmtad_ckpt}...")
    model = LMTADTeacher(
        repo_path=str(lmtad_repo),
        ckpt_path=str(lmtad_ckpt),
        device=device,
        dtype="float16",
        window=256,
    )
    logger.info("LM-TAD teacher loaded successfully")

    # Create grid mapper using the preferred conversion method:
    # - boundary from all polyline points
    # - centroid as mean-of-points
    road_to_token, (grid_h, grid_w) = _create_grid_mapper(
        roadmap_file=roadmap_file, device=device
    )
    logger.info("Created grid mapper: %dx%d cells", grid_h, grid_w)

    all_results: Dict[str, Dict[str, Any]] = {}
    writer = ResultsWriter(output_dir)

    for split, csv_file in _iter_splits(data_dir, splits):
        if not csv_file.exists():
            logger.warning("Split file not found, skipping: %s", csv_file)
            continue

        # Stream-sample CSV rows using Bernoulli sampling to avoid loading the
        # entire file into memory. This is best-effort: each input row is
        # independently included with probability `sample_frac`.
        try:
            target_csv = _prepare_target_csv(
                csv_file=csv_file,
                out_dir=output_dir,
                sample_frac=sample_frac,
                sample_seed=sample_seed,
            )
        except Exception as e:
            logger.error("Sampling failed for %s: %s", csv_file, e)
            target_csv = csv_file

        if target_csv is None:
            logger.warning("Sampled 0 rows from %s, skipping", csv_file)
            continue

        abnormality_stats = _abnormality_info_stats(target_csv)
        if abnormality_stats is not None:
            logger.info(
                "Split '%s': abnormal_fraction=%.2f%% (%d/%d rows have %s)",
                split,
                float(abnormality_stats["abnormal_fraction"]) * 100.0,
                int(abnormality_stats["num_abnormal"]),
                int(abnormality_stats["num_rows"]),
                str(abnormality_stats["column"]),
            )

        eval_out = _evaluate_csv(
            csv_path=target_csv,
            model=model,
            road_to_token=road_to_token,
            device=device,
            batch_size=batch_size,
        )
        if eval_out is None:
            logger.warning("No valid trajectories in %s, skipping", csv_file)
            continue

        perplexities, within_split_outliers, num_trajectories = eval_out
        outliers, outlier_method, within_split_outlier_rate, baseline_calibrated = (
            calibrator.apply(
                split=split,
                perplexities=perplexities,
                within_split_outliers=within_split_outliers,
            )
        )

        all_results[split] = _build_split_result(
            num_trajectories=num_trajectories,
            perplexities=perplexities,
            outliers=outliers,
            outlier_method=outlier_method,
            within_split_outliers=within_split_outliers,
            within_split_outlier_rate=within_split_outlier_rate,
            baseline_calibrated=baseline_calibrated,
            abnormality_stats=abnormality_stats,
        )

        logger.info(
            "Split '%s': mean_log_perplexity=%.4f, outlier_rate=%.2f%% (%s)",
            split,
            all_results[split]["mean_log_perplexity"],
            all_results[split]["outlier_rate"] * 100.0,
            outlier_method,
        )

        writer.write_split(split=split, payload=all_results[split])

    logger.info("Streaming results appended to: %s", writer.jsonl_path)
    logger.info("Aggregated results written to: %s", writer.agg_json_path)

    if write_baseline:
        out_path = baseline_out if baseline_out is not None else writer.baseline_path
        _write_baseline_eval_json(path=out_path, results_by_split=all_results)
        logger.info("Baseline eval written to: %s", out_path)
        logger.info(
            "Reuse it via: --baseline-eval %s --baseline-quantile <q>",
            out_path.parent,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate dataset CSV splits with LM-TAD"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Dataset name (used for default paths). "
            "In multi-dataset mode, use --baseline-dataset/--target-datasets instead."
        ),
    )

    # Multi-dataset convenience mode: run baseline then one or more targets.
    parser.add_argument(
        "--baseline-dataset",
        type=str,
        default=None,
        help="Baseline dataset name (e.g., Beijing).",
    )
    parser.add_argument(
        "--target-datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of target datasets to evaluate using the baseline threshold "
            "(e.g., Beijing_abnormal_3)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing per-dataset folders.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("tools_eval_lmtad"),
        help="Root directory for per-dataset evaluation outputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to dataset CSVs (overrides default)",
    )
    parser.add_argument(
        "--roadmap",
        type=Path,
        default=None,
        help=(
            "Path to roadmap.geo used for mapping roads to grid tokens. "
            "Default: <data-dir>/roadmap.geo if it exists."
        ),
    )
    parser.add_argument(
        "--lmtad-checkpoint",
        "--ckpt",
        dest="lmtad_checkpoint",
        type=Path,
        required=True,
        help="LM-TAD checkpoint path",
    )
    parser.add_argument(
        "--lmtad-repo",
        type=Path,
        default=Path("/home/mka299/LMTAD"),
        help="LM-TAD repo path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for evaluation"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to evaluate",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Directory to write results"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.1,
        help="Fraction of rows to sample from each split (0.0-1.0); 1.0 = full",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for sampling",
    )

    # Baseline-calibrated outlier thresholding.
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        default=None,
        help=(
            "Optional: baseline eval file or directory. If a directory is given, it should contain "
            "baseline_eval.json (preferred) or evaluation_results.json. "
            "If provided, outliers are computed using a fixed baseline-calibrated threshold (recommended)."
        ),
    )
    parser.add_argument(
        "--baseline-quantile",
        type=float,
        default=0.95,
        help=(
            "Baseline quantile q used to set the threshold when --baseline-eval is provided. "
            "Example: 0.95 targets ~5%% baseline outliers."
        ),
    )
    parser.add_argument(
        "--baseline-split",
        type=str,
        default="train",
        help=(
            "Fallback split name used to calibrate the baseline threshold when the evaluated split "
            "is not present in --baseline-eval."
        ),
    )

    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=(
            "Write a compact baseline file (baseline_eval.json) into --output-dir after evaluation. "
            "Use this on your first run over a normal baseline dataset."
        ),
    )
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=None,
        help=(
            "Optional: override where to write the baseline eval JSON when --write-baseline is set. "
            "Defaults to <output-dir>/baseline_eval.json."
        ),
    )

    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    multi_mode = bool(args.baseline_dataset or args.target_datasets)
    if multi_mode:
        if not args.baseline_dataset or not args.target_datasets:
            raise SystemExit(
                "Multi-dataset mode requires both --baseline-dataset and --target-datasets."
            )
        if args.data_dir is not None or args.output_dir is not None or args.baseline_eval is not None:
            logger.warning(
                "Ignoring --data-dir/--output-dir/--baseline-eval in multi-dataset mode; "
                "using --data-root/--out-root and the baseline run output as baseline_eval."
            )

        baseline_data_dir = args.data_root / str(args.baseline_dataset)
        baseline_output_dir = args.out_root / str(args.baseline_dataset)

        roadmap = args.roadmap
        if roadmap is None:
            candidate = baseline_data_dir / "roadmap.geo"
            if candidate.exists():
                roadmap = candidate
            else:
                raise SystemExit(
                    "--roadmap is required unless <baseline-data-dir>/roadmap.geo exists. "
                    f"Tried: {candidate}"
                )

        target_datasets = [
            s.strip() for s in str(args.target_datasets).split(",") if s.strip()
        ]
        if not target_datasets:
            raise SystemExit("--target-datasets parsed to an empty list")

        logger.info(
            "Running baseline dataset '%s' then %d target dataset(s)",
            str(args.baseline_dataset),
            len(target_datasets),
        )

        # 1) Baseline run (writes baseline_eval.json).
        evaluate_splits(
            data_dir=baseline_data_dir,
            roadmap_file=roadmap,
            lmtad_ckpt=args.lmtad_checkpoint,
            lmtad_repo=args.lmtad_repo,
            device=args.device,
            batch_size=args.batch_size,
            splits=splits,
            output_dir=baseline_output_dir,
            sample_frac=args.sample_frac,
            sample_seed=args.sample_seed,
            baseline_eval=None,
            baseline_quantile=args.baseline_quantile,
            baseline_split=args.baseline_split,
            write_baseline=True,
            baseline_out=None,
        )

        baseline_eval_dir = baseline_output_dir
        logger.info(
            "Reusing baseline eval from %s (q=%.3f)",
            baseline_eval_dir,
            float(args.baseline_quantile),
        )

        # 2) Target runs (use baseline threshold).
        for target in target_datasets:
            target_data_dir = args.data_root / target
            target_output_dir = args.out_root / target
            logger.info("Evaluating target dataset '%s'", target)
            evaluate_splits(
                data_dir=target_data_dir,
                roadmap_file=roadmap,
                lmtad_ckpt=args.lmtad_checkpoint,
                lmtad_repo=args.lmtad_repo,
                device=args.device,
                batch_size=args.batch_size,
                splits=splits,
                output_dir=target_output_dir,
                sample_frac=args.sample_frac,
                sample_seed=args.sample_seed,
                baseline_eval=baseline_eval_dir,
                baseline_quantile=args.baseline_quantile,
                baseline_split=args.baseline_split,
                write_baseline=False,
                baseline_out=None,
            )

        return

    # Single-dataset mode (backwards compatible).
    if not args.dataset:
        raise SystemExit(
            "Single-dataset mode requires --dataset (or use --baseline-dataset/--target-datasets)."
        )

    data_dir = args.data_dir if args.data_dir is not None else Path("data") / args.dataset
    roadmap = args.roadmap
    if roadmap is None:
        candidate = Path(data_dir) / "roadmap.geo"
        if candidate.exists():
            roadmap = candidate
        else:
            raise SystemExit(
                "--roadmap is required unless <data-dir>/roadmap.geo exists. "
                f"Tried: {candidate}"
            )

    evaluate_splits(
        data_dir=data_dir,
        roadmap_file=roadmap,
        lmtad_ckpt=args.lmtad_checkpoint,
        lmtad_repo=args.lmtad_repo,
        device=args.device,
        batch_size=args.batch_size,
        splits=splits,
        output_dir=args.output_dir,
        sample_frac=args.sample_frac,
        sample_seed=args.sample_seed,
        baseline_eval=args.baseline_eval,
        baseline_quantile=args.baseline_quantile,
        baseline_split=args.baseline_split,
        write_baseline=bool(args.write_baseline),
        baseline_out=args.baseline_out,
    )


if __name__ == "__main__":
    main()
