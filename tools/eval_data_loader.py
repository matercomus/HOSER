#!/usr/bin/env python3
"""Shared evaluation data loading utilities.

These helpers centralize loading of evaluation results, generation performance
metadata, and trajectory-level metrics so that plotting scripts can operate
without re-running the pipeline. The module intentionally contains no plotting
code, keeping data access testable and reusable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tools.model_detection import (
    get_display_name,
    get_model_color,
    get_model_line_style,
)


@dataclass
class GenerationPerformance:
    """Structured view of generation_performance metrics."""

    raw: Dict[str, Any]
    throughput_traj_per_sec: Optional[float] = None
    total_time_mean: Optional[float] = None
    total_time_median: Optional[float] = None
    total_time_p95: Optional[float] = None
    total_time_max: Optional[float] = None
    total_generation_time: Optional[float] = None
    forward_count_mean: Optional[float] = None
    forward_time_per_step_mean: Optional[float] = None
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.throughput_traj_per_sec = self.raw.get("throughput_traj_per_sec")
        self.total_time_mean = self.raw.get("total_time_mean")
        self.total_time_median = self.raw.get("total_time_median")
        self.total_time_p95 = self.raw.get("total_time_p95")
        self.total_time_max = self.raw.get("total_time_max")
        self.total_generation_time = self.raw.get("total_generation_time")
        self.forward_count_mean = self.raw.get("forward_count_mean")
        self.forward_time_per_step_mean = self.raw.get("forward_time_per_step_mean")
        self.device = self.raw.get("device")


@dataclass
class EvalRun:
    """Represents a single evaluation result directory."""

    path: Path
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    model_name: str
    od_source: str
    display_name: str
    color: str
    line_style: str
    generation_performance: Optional[GenerationPerformance] = None

    def as_row(self) -> Dict[str, Any]:
        """Flatten run data for tabular consumption."""

        row = {
            "model_name": self.model_name,
            "od_source": self.od_source,
            "display_name": self.display_name,
            "color": self.color,
            "line_style": self.line_style,
            "path": str(self.path),
        }
        row.update(self.metrics)
        if self.generation_performance:
            row.update(self.generation_performance.raw)
        return row


@dataclass
class TrajectoryMetrics:
    """Convenience wrapper for per-trajectory metrics collections."""

    run_path: Path
    records: List[Dict[str, Any]] = field(default_factory=list)


def _iter_eval_dirs(eval_dir: Path) -> Iterable[Path]:
    """Yield timestamped evaluation directories that have results."""

    eval_root = eval_dir / "eval"
    if not eval_root.exists():
        return []

    for subdir in sorted(eval_root.iterdir()):
        if not subdir.is_dir():
            continue
        results_file = subdir / "results.json"
        if results_file.exists():
            yield subdir


def load_eval_runs(
    eval_dir: Path,
    *,
    include_cross_dataset: bool = False,
    include_abnormal: bool = False,
) -> List[EvalRun]:
    """Load all evaluation runs meeting the provided filters."""

    runs: List[EvalRun] = []
    for run_dir in _iter_eval_dirs(eval_dir):
        results_file = run_dir / "results.json"
        try:
            with open(results_file, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except json.JSONDecodeError:
            continue

        metadata = metrics.get("metadata", {})
        if not include_cross_dataset and metadata.get("cross_dataset"):
            continue
        generated_file = metadata.get("generated_file", "")
        if not include_abnormal and "abnormal" in generated_file.lower():
            continue

        model_name = metadata.get("model_type") or _infer_model(generated_file)
        od_source = metadata.get("od_source", "unknown")

        perf = metrics.get("generation_performance")
        generation_perf = GenerationPerformance(perf) if perf else None

        display_name = get_display_name(model_name)
        color = get_model_color(model_name)
        line_style = get_model_line_style(model_name)

        runs.append(
            EvalRun(
                path=run_dir,
                metrics=metrics,
                metadata=metadata,
                model_name=model_name,
                od_source=od_source,
                display_name=display_name,
                color=color,
                line_style=line_style,
                generation_performance=generation_perf,
            )
        )

    return runs


def _infer_model(generated_file: str) -> str:
    """Fallback model detection when metadata omitted model_type."""

    if not generated_file:
        return "unknown"
    from tools.model_detection import extract_model_name

    return extract_model_name(generated_file)


def build_performance_table(runs: Iterable[EvalRun]) -> List[Dict[str, Any]]:
    """Convert loaded runs into a list of flat dicts for DataFrame creation."""

    table: List[Dict[str, Any]] = []
    for run in runs:
        # Skip runs without generation performance payloads
        if not run.generation_performance:
            continue
        table.append(run.as_row())
    return table


def load_trajectory_metrics(run: EvalRun) -> TrajectoryMetrics:
    """Load trajectory_metrics.json for a given run if it exists."""

    traj_file = run.path / "trajectory_metrics.json"
    records: List[Dict[str, Any]] = []
    if traj_file.exists():
        with open(traj_file, "r", encoding="utf-8") as handle:
            try:
                records = json.load(handle)
            except json.JSONDecodeError:
                records = []
    return TrajectoryMetrics(run_path=run.path, records=records)
