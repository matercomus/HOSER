#!/usr/bin/env python3
"""Standalone generation performance visualization tool.

This script consumes existing evaluation outputs and renders performance-focused
figures (throughput, latency, correlations) without re-running the pipeline.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from tools.eval_data_loader import (
        EvalRun,
        build_performance_table,
        load_eval_runs,
    )
    from tools.model_detection import extract_model_name, get_model_color
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tools.eval_data_loader import (
        EvalRun,
        build_performance_table,
        load_eval_runs,
    )
    from tools.model_detection import extract_model_name, get_model_color


PLOT_REGISTRY = {
    "efficiency": "plot_efficiency_tradeoff",
    "latency": "plot_latency_bars",
    "latency_percentiles": "plot_latency_percentiles",
    "speed_ranking": "plot_throughput_ranking",
    "heatmap": "plot_performance_heatmap",
    "slope": "plot_throughput_slope",
    "length_speed": "plot_length_vs_speed",
    "length_accuracy": "plot_length_vs_accuracy",
    "param_vs_throughput": "plot_parameter_count_vs_throughput",
    "parameter_counts": "plot_parameter_counts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate generation-performance plots without re-running evals"
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Evaluation directory containing eval/*/results.json",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write figures (default: <eval-dir>/figures/performance)",
    )
    parser.add_argument(
        "--plots",
        default="all",
        help=(
            "Comma-separated list of plots: efficiency,latency,latency_percentiles,"
            "speed_ranking,heatmap,slope,length_speed,length_accuracy,param_vs_throughput,parameter_counts"
        ),
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        dest="model_dirs",
        help="Directories containing .pth checkpoints for parameter counting (can repeat)",
    )
    parser.add_argument(
        "--style",
        help="Optional Matplotlib style sheet (e.g., 'seaborn-v0_8-darkgrid')",
    )
    parser.add_argument(
        "--include-cross",
        action="store_true",
        help="Include cross-dataset evaluation results",
    )
    parser.add_argument(
        "--include-abnormal",
        action="store_true",
        help="Include abnormal OD evaluation runs",
    )
    return parser.parse_args()


class PerformanceVisualizer:
    """Generate performance-focused plots from evaluation runs."""

    def __init__(
        self,
        runs: Sequence[EvalRun],
        output_dir: Path,
        param_counts: Optional[Dict[str, int]] = None,
    ):
        self.runs = list(runs)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows = build_performance_table(self.runs)
        self.param_counts = param_counts or {}

    def generate(self, requested: Iterable[str]) -> None:
        for plot in requested:
            method_name = PLOT_REGISTRY.get(plot)
            if not method_name:
                print(f"⚠️  Unknown plot key: {plot}")
                continue
            method = getattr(self, method_name, None)
            if not method:
                print(f"⚠️  Plot method missing: {method_name}")
                continue
            if not self.rows:
                print("⚠️  No runs with generation_performance; skipping plots")
                return
            method()

    def plot_efficiency_tradeoff(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        xs = []
        ys = []
        labels = []
        colors = []
        for row in self.rows:
            throughput = row.get("throughput_traj_per_sec")
            edr = row.get("EDR")
            if throughput is None or edr is None:
                continue
            xs.append(throughput)
            ys.append(1.0 - edr)
            colors.append(row.get("color"))
            labels.append(f"{row['display_name']} ({row['od_source']})")
        if not xs:
            plt.close(fig)
            return
        ax.scatter(xs, ys, c=colors, s=80, edgecolor="k", linewidth=0.5)
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, label, fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("Throughput (trajectories / second)")
        ax.set_ylabel("1 - EDR (higher is better)")
        ax.set_title("Efficiency vs Similarity Trade-off")
        ax.grid(alpha=0.3)
        self._save(fig, "efficiency_tradeoff")

    def plot_latency_bars(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        models = [f"{row['display_name']} ({row['od_source']})" for row in self.rows]
        medians = [row.get("total_time_median") for row in self.rows]
        p95s = [row.get("total_time_p95") for row in self.rows]
        if not any(medians):
            plt.close(fig)
            return
        x = np.arange(len(models))
        ax.bar(x, medians, color=[row.get("color") for row in self.rows])
        for idx, val in enumerate(p95s):
            if val is not None:
                ax.errorbar(
                    x[idx],
                    medians[idx] if medians[idx] is not None else 0,
                    yerr=val - (medians[idx] or 0),
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                )
        ax.set_ylabel("Time per trajectory (s)")
        ax.set_title("Latency Summary (median with 95th percentile whiskers)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "latency_summary")

    def plot_throughput_ranking(self) -> None:
        data = [
            (
                row.get("throughput_traj_per_sec"),
                f"{row['display_name']} ({row['od_source']})",
                row.get("color") or "#333333",
            )
            for row in self.rows
            if row.get("throughput_traj_per_sec") is not None
        ]
        if not data:
            return
        data.sort(key=lambda item: item[0], reverse=True)
        speeds, labels, colors = zip(*data)
        fig_height = max(4.5, 0.45 * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, speeds, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        for idx, value in enumerate(speeds):
            ax.text(
                value + max(speeds) * 0.01,
                y_pos[idx],
                f"{value:.2f}",
                va="center",
                fontsize=9,
            )
        ax.set_xlabel("Throughput (trajectories / second)")
        ax.set_title("Generation Speed Ranking (highest to lowest)")
        ax.grid(axis="x", alpha=0.3)
        self._save(fig, "throughput_ranking")

    def plot_latency_percentiles(self) -> None:
        data = [
            (
                row.get("total_time_median"),
                row.get("total_time_p95"),
                f"{row['display_name']} ({row['od_source']})",
                row.get("color") or "#333333",
            )
            for row in self.rows
            if row.get("total_time_median") is not None
            and row.get("total_time_p95") is not None
        ]
        if not data:
            return
        data.sort(key=lambda item: item[0])
        medians, p95s, labels, colors = zip(*data)
        fig_height = max(4.5, 0.45 * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        y_pos = np.arange(len(labels))
        for idx, (median, p95, color) in enumerate(zip(medians, p95s, colors)):
            ax.hlines(y_pos[idx], median, p95, color=color, linewidth=3)
        ax.scatter(medians, y_pos, color="black", marker="o", label="Median")
        ax.scatter(
            p95s, y_pos, color="black", marker="|", s=120, label="95th percentile"
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Latency per trajectory (seconds)")
        ax.set_title("Latency Percentiles by Model")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right")
        self._save(fig, "latency_percentiles")

    def plot_performance_heatmap(self) -> None:
        columns = [
            "throughput_traj_per_sec",
            "total_time_median",
            "total_time_p95",
            "total_time_max",
            "total_generation_time",
            "forward_count_mean",
        ]
        data = np.array([[row.get(col) or 0.0 for col in columns] for row in self.rows])
        if not data.size:
            return
        norm = np.zeros_like(data, dtype=float)
        for col_idx in range(data.shape[1]):
            col = data[:, col_idx]
            min_val, max_val = np.min(col), np.max(col)
            if math.isclose(max_val, min_val):
                norm[:, col_idx] = 0.0
            else:
                norm[:, col_idx] = (col - min_val) / (max_val - min_val)
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(self.rows) + 2))
        heatmap = ax.imshow(norm.T, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(self.rows)))
        labels = [f"{row['display_name']}\n({row['od_source']})" for row in self.rows]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns)
        for i in range(len(self.rows)):
            for j, col in enumerate(columns):
                value = self.rows[i].get(col)
                if value is None:
                    continue
                ax.text(i, j, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(heatmap, ax=ax, shrink=0.6, label="Normalized Score")
        ax.set_title("Generation Performance Overview")
        fig.tight_layout()
        self._save(fig, "performance_heatmap")

    def plot_throughput_slope(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        grouped: Dict[str, Dict[str, float]] = {}
        for row in self.rows:
            grouped.setdefault(row["model_name"], {})[row["od_source"]] = row.get(
                "throughput_traj_per_sec"
            )
        for model, values in grouped.items():
            train = values.get("train")
            test = values.get("test")
            if train is None or test is None:
                continue
            x = [0, 1]
            y = [train, test]
            ax.plot(
                x,
                y,
                marker="o",
                label=model,
                color=self._pick_color(model),
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Train", "Test"])
        ax.set_ylabel("Throughput (traj/s)")
        ax.set_title("Throughput Generalization (Train vs Test)")
        ax.grid(alpha=0.3)
        ax.legend()
        self._save(fig, "throughput_slope")

    def plot_length_vs_speed(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        xs = []
        ys = []
        colors = []
        labels = []
        for row in self.rows:
            dist = row.get("Distance_gen_mean")
            throughput = row.get("throughput_traj_per_sec")
            if dist is None or throughput is None:
                continue
            xs.append(dist)
            ys.append(throughput)
            colors.append(row.get("color"))
            labels.append(f"{row['display_name']} ({row['od_source']})")
        if not xs:
            plt.close(fig)
            return
        ax.scatter(xs, ys, c=colors, s=80, edgecolor="black", linewidth=0.5)
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, label, fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("Average generated distance (km)")
        ax.set_ylabel("Throughput (traj/s)")
        ax.set_title("Trajectory Length vs Generation Speed")
        ax.grid(alpha=0.3)
        self._save(fig, "length_vs_speed")

    def plot_length_vs_accuracy(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        xs = []
        ys = []
        colors = []
        labels = []
        for row in self.rows:
            dist = row.get("Distance_gen_mean")
            hausdorff = row.get("Hausdorff_km")
            if dist is None or hausdorff is None:
                continue
            xs.append(dist)
            ys.append(hausdorff)
            colors.append(row.get("color"))
            labels.append(f"{row['display_name']} ({row['od_source']})")
        if not xs:
            plt.close(fig)
            return
        ax.scatter(xs, ys, c=colors, s=80, edgecolor="black", linewidth=0.5)
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, label, fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("Average generated distance (km)")
        ax.set_ylabel("Hausdorff distance (km)")
        ax.set_title("Trajectory Length vs Accuracy")
        ax.grid(alpha=0.3)
        self._save(fig, "length_vs_accuracy")

    def plot_parameter_count_vs_throughput(self) -> None:
        if not self.param_counts:
            print(
                "⚠️  Parameter counts unavailable; skipping model-size vs throughput plot"
            )
            return
        points: List[tuple[float, float, str, str]] = []
        for row in self.rows:
            throughput = row.get("throughput_traj_per_sec")
            if throughput is None:
                continue
            param_count = self._resolve_param_count_for_row(row)
            if param_count is None:
                continue
            label = f"{row['display_name']} ({row['od_source']})"
            color = row.get("color") or "#333333"
            points.append((param_count, throughput, label, color))
        if not points:
            print("⚠️  No runs had both throughput and parameter counts; skipping plot")
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        for param_count, throughput, label, color in points:
            x_val = param_count / 1_000_000.0
            ax.scatter(
                x_val,
                throughput,
                color=color,
                s=80,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.text(x_val, throughput, label, fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("Parameter count (millions)")
        ax.set_ylabel("Throughput (traj/s)")
        ax.set_title("Model Size vs Generation Speed")
        ax.grid(alpha=0.3)
        self._save(fig, "parameter_vs_throughput")

    def plot_parameter_counts(self) -> None:
        if not self.param_counts:
            print("⚠️  Parameter counts unavailable; skipping parameter count summary")
            return
        items = sorted(
            self.param_counts.items(), key=lambda pair: pair[1], reverse=True
        )
        names = [name for name, _ in items]
        counts = [value for _, value in items]
        counts_millions = [count / 1_000_000.0 for count in counts]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
        x = np.arange(len(names))
        bar_colors = [get_model_color(name) for name in names]
        bars = ax.bar(x, counts_millions, color=bar_colors)
        for bar, value in zip(bars, counts_millions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}M",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Parameters (millions)")
        ax.set_title("Model Parameter Counts")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "parameter_counts")

    def _save(self, fig: plt.Figure, stem: str) -> None:
        fig.tight_layout()
        png_path = self.output_dir / f"{stem}.png"
        pdf_path = self.output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)

    def _pick_color(self, model_name: str) -> str:
        for row in self.rows:
            if row["model_name"] == model_name:
                return row.get("color") or "#333333"
        return "#333333"

    def _resolve_param_count_for_row(self, row: Dict[str, Any]) -> Optional[int]:
        for candidate in self._candidate_model_names(row):
            count = self.param_counts.get(candidate)
            if count is not None:
                return count
        return None

    def _candidate_model_names(self, row: Dict[str, Any]) -> List[str]:
        metadata = row.get("metadata") or {}
        candidates: List[str] = []
        seen: set[str] = set()
        base_names = []
        model_name = row.get("model_name")
        if model_name:
            base_names.append(str(model_name))
        metadata_model = metadata.get("model_type")
        if metadata_model and metadata_model not in base_names:
            base_names.append(str(metadata_model))
        seed = metadata.get("seed")
        for base in base_names:
            normalized = extract_model_name(base) or base
            if normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)
            if seed is not None:
                suffix = f"seed{seed}"
                if not normalized.endswith(suffix):
                    seeded = f"{normalized}_{suffix}"
                    if seeded not in seen:
                        seen.add(seeded)
                        candidates.append(seeded)
        return candidates


def load_parameter_counts(model_dirs: Iterable[Path]) -> Dict[str, int]:
    """Return normalized model names mapped to their parameter counts."""

    counts: Dict[str, int] = {}
    for directory in model_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pth")):
            model_key = extract_model_name(path.stem)
            if not model_key:
                continue
            try:
                checkpoint = torch.load(path, map_location="cpu")
            except Exception as exc:
                print(f"⚠️  Failed to load {path.name}: {exc}")
                continue
            state = _extract_state_dict_from_checkpoint(checkpoint)
            if not state:
                continue
            counts[model_key] = _count_parameters(state)
    return counts


def _extract_state_dict_from_checkpoint(
    checkpoint: Any,
) -> Optional[Dict[str, torch.Tensor]]:
    """Find the underlying state dict inside a checkpoint payload."""

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        return None
    for key in ("state_dict", "model", "network", "state"):
        candidate = checkpoint.get(key)
        if candidate is None:
            continue
        if isinstance(candidate, dict):
            return _extract_state_dict_from_checkpoint(candidate)
        if isinstance(candidate, torch.nn.Module):
            return candidate.state_dict()
    tensor_entries = {
        name: value
        for name, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if tensor_entries:
        return tensor_entries
    return None


def _count_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in state_dict.values())


def resolve_plot_list(arg_value: str) -> List[str]:
    if arg_value.strip().lower() in {"all", "*"}:
        return list(PLOT_REGISTRY.keys())
    return [token.strip() for token in arg_value.split(",") if token.strip()]


def main() -> None:
    args = parse_args()
    if args.style:
        plt.style.use(args.style)
    eval_dir = Path(args.eval_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else eval_dir / "figures" / "performance"
    )
    runs = load_eval_runs(
        eval_dir,
        include_cross_dataset=args.include_cross,
        include_abnormal=args.include_abnormal,
    )
    if not runs:
        raise SystemExit("No evaluation runs found; nothing to plot")
    model_dir_candidates: List[Path] = []
    if args.model_dirs:
        model_dir_candidates.extend(Path(d).resolve() for d in args.model_dirs)
    inferred_dir = eval_dir / "models"
    if inferred_dir.is_dir():
        model_dir_candidates.append(inferred_dir)

    resolved_model_dirs: List[Path] = []
    seen_dirs = set()
    for candidate in model_dir_candidates:
        if not candidate.exists():
            print(f"⚠️  Model directory {candidate} not found; skipping")
            continue
        if not candidate.is_dir():
            print(f"⚠️  Model path {candidate} is not a directory; skipping")
            continue
        candidate_str = str(candidate)
        if candidate_str in seen_dirs:
            continue
        seen_dirs.add(candidate_str)
        resolved_model_dirs.append(candidate)

    param_counts: Dict[str, int] = (
        load_parameter_counts(resolved_model_dirs) if resolved_model_dirs else {}
    )
    visualizer = PerformanceVisualizer(runs, output_dir, param_counts)
    visualizer.generate(resolve_plot_list(args.plots))


if __name__ == "__main__":
    main()
