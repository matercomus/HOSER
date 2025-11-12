#!/usr/bin/env python3
"""
Plot Abnormal OD Analysis Results

This module provides plotting functionality for real abnormal trajectory analysis.
Plots include OD distribution, category summaries, temporal delay analysis,
and OD heatmaps (abnormal, normal, and comparison).

Usage (Module):
    from tools.plot_abnormal_analysis import plot_analysis_from_files
    
    plot_analysis_from_files(
        abnormal_od_pairs_file=Path("abnormal_od_pairs_porto_hoser.json"),
        real_data_files=[Path("data/porto_hoser/train.csv"), Path("data/porto_hoser/test.csv")],
        detection_results_files=[Path("abnormal/porto_hoser/train/real_data/detection_results.json"), ...],
        samples_dir=Path("abnormal/porto_hoser"),
        output_dir=Path("figures/abnormal_od/porto_hoser"),
        dataset="porto_hoser"
    )

Usage (CLI):
    uv run python tools/plot_abnormal_analysis.py \\
        --abnormal-od-pairs abnormal_od_pairs_porto_hoser.json \\
        --real-data-dir data/porto_hoser \\
        --detection-results-dir abnormal/porto_hoser \\
        --output-dir figures/abnormal_od/porto_hoser \\
        --dataset porto_hoser \\
        --include-normal
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import numpy as np

# Import extraction functions
from tools.extract_abnormal_od_pairs import (
    extract_normal_od_pairs,
    load_detection_results,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set publication-quality plot defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "font.family": "sans-serif",
    }
)


@dataclass
class AnalysisPlotConfig:
    """Configuration for abnormal analysis plotting"""

    top_n_origins: int = 20
    top_n_destinations: int = 20
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300


def load_abnormal_od_pairs(od_pairs_file: Path) -> Dict[str, Any]:
    """Load abnormal OD pairs from JSON file with fail-fast assertions

    Args:
        od_pairs_file: Path to abnormal OD pairs JSON file

    Returns:
        Dictionary containing OD pairs data

    Raises:
        AssertionError: If file doesn't exist or required keys are missing
    """
    assert od_pairs_file.exists(), f"OD pairs file not found: {od_pairs_file}"

    logger.info(f"📂 Loading abnormal OD pairs from {od_pairs_file}")

    with open(od_pairs_file, "r") as f:
        od_data = json.load(f)

    assert "od_pairs_by_category" in od_data, (
        f"OD pairs file must contain 'od_pairs_by_category' key. "
        f"Found keys: {list(od_data.keys())}"
    )

    # Convert lists to tuples (JSON doesn't preserve tuple types)
    for category, pairs in od_data["od_pairs_by_category"].items():
        od_data["od_pairs_by_category"][category] = [
            tuple(pair) if isinstance(pair, list) else pair for pair in pairs
        ]

    logger.info(
        f"✅ Loaded OD pairs with {len(od_data['od_pairs_by_category'])} categories"
    )
    return od_data


def extract_normal_od_pairs_from_data(
    real_data_files: List[Path],
    abnormal_traj_ids: Set[int],
    max_trajectories: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Extract normal OD pairs from trajectory CSV files

    Args:
        real_data_files: List of paths to real trajectory CSV files
        abnormal_traj_ids: Set of abnormal trajectory IDs to exclude
        max_trajectories: Optional limit on number of trajectories to process per file

    Returns:
        List of (origin, destination) tuples from normal trajectories

    Raises:
        AssertionError: If file list is empty or files don't exist
    """
    assert real_data_files, "Real data files list cannot be empty"
    for file in real_data_files:
        assert file.exists(), f"Real data file not found: {file}"

    logger.info(f"📂 Extracting normal OD pairs from {len(real_data_files)} files")

    all_normal_od_pairs = []

    for real_data_file in real_data_files:
        normal_od_pairs = extract_normal_od_pairs(
            real_data_file, abnormal_traj_ids, max_trajectories
        )
        all_normal_od_pairs.extend(normal_od_pairs)

    # Deduplicate
    unique_pairs = list(set(all_normal_od_pairs))

    logger.info(f"✅ Extracted {len(unique_pairs)} unique normal OD pairs")
    return unique_pairs


def compute_od_heatmap_matrix(
    od_pairs: List[Tuple[int, int]],
    top_origins: Optional[List[int]] = None,
    top_dests: Optional[List[int]] = None,
    top_n: int = 20,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Compute heatmap matrix from OD pair counts

    Args:
        od_pairs: List of (origin, destination) tuples
        top_origins: Optional list of top origins to use (if None, computed from data)
        top_dests: Optional list of top destinations to use (if None, computed from data)
        top_n: Number of top origins/destinations to use if not specified

    Returns:
        Tuple of (matrix, origins_list, destinations_list)

    Raises:
        AssertionError: If od_pairs is empty or top_n is invalid
    """
    assert od_pairs, "OD pairs list cannot be empty"
    assert top_n > 0, "top_n must be positive"

    # Count frequency of each OD pair
    od_counts = {}
    origin_counts = {}
    dest_counts = {}

    for origin, dest in od_pairs:
        od_key = (origin, dest)
        od_counts[od_key] = od_counts.get(od_key, 0) + 1
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        dest_counts[dest] = dest_counts.get(dest, 0) + 1

    # Determine top origins and destinations
    if top_origins is None:
        top_origins = sorted(origin_counts.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        top_origins = [origin for origin, _ in top_origins]

    if top_dests is None:
        top_dests = sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        top_dests = [dest for dest, _ in top_dests]

    # Create adjacency matrix
    matrix = np.zeros((len(top_origins), len(top_dests)))
    for i, origin in enumerate(top_origins):
        for j, dest in enumerate(top_dests):
            count = od_counts.get((origin, dest), 0)
            matrix[i, j] = count

    return matrix, top_origins, top_dests


def plot_abnormal_od_distribution(
    od_data: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[AnalysisPlotConfig] = None,
) -> None:
    """Plot distribution of abnormal OD pairs (origins and destinations)

    Args:
        od_data: Dictionary containing OD pairs by category
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If required keys are missing
    """
    assert "od_pairs_by_category" in od_data, (
        "OD data must contain 'od_pairs_by_category' key"
    )

    if config is None:
        config = AnalysisPlotConfig()

    logger.info("📊 Plotting abnormal OD distribution...")

    # Count frequency of each origin and destination
    origin_counts = {}
    dest_counts = {}

    for category, pairs in od_data.get("od_pairs_by_category", {}).items():
        for origin, dest in pairs:
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            dest_counts[dest] = dest_counts.get(dest, 0) + 1

    # Get top N most frequent origins and destinations
    top_origins = sorted(origin_counts.items(), key=lambda x: x[1], reverse=True)[
        : config.top_n_origins
    ]
    top_dests = sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[
        : config.top_n_destinations
    ]

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot origins
    origins, origin_freqs = zip(*top_origins) if top_origins else ([], [])
    if origins:
        bars1 = ax1.barh(range(len(origins)), origin_freqs, color="#3498db", alpha=0.8)
        ax1.set_yticks(range(len(origins)))
        ax1.set_yticklabels([f"Road {o}" for o in origins])
        ax1.set_xlabel("Frequency in Abnormal Trajectories")
        ax1.set_title(f"Top {len(origins)} Abnormal Origins", fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)

        # Add frequency labels
        for i, (bar, freq) in enumerate(zip(bars1, origin_freqs)):
            width = bar.get_width()
            ax1.text(
                width + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{freq}",
                ha="left",
                va="center",
                fontsize=9,
            )

    # Plot destinations
    dests, dest_freqs = zip(*top_dests) if top_dests else ([], [])
    if dests:
        bars2 = ax2.barh(range(len(dests)), dest_freqs, color="#e74c3c", alpha=0.8)
        ax2.set_yticks(range(len(dests)))
        ax2.set_yticklabels([f"Road {d}" for d in dests])
        ax2.set_xlabel("Frequency in Abnormal Trajectories")
        ax2.set_title(f"Top {len(dests)} Abnormal Destinations", fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        # Add frequency labels
        for i, (bar, freq) in enumerate(zip(bars2, dest_freqs)):
            width = bar.get_width()
            ax2.text(
                width + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{freq}",
                ha="left",
                va="center",
                fontsize=9,
            )

    plt.suptitle(
        f"Abnormal OD Distribution - {dataset}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_abnormal_categories_summary(
    od_data: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[AnalysisPlotConfig] = None,
) -> None:
    """Plot pie chart of abnormal categories distribution

    Args:
        od_data: Dictionary containing OD pairs by category
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If categories are empty
    """
    assert "od_pairs_by_category" in od_data, (
        "OD data must contain 'od_pairs_by_category' key"
    )

    categories = list(od_data.get("od_pairs_by_category", {}).keys())
    counts = [len(pairs) for pairs in od_data.get("od_pairs_by_category", {}).values()]

    assert categories, "No abnormal categories found in data"

    if config is None:
        config = AnalysisPlotConfig()

    logger.info("📊 Plotting abnormal categories summary...")

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    wedges, texts, autotexts = ax.pie(
        counts, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90
    )

    # Enhance appearance
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title(
        f"Abnormal Categories Distribution - {dataset}\n"
        f"Total: {sum(counts):,} unique OD pairs",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_temporal_delay_analysis(
    samples_files: List[Path],
    output_file: Path,
    dataset: str,
    config: Optional[AnalysisPlotConfig] = None,
) -> None:
    """Plot temporal delay analysis from Wang samples

    Args:
        samples_files: List of paths to Wang sample JSON files
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If no sample files exist
    """
    assert samples_files, "Samples files list cannot be empty"
    assert any(f.exists() for f in samples_files), (
        f"No samples files found: {samples_files}"
    )

    if config is None:
        config = AnalysisPlotConfig()

    logger.info("📊 Plotting temporal delay analysis...")

    all_samples = []
    for samples_file in samples_files:
        if samples_file.exists():
            with open(samples_file, "r") as f:
                samples = json.load(f)
                all_samples.extend(samples)

    if not all_samples:
        logger.warning("No temporal delay samples found for analysis")
        return

    # Extract time deviations
    time_deviations = []
    length_deviations = []
    baseline_types = []

    for sample in all_samples:
        details = sample.get("details", {})
        time_deviations.append(details.get("time_deviation_sec", 0))
        length_deviations.append(details.get("length_deviation_m", 0))
        baseline_types.append(details.get("baseline_type", "unknown"))

    # Create analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Time deviation histogram
    ax1.hist(time_deviations, bins=30, color="#3498db", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Time Deviation (seconds)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Time Deviations")
    ax1.grid(alpha=0.3)

    # Length deviation histogram
    ax2.hist(length_deviations, bins=30, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Length Deviation (meters)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Length Deviations")
    ax2.grid(alpha=0.3)

    # Baseline types distribution
    baseline_counts = {}
    for bt in baseline_types:
        baseline_counts[bt] = baseline_counts.get(bt, 0) + 1

    ax3.bar(
        baseline_counts.keys(), baseline_counts.values(), color="#2ecc71", alpha=0.8
    )
    ax3.set_xlabel("Baseline Type")
    ax3.set_ylabel("Count")
    ax3.set_title("Baseline Types in Abnormal Detection")
    ax3.tick_params(axis="x", rotation=45)

    # Time vs Length deviation scatter
    ax4.scatter(length_deviations, time_deviations, alpha=0.6, color="#9b59b6")
    ax4.set_xlabel("Length Deviation (meters)")
    ax4.set_ylabel("Time Deviation (seconds)")
    ax4.set_title("Time vs Length Deviations")
    ax4.grid(alpha=0.3)

    plt.suptitle(f"Temporal Delay Analysis - {dataset}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_abnormal_od_heatmap(
    od_data: Dict[str, Any],
    output_file: Path,
    dataset: str,
    config: Optional[AnalysisPlotConfig] = None,
) -> None:
    """Plot abnormal OD heatmap

    Args:
        od_data: Dictionary containing OD pairs by category
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If required keys are missing
    """
    assert "od_pairs_by_category" in od_data, (
        "OD data must contain 'od_pairs_by_category' key"
    )

    if config is None:
        config = AnalysisPlotConfig()

    logger.info("📊 Plotting abnormal OD heatmap...")

    # Count frequency of each OD pair (expand pairs to include frequency)
    od_pairs_with_counts = []
    for category, pairs in od_data.get("od_pairs_by_category", {}).items():
        for origin, dest in pairs:
            od_pairs_with_counts.append((origin, dest))

    if not od_pairs_with_counts:
        logger.warning("No OD pairs found for heatmap")
        return

    # Compute heatmap matrix (function will count frequencies internally)
    matrix, top_origins, top_dests = compute_od_heatmap_matrix(
        od_pairs_with_counts, top_n=config.top_n_origins
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=config.figure_size)

    im = ax.imshow(matrix, cmap="Reds", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(top_dests)))
    ax.set_yticks(np.arange(len(top_origins)))
    ax.set_xticklabels([f"R{d}" for d in top_dests], rotation=45, ha="right")
    ax.set_yticklabels([f"R{o}" for o in top_origins])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Frequency in Abnormal Trajectories", rotation=270, labelpad=20)

    ax.set_title(
        f"Top Abnormal OD Pairs Heatmap - {dataset}\n"
        f"(Top {len(top_origins)} origins × {len(top_dests)} destinations)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Destination Roads")
    ax.set_ylabel("Origin Roads")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_normal_od_heatmap(
    normal_od_pairs: List[Tuple[int, int]],
    output_file: Path,
    dataset: str,
    top_origins: Optional[List[int]] = None,
    top_dests: Optional[List[int]] = None,
    config: Optional[AnalysisPlotConfig] = None,
) -> None:
    """Plot normal OD heatmap for comparison

    Args:
        normal_od_pairs: List of (origin, destination) tuples from normal trajectories
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        top_origins: Optional list of origins to use (for comparison with abnormal)
        top_dests: Optional list of destinations to use (for comparison with abnormal)
        config: Optional plotting configuration

    Raises:
        AssertionError: If normal_od_pairs is empty
    """
    assert normal_od_pairs, "Normal OD pairs list cannot be empty"

    if config is None:
        config = AnalysisPlotConfig()

    logger.info("📊 Plotting normal OD heatmap...")

    # Compute heatmap matrix
    matrix, used_origins, used_dests = compute_od_heatmap_matrix(
        normal_od_pairs,
        top_origins=top_origins,
        top_dests=top_dests,
        top_n=config.top_n_origins,
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=config.figure_size)

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(used_dests)))
    ax.set_yticks(np.arange(len(used_origins)))
    ax.set_xticklabels([f"R{d}" for d in used_dests], rotation=45, ha="right")
    ax.set_yticklabels([f"R{o}" for o in used_origins])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Frequency in Normal Trajectories", rotation=270, labelpad=20)

    ax.set_title(
        f"Top Normal OD Pairs Heatmap - {dataset}\n"
        f"(Top {len(used_origins)} origins × {len(used_dests)} destinations)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Destination Roads")
    ax.set_ylabel("Origin Roads")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_od_heatmap_comparison(
    abnormal_od_pairs: List[Tuple[int, int]],
    normal_od_pairs: List[Tuple[int, int]],
    output_file: Path,
    dataset: str,
    config: Optional[AnalysisPlotConfig] = None,
) -> None:
    """Plot side-by-side abnormal vs normal OD heatmaps

    Args:
        abnormal_od_pairs: List of (origin, destination) tuples from abnormal trajectories
        normal_od_pairs: List of (origin, destination) tuples from normal trajectories
        output_file: Path to save the plot (PNG and SVG)
        dataset: Dataset name for title
        config: Optional plotting configuration

    Raises:
        AssertionError: If either list is empty
    """
    assert abnormal_od_pairs, "Abnormal OD pairs list cannot be empty"
    assert normal_od_pairs, "Normal OD pairs list cannot be empty"

    if config is None:
        config = AnalysisPlotConfig()

    logger.info("📊 Plotting OD heatmap comparison...")

    # Compute top origins/destinations from abnormal data
    abnormal_matrix, top_origins, top_dests = compute_od_heatmap_matrix(
        abnormal_od_pairs, top_n=config.top_n_origins
    )

    # Compute normal matrix using same origins/destinations for comparison
    normal_matrix, _, _ = compute_od_heatmap_matrix(
        normal_od_pairs, top_origins=top_origins, top_dests=top_dests
    )

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Abnormal heatmap
    im1 = ax1.imshow(abnormal_matrix, cmap="Reds", aspect="auto")
    ax1.set_xticks(np.arange(len(top_dests)))
    ax1.set_yticks(np.arange(len(top_origins)))
    ax1.set_xticklabels([f"R{d}" for d in top_dests], rotation=45, ha="right")
    ax1.set_yticklabels([f"R{o}" for o in top_origins])
    ax1.set_title("Abnormal OD Pairs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Destination Roads")
    ax1.set_ylabel("Origin Roads")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Frequency", rotation=270, labelpad=20)

    # Normal heatmap
    im2 = ax2.imshow(normal_matrix, cmap="Blues", aspect="auto")
    ax2.set_xticks(np.arange(len(top_dests)))
    ax2.set_yticks(np.arange(len(top_origins)))
    ax2.set_xticklabels([f"R{d}" for d in top_dests], rotation=45, ha="right")
    ax2.set_yticklabels([f"R{o}" for o in top_origins])
    ax2.set_title("Normal OD Pairs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Destination Roads")
    ax2.set_ylabel("Origin Roads")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Frequency", rotation=270, labelpad=20)

    plt.suptitle(
        f"OD Pairs Heatmap Comparison - {dataset}\n"
        f"(Top {len(top_origins)} origins × {len(top_dests)} destinations)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=config.dpi, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Saved: {output_file}")


def plot_analysis_from_files(
    abnormal_od_pairs_file: Path,
    real_data_files: List[Path],
    detection_results_files: List[Path],
    samples_dir: Path,
    output_dir: Path,
    dataset: str,
    config: Optional[AnalysisPlotConfig] = None,
    include_normal: bool = False,
) -> Dict[str, Path]:
    """Main entry point: generate all analysis plots from files

    Args:
        abnormal_od_pairs_file: Path to abnormal OD pairs JSON file
        real_data_files: List of paths to real trajectory CSV files
        detection_results_files: List of paths to detection results JSON files
        samples_dir: Directory containing Wang samples (train/real_data/samples/, test/real_data/samples/)
        output_dir: Directory to save all plots
        dataset: Dataset name
        config: Optional plotting configuration
        include_normal: Whether to generate normal OD heatmap and comparison

    Returns:
        Dictionary mapping plot names to their file paths

    Raises:
        AssertionError: If required files don't exist
    """
    assert abnormal_od_pairs_file.exists(), (
        f"OD pairs file not found: {abnormal_od_pairs_file}"
    )
    assert real_data_files, "Real data files list cannot be empty"
    for file in real_data_files:
        assert file.exists(), f"Real data file not found: {file}"
    assert output_dir.parent.exists(), (
        f"Output directory parent does not exist: {output_dir.parent}"
    )

    if config is None:
        config = AnalysisPlotConfig()

    logger.info("=" * 80)
    logger.info("📊 Generating Analysis Plots")
    logger.info("=" * 80)

    # Load abnormal OD pairs
    od_data = load_abnormal_od_pairs(abnormal_od_pairs_file)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate analysis plots
    plot_files = {}

    # 1. OD Distribution
    plot_files["abnormal_od_distribution"] = output_dir / "abnormal_od_distribution.png"
    plot_abnormal_od_distribution(
        od_data, plot_files["abnormal_od_distribution"], dataset, config
    )

    # 2. Categories Summary
    plot_files["abnormal_categories_summary"] = (
        output_dir / "abnormal_categories_summary.png"
    )
    plot_abnormal_categories_summary(
        od_data, plot_files["abnormal_categories_summary"], dataset, config
    )

    # 3. Temporal Delay Analysis
    samples_files = []
    for split in ["train", "test"]:
        samples_file = (
            samples_dir
            / split
            / "real_data"
            / "samples"
            / "wang_temporal_delay_samples.json"
        )
        if samples_file.exists():
            samples_files.append(samples_file)

    if samples_files:
        plot_files["temporal_delay_analysis"] = (
            output_dir / "temporal_delay_analysis.png"
        )
        plot_temporal_delay_analysis(
            samples_files, plot_files["temporal_delay_analysis"], dataset, config
        )

    # 4. Abnormal OD Heatmap
    plot_files["abnormal_od_heatmap"] = output_dir / "abnormal_od_heatmap.png"

    # Extract abnormal OD pairs and compute top origins/destinations for comparison
    abnormal_od_pairs = []
    for category, pairs in od_data.get("od_pairs_by_category", {}).items():
        abnormal_od_pairs.extend(pairs)

    # Compute top origins/destinations from abnormal data
    _, top_origins, top_dests = compute_od_heatmap_matrix(
        abnormal_od_pairs, top_n=config.top_n_origins
    )

    plot_abnormal_od_heatmap(
        od_data, plot_files["abnormal_od_heatmap"], dataset, config
    )

    # 5. Normal OD Heatmap and Comparison (if requested)
    if include_normal:
        # Collect all abnormal trajectory IDs
        all_abnormal_traj_ids = set()
        for det_file in detection_results_files:
            if det_file.exists():
                results = load_detection_results(det_file)
                abnormal_indices = results.get("abnormal_indices", {})
                for traj_ids in abnormal_indices.values():
                    all_abnormal_traj_ids.update(traj_ids)

        # Extract normal OD pairs
        normal_od_pairs = extract_normal_od_pairs_from_data(
            real_data_files, all_abnormal_traj_ids, max_trajectories=None
        )

        if normal_od_pairs:
            # Normal OD Heatmap
            plot_files["normal_od_heatmap"] = output_dir / "normal_od_heatmap.png"
            plot_normal_od_heatmap(
                normal_od_pairs,
                plot_files["normal_od_heatmap"],
                dataset,
                top_origins=top_origins,
                top_dests=top_dests,
                config=config,
            )

            # Comparison Heatmap
            plot_files["od_heatmap_comparison"] = (
                output_dir / "od_heatmap_comparison.png"
            )
            plot_od_heatmap_comparison(
                abnormal_od_pairs,
                normal_od_pairs,
                plot_files["od_heatmap_comparison"],
                dataset,
                config,
            )

    logger.info(f"✅ Generated {len(plot_files)} analysis plots in {output_dir}")
    return plot_files


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate analysis plots for abnormal OD workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all analysis plots
  uv run python tools/plot_abnormal_analysis.py \\
    --abnormal-od-pairs abnormal_od_pairs_porto_hoser.json \\
    --real-data-dir data/porto_hoser \\
    --detection-results-dir abnormal/porto_hoser \\
    --samples-dir abnormal/porto_hoser \\
    --output-dir figures/abnormal_od/porto_hoser \\
    --dataset porto_hoser

  # Include normal OD heatmap and comparison
  uv run python tools/plot_abnormal_analysis.py \\
    --abnormal-od-pairs abnormal_od_pairs_porto_hoser.json \\
    --real-data-dir data/porto_hoser \\
    --detection-results-dir abnormal/porto_hoser \\
    --samples-dir abnormal/porto_hoser \\
    --output-dir figures/abnormal_od/porto_hoser \\
    --dataset porto_hoser \\
    --include-normal
        """,
    )

    parser.add_argument(
        "--abnormal-od-pairs",
        type=Path,
        required=True,
        help="Path to abnormal OD pairs JSON file",
    )
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        required=True,
        help="Directory containing train.csv and test.csv",
    )
    parser.add_argument(
        "--detection-results-dir",
        type=Path,
        required=True,
        help="Directory containing detection_results.json files (train/real_data/, test/real_data/)",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=False,
        help="Directory containing Wang samples (optional, for temporal delay analysis)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'porto_hoser', 'BJUT_Beijing')",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top origins/destinations for heatmaps (default: 20)",
    )
    parser.add_argument(
        "--include-normal",
        action="store_true",
        help="Also generate normal OD heatmap and comparison",
    )

    args = parser.parse_args()

    # Fail-fast assertions
    assert args.abnormal_od_pairs.exists(), (
        f"OD pairs file not found: {args.abnormal_od_pairs}"
    )
    assert args.real_data_dir.exists(), (
        f"Real data directory not found: {args.real_data_dir}"
    )
    train_csv = args.real_data_dir / "train.csv"
    test_csv = args.real_data_dir / "test.csv"
    assert train_csv.exists() or test_csv.exists(), (
        f"Neither train.csv nor test.csv found in {args.real_data_dir}"
    )

    # Build file lists
    real_data_files = []
    if train_csv.exists():
        real_data_files.append(train_csv)
    if test_csv.exists():
        real_data_files.append(test_csv)

    detection_results_files = []
    if (
        args.detection_results_dir / "train" / "real_data" / "detection_results.json"
    ).exists():
        detection_results_files.append(
            args.detection_results_dir
            / "train"
            / "real_data"
            / "detection_results.json"
        )
    if (
        args.detection_results_dir / "test" / "real_data" / "detection_results.json"
    ).exists():
        detection_results_files.append(
            args.detection_results_dir / "test" / "real_data" / "detection_results.json"
        )

    samples_dir = args.samples_dir if args.samples_dir else args.detection_results_dir

    config = AnalysisPlotConfig(
        top_n_origins=args.top_n,
        top_n_destinations=args.top_n,
    )

    try:
        plot_files = plot_analysis_from_files(
            abnormal_od_pairs_file=args.abnormal_od_pairs,
            real_data_files=real_data_files,
            detection_results_files=detection_results_files,
            samples_dir=samples_dir,
            output_dir=args.output_dir,
            dataset=args.dataset,
            config=config,
            include_normal=args.include_normal,
        )

        print("\n✅ Analysis plots generated successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📊 Generated {len(plot_files)} plots")

    except Exception as e:
        logger.error(f"❌ Plotting failed: {e}")
        raise


if __name__ == "__main__":
    main()
