"""
Convert HOSER Trajectories to LM-TAD Format
==========================================

This module provides tools to convert HOSER-generated trajectories (road IDs)
to LM-TAD format (grid tokens) for teacher model evaluation.

Key Features:
- Reuses existing GridMapper infrastructure from critics/grid_mapper.py
- Supports batch processing for memory efficiency
- Handles both Porto and Beijing datasets
- Compatible with pre-converted LM-TAD datasets
- Ensures grid dimension consistency with teacher model

Usage:
    from tools.convert_to_lmtad_format import convert_hoser_to_lmtad_format

    convert_hoser_to_lmtad_format(
        trajectory_file=Path("generated_trajectories.csv"),
        roadmap_file=Path("data/porto_hoser/roadmap.geo"),
        output_file=Path("trajectories_lmtad_format.csv"),
        vocab_file=Path("vocab.json"),
        dataset="porto_hoser"
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from critics.grid_mapper import GridConfig, GridMapper

logger = logging.getLogger(__name__)

# Dataset grid configurations from pre-converted LM-TAD data
DATASET_CONFIGS = {
    "porto_hoser": {
        "grid_size": 0.001,
        "downsample_factor": 1,
        "vocab_path": "/home/matt/Dev/LMTAD/data/porto_hoser/vocab.json",
    },
    "beijing_hoser_reference": {
        "grid_size": 0.001,
        "downsample_factor": 1,
        "vocab_path": "/home/matt/Dev/LMTAD/data/beijing_hoser_reference/vocab.json",
    },
}


def extract_road_centroids(roadmap_file: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract road centroids and boundaries from roadmap.geo.

    Parameters
    ----------
    roadmap_file : Path
        Path to roadmap.geo containing road network geometry

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        - road_centroids: Shape (N, 2) array of (lat, lng) for each road
        - boundary: Dict with min_lat, max_lat, min_lng, max_lng

    Notes
    -----
    Centroids are computed as the mean of all coordinate points for each road,
    matching the LM-TAD preprocessing approach.
    """
    if not roadmap_file.exists():
        raise FileNotFoundError(f"Roadmap file not found: {roadmap_file}")

    logger.info("Loading road network geometry...")

    # Load roadmap with schema overrides for problematic columns
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

    roadmap = pd.read_csv(roadmap_file, dtype=schema_overrides)
    centroids = []
    min_lat, max_lat = float("inf"), float("-inf")
    min_lng, max_lng = float("inf"), float("-inf")

    for coords_str in tqdm(roadmap["coordinates"], desc="Processing roads"):
        coords = json.loads(coords_str)

        # Calculate centroid
        lats = [coord[1] for coord in coords]
        lngs = [coord[0] for coord in coords]
        centroid_lat = sum(lats) / len(lats)
        centroid_lng = sum(lngs) / len(lngs)

        centroids.append([centroid_lat, centroid_lng])

        # Update boundaries
        min_lat = min(min_lat, min(lats))
        max_lat = max(max_lat, max(lats))
        min_lng = min(min_lng, min(lngs))
        max_lng = max(max_lng, max(lngs))

    road_centroids = np.array(centroids, dtype=np.float64)

    boundary = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lng": min_lng,
        "max_lng": max_lng,
    }

    return road_centroids, boundary


def create_grid_mapper(
    dataset: str,
    road_centroids: np.ndarray,
    boundary: Dict[str, float],
) -> Tuple[GridMapper, Dict[str, int]]:
    """
    Create GridMapper with dataset-specific configuration.

    Parameters
    ----------
    dataset : str
        Dataset name: "porto_hoser" or "beijing_hoser_reference"
    road_centroids : np.ndarray
        (N, 2) array of (lat, lng) for each road centroid
    boundary : Dict[str, float]
        Geographic boundaries from roadmap

    Returns
    -------
    Tuple[GridMapper, Dict[str, int]]
        - mapper: Configured GridMapper instance
        - vocab: Grid token vocabulary mapping

    Notes
    -----
    Uses the same grid configuration as pre-converted LM-TAD datasets
    to ensure token consistency between HOSER and LM-TAD trajectories.
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset]

    # Create GridConfig for mapper
    grid_config = GridConfig(
        min_lat=float(boundary["min_lat"]),
        max_lat=float(boundary["max_lat"]),
        min_lng=float(boundary["min_lng"]),
        max_lng=float(boundary["max_lng"]),
        grid_size=config["grid_size"],
        downsample_factor=config["downsample_factor"],
    )

    # Load reference vocab to verify grid dimensions
    ref_vocab_path = Path(config["vocab_path"])
    if ref_vocab_path.exists():
        # Calculate grid dimensions
        lat_span = boundary["max_lat"] - boundary["min_lat"]
        lng_span = boundary["max_lng"] - boundary["min_lng"]
        grid_h = int(lat_span / config["grid_size"]) + 1
        grid_w = int(lng_span / config["grid_size"]) + 1

        # Apply downsampling
        if config["downsample_factor"] > 1:
            grid_h //= config["downsample_factor"]
            grid_w //= config["downsample_factor"]

        verify_hw = (grid_h, grid_w)
        logger.info(f"Grid verification from reference vocab: {verify_hw}")
    else:
        verify_hw = None
        logger.warning(f"Reference vocab not found: {ref_vocab_path}")
        logger.warning("Proceeding without grid verification")

    # Create mapper and vocabulary
    mapper = GridMapper(
        boundary=grid_config, road_centroids=road_centroids, verify_hw=verify_hw
    )

    # Create vocabulary
    max_token = mapper.grid_h * mapper.grid_w - 1
    vocab = {str(i): i for i in range(max_token + 1)}
    vocab["PAD"] = len(vocab)
    vocab["EOT"] = len(vocab)
    vocab["SOT"] = len(vocab)

    return mapper, vocab


def convert_trajectory_batch(
    trajectory_file: Path,
    mapper: GridMapper,
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Convert HOSER trajectories to grid tokens in batches.

    Parameters
    ----------
    trajectory_file : Path
        HOSER trajectory CSV with rid_list column
    mapper : GridMapper
        Configured GridMapper instance
    batch_size : int, default=10000
        Batch size for trajectory processing

    Returns
    -------
    pd.DataFrame
        DataFrame with converted grid token trajectories

    Notes
    -----
    Supports both string list "[123, 456]" and comma-separated "123,456" formats
    for rid_list column. Uses vectorized road-to-token mapping for efficiency.
    """
    df = pd.read_csv(trajectory_file)

    if "rid_list" not in df.columns:
        raise ValueError(
            f"Column 'rid_list' not found in {trajectory_file}. "
            f"Found: {df.columns.tolist()}"
        )

    # Get road-to-grid mapping
    road_to_grid = mapper.map_all()

    # Convert trajectories
    converted = []
    skipped = 0

    logger.info(f"Converting {len(df)} trajectories...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        try:
            # Parse rid_list (handle both "[123,456]" and "123,456" formats)
            rid_list_str = row["rid_list"]
            if isinstance(rid_list_str, str):
                rid_list_str = rid_list_str.strip()
                if rid_list_str.startswith("["):
                    # List string format: "[123, 456]"
                    rid_list = eval(rid_list_str)
                else:
                    # Comma-separated format: "123,456"
                    rid_list = [int(rid.strip()) for rid in rid_list_str.split(",")]

            # Convert to numpy array for vectorized mapping
            road_ids = np.array(rid_list, dtype=np.int64)

            # Map road IDs to grid tokens and convert to Python ints
            grid_tokens = [int(t) for t in road_to_grid[road_ids]]

            # Add to results
            converted.append(grid_tokens)

        except Exception as e:
            logger.warning(f"Failed to convert trajectory {idx}: {e}")
            skipped += 1
            continue

    if skipped > 0:
        logger.warning(f"Skipped {skipped} trajectories due to errors")

    converted_df = pd.DataFrame({"trajectory_tokens": converted})

    logger.info(f"Successfully converted {len(converted_df)} trajectories")

    return converted_df


def save_lmtad_format(
    df: pd.DataFrame,
    output_file: Path,
    vocab_file: Path,
    vocab: Dict[str, int],
) -> None:
    """
    Save trajectories and vocabulary in LM-TAD format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trajectory_tokens column
    output_file : Path
        Output file for trajectories
    vocab_file : Path
        Output file for vocabulary
    vocab : Dict[str, int]
        Grid token vocabulary mapping
    """
    import tempfile
    import os
    import shutil

    # Create output directories safely
    os.makedirs(output_file.parent, exist_ok=True)
    os.makedirs(vocab_file.parent, exist_ok=True)

    # Save trajectories atomically using a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=output_file.parent
    ) as tmp:
        for tokens in df["trajectory_tokens"]:
            tmp.write(f"{tokens}\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    shutil.move(tmp.name, output_file)

    # Save vocabulary atomically using a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=vocab_file.parent
    ) as tmp:
        json.dump(vocab, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
    shutil.move(tmp.name, vocab_file)

    logger.info(f"Saved {len(df)} trajectories to {output_file}")
    logger.info(f"Saved vocabulary ({len(vocab)} tokens) to {vocab_file}")


def convert_hoser_to_lmtad_format(
    trajectory_file: Path,
    roadmap_file: Path,
    output_file: Path,
    vocab_file: Path,
    dataset: str,
    batch_size: int = 10000,
) -> Path:
    """
    Convert HOSER trajectories to LM-TAD format.

    This is the main entry point for converting HOSER-generated trajectories
    (road IDs) to LM-TAD format (grid tokens) using the same grid mapping
    as the teacher model.

    Parameters
    ----------
    trajectory_file : Path
        HOSER CSV file with rid_list column (format: "[123,456,789]")
    roadmap_file : Path
        roadmap.geo file for centroid extraction
    output_file : Path
        Output CSV file for LM-TAD format trajectories
    vocab_file : Path
        Output vocab.json file with grid configuration
    dataset : str
        Dataset name: "porto_hoser" or "beijing_hoser_reference"
    batch_size : int, default=10000
        Batch size for processing trajectories

    Returns
    -------
    Path
        Path to the converted output file

    Examples
    --------
    >>> convert_hoser_to_lmtad_format(
    ...     trajectory_file=Path("generated_trajectories.csv"),
    ...     roadmap_file=Path("data/porto_hoser/roadmap.geo"),
    ...     output_file=Path("trajectories_lmtad_format.csv"),
    ...     vocab_file=Path("vocab.json"),
    ...     dataset="porto_hoser"
    ... )
    """
    logger.info("=" * 80)
    logger.info("Converting HOSER trajectories to LM-TAD format")
    logger.info("=" * 80)
    logger.info(f"Input: {trajectory_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Dataset: {dataset}")

    # Step 1: Extract road centroids
    logger.info("\n[1/4] Extracting road centroids...")
    road_centroids, boundary = extract_road_centroids(roadmap_file)
    logger.info(f"  Extracted {len(road_centroids)} road centroids")
    logger.info(
        f"  Boundary: lat=[{boundary['min_lat']:.6f}, {boundary['max_lat']:.6f}], "
        f"lng=[{boundary['min_lng']:.6f}, {boundary['max_lng']:.6f}]"
    )

    # Step 2: Create grid mapper
    logger.info("\n[2/4] Creating grid mapper...")
    grid_mapper, vocab = create_grid_mapper(dataset, road_centroids, boundary)
    logger.info(f"  Grid dimensions: {grid_mapper.grid_h} x {grid_mapper.grid_w}")
    logger.info(f"  Vocabulary size: {len(vocab)} tokens")

    # Step 3: Convert trajectories
    logger.info("\n[3/4] Converting trajectories...")
    converted_df = convert_trajectory_batch(
        trajectory_file, grid_mapper, batch_size=batch_size
    )
    logger.info(f"  Converted {len(converted_df)} trajectories")

    # Step 4: Save LM-TAD format
    logger.info("\n[4/4] Saving LM-TAD format...")
    save_lmtad_format(converted_df, output_file, vocab_file, vocab)
    logger.info(f"  Saved trajectories: {output_file}")
    logger.info(f"  Saved vocabulary: {vocab_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Conversion complete!")
    logger.info("=" * 80)

    return output_file


def main() -> None:
    """Command-line interface for trajectory conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HOSER trajectories to LM-TAD format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Porto generated trajectories
  python tools/convert_to_lmtad_format.py \\
    --trajectory-file eval_dir/generated/hoser_abnormal_od.csv \\
    --roadmap-file data/porto_hoser/roadmap.geo \\
    --output-file eval_dir/lmtad/trajectories.csv \\
    --vocab-file eval_dir/lmtad/vocab.json \\
    --dataset porto_hoser

  # Convert Beijing generated trajectories
  python tools/convert_to_lmtad_format.py \\
    --trajectory-file eval_dir/generated/hoser_abnormal_od.csv \\
    --roadmap-file data/beijing_hoser_reference/roadmap.geo \\
    --output-file eval_dir/lmtad/trajectories.csv \\
    --vocab-file eval_dir/lmtad/vocab.json \\
    --dataset beijing_hoser_reference
        """,
    )

    parser.add_argument(
        "--trajectory-file",
        type=Path,
        required=True,
        help="HOSER CSV file with rid_list column",
    )
    parser.add_argument(
        "--roadmap-file",
        type=Path,
        required=True,
        help="roadmap.geo file for centroid extraction",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output CSV file for LM-TAD format",
    )
    parser.add_argument(
        "--vocab-file", type=Path, required=True, help="Output vocab.json file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["porto_hoser", "beijing_hoser_reference"],
        help="Dataset name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    # Run conversion
    convert_hoser_to_lmtad_format(
        trajectory_file=args.trajectory_file,
        roadmap_file=args.roadmap_file,
        output_file=args.output_file,
        vocab_file=args.vocab_file,
        dataset=args.dataset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
