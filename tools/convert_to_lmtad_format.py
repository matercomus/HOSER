#!/usr/bin/env python3
"""
Convert HOSER Trajectories to LM-TAD Grid Token Format
=======================================================

Purpose
-------
Converts HOSER-format trajectories (road IDs) to LM-TAD grid token format for
teacher model evaluation and comparison. This module enables:
- Fair comparison between HOSER and LM-TAD by using consistent input format
- Evaluation of HOSER-generated trajectories using LM-TAD teacher model
- Cross-dataset trajectory translation workflows

Input Format (HOSER CSV)
------------------------
CSV file with columns: mm_id, entity_id, traj_id, rid_list, time_list
- rid_list: Comma-separated road IDs, e.g., "29750,22077,22080,33716"
- time_list: ISO timestamps, e.g., "2015-11-05T13:14:32Z,2015-11-05T13:15:16Z"

Output Format (LM-TAD CSV)
--------------------------
CSV file with columns: mm_id, entity_id, traj_id, grid_token_list, time_list
- grid_token_list: Comma-separated grid tokens, e.g., "1234,1235,1238,1240"
- time_list: Preserved from input (same format)

Grid Mapping Strategy
---------------------
Uses the same centroid-to-grid formula as LM-TAD preprocessing:
1. Compute road centroids from roadmap.geo coordinates
2. Map each centroid (lat, lng) to grid cell using GridMapper
3. Apply same grid_size and downsample_factor as teacher training

Dataset Support
---------------
- Porto: grid_size=0.001, downsample=1
- Beijing: grid_size=0.001, downsample=1 (or dataset-specific config)

Usage Examples
--------------
Basic conversion:
    >>> python tools/convert_to_lmtad_format.py \\
    ...     --input gene/Beijing/seed42/generated.csv \\
    ...     --output gene_lmtad/Beijing/seed42/generated.csv \\
    ...     --dataset Beijing

Batch conversion with config:
    >>> python tools/convert_to_lmtad_format.py \\
    ...     --input gene/Beijing/seed42/*.csv \\
    ...     --output gene_lmtad/Beijing/seed42/ \\
    ...     --config config/Beijing.yaml

Programmatic usage:
    >>> from tools.convert_to_lmtad_format import convert_hoser_to_lmtad_format
    >>>
    >>> convert_hoser_to_lmtad_format(
    ...     input_csv="gene/Beijing/generated.csv",
    ...     output_csv="gene_lmtad/Beijing/generated.csv",
    ...     dataset="Beijing",
    ...     data_dir=Path("data/Beijing")
    ... )
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Allow importing GridMapper without relative path issues
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from critics.grid_mapper import GridConfig, GridMapper  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def extract_road_centroids(geo_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract road centroids from roadmap.geo file.

    Parameters
    ----------
    geo_path : Path
        Path to roadmap.geo file containing road network geometry.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame]
        - road_centroids: Shape (N, 2) array of (lat, lng) for each road
        - geo_df: Original GeoDataFrame for reference

    Notes
    -----
    Centroids are computed as the mean of all coordinate points for each road.
    This matches the LM-TAD preprocessing approach for consistent grid mapping.
    """
    if not geo_path.exists():
        raise FileNotFoundError(f"roadmap.geo not found at {geo_path}")

    logger.info(f"Loading road network from {geo_path}")
    geo_df = pd.read_csv(geo_path)

    # Parse coordinates and compute centroids
    coords_series = geo_df["coordinates"].apply(eval)
    road_centroids_lat = np.array(
        [np.mean([pt[1] for pt in coords]) for coords in coords_series],
        dtype=np.float32,
    )
    road_centroids_lng = np.array(
        [np.mean([pt[0] for pt in coords]) for coords in coords_series],
        dtype=np.float32,
    )

    road_centroids = np.stack([road_centroids_lat, road_centroids_lng], axis=1)
    logger.info(f"Extracted {len(road_centroids)} road centroids")

    return road_centroids, geo_df


def create_grid_mapper(
    geo_df: pd.DataFrame,
    road_centroids: np.ndarray,
    grid_size: float = 0.001,
    downsample_factor: int = 1,
    verify_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[GridMapper, np.ndarray]:
    """Create GridMapper and road-to-token mapping.

    Parameters
    ----------
    geo_df : pd.DataFrame
        GeoDataFrame containing road network geometry.
    road_centroids : np.ndarray
        Shape (N, 2) array of (lat, lng) for each road's centroid.
    grid_size : float, optional
        Grid cell size in degrees, by default 0.001 (matches LM-TAD).
    downsample_factor : int, optional
        Grid downsampling factor, by default 1 (no downsampling).
    verify_hw : Optional[Tuple[int, int]], optional
        Optional (height, width) to verify grid dimensions match teacher.

    Returns
    -------
    Tuple[GridMapper, np.ndarray]
        - mapper: GridMapper instance for converting coordinates to tokens
        - road_to_token: Shape (N,) array mapping road_id to grid_token_id

    Notes
    -----
    The grid boundaries are computed from the road network extents to ensure
    all roads are mapped to valid grid cells.
    """
    # Compute boundaries from road coordinates
    coords_series = geo_df["coordinates"].apply(eval)
    min_lng = min(min(pt[0] for pt in coords) for coords in coords_series)
    max_lng = max(max(pt[0] for pt in coords) for coords in coords_series)
    min_lat = min(min(pt[1] for pt in coords) for coords in coords_series)
    max_lat = max(max(pt[1] for pt in coords) for coords in coords_series)

    logger.info(
        f"Grid boundaries: lat=[{min_lat:.6f}, {max_lat:.6f}], "
        f"lng=[{min_lng:.6f}, {max_lng:.6f}]"
    )

    # Create grid configuration
    grid_cfg = GridConfig(
        min_lat=float(min_lat),
        max_lat=float(max_lat),
        min_lng=float(min_lng),
        max_lng=float(max_lng),
        grid_size=grid_size,
        downsample_factor=downsample_factor,
    )

    # Create mapper and compute road-to-token mapping
    mapper = GridMapper(grid_cfg, road_centroids, verify_hw=verify_hw)
    road_to_token = mapper.map_all()

    logger.info(
        f"Created grid mapper: {mapper.grid_h}x{mapper.grid_w} grid "
        f"({mapper.grid_h * mapper.grid_w} total tokens)"
    )

    return mapper, road_to_token.astype(np.int64)


def convert_trajectory_batch(
    traj_df: pd.DataFrame,
    road_to_token: np.ndarray,
) -> pd.DataFrame:
    """Convert batch of trajectories from road IDs to grid tokens.

    Parameters
    ----------
    traj_df : pd.DataFrame
        DataFrame with columns: mm_id, entity_id, traj_id, rid_list, time_list
    road_to_token : np.ndarray
        Shape (N,) array mapping road_id to grid_token_id

    Returns
    -------
    pd.DataFrame
        Converted DataFrame with columns: mm_id, entity_id, traj_id,
        grid_token_list, time_list

    Notes
    -----
    - Preserves all metadata columns (mm_id, entity_id, traj_id, time_list)
    - Converts rid_list to grid_token_list using road_to_token mapping
    - Handles invalid road IDs gracefully by skipping those trajectories
    - Progress bar shows conversion progress for large datasets
    """
    converted_rows = []
    skipped_count = 0
    invalid_road_ids = set()

    logger.info(f"Converting {len(traj_df)} trajectories...")

    for idx, row in tqdm(
        traj_df.iterrows(),
        total=len(traj_df),
        desc="Converting trajectories",
        disable=len(traj_df) < 100,
    ):
        try:
            # Parse road ID list
            rid_list_str = row["rid_list"]
            if isinstance(rid_list_str, str):
                # Handle string representation of list
                if rid_list_str.startswith("["):
                    rid_list = eval(rid_list_str)
                else:
                    # Comma-separated values
                    rid_list = [int(x.strip()) for x in rid_list_str.split(",")]
            else:
                # Already a list
                rid_list = rid_list_str

            # Convert to numpy array
            rid_array = np.array(rid_list, dtype=np.int64)

            # Check for invalid road IDs
            invalid_mask = (rid_array < 0) | (rid_array >= len(road_to_token))
            if invalid_mask.any():
                invalid_ids = rid_array[invalid_mask]
                invalid_road_ids.update(invalid_ids.tolist())
                skipped_count += 1
                continue

            # Map road IDs to grid tokens
            grid_tokens = road_to_token[rid_array]

            # Create output row
            converted_row = {
                "mm_id": row["mm_id"],
                "entity_id": row["entity_id"],
                "traj_id": row["traj_id"],
                "grid_token_list": ",".join(map(str, grid_tokens)),
                "time_list": row["time_list"],
            }
            converted_rows.append(converted_row)

        except Exception as e:
            logger.warning(f"Failed to convert trajectory at index {idx}: {e}")
            skipped_count += 1
            continue

    if invalid_road_ids:
        logger.warning(
            f"Skipped {skipped_count} trajectories with invalid road IDs. "
            f"Invalid IDs: {sorted(list(invalid_road_ids))[:10]}..."
            if len(invalid_road_ids) > 10
            else f"Invalid IDs: {sorted(list(invalid_road_ids))}"
        )
    elif skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} trajectories due to conversion errors")

    converted_df = pd.DataFrame(converted_rows)
    logger.info(
        f"Successfully converted {len(converted_df)}/{len(traj_df)} trajectories"
    )

    return converted_df


def save_lmtad_format(
    converted_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save converted trajectories to LM-TAD format CSV.

    Parameters
    ----------
    converted_df : pd.DataFrame
        DataFrame with columns: mm_id, entity_id, traj_id, grid_token_list, time_list
    output_path : Path
        Output CSV file path

    Notes
    -----
    Creates parent directories if they don't exist.
    Saves in CSV format compatible with LM-TAD training/evaluation code.
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    converted_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(converted_df)} trajectories to {output_path}")


def convert_hoser_to_lmtad_format(
    input_csv: Path,
    output_csv: Path,
    dataset: str,
    data_dir: Optional[Path] = None,
    config: Optional[dict] = None,
    verify_hw: Optional[Tuple[int, int]] = None,
) -> None:
    """Main conversion function: HOSER trajectories -> LM-TAD grid tokens.

    Parameters
    ----------
    input_csv : Path
        Input CSV file with HOSER trajectories (road IDs).
    output_csv : Path
        Output CSV file for LM-TAD format (grid tokens).
    dataset : str
        Dataset name ('Beijing' or 'porto_hoser').
    data_dir : Optional[Path], optional
        Data directory containing roadmap.geo. If None, uses data/{dataset}.
    config : Optional[dict], optional
        Configuration dict with 'distill.grid_size' and 'distill.downsample'.
        If None, uses defaults (grid_size=0.001, downsample=1).
    verify_hw : Optional[Tuple[int, int]], optional
        Optional (height, width) to verify grid dimensions match teacher.

    Raises
    ------
    FileNotFoundError
        If input_csv or roadmap.geo not found.
    ValueError
        If grid dimensions don't match verify_hw.

    Examples
    --------
    Basic usage with defaults:
        >>> convert_hoser_to_lmtad_format(
        ...     input_csv=Path("gene/Beijing/seed42/generated.csv"),
        ...     output_csv=Path("gene_lmtad/Beijing/seed42/generated.csv"),
        ...     dataset="Beijing"
        ... )

    With custom config:
        >>> config = {"distill": {"grid_size": 0.001, "downsample": 1}}
        >>> convert_hoser_to_lmtad_format(
        ...     input_csv=Path("gene/porto/generated.csv"),
        ...     output_csv=Path("gene_lmtad/porto/generated.csv"),
        ...     dataset="porto_hoser",
        ...     config=config
        ... )
    """
    # Resolve paths
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Determine data directory
    if data_dir is None:
        data_dir = ROOT / "data" / dataset
    else:
        data_dir = Path(data_dir)

    # Load grid configuration
    if config is None:
        grid_size = 0.001
        downsample_factor = 1
        logger.info(
            f"Using default grid config: grid_size={grid_size}, downsample={downsample_factor}"
        )
    else:
        grid_size = float(config.get("distill", {}).get("grid_size", 0.001))
        downsample_factor = int(config.get("distill", {}).get("downsample", 1))
        logger.info(
            f"Using config grid settings: grid_size={grid_size}, downsample={downsample_factor}"
        )

    # Extract road centroids
    geo_path = data_dir / "roadmap.geo"
    road_centroids, geo_df = extract_road_centroids(geo_path)

    # Create grid mapper and road-to-token mapping
    mapper, road_to_token = create_grid_mapper(
        geo_df,
        road_centroids,
        grid_size=grid_size,
        downsample_factor=downsample_factor,
        verify_hw=verify_hw,
    )

    # Load input trajectories
    logger.info(f"Loading trajectories from {input_csv}")
    traj_df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(traj_df)} trajectories")

    # Convert trajectories
    converted_df = convert_trajectory_batch(traj_df, road_to_token)

    # Save output
    save_lmtad_format(converted_df, output_csv)

    logger.info("✅ Conversion complete!")


def main() -> None:
    """Command-line interface for trajectory format conversion."""
    parser = argparse.ArgumentParser(
        description="Convert HOSER trajectories (road IDs) to LM-TAD format (grid tokens)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file with defaults
  %(prog)s \\
      --input gene/Beijing/seed42/generated.csv \\
      --output gene_lmtad/Beijing/seed42/generated.csv \\
      --dataset Beijing

  # Convert with custom config
  %(prog)s \\
      --input gene/porto/generated.csv \\
      --output gene_lmtad/porto/generated.csv \\
      --dataset porto_hoser \\
      --config config/porto.yaml

  # Verify grid dimensions match teacher
  %(prog)s \\
      --input gene/Beijing/generated.csv \\
      --output gene_lmtad/Beijing/generated.csv \\
      --dataset Beijing \\
      --verify-grid-hw 177 159
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input HOSER trajectory CSV file (with road IDs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output LM-TAD format CSV file (with grid tokens)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Beijing", "porto_hoser", "BJUT_Beijing"],
        help="Dataset name (determines data directory)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override data directory (default: data/{dataset})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file with distill.grid_size and distill.downsample",
    )
    parser.add_argument(
        "--verify-grid-hw",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Verify grid dimensions match teacher (height width)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config if provided
    config = None
    if args.config:
        config_path = args.config
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Convert verify_hw to tuple if provided
    verify_hw = tuple(args.verify_grid_hw) if args.verify_grid_hw else None

    # Run conversion
    try:
        convert_hoser_to_lmtad_format(
            input_csv=args.input,
            output_csv=args.output,
            dataset=args.dataset,
            data_dir=args.data_dir,
            config=config,
            verify_hw=verify_hw,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
