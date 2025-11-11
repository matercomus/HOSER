#!/usr/bin/env python3
"""
Shared Configuration Loader for Evaluation Workflows

This module provides a centralized way to load and manage evaluation configuration
from YAML files, used by both python_pipeline.py and run_abnormal_od_workflow.py.

Usage:
    from tools.config_loader import load_evaluation_config, EvaluationConfig

    # Load config from eval directory
    config = load_evaluation_config(eval_dir=Path("hoser-eval-xyz"))

    # Access config values
    data_dir = config.get_data_dir()
    dataset = config.dataset
    seed = config.seed
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration object for evaluation workflows"""

    eval_dir: Path
    dataset: str = "Beijing"
    data_dir: Optional[Path] = None
    seed: int = 42
    cuda_device: int = 0
    num_gene: int = 100
    beam_width: int = 4
    beam_search: bool = True
    grid_size: float = 0.001
    edr_eps: float = 100.0

    # Translation configuration (new)
    source_dataset: Optional[str] = (
        None  # Source dataset name (where OD pairs come from)
    )
    target_dataset: Optional[str] = (
        None  # Target dataset name (where models are trained)
    )
    translation_max_distance: float = (
        20.0  # Max distance threshold for quality filtering (meters)
    )
    translation_mapping_file: Optional[Path] = None  # Path to mapping file
    translation_distance_file: Optional[Path] = (
        None  # Path to distance file (deprecated, distances now in mapping)
    )

    # Analysis-focused workflow configuration
    skip_generation: bool = (
        False  # Skip trajectory generation, focus on analysis of real data
    )

    # Additional config values stored as dict
    _raw_config: Dict[str, Any] = field(default_factory=dict)

    def get_data_dir(self) -> Path:
        """Get resolved data directory path"""
        if self.data_dir is None:
            # Fallback: use dataset name
            return self.eval_dir.parent / "data" / self.dataset

        data_dir_path = Path(self.data_dir)
        if not data_dir_path.is_absolute():
            # Relative to eval directory
            data_dir_path = self.eval_dir / data_dir_path

        # Resolve symlinks if needed
        if data_dir_path.is_symlink():
            data_dir_path = data_dir_path.resolve()

        return data_dir_path

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value from raw config"""
        return self._raw_config.get(key, default)

    def get_translation_mapping_file(self) -> Optional[Path]:
        """Auto-detect translation mapping file path based on source/target datasets

        Returns:
            Path to mapping file if it can be determined, None otherwise
        """
        if self.translation_mapping_file:
            return self.translation_mapping_file

        # Auto-detect from source and target datasets
        if self.source_dataset and self.target_dataset:
            mapping_filename = (
                f"road_mapping_{self.source_dataset}_to_{self.target_dataset}.json"
            )
            return self.eval_dir / mapping_filename

        return None

    def get_translation_distance_file(self) -> Optional[Path]:
        """Auto-detect translation distance file path (deprecated, distances now in mapping)

        Returns:
            Path to distance file if specified, None otherwise
        """
        if self.translation_distance_file:
            return self.translation_distance_file

        # Auto-detect from source and target datasets
        if self.source_dataset and self.target_dataset:
            distance_filename = f"road_mapping_{self.source_dataset}_to_{self.target_dataset}_distances.json"
            return self.eval_dir / distance_filename

        return None


def load_evaluation_config(
    eval_dir: Path,
    config_path: Optional[Path] = None,
) -> EvaluationConfig:
    """
    Load evaluation configuration from YAML file.

    Args:
        eval_dir: Evaluation directory path
        config_path: Optional path to config file (default: eval_dir/config/evaluation.yaml)

    Returns:
        EvaluationConfig object with loaded configuration

    Example:
        >>> from pathlib import Path
        >>> from tools.config_loader import load_evaluation_config
        >>>
        >>> config = load_evaluation_config(eval_dir=Path("hoser-eval-xyz"))
        >>> print(f"Dataset: {config.dataset}")
        >>> print(f"Data dir: {config.get_data_dir()}")
    """
    eval_dir = Path(eval_dir).resolve()

    # Determine config file path
    if config_path is None:
        config_path = eval_dir / "config" / "evaluation.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = eval_dir / config_path

    # Load YAML config
    raw_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(
                f"Failed to load config from {config_path}: {e}, using defaults"
            )
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # Extract common fields
    dataset = raw_config.get("dataset", "Beijing")
    seed = raw_config.get("seed", 42)
    cuda_device = raw_config.get("cuda_device", 0)
    num_gene = raw_config.get("num_gene", 100)
    beam_width = raw_config.get("beam_width", 4)
    beam_search = raw_config.get("beam_search", False)  # Default to A* search
    grid_size = raw_config.get("grid_size", 0.001)
    edr_eps = raw_config.get("edr_eps", 100.0)

    # Handle data_dir
    data_dir = raw_config.get("data_dir")
    if data_dir:
        data_dir = Path(data_dir)

    # Extract translation fields
    source_dataset = raw_config.get("source_dataset") or raw_config.get(
        "cross_dataset_name"
    )  # Support legacy cross_dataset_name
    target_dataset = (
        raw_config.get("target_dataset") or dataset
    )  # Default to main dataset
    translation_max_distance = raw_config.get("translation_max_distance", 20.0)
    translation_mapping_file = raw_config.get("translation_mapping_file")
    translation_distance_file = raw_config.get(
        "translation_distance_file"
    )  # Deprecated field

    # Handle translation file paths
    if translation_mapping_file:
        translation_mapping_file = Path(translation_mapping_file)
        if not translation_mapping_file.is_absolute():
            translation_mapping_file = eval_dir / translation_mapping_file

    if translation_distance_file:
        translation_distance_file = Path(translation_distance_file)
        if not translation_distance_file.is_absolute():
            translation_distance_file = eval_dir / translation_distance_file

    # Extract skip_generation
    skip_generation = raw_config.get("skip_generation", False)

    # Create config object
    config = EvaluationConfig(
        eval_dir=eval_dir,
        dataset=dataset,
        data_dir=data_dir,
        seed=seed,
        cuda_device=cuda_device,
        num_gene=num_gene,
        beam_width=beam_width,
        beam_search=beam_search,
        grid_size=grid_size,
        edr_eps=edr_eps,
        # Translation configuration
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        translation_max_distance=translation_max_distance,
        translation_mapping_file=translation_mapping_file,
        translation_distance_file=translation_distance_file,
        skip_generation=skip_generation,
        _raw_config=raw_config,
    )

    return config
