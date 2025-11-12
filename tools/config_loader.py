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

YAML Configuration Format:
    # Standard HOSER Student Evaluation
    dataset: Beijing
    data_dir: /path/to/HOSER-dataset
    seed: 42
    cuda_device: 0
    num_gene: 5000
    beam_width: 4
    beam_search: true
    grid_size: 0.001
    edr_eps: 100.0

    # Optional: Cross-dataset translation (OD pair mapping)
    source_dataset: porto_hoser      # Source dataset (where OD pairs come from)
    target_dataset: BJUT_Beijing     # Target dataset (where models are trained)
    translation_max_distance: 20.0   # Max distance for quality filtering (meters)
    translation_mapping_file: road_mapping_porto_to_beijing.json  # Optional explicit path

    # Optional: Analysis-focused workflow (skip generation)
    skip_generation: false  # Set to true to skip trajectory generation

    # Optional: LM-TAD Teacher Baseline Evaluation
    lmtad_evaluation: true
    lmtad_repo: /home/matt/Dev/LMTAD
    lmtad_checkpoint: code/results/LMTAD/beijing_hoser_reference/run_20250928_202718/.../weights_only.pt
    lmtad_real_data_dir: /home/matt/Dev/LMTAD/data/beijing_hoser_reference  # Optional (auto-detected)

Configuration Priority:
    1. Explicit values in YAML file
    2. Auto-detection based on context (e.g., lmtad_real_data_dir from lmtad_repo + dataset)
    3. Sensible defaults (e.g., seed=42, beam_width=4)

Path Resolution:
    - Absolute paths are used as-is
    - Relative paths are resolved relative to eval_dir
    - LM-TAD checkpoint paths are searched in: lmtad_repo, eval_dir, then CWD

Backward Compatibility:
    - All new fields have sensible defaults
    - Existing configs continue to work without modification
    - Legacy field names are supported (e.g., cross_dataset_name → source_dataset)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration object for evaluation workflows

    This config supports both HOSER student model evaluation and LM-TAD teacher
    baseline evaluation. LM-TAD evaluation uses the same workflow but points to
    teacher model checkpoints and data.

    Example YAML for LM-TAD evaluation:
        ```yaml
        # LM-TAD Teacher Baseline Evaluation
        dataset: Beijing
        data_dir: /home/matt/Dev/LMTAD/data/beijing_hoser_reference

        # LM-TAD specific settings
        lmtad_evaluation: true
        lmtad_repo: /home/matt/Dev/LMTAD
        lmtad_checkpoint: code/results/LMTAD/beijing_hoser_reference/run_20250928_202718/.../weights_only.pt

        # Evaluation settings (same as HOSER)
        num_gene: 5000
        beam_width: 4
        seed: 42
        grid_size: 0.001
        edr_eps: 100.0
        ```
    """

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

    # LM-TAD teacher evaluation configuration
    lmtad_evaluation: bool = False  # Enable LM-TAD teacher baseline evaluation
    lmtad_repo: Optional[Path] = None  # Path to LM-TAD repository root
    lmtad_checkpoint: Optional[Path] = None  # Path to LM-TAD checkpoint (.pt file)
    lmtad_real_data_dir: Optional[Path] = (
        None  # Real data directory for teacher (usually different from student data_dir)
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

    def get_lmtad_checkpoint(self) -> Optional[Path]:
        """Get resolved LM-TAD checkpoint path

        Returns:
            Absolute path to LM-TAD checkpoint if configured, None otherwise

        Raises:
            FileNotFoundError: If lmtad_evaluation=True but checkpoint not found

        Example:
            >>> config = load_evaluation_config(eval_dir=Path("hoser-eval"))
            >>> if config.lmtad_evaluation:
            ...     ckpt = config.get_lmtad_checkpoint()
            ...     print(f"Teacher checkpoint: {ckpt}")
        """
        if not self.lmtad_evaluation:
            return None

        if self.lmtad_checkpoint is None:
            raise ValueError(
                "lmtad_evaluation=True but lmtad_checkpoint not specified in config"
            )

        # Resolve checkpoint path
        ckpt_path = Path(self.lmtad_checkpoint)

        # If relative path, try resolving relative to:
        # 1. LM-TAD repo root (if lmtad_repo specified)
        # 2. Eval directory
        # 3. Current working directory
        if not ckpt_path.is_absolute():
            if self.lmtad_repo:
                # Try relative to repo root first
                repo_ckpt = Path(self.lmtad_repo) / ckpt_path
                if repo_ckpt.exists():
                    ckpt_path = repo_ckpt.resolve()
                else:
                    # Try relative to eval dir
                    eval_ckpt = self.eval_dir / ckpt_path
                    if eval_ckpt.exists():
                        ckpt_path = eval_ckpt.resolve()
                    else:
                        # Assume it's relative to CWD (absolute path will fail below)
                        ckpt_path = ckpt_path.resolve()
            else:
                # No repo specified, try eval dir then CWD
                eval_ckpt = self.eval_dir / ckpt_path
                if eval_ckpt.exists():
                    ckpt_path = eval_ckpt.resolve()
                else:
                    ckpt_path = ckpt_path.resolve()

        # Verify checkpoint exists
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"LM-TAD checkpoint not found: {ckpt_path}\n"
                f"Original config value: {self.lmtad_checkpoint}\n"
                f"lmtad_repo: {self.lmtad_repo}"
            )

        return ckpt_path

    def get_lmtad_real_data_path(self) -> Optional[Path]:
        """Get real data directory for LM-TAD teacher evaluation

        This is typically different from student data_dir because:
        - Teacher uses LM-TAD format data (grid-based tokenization)
        - Student uses HOSER format data (road network graphs)

        Returns:
            Path to real data directory for teacher evaluation, or None if not applicable

        Example:
            >>> config = load_evaluation_config(eval_dir=Path("hoser-eval"))
            >>> if config.lmtad_evaluation:
            ...     real_data = config.get_lmtad_real_data_path()
            ...     train_csv = real_data / "train.csv"
            ...     test_csv = real_data / "test.csv"
        """
        if not self.lmtad_evaluation:
            return None

        # Priority order:
        # 1. Explicit lmtad_real_data_dir
        # 2. Fall back to data_dir (if it's LM-TAD format)
        # 3. Auto-detect from lmtad_repo + dataset name
        if self.lmtad_real_data_dir:
            real_data_path = Path(self.lmtad_real_data_dir)
        elif self.data_dir:
            # Use configured data_dir (caller must ensure it's LM-TAD format)
            real_data_path = self.get_data_dir()
            return real_data_path
        elif self.lmtad_repo and self.dataset:
            # Auto-detect: LMTAD_REPO/data/{dataset}_hoser_reference
            # This matches the naming convention used in teacher training
            dataset_name = (
                f"{self.dataset.lower()}_hoser_reference"
                if not self.dataset.endswith("_hoser_reference")
                else self.dataset
            )
            real_data_path = Path(self.lmtad_repo) / "data" / dataset_name
        else:
            raise ValueError(
                "Cannot determine LM-TAD real data path. Please specify one of:\n"
                "  - lmtad_real_data_dir (preferred)\n"
                "  - data_dir (if it's LM-TAD format)\n"
                "  - lmtad_repo + dataset (for auto-detection)"
            )

        # Resolve relative paths
        if not real_data_path.is_absolute():
            real_data_path = self.eval_dir / real_data_path

        real_data_path = real_data_path.resolve()

        # Verify data directory exists
        if not real_data_path.exists():
            logger.warning(
                f"LM-TAD real data directory not found: {real_data_path}\n"
                f"This may cause issues if real data is needed for evaluation."
            )

        return real_data_path


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

    # Extract LM-TAD teacher evaluation configuration
    lmtad_evaluation = raw_config.get("lmtad_evaluation", False)
    lmtad_repo = raw_config.get("lmtad_repo")
    lmtad_checkpoint = raw_config.get("lmtad_checkpoint")
    lmtad_real_data_dir = raw_config.get("lmtad_real_data_dir")

    # Handle LM-TAD path fields (convert to Path objects)
    if lmtad_repo:
        lmtad_repo = Path(lmtad_repo)
        if not lmtad_repo.is_absolute():
            lmtad_repo = eval_dir / lmtad_repo

    if lmtad_checkpoint:
        lmtad_checkpoint = Path(lmtad_checkpoint)
        # Note: Resolution happens in get_lmtad_checkpoint() to handle multiple search paths

    if lmtad_real_data_dir:
        lmtad_real_data_dir = Path(lmtad_real_data_dir)
        if not lmtad_real_data_dir.is_absolute():
            lmtad_real_data_dir = eval_dir / lmtad_real_data_dir

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
        # LM-TAD teacher evaluation configuration
        lmtad_evaluation=lmtad_evaluation,
        lmtad_repo=lmtad_repo,
        lmtad_checkpoint=lmtad_checkpoint,
        lmtad_real_data_dir=lmtad_real_data_dir,
        _raw_config=raw_config,
    )

    return config
