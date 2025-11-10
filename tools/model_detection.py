#!/usr/bin/env python3
"""
Model Detection Utility

Shared utility for detecting and parsing model names from generated trajectory files.
Handles multiple naming conventions (Beijing distilled, Porto phase1/2) and seed variants.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Model naming patterns (order matters - check longer names first to avoid false matches)
MODEL_PATTERNS = [
    # Porto convention - phase 2 with seeds
    "distill_phase2_seed44",
    "distill_phase2_seed43",
    "distill_phase2",
    # Porto convention - phase 1 with seeds
    "distill_phase1_seed44",
    "distill_phase1_seed43",
    "distill_phase1",
    # Beijing convention - distilled with seeds
    "distilled_seed44",
    "distilled_seed43",
    "distilled",
    # Vanilla with seeds (both conventions)
    "vanilla_seed44",
    "vanilla_seed43",
    "vanilla",
]


# Model display names for visualizations and reports
MODEL_DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "vanilla_seed43": "Vanilla (seed 43)",
    "vanilla_seed44": "Vanilla (seed 44)",
    "distilled": "Distilled (seed 42)",
    "distilled_seed43": "Distilled (seed 43)",
    "distilled_seed44": "Distilled (seed 44)",
    "distill_phase1": "Distill Phase 1 (seed 42)",
    "distill_phase1_seed43": "Distill Phase 1 (seed 43)",
    "distill_phase1_seed44": "Distill Phase 1 (seed 44)",
    "distill_phase2": "Distill Phase 2 (seed 42)",
    "distill_phase2_seed43": "Distill Phase 2 (seed 43)",
    "distill_phase2_seed44": "Distill Phase 2 (seed 44)",
    "real": "Real Trajectory",
}


@dataclass
class ModelFile:
    """Detected model file metadata"""

    path: Path  # Full path to file
    filename: str  # Just the filename
    model: str  # Model name (e.g., "distill_phase1", "vanilla_seed44")
    od_type: Optional[str] = None  # "train" or "test" (None if not applicable)
    split: Optional[str] = None  # Alias for od_type for backward compatibility

    def __post_init__(self):
        """Set split as alias for od_type"""
        if self.split is None and self.od_type is not None:
            self.split = self.od_type


def extract_model_name(filename: str) -> Optional[str]:
    """
    Extract model name from filename using pattern matching.

    Checks patterns in order of specificity (longest first) to avoid false matches.

    Args:
        filename: File name or path string

    Returns:
        Model name if detected, None otherwise

    Examples:
        >>> extract_model_name("2025-11-07_03-23-44_distill_phase1_test.csv")
        'distill_phase1'
        >>> extract_model_name("distilled_seed44_train.csv")
        'distilled_seed44'
        >>> extract_model_name("vanilla_test.csv")
        'vanilla'
    """
    filename_lower = filename.lower()

    for pattern in MODEL_PATTERNS:
        if pattern in filename_lower:
            return pattern

    return None


def extract_od_type(filename: str) -> Optional[str]:
    """
    Extract OD type (train/test) from filename.

    Args:
        filename: File name or path string

    Returns:
        "train" or "test" if detected, None otherwise

    Examples:
        >>> extract_od_type("distill_phase1_test.csv")
        'test'
        >>> extract_od_type("vanilla_train.csv")
        'train'
    """
    filename_lower = filename.lower()

    if "test" in filename_lower:
        return "test"
    elif "train" in filename_lower:
        return "train"

    return None


def detect_model_files(
    directory: Path,
    pattern: str = "*.csv",
    require_model: bool = True,
    require_od_type: bool = False,
    recursive: bool = True,
) -> List[ModelFile]:
    """
    Detect and parse model files in a directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for files (default: "*.csv")
        require_model: Only return files with detected model names
        require_od_type: Only return files with detected OD type
        recursive: Search recursively in subdirectories

    Returns:
        List of ModelFile objects with detected metadata

    Examples:
        >>> files = detect_model_files(Path("gene/porto_hoser/seed42"))
        >>> len(files)
        18
        >>> files[0].model
        'distill_phase1'
        >>> files[0].od_type
        'test'
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    logger.info(f"🔍 Detecting model files in {directory}")

    # Search for files
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    detected_files = []

    for file_path in files:
        filename = file_path.name

        # Extract model name
        model = extract_model_name(filename)

        if require_model and model is None:
            continue

        # Extract OD type
        od_type = extract_od_type(filename)

        if require_od_type and od_type is None:
            continue

        detected_files.append(
            ModelFile(
                path=file_path,
                filename=filename,
                model=model,
                od_type=od_type,
            )
        )

        if model:
            logger.debug(
                f"  Found: {model}"
                + (f" - {od_type}" if od_type else "")
                + f" ({filename})"
            )

    logger.info(f"✅ Detected {len(detected_files)} model files")

    return detected_files


def group_by_model(model_files: List[ModelFile]) -> Dict[str, List[ModelFile]]:
    """
    Group model files by model name.

    Args:
        model_files: List of detected model files

    Returns:
        Dictionary mapping model_name -> list of ModelFile objects

    Examples:
        >>> files = detect_model_files(Path("gene/porto_hoser/seed42"))
        >>> grouped = group_by_model(files)
        >>> sorted(grouped.keys())
        ['distill_phase1', 'distill_phase1_seed43', 'distill_phase1_seed44', ...]
    """
    grouped = {}

    for model_file in model_files:
        if model_file.model not in grouped:
            grouped[model_file.model] = []
        grouped[model_file.model].append(model_file)

    return grouped


def group_by_od_type(model_files: List[ModelFile]) -> Dict[str, List[ModelFile]]:
    """
    Group model files by OD type (train/test).

    Args:
        model_files: List of detected model files

    Returns:
        Dictionary mapping od_type -> list of ModelFile objects

    Examples:
        >>> files = detect_model_files(Path("gene/porto_hoser/seed42"))
        >>> grouped = group_by_od_type(files)
        >>> len(grouped['test'])
        9
        >>> len(grouped['train'])
        9
    """
    grouped = {}

    for model_file in model_files:
        od_type = model_file.od_type
        if od_type is None:
            od_type = "unknown"

        if od_type not in grouped:
            grouped[od_type] = []
        grouped[od_type].append(model_file)

    return grouped


def get_display_name(model: str) -> str:
    """
    Get human-readable display name for a model.

    Args:
        model: Model identifier

    Returns:
        Display name for visualization and reports

    Examples:
        >>> get_display_name("distill_phase1")
        'Distill Phase 1 (seed 42)'
        >>> get_display_name("vanilla_seed44")
        'Vanilla (seed 44)'
    """
    return MODEL_DISPLAY_NAMES.get(model, model.replace("_", " ").title())


def main():
    """Command-line interface for testing model detection"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Test model file detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect models in evaluation directory
  uv run python tools/model_detection.py hoser-distill-optuna-porto-eval-xyz/gene/porto_hoser

  # Show grouped by model
  uv run python tools/model_detection.py gene/Beijing/seed42 --group-by model
        """,
    )
    parser.add_argument("directory", type=Path, help="Directory to search")
    parser.add_argument(
        "--pattern", default="*.csv", help="File pattern (default: *.csv)"
    )
    parser.add_argument(
        "--group-by",
        choices=["model", "od_type"],
        help="Group results by model or OD type",
    )
    parser.add_argument(
        "--require-od-type",
        action="store_true",
        help="Only show files with detected OD type",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Detect files
    files = detect_model_files(
        args.directory,
        pattern=args.pattern,
        require_od_type=args.require_od_type,
    )

    if not files:
        print("❌ No model files detected")
        sys.exit(1)

    # Display results
    print("\n" + "=" * 70)
    print(f"Detected {len(files)} model files")
    print("=" * 70)

    if args.group_by == "model":
        grouped = group_by_model(files)
        for model, model_files in sorted(grouped.items()):
            print(f"\n📦 {get_display_name(model)} ({len(model_files)} files):")
            for mf in model_files:
                od_str = f" ({mf.od_type})" if mf.od_type else ""
                print(f"  • {mf.filename}{od_str}")

    elif args.group_by == "od_type":
        grouped = group_by_od_type(files)
        for od_type, model_files in sorted(grouped.items()):
            print(f"\n📊 {od_type.upper()} ({len(model_files)} files):")
            for mf in model_files:
                model_str = f" [{mf.model}]" if mf.model else ""
                print(f"  • {mf.filename}{model_str}")

    else:
        # Flat list
        for mf in files:
            model_str = mf.model or "unknown"
            od_str = f" ({mf.od_type})" if mf.od_type else ""
            print(f"  • {model_str}{od_str}: {mf.filename}")

    print("=" * 70)


if __name__ == "__main__":
    main()
