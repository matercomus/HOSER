#!/usr/bin/env python3
"""
Model Detection Utility

Centralized utility for detecting and managing model names across the HOSER codebase.
Handles multiple naming conventions (Beijing distilled, Porto distill_phase1/phase2)
and supports all seed variants (seed42, seed43, seed44).

Usage as a module:
    from tools.model_detection import extract_model_name, get_display_name, get_model_color
    
    model = extract_model_name("hoser_distilled_seed44_trainod_gene.csv")
    display_name = get_display_name(model)
    color = get_model_color(model)

Usage as CLI:
    python tools/model_detection.py eval_dir/gene/dataset/seed42 --group-by model
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict


@dataclass
class ModelFile:
    """
    Structured representation of a model file with metadata.
    
    Attributes:
        path: Full path to the file
        model_name: Detected model name (e.g., "distilled_seed44", "distill_phase2_seed43")
        seed: Seed variant if present (e.g., "seed42", "seed43", "seed44")
        base_model: Base model name without seed (e.g., "distilled", "distill_phase2")
        filename: Original filename
    """
    path: Path
    model_name: str
    seed: Optional[str] = None
    base_model: Optional[str] = None
    filename: Optional[str] = None
    
    def __post_init__(self):
        if self.filename is None:
            self.filename = self.path.name


# Model name patterns - order matters, check most specific first
MODEL_PATTERNS = [
    # Porto distill_phase2 variants (most specific)
    "distill_phase2_seed44",
    "distill_phase2_seed43",
    "distill_phase2_seed42",
    "distill_phase2",
    
    # Porto distill_phase1 variants
    "distill_phase1_seed44",
    "distill_phase1_seed43", 
    "distill_phase1_seed42",
    "distill_phase1",
    
    # Beijing distilled variants
    "distilled_seed44",
    "distilled_seed43",
    "distilled_seed42",
    "distilled",
    
    # Vanilla variants
    "vanilla_seed44",
    "vanilla_seed43",
    "vanilla_seed42",
    "vanilla",
]

# Display names for visualizations
DISPLAY_NAMES = {
    # Beijing distilled models
    "distilled": "Distilled",
    "distilled_seed42": "Distilled (seed 42)",
    "distilled_seed43": "Distilled (seed 43)",
    "distilled_seed44": "Distilled (seed 44)",
    
    # Porto phase 1 models
    "distill_phase1": "Distill Phase 1",
    "distill_phase1_seed42": "Distill Phase 1 (seed 42)",
    "distill_phase1_seed43": "Distill Phase 1 (seed 43)",
    "distill_phase1_seed44": "Distill Phase 1 (seed 44)",
    
    # Porto phase 2 models
    "distill_phase2": "Distill Phase 2",
    "distill_phase2_seed42": "Distill Phase 2 (seed 42)",
    "distill_phase2_seed43": "Distill Phase 2 (seed 43)",
    "distill_phase2_seed44": "Distill Phase 2 (seed 44)",
    
    # Vanilla models
    "vanilla": "Vanilla",
    "vanilla_seed42": "Vanilla (seed 42)",
    "vanilla_seed43": "Vanilla (seed 43)",
    "vanilla_seed44": "Vanilla (seed 44)",
    
    # Special cases
    "real": "Real",
    "unknown": "Unknown",
}

# Color scheme for visualizations
MODEL_COLORS = {
    # Real data
    "real": "#34495e",  # Dark gray
    
    # Beijing distilled models (green family)
    "distilled": "#2ecc71",  # Green
    "distilled_seed42": "#2ecc71",  # Green
    "distilled_seed43": "#27ae60",  # Medium green
    "distilled_seed44": "#27ae60",  # Dark green
    
    # Porto phase 1 models (blue family)
    "distill_phase1": "#3498db",  # Blue
    "distill_phase1_seed42": "#3498db",  # Blue
    "distill_phase1_seed43": "#2980b9",  # Dark blue
    "distill_phase1_seed44": "#1f618d",  # Darker blue
    
    # Porto phase 2 models (purple family)
    "distill_phase2": "#9b59b6",  # Purple
    "distill_phase2_seed42": "#9b59b6",  # Purple
    "distill_phase2_seed43": "#8e44ad",  # Dark purple
    "distill_phase2_seed44": "#7d3c98",  # Darker purple
    
    # Vanilla models (red family)
    "vanilla": "#e74c3c",  # Red
    "vanilla_seed42": "#e74c3c",  # Red
    "vanilla_seed43": "#c0392b",  # Dark red
    "vanilla_seed44": "#a93226",  # Darker red
    
    # Unknown
    "unknown": "#95a5a6",  # Gray
}

# Line styles for visualizations
MODEL_LINE_STYLES = {
    "real": "-",
    "distilled": "-",
    "distilled_seed42": "-",
    "distilled_seed43": "-",
    "distilled_seed44": "-",
    "distill_phase1": "-",
    "distill_phase1_seed42": "-",
    "distill_phase1_seed43": "-",
    "distill_phase1_seed44": "-",
    "distill_phase2": "-",
    "distill_phase2_seed42": "-",
    "distill_phase2_seed43": "-",
    "distill_phase2_seed44": "-",
    "vanilla": "-",
    "vanilla_seed42": "-",
    "vanilla_seed43": "-",
    "vanilla_seed44": "-",
    "unknown": "--",
}


def extract_model_name(filename: str) -> str:
    """
    Extract model name from filename using pattern matching.
    
    Order matters: checks most specific patterns first (e.g., distill_phase2_seed44
    before distill_phase2, distilled_seed44 before distilled).
    
    Args:
        filename: Filename or path to extract model name from
        
    Returns:
        Model name string (e.g., "distilled_seed44", "distill_phase2_seed43", "vanilla")
        Returns "unknown" if no pattern matches.
        
    Examples:
        >>> extract_model_name("hoser_distilled_seed44_trainod_gene.csv")
        'distilled_seed44'
        >>> extract_model_name("hoser_distill_phase2_seed43_testod_gene.csv")
        'distill_phase2_seed43'
        >>> extract_model_name("hoser_vanilla_trainod_gene.csv")
        'vanilla'
    """
    filename_lower = str(filename).lower()
    
    for pattern in MODEL_PATTERNS:
        if pattern in filename_lower:
            return pattern
    
    return "unknown"


def get_display_name(model_name: str) -> str:
    """
    Get human-readable display name for a model.
    
    Args:
        model_name: Model name from extract_model_name()
        
    Returns:
        Display name suitable for plots and visualizations
        
    Examples:
        >>> get_display_name("distilled_seed44")
        'Distilled (seed 44)'
        >>> get_display_name("distill_phase2_seed43")
        'Distill Phase 2 (seed 43)'
    """
    return DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())


def get_model_color(model_name: str) -> str:
    """
    Get color code for a model for consistent visualization.
    
    Args:
        model_name: Model name from extract_model_name()
        
    Returns:
        Hex color code
        
    Examples:
        >>> get_model_color("distilled_seed44")
        '#27ae60'
        >>> get_model_color("distill_phase2_seed43")
        '#8e44ad'
    """
    return MODEL_COLORS.get(model_name, MODEL_COLORS["unknown"])


def get_model_line_style(model_name: str) -> str:
    """
    Get line style for a model for consistent visualization.
    
    Args:
        model_name: Model name from extract_model_name()
        
    Returns:
        Matplotlib line style string
        
    Examples:
        >>> get_model_line_style("distilled_seed44")
        '-'
        >>> get_model_line_style("unknown")
        '--'
    """
    return MODEL_LINE_STYLES.get(model_name, MODEL_LINE_STYLES["unknown"])


def parse_model_components(model_name: str) -> Dict[str, Optional[str]]:
    """
    Parse model name into components.
    
    Args:
        model_name: Model name from extract_model_name()
        
    Returns:
        Dictionary with 'base_model' and 'seed' keys
        
    Examples:
        >>> parse_model_components("distilled_seed44")
        {'base_model': 'distilled', 'seed': 'seed44'}
        >>> parse_model_components("distill_phase2_seed43")
        {'base_model': 'distill_phase2', 'seed': 'seed43'}
        >>> parse_model_components("vanilla")
        {'base_model': 'vanilla', 'seed': None}
    """
    # Check for seed pattern
    seed = None
    base_model = model_name
    
    for seed_num in ["42", "43", "44"]:
        seed_pattern = f"_seed{seed_num}"
        if seed_pattern in model_name:
            seed = f"seed{seed_num}"
            base_model = model_name.replace(seed_pattern, "")
            break
    
    return {
        "base_model": base_model,
        "seed": seed,
    }


def detect_model_files(directory: Path, pattern: str = "*.csv") -> List[ModelFile]:
    """
    Detect all model files in a directory and extract metadata.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (default: "*.csv")
        
    Returns:
        List of ModelFile objects with detected metadata
        
    Examples:
        >>> files = detect_model_files(Path("eval_dir/gene/porto/seed42"))
        >>> for f in files:
        ...     print(f.model_name, f.seed, f.base_model)
    """
    directory = Path(directory)
    model_files = []
    
    for file_path in directory.glob(pattern):
        model_name = extract_model_name(file_path.name)
        components = parse_model_components(model_name)
        
        model_file = ModelFile(
            path=file_path,
            model_name=model_name,
            seed=components["seed"],
            base_model=components["base_model"],
            filename=file_path.name,
        )
        model_files.append(model_file)
    
    return model_files


def main():
    """CLI interface for testing and using model detection utility."""
    parser = argparse.ArgumentParser(
        description="Model Detection Utility - detect and analyze model files"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan for model files",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to match (default: *.csv)",
    )
    parser.add_argument(
        "--group-by",
        choices=["model", "seed", "base_model"],
        default="model",
        help="Group files by model, seed, or base_model",
    )
    
    args = parser.parse_args()
    
    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        return 1
    
    print(f"Scanning {args.directory} for {args.pattern} files...")
    print()
    
    model_files = detect_model_files(args.directory, args.pattern)
    
    if not model_files:
        print("No model files found.")
        return 0
    
    print(f"Found {len(model_files)} files")
    print()
    
    # Group by specified attribute
    groups = {}
    for mf in model_files:
        if args.group_by == "model":
            key = mf.model_name
        elif args.group_by == "seed":
            key = mf.seed or "no-seed"
        else:  # base_model
            key = mf.base_model or "unknown"
        
        if key not in groups:
            groups[key] = []
        groups[key].append(mf)
    
    # Print grouped results
    for group_name in sorted(groups.keys()):
        files = groups[group_name]
        print(f"{group_name}: {len(files)} files")
        
        # Show display name and color
        if args.group_by == "model":
            display = get_display_name(group_name)
            color = get_model_color(group_name)
            print(f"  Display: {display}")
            print(f"  Color: {color}")
        
        # Show first few files
        for mf in files[:3]:
            print(f"  - {mf.filename}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
