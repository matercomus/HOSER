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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import colorsys

DEFAULT_SEED_TOKEN = "default"
DEFAULT_SEED_NUMBER = 42
PHASE_ENABLED_PREFIXES = ("porto",)


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


@dataclass
class ModelFamily:
    """Encapsulate palette, display, and styling for a model family."""

    name: str
    base_display: str
    line_style: str = "-"
    palette_name: str = "husl"
    palette_size: int = 5
    variant_label: Optional[str] = "seed"
    palette: List[str] = field(default_factory=list, init=False)
    palette_kwargs: Optional[Dict[str, Any]] = None
    mix_with_white: float = 0.0
    base_offset: int = 0

    def get_display_name(self, seed: Optional[str]) -> str:
        """Return a human-friendly display name for this family/variant."""

        if seed and self.variant_label:
            seed_num = _extract_digits(seed)
            return f"{self.base_display} ({self.variant_label} {seed_num})"
        return self.base_display

    def get_color(self, seed: Optional[str]) -> str:
        """Return a deterministic color for the given variant."""

        palette = self._get_palette()
        index = _seed_to_index(seed)
        return palette[(index + self.base_offset) % len(palette)]

    def get_line_style(self) -> str:
        """Return the preferred matplotlib line style for this family."""

        return self.line_style

    def _get_palette(self) -> List[str]:
        if not self.palette:
            kwargs = self.palette_kwargs or {}
            colors = _build_color_palette(
                self.palette_name,
                self.palette_size,
                palette_kwargs=kwargs,
            )
            if self.mix_with_white:
                mix = self.mix_with_white
                colors = [
                    (
                        min(1.0, c[0] + (1.0 - c[0]) * mix),
                        min(1.0, c[1] + (1.0 - c[1]) * mix),
                        min(1.0, c[2] + (1.0 - c[2]) * mix),
                    )
                    for c in colors
                ]
            self.palette = [_rgb_to_hex(color) for color in colors]
        return self.palette


def _rgb_to_hex(color: tuple[float, float, float]) -> str:
    """Convert an RGB tuple in [0,1] to a hex string (#rrggbb)."""

    r, g, b = color
    r_i = max(0, min(255, int(round(r * 255))))
    g_i = max(0, min(255, int(round(g * 255))))
    b_i = max(0, min(255, int(round(b * 255))))
    return f"#{r_i:02x}{g_i:02x}{b_i:02x}"


def _build_color_palette(
    palette_name: str,
    palette_size: int,
    *,
    palette_kwargs: Optional[Dict[str, Any]] = None,
) -> List[tuple[float, float, float]]:
    """Build an RGB palette, preferring seaborn but falling back to matplotlib.

    Tests and non-plotting utilities should not require seaborn to be installed.
    When seaborn isn't available, we approximate palettes using matplotlib
    colormaps.
    """

    palette_kwargs = palette_kwargs or {}

    # Prefer seaborn palettes when available.
    try:
        import seaborn as sns  # type: ignore

        return sns.color_palette(palette_name, palette_size, **palette_kwargs)
    except Exception:
        pass

    # Next, try matplotlib colormaps if available.
    try:
        from matplotlib import colormaps

        try:
            cmap = colormaps.get_cmap(palette_name)
        except Exception:
            cmap = colormaps.get_cmap("viridis")

        if palette_size <= 1:
            rgba = cmap(0.5)
            return [(float(rgba[0]), float(rgba[1]), float(rgba[2]))]

        xs = [0.1 + (0.8 * i / (palette_size - 1)) for i in range(palette_size)]
        rgb = []
        for x in xs:
            rgba = cmap(x)
            rgb.append((float(rgba[0]), float(rgba[1]), float(rgba[2])))
        return rgb
    except Exception:
        pass

    # Final fallback: generate a simple HSL palette (deterministic, no deps).
    if palette_size <= 0:
        return []
    if palette_size == 1:
        return [(0.2, 0.4, 0.8)]

    palette_name_lower = palette_name.lower()
    if palette_name_lower in {"greys", "grays", "grey", "gray"}:
        # Deterministic grayscale ramp from darker to lighter.
        if palette_size == 1:
            return [(0.5, 0.5, 0.5)]
        xs = [0.2 + (0.6 * i / (palette_size - 1)) for i in range(palette_size)]
        return [(float(x), float(x), float(x)) for x in xs]

    if palette_name_lower in {"ylorbr", "ylorbr_r"}:
        # Approximate a yellow→orange→brown sequential palette.
        # Hue ~40° (yellow/orange), with decreasing lightness.
        hues = 40.0 / 360.0
        lights = (
            [0.85]
            if palette_size == 1
            else [0.85 - (0.5 * i / (palette_size - 1)) for i in range(palette_size)]
        )
        sat = 0.85
        rgb: List[tuple[float, float, float]] = []
        for light in lights:
            r, g, b = colorsys.hls_to_rgb(hues, float(light), sat)
            rgb.append((float(r), float(g), float(b)))
        return rgb

    palette: List[tuple[float, float, float]] = []
    # Use the palette name to pick a stable starting hue.
    name_seed = sum(ord(ch) for ch in palette_name) % 360
    base_hue = name_seed / 360.0
    for idx in range(palette_size):
        hue = (base_hue + (idx / palette_size)) % 1.0
        light = 0.55
        sat = 0.75
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        palette.append((float(r), float(g), float(b)))
    return palette


# Model naming convention patterns (using regex for automatic detection)
# Order matters: more specific patterns should be checked first
MODEL_CONVENTIONS = [
    # Abnormal evaluation outputs (must come before non-abnormal patterns)
    (r"distilled_.*seed(\d+).*_l1.*abnormal", "distilled_l1_abnormal_seed{}"),
    (r"distilled.*_l1.*abnormal", "distilled_l1_abnormal"),
    (r"distilled_.*seed(\d+).*abnormal", "distilled_abnormal_seed{}"),
    (r"distilled.*abnormal", "distilled_abnormal"),
    (r"vanilla_.*seed(\d+).*abnormal", "vanilla_abnormal_seed{}"),
    (r"vanilla.*abnormal", "vanilla_abnormal"),
    # Porto distill_phase<N>_seed<M> pattern
    (r"distill_phase(\d+)_seed(\d+)", "distill_phase{}_seed{}"),
    # Porto distill_phase<N> pattern (no seed)
    (r"distill_phase(\d+)(?!_seed)", "distill_phase{}"),
    # Beijing distilled *_L1 variants (their own family)
    (r"distilled_\d+epoch_seed(\d+)_l1", "distilled_l1_seed{}"),
    (r"distilled_seed(\d+)_l1", "distilled_l1_seed{}"),
    (r"distilled_.*seed(\d+)_l1", "distilled_l1_seed{}"),
    # Beijing distilled_l1 explicit pattern (if present in filenames)
    (r"distilled_l1_seed(\d+)", "distilled_l1_seed{}"),
    (r"distilled_l1(?!_seed)", "distilled_l1"),
    # Beijing distilled_<N>epoch_seed<M> pattern (normalize to distilled_seed<M>)
    (r"distilled_\d+epoch_seed(\d+)", "distilled_seed{}"),
    # Beijing distilled_seed<M> pattern
    (r"distilled_seed(\d+)", "distilled_seed{}"),
    # Beijing distilled_.*seed<M> pattern (handles intermediate text like _25epoch_)
    (r"distilled_.*seed(\d+)", "distilled_seed{}"),
    # Beijing distilled pattern (no seed)
    (r"distilled(?!_seed)", "distilled"),
    # Vanilla_<N>epoch_seed<M> pattern (normalize to vanilla_seed<M>)
    (r"vanilla_\d+epoch_seed(\d+)", "vanilla_seed{}"),
    # Vanilla_seed<M> pattern
    (r"vanilla_seed(\d+)", "vanilla_seed{}"),
    # Vanilla pattern (no seed)
    (r"vanilla_.*seed(\d+)", "vanilla_seed{}"),
    (r"vanilla(?!_seed)", "vanilla"),
]

# Known model patterns for backward compatibility and testing
# These are examples of models that follow the conventions above
KNOWN_MODEL_PATTERNS = [
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
    "distilled_25epoch_seed44",
    "distilled_seed44",
    "distilled_seed43",
    "distilled_seed42",
    "distilled",
    # Vanilla variants
    "vanilla_25epoch_seed44",
    "vanilla_seed44",
    "vanilla_seed43",
    "vanilla_seed42",
    "vanilla",
]

# Keep MODEL_PATTERNS for backward compatibility
MODEL_PATTERNS = KNOWN_MODEL_PATTERNS
MODEL_FAMILIES: Dict[str, ModelFamily] = {
    "distilled": ModelFamily(
        name="distilled",
        base_display="Distilled",
        palette_name="crest",
        palette_size=8,
        base_offset=2,
    ),
    "distilled_l1": ModelFamily(
        name="distilled_l1",
        base_display="Distilled L1",
        palette_name="rocket",
        palette_size=8,
        base_offset=2,
    ),
    "distilled_abnormal": ModelFamily(
        name="distilled_abnormal",
        base_display="Distilled Abnormal",
        palette_name="crest",
        palette_size=8,
        base_offset=2,
    ),
    "distilled_l1_abnormal": ModelFamily(
        name="distilled_l1_abnormal",
        base_display="Distilled L1 Abnormal",
        palette_name="rocket",
        palette_size=8,
        base_offset=2,
    ),
    "vanilla": ModelFamily(
        name="vanilla",
        base_display="Vanilla",
        palette_name="flare",
        palette_size=8,
        base_offset=2,
    ),
    "vanilla_abnormal": ModelFamily(
        name="vanilla_abnormal",
        base_display="Vanilla Abnormal",
        palette_name="flare",
        palette_size=8,
        base_offset=2,
    ),
    "distill_phase1": ModelFamily(
        name="distill_phase1",
        base_display="Distill Phase 1",
        palette_name="mako",
        palette_size=8,
        base_offset=1,
    ),
    "distill_phase2": ModelFamily(
        name="distill_phase2",
        base_display="Distill Phase 2",
        palette_name="rocket",
        palette_size=8,
        base_offset=2,
    ),
    # Backward-compat alias (some callers may use mixed-case token).
    "distilled_L1": ModelFamily(
        name="distilled_l1",
        base_display="Distilled L1",
        palette_name="rocket",
        palette_size=8,
        base_offset=2,
    ),
    "real": ModelFamily(
        name="real",
        base_display="Real",
        palette_name="YlOrBr",
        palette_size=6,
        variant_label=None,
        mix_with_white=0.1,
        base_offset=1,
    ),
    "unknown": ModelFamily(
        name="unknown",
        base_display="Unknown",
        line_style="--",
        palette_name="Greys",
        palette_size=3,
        variant_label=None,
        base_offset=1,
    ),
}


@dataclass(frozen=True)
class ModelMetadata:
    """Canonical metadata extracted from a model identifier."""

    model_name: str
    base_model: str
    normalized_base: str
    seed_label: Optional[str]
    seed_number: Optional[int]
    phase_label: Optional[str]


def extract_model_name(filename: str) -> str:
    """
    Extract model name from filename using pattern matching.

    Automatically detects models following naming conventions:
    - distill_phase<N>_seed<M> (e.g., distill_phase2_seed44, distill_phase3_seed45)
    - distill_phase<N> (e.g., distill_phase1, distill_phase2)
    - distilled_seed<M> (e.g., distilled_seed42, distilled_seed45)
    - distilled_l1_seed<M> (e.g., distilled_l1_seed42)
    - distilled
    - vanilla_seed<M> (e.g., vanilla_seed43)
    - vanilla

    New models following these conventions are automatically supported without
    requiring updates to the MODEL_PATTERNS list.

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
        >>> extract_model_name("hoser_distill_phase3_seed45_trainod_gene.csv")
        'distill_phase3_seed45'
        >>> extract_model_name("hoser_vanilla_trainod_gene.csv")
        'vanilla'
    """
    filename_lower = str(filename).lower()

    # Try each convention pattern
    for pattern, template in MODEL_CONVENTIONS:
        match = re.search(pattern, filename_lower)
        if match:
            # Format the template with captured groups
            groups = match.groups()
            if groups:
                return template.format(*groups)
            else:
                return template

    return "unknown"


def get_display_name(model_name: str) -> str:
    """Return human-readable display name for a model."""

    components = parse_model_components(model_name)
    family = _get_or_create_family(components["base_model"])
    return family.get_display_name(components["seed"])


def get_model_color(model_name: str) -> str:
    """Return a hex color suitable for plotting the given model."""

    components = parse_model_components(model_name)
    family = _get_or_create_family(components["base_model"])
    return family.get_color(components["seed"])


def get_model_line_style(model_name: str) -> str:
    """Return matplotlib line style for the given model."""

    components = parse_model_components(model_name)
    family = _get_or_create_family(components["base_model"])
    return family.get_line_style()


def parse_model_components(model_name: str) -> Dict[str, Optional[str]]:
    """
    Parse model name into components.

    Automatically extracts seed numbers from models following conventions,
    supporting any seed number (not just 42, 43, 44).

    Args:
        model_name: Model name from extract_model_name()

    Returns:
        Dictionary with 'base_model' and 'seed' keys

    Examples:
        >>> parse_model_components("distilled_seed44")
        {'base_model': 'distilled', 'seed': 'seed44'}
        >>> parse_model_components("distill_phase2_seed43")
        {'base_model': 'distill_phase2', 'seed': 'seed43'}
        >>> parse_model_components("distill_phase3_seed45")
        {'base_model': 'distill_phase3', 'seed': 'seed45'}
        >>> parse_model_components("vanilla")
        {'base_model': 'vanilla', 'seed': None}
    """
    normalized = model_name.lower()

    # Support legacy strings like "distilled_seed44_L1" by mapping them onto the
    # canonical base model "distilled_l1".
    is_l1 = False
    if normalized.endswith("_l1"):
        normalized = normalized[: -len("_l1")]
        is_l1 = True

    # Check for seed pattern using regex to support any seed number
    match = re.search(r"_seed(\d+)", normalized)
    if match:
        seed_num = match.group(1)
        seed = f"seed{seed_num}"
        base_model = normalized.replace(f"_seed{seed_num}", "")
        if is_l1 and base_model == "distilled":
            base_model = "distilled_l1"
        return {
            "base_model": base_model,
            "seed": seed,
        }

    # No seed found
    base_model = normalized
    if is_l1 and base_model == "distilled":
        base_model = "distilled_l1"
    return {"base_model": base_model, "seed": None}


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


def _get_or_create_family(base_model: str) -> ModelFamily:
    """Return an existing family or create a dynamic one for new phases."""

    if base_model in MODEL_FAMILIES:
        return MODEL_FAMILIES[base_model]

    match = re.match(r"distill_phase(\d+)", base_model)
    if match:
        phase = match.group(1)
        family = ModelFamily(
            name=base_model,
            base_display=f"Distill Phase {phase}",
            palette_name="husl",
            palette_size=8,
            mix_with_white=0.3,
            base_offset=3,
        )
        MODEL_FAMILIES[base_model] = family
        return family

    if base_model == "distilled_l1":
        return MODEL_FAMILIES["distilled_l1"]

    return MODEL_FAMILIES["unknown"]


def _extract_digits(seed: str) -> str:
    """Return the numeric component of a seed string, if present."""

    match = re.search(r"(\d+)", seed)
    return match.group(1) if match else seed


def _seed_to_index(seed: Optional[str]) -> int:
    """Map a seed identifier to a stable palette index."""

    if not seed:
        return 0
    match = re.search(r"(\d+)", seed)
    if match:
        return int(match.group(1))
    return sum(ord(ch) for ch in seed)


def normalize_base_model(base_model: Optional[str]) -> str:
    """Normalize base model aliases for trio validation and grouping."""

    if not base_model:
        return "unknown"
    lowered = base_model.lower()

    # Treat all distilled variants (e.g., distilled_l1, distilled_abnormal) as distilled.
    if lowered.startswith("distilled"):
        return "distilled"

    # Porto phase variants should normalize to the distilled umbrella.
    if lowered.startswith("distill"):
        return "distilled"

    # Treat all vanilla variants (e.g., vanilla_abnormal) as vanilla.
    if lowered.startswith("vanilla"):
        return "vanilla"

    return lowered


def get_phase_label_from_base(base_model: Optional[str]) -> Optional[str]:
    """Return a concise phase label derived from the base model name."""

    if not base_model:
        return None
    if base_model.startswith("distill_phase"):
        return base_model.replace("distill_", "")
    return None


def seed_label_to_number(seed_label: Optional[str]) -> Optional[int]:
    """Convert canonical seed tokens (seed42/default) into integers."""

    if not seed_label:
        return None

    digits = _extract_digits(seed_label)
    if digits.isdigit():
        return int(digits)

    if seed_label.lower() == DEFAULT_SEED_TOKEN:
        return DEFAULT_SEED_NUMBER

    return None


def format_seed_label(seed_label: Optional[str]) -> Optional[str]:
    """Return a human-friendly seed display label."""

    seed_number = seed_label_to_number(seed_label)
    if seed_number is not None:
        return f"Seed {seed_number}"

    if not seed_label:
        return None

    display = seed_label.replace("_", " ").title()
    return f"Seed {display}".strip()


def format_phase_display(phase_label: Optional[str]) -> Optional[str]:
    """Return a spaced, title-cased phase label for display."""

    if not phase_label:
        return None

    text = phase_label.replace("_", " ").title()
    chars: List[str] = []
    for idx, char in enumerate(text):
        if idx > 0 and char.isdigit() and text[idx - 1].isalpha():
            chars.append(" ")
        chars.append(char)

    return "".join(chars).strip()


def build_model_metadata(model_name: str) -> ModelMetadata:
    """Build a reusable metadata snapshot for a model identifier."""

    components = parse_model_components(model_name)
    base_model = components.get("base_model") or "unknown"
    normalized = normalize_base_model(base_model)
    seed_label = components.get("seed")
    phase_label = get_phase_label_from_base(base_model)
    seed_number = seed_label_to_number(seed_label)

    return ModelMetadata(
        model_name=model_name,
        base_model=base_model,
        normalized_base=normalized,
        seed_label=seed_label,
        seed_number=seed_number,
        phase_label=phase_label,
    )


def dataset_supports_phases(dataset_name: Optional[str]) -> bool:
    """Return True if the dataset encodes distinct distillation phases."""

    if not dataset_name:
        return False

    slug = dataset_name.lower()
    return any(slug.startswith(prefix) for prefix in PHASE_ENABLED_PREFIXES)


DEFAULT_MODEL_NAMES = [
    "real",
    "unknown",
    "distilled",
    "distilled_seed42",
    "distilled_seed43",
    "distilled_seed44",
    "distilled_l1",
    "distilled_l1_seed42",
    "distilled_l1_seed43",
    "distilled_l1_seed44",
    "distilled_abnormal",
    "distilled_abnormal_seed42",
    "distilled_abnormal_seed43",
    "distilled_abnormal_seed44",
    "distilled_l1_abnormal",
    "distilled_l1_abnormal_seed42",
    "distilled_l1_abnormal_seed43",
    "distilled_l1_abnormal_seed44",
    "vanilla_abnormal",
    "vanilla_abnormal_seed42",
    "vanilla_abnormal_seed43",
    "vanilla_abnormal_seed44",
    "distill_phase1",
    "distill_phase1_seed42",
    "distill_phase1_seed43",
    "distill_phase1_seed44",
    "distill_phase2",
    "distill_phase2_seed42",
    "distill_phase2_seed43",
    "distill_phase2_seed44",
    "vanilla",
    "vanilla_seed42",
    "vanilla_seed43",
    "vanilla_seed44",
]


def _build_display_names() -> Dict[str, str]:
    """Build display-name lookup table for commonly referenced models."""

    return {name: get_display_name(name) for name in DEFAULT_MODEL_NAMES}


def _build_model_colors() -> Dict[str, str]:
    """Build color lookup table for commonly referenced models."""

    return {name: get_model_color(name) for name in DEFAULT_MODEL_NAMES}


def _build_line_styles() -> Dict[str, str]:
    """Build line-style lookup table for commonly referenced models."""

    return {name: get_model_line_style(name) for name in DEFAULT_MODEL_NAMES}


# Display/look-up tables for backward compatibility (exported via tools.__init__)
DISPLAY_NAMES = _build_display_names()
MODEL_COLORS = _build_model_colors()
MODEL_LINE_STYLES = _build_line_styles()


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
