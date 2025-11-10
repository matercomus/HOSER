# Model Detection Utility

Centralized utility for detecting and managing model names across the HOSER codebase.

## Overview

The `model_detection.py` module provides a **single source of truth** for model name patterns, display names, and visualization colors. This ensures consistency across all analysis and visualization scripts.

### Automatic Model Detection

The utility uses **regex-based pattern matching** to automatically detect new models following existing naming conventions. This means:

- ✅ **No manual updates needed** for new seed variants (e.g., `seed45`, `seed99`)
- ✅ **No manual updates needed** for new phase models (e.g., `distill_phase3`, `distill_phase4`)
- ✅ **Automatic display names** generated for new models
- ✅ **Automatic color assignment** based on model family

### Supported Naming Conventions

The utility automatically recognizes models following these patterns:

- **Beijing Models**: `distilled`, `distilled_seed<N>` (e.g., `distilled_seed42`, `distilled_seed99`)
- **Porto Phases**: `distill_phase<N>`, `distill_phase<N>_seed<M>` (e.g., `distill_phase1`, `distill_phase3_seed45`)
- **Vanilla Models**: `vanilla`, `vanilla_seed<N>` (e.g., `vanilla_seed42`, `vanilla_seed100`)

Where `<N>` and `<M>` can be **any number**.

## Features

✅ **Automatic detection** of new models following conventions  
✅ **Handles multiple naming conventions** (Beijing `distilled`, Porto `distill_phase<N>`)  
✅ **Supports any seed variant** (`seed42`, `seed43`, `seed99`, etc.)  
✅ **Dynamic display names** generated automatically  
✅ **Smart color assignment** based on model family  
✅ **Structured `ModelFile` dataclass** with metadata  
✅ **Fully documented** with CLI testing support  
✅ **Type-safe** with dataclasses

## Usage

### As a Module

```python
from tools.model_detection import extract_model_name, get_display_name, get_model_color

# Extract model name from filename
filename = "hoser_distilled_seed44_trainod_gene.csv"
model = extract_model_name(filename)  # Returns: "distilled_seed44"

# Works with new models automatically!
filename_new = "hoser_distill_phase3_seed99_trainod.csv"
model_new = extract_model_name(filename_new)  # Returns: "distill_phase3_seed99"

# Get display name for plots (automatically generated)
display_name = get_display_name(model)  # Returns: "Distilled (seed 44)"
display_name_new = get_display_name(model_new)  # Returns: "Distill Phase 3 (seed 99)"

# Get color for visualizations (automatically assigned)
color = get_model_color(model)  # Returns: "#27ae60"
color_new = get_model_color(model_new)  # Returns a color from phase family
```

### Detection from Directory

```python
from tools.model_detection import detect_model_files
from pathlib import Path

# Scan directory for model files
files = detect_model_files(Path("eval_dir/gene/porto/seed42"))

for model_file in files:
    print(f"File: {model_file.filename}")
    print(f"Model: {model_file.model_name}")
    print(f"Seed: {model_file.seed}")
    print(f"Base: {model_file.base_model}")
    print(f"Display: {get_display_name(model_file.model_name)}")
    print()
```

### CLI Interface

Test the utility from command line:

```bash
# Scan directory and group by model
python tools/model_detection.py eval_dir/gene/dataset/seed42 --group-by model

# Group by seed variant
python tools/model_detection.py eval_dir/gene/dataset/seed42 --group-by seed

# Group by base model (without seed)
python tools/model_detection.py eval_dir/gene/dataset/seed42 --group-by base_model

# Scan for specific pattern
python tools/model_detection.py eval_dir/gene/dataset/seed42 --pattern "*.json"
```

## Migration Guide

### Before: Manual Pattern Matching (60+ lines)

```python
# OLD CODE - Don't do this anymore!
if "distill_phase2_seed44" in filename:
    model = "distill_phase2_seed44"
elif "distill_phase2_seed43" in filename:
    model = "distill_phase2_seed43"
elif "distill_phase2_seed42" in filename:
    model = "distill_phase2_seed42"
elif "distill_phase2" in filename:
    model = "distill_phase2"
elif "distill_phase1_seed44" in filename:
    model = "distill_phase1_seed44"
# ... 20+ more elif statements
elif "distilled_seed44" in filename:
    model = "distilled_seed44"
elif "distilled_seed43" in filename:
    model = "distilled_seed43"
elif "distilled" in filename:
    model = "distilled"
elif "vanilla" in filename:
    model = "vanilla"
else:
    model = "unknown"
```

### After: Single Line

```python
# NEW CODE - Clean and maintainable!
from tools.model_detection import extract_model_name

model = extract_model_name(filename)
```

### Hardcoded Color Dictionaries

**Before:**
```python
# OLD CODE - Duplicated across files
MODEL_COLORS = {
    "distilled": "#2ecc71",
    "distilled_seed44": "#27ae60",
    "vanilla": "#e74c3c",
    "distill_phase1": "#3498db",
    "distill_phase2": "#9b59b6",
    # ... many more
}

color = MODEL_COLORS.get(model, "#000000")
```

**After:**
```python
# NEW CODE - Consistent everywhere
from tools.model_detection import get_model_color

color = get_model_color(model)
```

### Display Names

**Before:**
```python
# OLD CODE - Hardcoded mappings
if model == "distilled_seed44":
    display = "Distilled (seed 44)"
elif model == "distill_phase2_seed43":
    display = "Distill Phase 2 (seed 43)"
# ... etc
```

**After:**
```python
# NEW CODE - Automatic
from tools.model_detection import get_display_name

display = get_display_name(model)
```

## API Reference

### Functions

#### `extract_model_name(filename: str) -> str`

Extract model name from filename using regex-based pattern matching.

**Automatically detects:**
- New seed variants (e.g., `seed45`, `seed99`)
- New phase models (e.g., `distill_phase3`, `distill_phase4`)
- Any combination following conventions

**Args:**
- `filename`: Filename or path to extract model name from

**Returns:**
- Model name string (e.g., "distilled_seed44", "distill_phase2_seed43", "distill_phase3_seed99")
- Returns "unknown" if no pattern matches

**Examples:**
```python
>>> extract_model_name("hoser_distilled_seed44_trainod_gene.csv")
'distilled_seed44'
>>> extract_model_name("hoser_distill_phase2_seed43_testod_gene.csv")
'distill_phase2_seed43'
>>> extract_model_name("hoser_distill_phase3_seed99_trainod.csv")  # New model!
'distill_phase3_seed99'
```

#### `get_display_name(model_name: str) -> str`

Get human-readable display name for visualizations.

**Automatically generates** display names for new models following conventions.

**Args:**
- `model_name`: Model name from `extract_model_name()`

**Returns:**
- Display name suitable for plots and visualizations

**Examples:**
```python
>>> get_display_name("distilled_seed44")
'Distilled (seed 44)'
>>> get_display_name("distill_phase2_seed43")
'Distill Phase 2 (seed 43)'
>>> get_display_name("distill_phase3_seed99")  # Automatic!
'Distill Phase 3 (seed 99)'
```

#### `get_model_color(model_name: str) -> str`

Get color code for consistent visualization.

**Automatically assigns** colors to new models based on their family:
- `distilled*` → green family
- `distill_phase1*` → blue family
- `distill_phase2*` → purple family
- `distill_phase3+` → cycles through color palette
- `vanilla*` → red family

**Args:**
- `model_name`: Model name from `extract_model_name()`

**Returns:**
- Hex color code

**Examples:**
```python
>>> get_model_color("distilled_seed44")
'#27ae60'
>>> get_model_color("distill_phase2_seed43")
'#8e44ad'
```

#### `get_model_line_style(model_name: str) -> str`

Get line style for consistent visualization.

**Args:**
- `model_name`: Model name from `extract_model_name()`

**Returns:**
- Matplotlib line style string

#### `parse_model_components(model_name: str) -> Dict[str, Optional[str]]`

Parse model name into base model and seed components.

**Args:**
- `model_name`: Model name from `extract_model_name()`

**Returns:**
- Dictionary with 'base_model' and 'seed' keys

**Examples:**
```python
>>> parse_model_components("distilled_seed44")
{'base_model': 'distilled', 'seed': 'seed44'}
>>> parse_model_components("vanilla")
{'base_model': 'vanilla', 'seed': None}
```

#### `detect_model_files(directory: Path, pattern: str = "*.csv") -> List[ModelFile]`

Detect all model files in a directory and extract metadata.

**Args:**
- `directory`: Directory to search
- `pattern`: File pattern to match (default: "*.csv")

**Returns:**
- List of `ModelFile` objects with detected metadata

### Data Classes

#### `ModelFile`

Structured representation of a model file with metadata.

**Attributes:**
- `path`: Full path to the file (Path)
- `model_name`: Detected model name (str)
- `seed`: Seed variant if present (Optional[str])
- `base_model`: Base model name without seed (Optional[str])
- `filename`: Original filename (Optional[str])

### Constants

#### `MODEL_CONVENTIONS`

List of regex patterns and templates used for automatic model detection. These define the naming conventions that are automatically recognized.

#### `KNOWN_MODEL_PATTERNS` (backward compatibility)

List of explicitly known model patterns. New models don't need to be added here - they're detected automatically via `MODEL_CONVENTIONS`.

#### `DISPLAY_NAMES`

Dictionary mapping known model names to human-readable display names. New models get automatic names via pattern matching.

#### `MODEL_COLORS`

Dictionary mapping known model names to hex color codes. New models get automatic colors based on their family.

#### `MODEL_LINE_STYLES`

Dictionary mapping model names to matplotlib line styles.

## Benefits

### Single Source of Truth

All model patterns, names, and colors are defined in one place. No more hunting through multiple files to update model definitions.

### Automatic Support for New Models

**No code changes needed!** New models following existing conventions are automatically:
- ✅ Detected and extracted from filenames
- ✅ Given appropriate display names
- ✅ Assigned colors from the correct family
- ✅ Handled consistently across all scripts

**Example:** Adding a new phase model:
```python
# Old way (would require updates in multiple places):
# 1. Add to MODEL_PATTERNS
# 2. Add to DISPLAY_NAMES
# 3. Add to MODEL_COLORS
# 4. Update 4+ different scripts

# New way: NOTHING! Just use it:
filename = "hoser_distill_phase5_seed99_trainod.csv"
model = extract_model_name(filename)  # Works immediately!
display = get_display_name(model)     # "Distill Phase 5 (seed 99)"
color = get_model_color(model)        # Assigned automatically
```

### Easy to Add New Conventions (if needed)

If you need to add a completely new naming convention (not phase/seed variant):
```python
# In MODEL_CONVENTIONS, add one line:
(r'new_pattern_(\d+)', 'new_pattern_{}'),
"new_model_seed45": "#hexcode",
```

Instead of updating 4+ different scripts with if/elif chains.

### Consistent Behavior

All scripts use the same detection logic, so model names are consistent across:
- Scenario analysis
- Wang detection
- Trajectory visualization
- Statistical analysis
- And all future tools

### Better Maintainability

- Clear, documented API
- Type-safe with dataclasses
- Easy to test
- Self-documenting code

### Type Safety

Using dataclasses ensures proper type hints and IDE support:
```python
model_file: ModelFile = detect_model_files(path)[0]
model_file.model_name  # IDE knows this is a str
model_file.seed        # IDE knows this is Optional[str]
```

## Testing

### Manual Testing

```bash
# Test model detection on real data
python tools/model_detection.py eval_dir/gene/porto/seed42 --group-by model

# Expected output:
# Found X files
# 
# distilled_seed44: Y files
#   Display: Distilled (seed 44)
#   Color: #27ae60
#   - hoser_distilled_seed44_trainod_gene.csv
#   ...
```

### Unit Testing

```python
import pytest
from tools.model_detection import extract_model_name, get_display_name

def test_extract_beijing_models():
    assert extract_model_name("hoser_distilled_seed44_trainod.csv") == "distilled_seed44"
    assert extract_model_name("hoser_distilled_trainod.csv") == "distilled"

def test_extract_porto_models():
    assert extract_model_name("hoser_distill_phase2_seed43_testod.csv") == "distill_phase2_seed43"
    assert extract_model_name("hoser_distill_phase1_trainod.csv") == "distill_phase1"

def test_display_names():
    assert get_display_name("distilled_seed44") == "Distilled (seed 44)"
    assert get_display_name("distill_phase2_seed43") == "Distill Phase 2 (seed 43)"
```

## Migration Checklist

When migrating a file to use the model detection utility:

- [ ] Import the necessary functions at the top of the file
- [ ] Replace manual if/elif pattern matching with `extract_model_name()`
- [ ] Replace hardcoded color dictionaries with `get_model_color()`
- [ ] Replace manual display name logic with `get_display_name()`
- [ ] Test that the script still works correctly
- [ ] Remove old pattern matching code
- [ ] Update any documentation

## Examples from Real Migrations

### Example 1: analyze_scenarios.py

**Before (lines 885-890):**
```python
if model_name is None:
    if "distilled_seed44" in generated_file.name:
        model_name = "distilled_seed44"
    elif "distilled_seed43" in generated_file.name:
        model_name = "distilled_seed43"
    elif "distilled" in generated_file.name:
        model_name = "distilled"
    elif "vanilla" in generated_file.name:
        model_name = "vanilla"
    else:
        model_name = "unknown"
```

**After:**
```python
from tools.model_detection import extract_model_name

if model_name is None:
    model_name = extract_model_name(generated_file.name)
```

### Example 2: visualize_wang_results.py

**Before (lines 49-58):**
```python
COLORS = {
    "real": "#34495e",
    "distilled": "#2ecc71",
    "distilled_seed44": "#27ae60",
    "vanilla": "#e74c3c",
    "distill_phase1": "#3498db",
    "distill_phase1_seed43": "#2980b9",
    "distill_phase1_seed44": "#1f618d",
    "distill_phase2": "#9b59b6",
    "distill_phase2_seed43": "#8e44ad",
    "distill_phase2_seed44": "#7d3c98",
}
```

**After:**
```python
from tools.model_detection import get_model_color

# Use get_model_color(model) wherever color is needed
# No need for COLORS dict anymore!
```

## Troubleshooting

### Q: Model detection returns "unknown"

**A:** Check that your filename contains one of the supported patterns. The detection is case-insensitive and checks for substrings. Supported patterns are listed in `MODEL_PATTERNS`.

### Q: Wrong model detected (e.g., "distilled" instead of "distilled_seed44")

**A:** The order of patterns matters. Make sure you're using the utility correctly and not manually checking patterns. The utility checks most specific patterns first.

### Q: Need to add a new model

**A:** Add the pattern to `MODEL_PATTERNS` (in the right position based on specificity), add display name to `DISPLAY_NAMES`, and add color to `MODEL_COLORS`. That's it!

### Q: How do I know what models are in a directory?

**A:** Use the CLI:
```bash
python tools/model_detection.py /path/to/directory --group-by model
```

## See Also

- `tools/analyze_scenarios.py` - Example usage in scenario analysis
- `tools/visualize_wang_results.py` - Example usage in visualization
- `visualize_trajectories.py` - Example usage in trajectory visualization
