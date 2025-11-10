# Model Detection Utility - Shared Module

## Overview

The `tools/model_detection.py` module provides a centralized, reusable utility for detecting and parsing model names from generated trajectory files. It handles multiple naming conventions (Beijing `distilled`, Porto `distill_phase1/phase2`) and seed variants.

## Why This Exists

Previously, model detection logic was duplicated across multiple scripts:
- `visualize_trajectories.py` - 60+ lines of nested if/elif statements
- `run_wang_detection_pipeline.py` - Custom parsing with underscore splitting
- `analyze_scenarios.py` - Separate detection logic (2+ locations)
- `tools/visualize_wang_results.py` - Hardcoded model color mappings

This led to:
- **Code duplication**: Same logic copy-pasted with slight variations
- **Maintenance burden**: Changes needed in multiple places
- **Inconsistency risk**: Different scripts might detect models differently
- **Harder to add new models**: New naming patterns required updates in many files

## What It Provides

### Core Functions

```python
from tools.model_detection import (
    detect_model_files,      # Main detection function
    extract_model_name,      # Extract model from filename
    extract_od_type,         # Extract train/test from filename
    get_display_name,        # Human-readable model names
    group_by_model,          # Group files by model
    group_by_od_type,        # Group files by OD type
    ModelFile,               # Dataclass for detected files
)
```

### Supported Model Patterns

**Automatically detects (order matters - checks longest first):**
- `distill_phase2_seed44`, `distill_phase2_seed43`, `distill_phase2`
- `distill_phase1_seed44`, `distill_phase1_seed43`, `distill_phase1`
- `distilled_seed44`, `distilled_seed43`, `distilled`
- `vanilla_seed44`, `vanilla_seed43`, `vanilla`

**Filename formats handled:**
- `2025-11-07_03-23-44_distill_phase1_test.csv` (Porto convention)
- `distilled_seed44_train.csv` (Beijing convention)
- Any file containing model pattern + train/test

## Usage Examples

### Basic Detection

```python
from tools.model_detection import detect_model_files

# Detect all model files in a directory
files = detect_model_files(
    Path("hoser-distill-optuna-porto-eval-xyz/gene/porto_hoser/seed42"),
    pattern="*.csv",
    require_model=True,
    require_od_type=True,
    recursive=True,
)

# Result: List of ModelFile objects
for f in files:
    print(f"{f.model} ({f.od_type}): {f.filename}")
```

### Grouping and Filtering

```python
from tools.model_detection import detect_model_files, group_by_model, group_by_od_type

# Detect files
files = detect_model_files(directory, pattern="*_test.csv")

# Group by model
by_model = group_by_model(files)
for model, model_files in by_model.items():
    print(f"{model}: {len(model_files)} files")

# Group by OD type
by_od = group_by_od_type(files)
test_files = by_od.get("test", [])
```

### Display Names for Visualization

```python
from tools.model_detection import get_display_name

# Get human-readable names
print(get_display_name("distill_phase1"))        # "Distill Phase 1 (seed 42)"
print(get_display_name("vanilla_seed44"))        # "Vanilla (seed 44)"
print(get_display_name("distilled_seed43"))      # "Distilled (seed 43)"
```

### CLI Testing

```bash
# Test detection on a directory
uv run python tools/model_detection.py hoser-distill-optuna-porto-eval-xyz/gene/porto_hoser/seed42

# Group by model
uv run python tools/model_detection.py eval_dir/gene/dataset/seed42 --group-by model

# Group by OD type
uv run python tools/model_detection.py eval_dir/gene/dataset/seed42 --group-by od_type

# Only show files with detected OD type
uv run python tools/model_detection.py eval_dir/gene/dataset/seed42 --require-od-type
```

## Current Integration

### ✅ Already Integrated

1. **`visualize_trajectories.py`**
   - Replaced 60+ lines of nested if/elif with clean utility call
   - Detection in `_detect_gene_files()` method

2. **`tools/run_wang_detection_pipeline.py`**
   - Replaced custom parsing logic with shared utility
   - Detection in `find_generated_models()` function

### 🔄 Recommended for Migration

1. **`tools/analyze_scenarios.py`**
   - Lines 885-890: Model name extraction in `_extract_scenarios_from_csv()`
   - Lines 1439-1445: Model name extraction in `_analyze_all_generated_models()`
   - **Benefit**: Consistency with other scripts, easier maintenance

2. **`tools/visualize_wang_results.py`**
   - Uses hardcoded `MODEL_COLORS` dict (lines 49-58)
   - **Could use**: `get_display_name()` for consistent naming

3. **Other potential candidates** (search for):
   - Any script with `"distill_phase" in filename` or similar patterns
   - Scripts that parse model names from file paths
   - Visualization scripts needing model display names

## Adding New Models

To add a new model naming pattern:

1. **Add to `MODEL_PATTERNS` list** in `tools/model_detection.py`:
   ```python
   MODEL_PATTERNS = [
       "new_model_seed44",  # Add longest variants first
       "new_model_seed43",
       "new_model",
       # ... existing patterns
   ]
   ```

2. **Add display name** to `MODEL_DISPLAY_NAMES`:
   ```python
   MODEL_DISPLAY_NAMES = {
       "new_model": "New Model (seed 42)",
       "new_model_seed43": "New Model (seed 43)",
       # ... existing names
   }
   ```

3. **That's it!** All scripts using the utility automatically support it.

## Architecture Benefits

### Before (Duplicated Logic)
```
visualize_trajectories.py         [60+ lines of model detection]
run_wang_detection_pipeline.py    [Custom parsing with underscore logic]
analyze_scenarios.py               [Separate if/elif chains in 2 places]
visualize_wang_results.py          [Hardcoded model colors]
```

### After (Shared Module)
```
tools/model_detection.py           [Single source of truth]
    ↑
    ├── visualize_trajectories.py     [1 function call]
    ├── run_wang_detection_pipeline.py [1 function call]
    ├── analyze_scenarios.py           [TODO: migrate]
    └── visualize_wang_results.py      [TODO: use display names]
```

### Key Advantages

1. **Single Source of Truth**: Model patterns defined once
2. **Consistency**: All scripts detect models identically
3. **Maintainability**: One place to update for new models
4. **Testability**: Utility can be tested independently
5. **Discoverability**: Clear API with docstrings and examples
6. **Type Safety**: Returns typed `ModelFile` dataclass

## Migration Guide

For scripts still using custom detection:

### Step 1: Import the utility
```python
from tools.model_detection import detect_model_files, extract_model_name
```

### Step 2: Replace custom logic

**Before:**
```python
if "distill_phase2_seed44" in filename:
    model = "distill_phase2_seed44"
elif "distill_phase2_seed43" in filename:
    model = "distill_phase2_seed43"
# ... 20+ more elif statements
```

**After:**
```python
model = extract_model_name(filename)
```

### Step 3: Use structured results

**Before:**
```python
models = []
for file in directory.glob("*.csv"):
    if "distill" in file.name or "vanilla" in file.name:
        models.append(file)
```

**After:**
```python
model_files = detect_model_files(directory, pattern="*.csv", require_model=True)
models = [mf.path for mf in model_files]
```

## Testing

The module includes a main CLI for testing:

```bash
# Quick test
uv run python tools/model_detection.py path/to/gene/dir

# Verify detection
uv run python tools/model_detection.py hoser-distill-optuna-porto-eval-xyz/gene/porto_hoser/seed42 --group-by model

# Expected output: 9 models detected (3 variants × 3 seeds each)
```

## Future Enhancements

Potential additions:
- [ ] Model metadata (training date, hyperparameters, etc.)
- [ ] Validation (check if model files are well-formed)
- [ ] Caching for repeated detections
- [ ] Configuration file for custom patterns
- [ ] Integration with evaluation pipeline config

## Questions?

See the docstrings in `tools/model_detection.py` or run:
```bash
uv run python tools/model_detection.py --help
```

