# Dynamic Model Comparison Heatmap Implementation

## Summary

Successfully implemented dynamic model detection and comprehensive heatmap generation for scenario analysis. The system now automatically detects all vanilla and distilled models and generates **both** individual comparison files **and** a comprehensive grid overview.

## Changes Made

### 1. Enhanced Data Loader (`scenario_plots/data_loader.py`)

Added three new utility functions:

#### `classify_models(data, od_source='train')`
- Automatically detects and classifies models as "vanilla" or "distilled"
- Returns sorted lists of vanilla and distilled models
- Works with any number of models in the dataset

#### `generate_model_colors(models)`
- Dynamically generates distinct colors for visualization
- Uses predefined colors for ≤3 models
- Uses seaborn's "husl" palette for larger model sets
- Returns dictionary mapping model names to hex colors

#### `generate_model_labels(models)`
- Converts model names to human-readable labels
- Preserves seed numbers in parentheses
- Example: `distilled_seed42` → `Distilled (seed 42)`

### 2. Application Plots (`scenario_plots/application_plots.py`)

#### New Function: `plot_improvement_heatmaps_individual()`
**Option A - Individual Files**
- Generates separate heatmap for each (distilled, vanilla) pair
- File naming: `improvement_heatmap_{distilled}_vs_{vanilla}.png/pdf`
- Each heatmap shows: scenarios (rows) × metrics (columns)
- Includes average improvement annotation

**Output for Beijing** (1 vanilla + 2 distilled):
- `improvement_heatmap_distilled_vs_vanilla.png`
- `improvement_heatmap_distilled_seed44_vs_vanilla.png`

**Output for Porto** (3 vanilla + 3 distilled):
- 9 separate files, one per comparison pair

#### New Function: `plot_improvement_heatmap_grid()`
**Option B - Comprehensive Grid**
- Single figure with subplot grid layout
- Rows: distilled models, Columns: vanilla models
- Each subplot: mini-heatmap of scenarios × metrics
- Shared colorbar for consistent scale
- Figure size scales dynamically based on number of models

**Output for Beijing**: 2×1 grid (2 distilled × 1 vanilla)
**Output for Porto**: 3×3 grid (3 distilled × 3 vanilla)

#### Updated: `plot_all()`
Now calls both new functions by default:
```python
def plot_all(data, output_dir, dpi=300):
    plot_application_radars(data, output_dir, dpi)
    plot_improvement_heatmaps_individual(data, output_dir, dpi)  # NEW
    plot_improvement_heatmap_grid(data, output_dir, dpi)        # NEW
```

### 3. Backward Compatibility

✅ **All existing code continues to work without changes:**
- `get_metric_value()` - unchanged
- `get_scenario_list()` - unchanged
- `calculate_improvement()` - unchanged
- `plot_improvement_heatmap()` - kept for compatibility (marked as deprecated)

**No changes required in:**
- `metrics_plots.py`
- `temporal_spatial_plots.py`
- `analysis_plots.py`
- `robustness_plots.py`
- `create_scenario_plots.py`

## Usage

### Automatic Mode (Recommended)
Simply run the existing command - it now generates all plots automatically:

```bash
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6
```

### What Gets Generated

For a dataset with M distilled models and N vanilla models:

**Individual Heatmaps (Option A):**
- M × N separate PNG/PDF files
- Detailed view of each comparison

**Grid Overview (Option B):**
- 1 PNG/PDF file with M×N subplot grid
- Comprehensive at-a-glance comparison

## Example Outputs

### Beijing Dataset
- Models detected: 1 vanilla, 2 distilled
- Individual files: 2 heatmaps
- Grid file: 1 figure (2×1 grid)

### Porto Dataset
- Models detected: 3 vanilla, 3 distilled
- Individual files: 9 heatmaps
- Grid file: 1 figure (3×3 grid)

## Technical Details

### Model Detection Logic
Models are classified by naming pattern (case-insensitive):
- **Vanilla**: contains "vanilla"
- **Distilled**: contains "distill"

### Color Palette
- Small sets (≤3 models): Predefined colors `[red, blue, green]`
- Large sets (>3 models): Seaborn "husl" palette (evenly spaced hues)

### Grid Layout Sizing
- Small grids (≤2×2): 8×6 inches per subplot
- Large grids (>2×2): 6×5 inches per subplot
- Font sizes scale down for readability

### Fallback Behavior
- If no vanilla models found: Log warning, skip plots
- If no distilled models found: Log warning, skip plots
- Graceful handling of missing data

## Testing

Validated with comprehensive test suite covering:
- ✅ Model classification (Beijing and Porto scenarios)
- ✅ Color generation (small and large model sets)
- ✅ Label generation (with seed preservation)
- ✅ Comparison counts (2 for Beijing, 9 for Porto)

All tests passed successfully.

## Benefits

1. **Scalability**: Works with any number of models
2. **Flexibility**: Generates both detailed and overview visualizations
3. **Automation**: No manual model specification needed
4. **Compatibility**: Doesn't break existing code
5. **Robustness**: Graceful fallback for missing data

## Bug Fixes (2025-01-29)

### Fixed Three Critical Bugs

**Bug 1: Color Generation Fallback Limitation**
- **Issue**: Fallback color list had only 9 colors, causing mismatch for >9 models
- **Fix**: Removed fallback entirely, always use seaborn's husl palette
- **Commit**: `16404c4` - Simplify color generation

**Bug 2 & 3: Hardcoded Scenarios and Metrics**
- **Issue**: Both individual and grid heatmap functions used hardcoded lists
- **Problem**: Silent failures when actual data had different scenario names or metrics
- **Fix**: Dynamic extraction from actual data using new functions:
  - `get_available_scenarios()` - Extracts scenarios from data
  - `get_available_metrics()` - Extracts metrics from data
  - `get_metric_display_labels()` - Formats metric names for display
- **Commits**: 
  - `3edff94` - Add dynamic data extraction functions
  - `2383dab` - Replace hardcoded scenarios/metrics with dynamic extraction
  - `a6e677f` - Fix metric label formatting order

### Testing
All fixes validated with comprehensive test suite covering:
- Scenario extraction from realistic data structures
- Metric extraction with missing data points
- Label formatting for various metric types
- Empty data handling
- Color generation scalability (1-15+ models)
- Real-world usage simulation

## Future Work

The hardcoded model names in other plotting modules (`metrics_plots.py`, `temporal_spatial_plots.py`, etc.) could be updated to use the new dynamic detection for consistency, but this is optional as they continue to work correctly with existing datasets.

