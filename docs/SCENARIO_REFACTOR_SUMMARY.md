# Scenario Analysis Refactor Summary

## Overview
Moved all scenario analysis computation from `visualize_trajectories.py` into `tools/analyze_scenarios.py`, producing comprehensive, well-labeled results that visualization scripts can consume without recomputation.

## Changes Made

### 1. `tools/analyze_scenarios.py` - Split Outputs Into Focused Files

**New Output Structure:**
```
scenarios/
└── {od_source}/        # train, test
    └── {model}/        # vanilla, distilled, etc.
        ├── trajectory_metadata.json     # Per-trajectory data with clear labels
        ├── scenario_metrics.json        # Individual scenario metrics  
        ├── scenario_combinations.json   # Combination analysis results
        └── trajectory_scenarios.json    # Index→tags mapping for viz sampling
```

**New Methods:**
- `_build_trajectory_metadata()`: Constructs per-trajectory metadata with OD, scenarios, and details
- `_build_trajectory_scenarios_mapping()`: Creates simple index→tags mapping
- `run_cross_model_scenario_analysis()`: Aggregates across all models, computes cross-model statistics

**Key Features:**
- All files include clear metadata (model, od_source, dataset)
- No more 77k line JSON monsters (removed bloated `_scenario_data`)
- Cross-model analysis pre-computes OD pairs by scenario, multi-scenario candidates
- CLI supports `--cross-model` flag for aggregation mode

### 2. `python_pipeline.py` - Automated Cross-Model Aggregation

**Updates:**
- `_run_scenario_analysis()` now runs two-step process:
  1. Individual model analysis for each OD source
  2. Cross-model aggregation analysis
- Passes `model_name` for clear data provenance
- Comprehensive error handling with tracebacks

### 3. `visualize_trajectories.py` - Read Pre-Computed Results

**Updates:**
- `_load_scenario_results()`: Loads from new file structure
  - Prefers `cross_model_analysis.json` (comprehensive)
  - Falls back to individual `scenario_metrics.json` + `trajectory_scenarios.json`
  - Backward compatible

- Removed `_save_scenario_statistics()` (redundant)

- Simplified `_generate_scenario_cross_model_comparisons()`:
  - No statistics tracking
  - Pure visualization
  - Reads pre-computed data from analyze_scenarios.py

- Updated `_generate_multi_scenario_comparisons()`:
  - No return type
  - Pure visualization function

## Benefits

1. **Clear Data Provenance**: All results clearly labeled with model/OD/dataset
2. **No Redundant Computation**: Visualization reads pre-computed statistics
3. **Smaller Files**: Focused JSON files instead of massive combined files
4. **Single Source of Truth**: analyze_scenarios.py owns all statistics
5. **Cleaner Separation**: Analysis vs. visualization concerns separated
6. **Faster Plotting**: No statistics overhead during visualization
7. **Easy Debugging**: Clear structure makes it easy to understand results later

## File Structure Example

**Individual Model Analysis:**
```json
// trajectory_metadata.json
{
  "model": "vanilla",
  "od_source": "train",
  "dataset": "Beijing",
  "trajectories": [
    {
      "index": 0,
      "origin_road_id": 12345,
      "destination_road_id": 67890,
      "timestamp": "2024-01-01T08:30:00Z",
      "scenario_tags": ["peak", "weekday", "city_center"],
      "scenario_details": {...}
    }
  ]
}
```

**Cross-Model Analysis:**
```json
// cross_model_analysis.json
{
  "dataset": "Beijing",
  "models": ["vanilla", "distilled", "distilled_seed44"],
  "od_sources": ["train", "test"],
  "analysis": {
    "train": {
      "models_loaded": {...},
      "scenarios": {
        "peak": {
          "models": ["vanilla", "distilled", "real"],
          "od_pairs": {...}
        }
      },
      "multi_scenario_candidates": {...}
    }
  }
}
```

## Usage

**Run individual model analysis:**
```bash
uv run python tools/analyze_scenarios.py \
    --eval-dir hoser-distill-optuna-6 \
    --config config/scenarios_beijing.yaml
```

**Run cross-model aggregation:**
```bash
uv run python tools/analyze_scenarios.py \
    --eval-dir hoser-distill-optuna-6 \
    --cross-model \
    --config config/scenarios_beijing.yaml
```

**Visualize (uses pre-computed results):**
```bash
uv run python visualize_trajectories.py \
    --eval-dir hoser-distill-optuna-6 \
    --scenario_cross_model
```

## Migration Notes

- Old `scenario_analysis.json` files are no longer generated
- Visualization scripts automatically detect and use new structure
- Backward compatibility maintained for old file structure
- No changes needed to existing visualization commands
