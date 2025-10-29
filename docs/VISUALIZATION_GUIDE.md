# Visualization Scripts Guide

This guide covers the usage of the visualization scripts for HOSER evaluation results.

## Overview

Three visualization scripts are available at the project root, all designed to work with evaluation directories created by `setup_evaluation.py`:

1. **`visualize_trajectories.py`** - Interactive trajectory visualizations with scenario-aware sampling
2. **`create_distribution_plots.py`** - Statistical distribution comparisons
3. **`create_analysis_figures.py`** - Publication-quality analysis figures including scenario breakdowns

## Prerequisites

All scripts automatically:
- Detect the dataset from `evaluation.yaml` in the eval directory
- Create output directories within the eval directory (`figures/`)
- Support both Beijing and Porto datasets
- Load scenario analysis results if available

## 1. Trajectory Visualization (`visualize_trajectories.py`)

### Features
- Visualize real vs. generated trajectories
- Multiple sampling strategies (random, length-based, representative, scenario)
- Scenario-aware sampling (uses scenario analysis results if available)
- Separate and overlaid comparison modes
- Cross-model comparisons for same OD pairs
- **NEW: Scenario-based cross-model comparisons** - Compare all models across different scenarios
- Road network overlay

### Basic Usage

```bash
# Visualize trajectories from an evaluation directory
uv run python visualize_trajectories.py --eval-dir hoser-distill-optuna-6

# Use scenario-based sampling (if scenario analysis was run)
uv run python visualize_trajectories.py --eval-dir hoser-distill-optuna-6 --sample_strategy scenario

# Generate cross-model comparisons
uv run python visualize_trajectories.py --eval-dir hoser-distill-optuna-6 --cross_model

# Only overlaid plots (no separate)
uv run python visualize_trajectories.py --eval-dir hoser-distill-optuna-6 --no_separate
```

### Sampling Strategies

- **`length_based`** (default): Samples short, medium, and long trajectories
- **`random`**: Random sampling from all trajectories
- **`representative`**: Median-length trajectory (most representative)
- **`scenario`**: Samples from top N scenarios (requires scenario analysis results)

### Options

```bash
--eval-dir EVAL_DIR           # Required: evaluation directory
--dataset DATASET             # Optional: override auto-detected dataset
--sample_strategy STRATEGY    # Sampling strategy (default: length_based)
--samples_per_type N          # Number of samples per type (default: 1)
--max_scenarios N             # Max scenarios to plot (default: 5, for scenario strategy)
--random_seed SEED            # Random seed (default: 42)
--no_separate                 # Skip separate plots
--no_overlaid                 # Skip overlaid plots
--cross_model                 # Generate cross-model comparisons
--no_real                     # Exclude real trajectories from cross-model
--dpi DPI                     # Output resolution (default: 300)
```

### Output

- Saved to: `{eval_dir}/figures/trajectories/`
- Formats: PNG images

## 2. Distribution Plots (`create_distribution_plots.py`)

### Features
- Distance distribution comparisons
- Duration distribution comparisons
- Statistical overlays (KDE, histograms)
- Real vs. generated comparisons across all models

### Basic Usage

```bash
# Generate distribution plots
uv run python create_distribution_plots.py --eval-dir hoser-distill-optuna-6

# Specify dataset explicitly
uv run python create_distribution_plots.py --eval-dir hoser-distill-optuna-6 --dataset Beijing

# Enable verbose logging
uv run python create_distribution_plots.py --eval-dir hoser-distill-optuna-6 --verbose
```

### Options

```bash
--eval-dir EVAL_DIR    # Required: evaluation directory
--dataset DATASET      # Optional: override auto-detected dataset
--verbose              # Enable verbose logging
```

### Output

- Saved to: `{eval_dir}/figures/distributions/`
- Formats: PNG images
- Plots: Distance distributions, duration distributions, KDE overlays

## 3. Scenario-Based Cross-Model Comparisons

### Overview
The scenario cross-model mode **filters trajectories by scenario**, then uses the same OD-matching logic as regular cross-model comparisons. This ensures fair comparisons where all models are evaluated on the **same routes** within each scenario context (peak/off-peak, weekday/weekend, city center, etc.).

### Features
- **OD-matched comparisons within scenarios**: Only compares trajectories with matching origin-destination pairs
- **Multiple plots per scenario**: Generates one plot for each common OD pair found within the scenario
- **Fair route comparisons**: All models navigate the same route in the same scenario conditions
- **No arbitrary filtering**: Processes all scenarios regardless of trajectory count
- **Reuses proven logic**: Same plot style and OD-matching as regular cross-model mode

### Usage

```bash
# Generate scenario-based cross-model comparisons (OD-matched within scenarios)
uv run python visualize_trajectories.py \
    --eval-dir hoser-distill-optuna-6 \
    --scenario_cross_model

# Combine with other modes (skip overlaid/separate, only do scenario cross-model)
uv run python visualize_trajectories.py \
    --eval-dir hoser-distill-optuna-6 \
    --scenario_cross_model \
    --no_separate --no_overlaid
```

### Options

```bash
--scenario_cross_model            # Enable scenario-based cross-model mode with OD matching
```

### Output

Per-scenario OD comparison plots organized by scenario:
- `scenario_cross_model/train/off_peak/train_od_comparison_1_origin997_dest26798.{pdf,png}`
- `scenario_cross_model/train/off_peak/train_od_comparison_2_origin17308_dest16404.{pdf,png}`
- `scenario_cross_model/train/weekday/train_od_comparison_1_origin997_dest26798.{pdf,png}`
- `scenario_cross_model/train/suburban/train_od_comparison_1_origin1154_dest36175.{pdf,png}`
- ... (multiple plots per scenario, one for each matching OD pair)

Directory structure:
```
scenario_cross_model/
├── train/
│   ├── off_peak/          # 3 OD comparisons
│   ├── weekday/           # 1 OD comparison  
│   ├── suburban/          # 3 OD comparisons
│   └── ...
└── test/
    ├── off_peak/          # 6 OD comparisons
    ├── weekday/           # 6 OD comparisons
    └── ...
```

Each plot shows all models (vanilla, distilled, distilled_seed44, real) for the same OD pair within the specific scenario context.

### Requirements

- Requires scenario analysis to be run first (using `tools/analyze_scenarios.py`)
- Or run `python_pipeline.py --run-scenarios` to generate scenarios automatically
- The scenario analysis must generate `trajectory_scenarios.json` mapping file

### Plot Titles

Titles include scenario context to make it clear what conditions are being compared:

```
"Off Peak Scenario - TRAIN OD: All Models - Origin 997 → Destination 26798"
"Weekday Scenario - TEST OD: All Models - Origin 832 → Destination 17361"
"City Center Scenario - TRAIN OD: All Models - Origin 17496 → Destination 33590"
```

### Multi-Scenario Comparisons

**New Feature**: For OD pairs that appear in 2+ scenarios, a concatenated multi-panel plot is automatically generated showing the same route under different conditions side-by-side.

Example output:
- `scenario_cross_model/train/multi_scenario/train_origin997_dest26798_multi_scenario.{pdf,png}`
  - Shows Origin 997 → Dest 26798 across 3 scenarios: Off Peak | Suburban | Weekday
  - Each panel shows all models (vanilla, distilled, distilled_seed44, real) for that scenario
  - Perfect for analyzing subtle differences in the same route under varying conditions

Benefits:
- **Side-by-side comparison** of same OD pair across scenarios
- **Identify scenario-specific model behavior** (e.g., does the model perform worse during peak hours?)
- **Publication-ready** multi-panel figures showing condition variations

Output structure:
```
scenario_cross_model/
├── train/
│   ├── multi_scenario/          # NEW: Multi-scenario concatenated plots
│   │   ├── train_origin997_dest26798_multi_scenario.{pdf,png}    # 3 scenarios
│   │   └── train_origin17308_dest16404_multi_scenario.{pdf,png}  # 2 scenarios
│   ├── off_peak/                # Individual scenario plots
│   └── ...
└── test/
    └── multi_scenario/          # NEW: Multi-scenario concatenated plots
        ├── test_origin31410_dest374_multi_scenario.{pdf,png}     # 3 scenarios
        └── ...
```

## 4. Comprehensive Scenario Plots (`create_scenario_plots.py`)

**NEW: YAML-driven plotting system** with configurable metric filtering and individual plot selection.

### Features
- 13 publication-quality analysis plots organized by type
- YAML-based configuration for extensibility
- Individual plot selection for fast iteration
- Dataset-specific metric filtering
- Plot groups for convenience
- Self-documenting with `--list-plots`

### Quick Start

```bash
# List all available plots and groups
uv run python create_scenario_plots.py --list-plots

# Generate all plots (default)
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6

# Generate only heatmap visualizations (fast)
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots heatmaps_only

# Generate a single plot (development/iteration)
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \
  --plots application.improvement_heatmaps

# Generate multiple specific plots
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \
  --plots metrics.scenario_heatmap,application.radar_charts
```

### Available Plots

**Metrics Plots (`metrics.*`)**
- `metrics.scenario_heatmap` - Scenario × Metric performance heatmap
- `metrics.model_comparison` - Cross-model bar chart comparison
- `metrics.aggregate_comparison` - Aggregate metrics across models
- `metrics.metric_distributions` - Distribution plots for key metrics

**Temporal/Spatial Plots (`temporal_spatial.*`)**
- `temporal_spatial.temporal_patterns` - Time-based performance patterns
- `temporal_spatial.spatial_patterns` - Spatial coverage and performance

**Robustness Plots (`robustness.*`)**
- `robustness.scenario_robustness` - Performance stability across scenarios
- `robustness.metric_sensitivity` - Metric sensitivity analysis

**Analysis Plots (`analysis.*`)**
- `analysis.difficulty_ranking` - Scenario difficulty ranking
- `analysis.metric_sensitivity_by_scenario` - Most sensitive metrics per scenario
- `analysis.duration_ceiling` - Duration prediction ceiling effect

**Application Plots (`application.*`)**
- `application.improvement_heatmaps` - Distance heatmaps showing raw metric values (lower=better, individual + grid)
- `application.radar_charts` - Application use case radar charts

### Plot Groups

Pre-defined groups for convenience:
- `all` - All plots (default)
- `core` - Quick overview (scenario_heatmap + distance heatmaps)
- `heatmaps_only` - Only heatmap visualizations
- `full_analysis` - Complete analysis suite
- `application` - Application-focused plots (distance heatmaps + radars)
- `metrics`, `analysis`, `robustness`, `temporal` - Type-specific groups

### Configuration

**Metric Filtering** (`eval_dir/config/scenarios_*.yaml`):
```yaml
plotting:
  metrics:
    exclude_patterns:
      - "*_real_*"  # Exclude metrics mirroring input data
    exclude:
      - "total_generated_od_pairs"
      - "matched_od_pairs"
```

**Plot-Specific Settings** (`config/scenario_plots.yaml`):
```yaml
plot_types:
  application:
    plots:
      improvement_heatmaps:
        config:
          colormap: "RdYlGn_r"  # Reversed Red-Yellow-Green (green=good, red=bad)
          percentile_range: [5, 95]
```

## Distance Heatmaps (application.improvement_heatmaps)

The distance heatmaps visualize how close each model's outputs are to real data across different scenarios and metrics.

### Key Features

**1. Raw Distance Values**
- Shows actual metric values (JSD, Hausdorff, DTW, EDR)
- Lower values = closer to real data (better performance)
- All metrics measure distance from reality where 0 = perfect match

**2. Per-Metric Color Normalization**
- Each metric column has its own color scale
- Green = best value for that metric (across all scenarios/models)
- Red = worst value for that metric
- Makes it easy to see relative performance within each metric
- Example: JSD of 0.001 might be green, Hausdorff of 5.0 might also be green (both are best in their respective columns)

**3. Two Visualization Modes**

**Individual Heatmaps** (`plot_improvement_heatmaps_individual`):
- One heatmap per distilled model
- Rows = scenarios (e.g., peak, off_peak, city_center)
- Columns = metrics (e.g., JSD, Hausdorff, DTW, EDR)
- Annotations show actual metric values
- Filename: `distance_heatmap_{model_name}.pdf`

**Grid Heatmap** (`plot_improvement_heatmap_grid`):
- All models shown side-by-side in a grid
- Enables easy comparison between vanilla and distilled models
- Same per-metric normalization applied consistently across all models
- Filename: `distance_heatmap_grid.pdf`

**4. Configurable Colormap**
- Default: `RdYlGn_r` (reversed Red-Yellow-Green)
- Can be changed via `config/scenario_plots.yaml`
- Any matplotlib colormap name is valid (e.g., `Spectral_r`, `viridis`, `coolwarm`)

### Interpretation

- **Green cells**: Model performs well on this metric/scenario combination
- **Yellow cells**: Model has average performance
- **Red cells**: Model performs poorly on this metric/scenario combination
- **Numbers**: Actual distance values for precise comparison
- **Colors**: Relative performance for visual pattern recognition

### Example Output

```
Distance heatmap showing Distill model:
                  JSD    Hausdorff  DTW    EDR
peak              0.012  4.5        3.2    0.15  (green/yellow/yellow/green)
off_peak          0.008  3.8        2.9    0.12  (green/green/green/green)
city_center       0.015  5.2        4.1    0.18  (yellow/red/red/yellow)
```

In this example:
- JSD values range 0.008-0.015 (all green/yellow = good)
- Hausdorff values range 3.8-5.2 (green to red = varies)
- Model performs best during off_peak hours
- City center scenarios are most challenging

### Options

```bash
--eval-dir EVAL_DIR        # Required: evaluation directory
--plots SELECTION          # Plot IDs, groups, or comma-separated (default: all)
--dpi DPI                  # Output resolution (default: 300)
--list-plots               # List all available plots and groups, then exit
```

### Output

- Saved to: `{eval_dir}/figures/scenarios/`
- Formats: PDF (publication-quality) and PNG
- All 13 plots with consistent styling

### Benefits

1. **Fast iteration** - Generate only the plot you're working on
2. **No code changes** - Add/remove plots via YAML configuration
3. **Dataset-specific** - Different metric filtering per dataset
4. **Self-documenting** - Use `--list-plots` to see all options
5. **Extensible** - Easy to add new plots without touching main code

### Examples

```bash
# Development: iterate on single plot
uv run python create_scenario_plots.py --eval-dir DIR \
  --plots application.improvement_heatmaps

# Publication: high-DPI core visualizations
uv run python create_scenario_plots.py --eval-dir DIR \
  --plots core --dpi 600

# Analysis: generate all heatmaps
uv run python create_scenario_plots.py --eval-dir DIR \
  --plots heatmaps_only

# Custom: specific plots you need
uv run python create_scenario_plots.py --eval-dir DIR \
  --plots metrics.scenario_heatmap,application.radar_charts,analysis.difficulty_ranking
```

See `scenario_plots/README.md` for detailed documentation on adding new plots and configuration options.

## 5. Analysis Figures (`create_analysis_figures.py`)

### Features
- Publication-quality PDF and PNG figures
- Metric comparisons across models
- Performance radar charts
- Scenario-based visualizations (if scenario analysis was run)

### Basic Usage

```bash
# Generate all analysis figures
uv run python create_analysis_figures.py --eval-dir hoser-distill-optuna-6

# Specify dataset explicitly
uv run python create_analysis_figures.py --eval-dir hoser-distill-optuna-6 --dataset Porto
```

### Options

```bash
--eval-dir EVAL_DIR    # Required: evaluation directory
--dataset DATASET      # Optional: override auto-detected dataset
```

### Output

- Saved to: `{eval_dir}/figures/analysis/`
- Formats: PDF (publication-quality) and PNG
- Figures:
  - Metric comparisons
  - Performance radar charts
  - Model comparisons

### Scenario Visualizations

If scenario analysis has been run (using `tools/analyze_scenarios.py`), the `ScenarioVisualizer` class provides additional plots:

```bash
# Scenario analysis creates its own plots automatically
uv run python tools/analyze_scenarios.py --eval-dir hoser-distill-optuna-6 --config config/scenarios_beijing.yaml

# Or trigger from python_pipeline.py with --run-scenarios flag
```

**Scenario plots include:**
- Scenario distribution bar charts
- Metric comparison heatmaps
- Hierarchical breakdown pie charts

**Output location:** `{eval_dir}/scenarios/{od_source}/{model}/`

## Integration with Scenario Analysis

All visualization scripts auto-detect and enhance their output if scenario analysis results are available:

1. **Trajectory Visualization**: 
   - Enables `--sample_strategy scenario` option
   - Samples trajectories from top N scenarios
   - Adds scenario labels to plots

2. **Distribution Plots**:
   - Can be filtered by scenario (future enhancement)

3. **Analysis Figures**:
   - Includes scenario-specific metric breakdowns
   - Uses `ScenarioVisualizer` class for scenario plots

## Workflow Example

```bash
# 1. Setup evaluation directory
uv run python setup_evaluation.py --dataset Beijing --models distilled vanilla

# 2. Run evaluation pipeline
cd eval_Beijing_20251024_123456
uv run python ../python_pipeline.py --run-scenarios

# 3. Generate visualizations (from project root)
cd ..
uv run python visualize_trajectories.py --eval-dir eval_Beijing_20251024_123456 --sample_strategy scenario
uv run python create_distribution_plots.py --eval-dir eval_Beijing_20251024_123456
uv run python create_scenario_plots.py --eval-dir eval_Beijing_20251024_123456
uv run python create_analysis_figures.py --eval-dir eval_Beijing_20251024_123456

# 4. All figures are now in eval_Beijing_20251024_123456/figures/
```

## Output Directory Structure

After running all visualization scripts:

```
eval_directory/
├── figures/
│   ├── trajectories/          # From visualize_trajectories.py
│   │   ├── separate/          # Individual model plots
│   │   ├── overlaid/          # Combined plots per model
│   │   ├── cross_model/       # Same OD pair, different models
│   │   └── scenario_cross_model/  # Scenario-based comparisons
│   │       ├── train/
│   │       │   ├── off_peak_comparison.{pdf,png}
│   │       │   ├── weekday_comparison.{pdf,png}
│   │       │   ├── city_center_comparison.{pdf,png}
│   │       │   └── all_scenarios_grid.{pdf,png}
│   │       └── test/
│   │           └── ... (same structure)
│   ├── distributions/         # From create_distribution_plots.py
│   │   ├── distance_distribution.png
│   │   └── duration_distribution.png
│   ├── scenarios/             # NEW: From create_scenario_plots.py (13 plots)
│   │   ├── improvement_heatmap_*.{pdf,png}
│   │   ├── improvement_heatmap_grid.{pdf,png}
│   │   ├── scenario_metrics_heatmap.{pdf,png}
│   │   ├── application_use_case_radar.{pdf,png}
│   │   ├── scenario_difficulty_ranking.{pdf,png}
│   │   └── ... (all 13 scenario analysis plots)
│   ├── analysis/              # From create_analysis_figures.py
│   │   ├── metric_comparison.pdf
│   │   ├── performance_radar.pdf
│   │   └── ...
│   └── old_scenarios/         # From ScenarioVisualizer (legacy, if scenarios run)
│       ├── train/
│       │   └── vanilla/
│       │       ├── scenario_distribution.png
│       │       ├── metric_comparison.png
│       │       └── hierarchical_*.png
│       └── test/
└── scenarios/                 # Raw scenario analysis results
    ├── train/
    │   └── vanilla/
    │       ├── scenario_analysis.json
    │       └── trajectory_scenarios.json
    └── test/
```

## Tips

1. **Run scenario analysis first** to enable scenario-aware sampling in trajectory visualization
2. **Use `--verbose`** for debugging if plots aren't generating as expected
3. **Check dataset detection** - scripts auto-detect from `evaluation.yaml`, but you can override with `--dataset`
4. **All scripts work from project root** - they find data and save outputs relative to the eval directory
5. **PDF outputs** are publication-quality from `create_analysis_figures.py`

## Troubleshooting

### "No generated files found"
- Ensure trajectory generation completed successfully
- Check that `gene/{dataset}/` exists in the eval directory
- Verify model names match expected patterns (vanilla, distilled, etc.)

### "Cannot load scenario results"
- Scenario analysis must be run first: `tools/analyze_scenarios.py`
- Or enable in pipeline: `python_pipeline.py --run-scenarios`
- Check that `scenarios/` directory exists with `scenario_analysis.json` files

### "Dataset not detected"
- Verify `config/evaluation.yaml` exists in eval directory
- Use `--dataset` flag to override: `--dataset Beijing` or `--dataset Porto`

### "Road network not found"
- Ensure data directory exists: `data/Beijing/` or `data/porto_hoser/`
- Check that `roadmap.geo` file is present
- Data paths are relative to project root, not eval directory

## Advanced Usage

### Batch Processing Multiple Eval Directories

```bash
# Process all eval directories with full visualization suite
for eval_dir in eval_*/; do
    echo "Processing $eval_dir..."
    uv run python visualize_trajectories.py --eval-dir "$eval_dir"
    uv run python create_distribution_plots.py --eval-dir "$eval_dir"
    uv run python create_scenario_plots.py --eval-dir "$eval_dir"
    uv run python create_analysis_figures.py --eval-dir "$eval_dir"
done

# Quick batch processing with only core plots
for eval_dir in eval_*/; do
    echo "Processing $eval_dir (core plots only)..."
    uv run python create_scenario_plots.py --eval-dir "$eval_dir" --plots core
done
```

### Custom Scenario Sampling

```python
# In visualize_trajectories.py, the ScenarioSampler can be extended
# to implement custom sampling logic based on scenario tags
```

## Related Documentation

- **Setup Guide**: `docs/SETUP_EVALUATION_GUIDE.md`
- **Scenario Analysis**: `docs/SCENARIO_ANALYSIS_GUIDE.md`
- **Evaluation Pipeline**: `docs/EVALUATION_PIPELINE_GUIDE.md`

