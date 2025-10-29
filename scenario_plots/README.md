# Scenario Analysis Visualization Package

**YAML-driven, modular plotting system** for generating comprehensive scenario analysis figures with configurable metric filtering and individual plot selection.

## Structure

```
scenario_plots/
├── __init__.py                   # Package initialization
├── config_loader.py              # YAML plot configuration loader
├── data_loader.py                # Data loading utilities with metric filtering
├── metrics_plots.py              # Plots #1-3, #8 (metric comparisons)
├── temporal_spatial_plots.py     # Plots #4-5 (temporal/spatial)
├── robustness_plots.py           # Plots #6-7 (seed/difficulty)
├── analysis_plots.py             # Plots #9-11 (advanced analysis)
└── application_plots.py          # Plots #12-13 (applications/improvement)

config/
├── scenario_plots.yaml           # Plot registry and configuration
├── scenarios_beijing.yaml        # Beijing-specific settings
└── scenarios_porto.yaml          # Porto-specific settings
```

## Quick Start

### List Available Plots

See all available plots and groups:

```bash
uv run python create_scenario_plots.py --list-plots
```

### Generate Plots

```bash
# Generate all plots
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6

# Generate only heatmap visualizations
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --plots heatmaps_only

# Generate a single plot (fast for development/iteration)
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \
  --plots application.improvement_heatmaps

# Generate multiple specific plots
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \
  --plots metrics.scenario_heatmap,application.radar_charts

# With custom DPI
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 \
  --plots application --dpi 600
```

## Available Plots

All plots are defined in `config/scenario_plots.yaml` and organized by type:

### Metrics Plots (`metrics.*`)
- **metrics.scenario_heatmap**: Scenario × Metric heatmap showing performance
- **metrics.model_comparison**: Cross-model bar chart comparison
- **metrics.aggregate_comparison**: Aggregate metrics across models
- **metrics.metric_distributions**: Distribution plots for key metrics

### Temporal/Spatial Plots (`temporal_spatial.*`)
- **temporal_spatial.temporal_patterns**: Time-based performance patterns
- **temporal_spatial.spatial_patterns**: Spatial coverage and performance

### Robustness Plots (`robustness.*`)
- **robustness.scenario_robustness**: Performance stability across scenarios
- **robustness.metric_sensitivity**: Metric sensitivity analysis

### Analysis Plots (`analysis.*`)
- **analysis.difficulty_ranking**: Scenario difficulty ranking by performance
- **analysis.metric_sensitivity_by_scenario**: Most sensitive metrics per scenario
- **analysis.duration_ceiling**: Duration prediction ceiling effect analysis

### Application Plots (`application.*`)
- **application.improvement_heatmaps**: Model quality heatmaps showing distance from real data (individual + grid)
- **application.radar_charts**: Application use case radar charts

## Plot Groups

Convenient groups for common use cases (defined in `config/scenario_plots.yaml`):

- **all**: All enabled plots (default)
- **core**: Quick overview (scenario_heatmap + model quality heatmaps)
- **heatmaps_only**: Only heatmap visualizations
- **full_analysis**: Complete analysis suite (metrics + application + analysis)
- **application**: Application-focused plots (heatmaps + radars)
- **metrics**: All metric plots
- **analysis**: All analysis plots

## Configuration

### Metric Filtering

Control which metrics appear in plots by editing the dataset's scenario config file (`eval_dir/config/scenarios_*.yaml`):

```yaml
# Visualization configuration
plotting:
  # Metric filtering (applied to all plots)
  metrics:
    # Exclude by pattern (supports wildcards with * and ?)
    exclude_patterns:
      - "*_real_*"  # Real metrics mirror input data, not useful for comparison
    
    # Exclude specific metrics by exact name
    exclude:
      - "total_generated_od_pairs"
      - "matched_od_pairs"
    
    # Optional: Only include specific metrics (overrides excludes)
    # include_only:
    #   - "DTW_km"
    #   - "Hausdorff_km"
```

**Benefits:**
- Removes clutter from visualizations
- Dataset-specific filtering (Beijing vs Porto can have different settings)
- No code changes needed
- Applied automatically to all plots

### Plot-Specific Configuration

Configure individual plot settings in `config/scenario_plots.yaml`:

```yaml
plot_types:
  application:
    plots:
      improvement_heatmaps:
        functions:
          - plot_improvement_heatmaps_individual
          - plot_improvement_heatmap_grid
        description: "Model quality heatmaps (distance from real data, 0=perfect)"
        enabled: true
        config:
          colormap: "RdWhGn"
          percentile_range: [5, 95]
```

Dataset-specific overrides can be added to `scenarios_*.yaml`:

```yaml
plotting:
  plot_overrides:
    application.improvement_heatmaps:
      percentile_range: [10, 90]  # More conservative for this dataset
```

## Adding New Plots

1. **Implement the plot function** in the appropriate module (e.g., `metrics_plots.py`):

```python
def plot_my_new_visualization(data: Dict, output_dir: Path, dpi: int, 
                               loader=None, config=None):
    """Plot description
    
    Args:
        data: Loaded scenario data
        output_dir: Directory to save plots
        dpi: Output resolution
        loader: Optional ScenarioDataLoader for metric filtering
        config: Optional plot-specific configuration
    """
    # Implementation
    pass
```

2. **Register the plot** in `config/scenario_plots.yaml`:

```yaml
plot_types:
  metrics:  # Choose appropriate type
    plots:
      my_new_plot:
        function: plot_my_new_visualization
        description: "Clear description of what this plot shows"
        enabled: true
        config:
          # Optional plot-specific settings
          some_parameter: value
```

3. **Add to a group** (optional):

```yaml
groups:
  my_custom_group:
    description: "Custom analysis plots"
    plots:
      - metrics.my_new_plot
      - analysis.difficulty_ranking
```

4. **Test the plot**:

```bash
# Test individually
uv run python create_scenario_plots.py --eval-dir DIR --plots metrics.my_new_plot

# Verify it appears in listings
uv run python create_scenario_plots.py --list-plots
```

## Output

All plots are saved to `{eval_dir}/figures/scenarios/` in both PDF and PNG formats.

## Requirements

- matplotlib >= 3.5
- seaborn >= 0.12
- numpy
- PyYAML
- pathlib (stdlib)

All dependencies are already available in the project environment.

## Architecture

### ScenarioDataLoader (`data_loader.py`)
- Loads scenario evaluation data from JSON files
- Loads and applies dataset-specific configuration
- Provides `get_filtered_metrics()` for metric filtering
- Provides `get_plot_override()` for plot-specific config

### PlotConfigLoader (`config_loader.py`)
- Loads plot registry from `config/scenario_plots.yaml`
- Resolves plot groups and wildcards (e.g., `metrics.*`)
- Provides `get_enabled_plots()` for dynamic plot selection
- Lists available plots and groups

### Main Script (`create_scenario_plots.py`)
- CLI interface with `--list-plots` and `--plots` options
- Dynamically imports modules based on YAML config
- Passes `loader` and `config` to plotting functions
- Supports individual plots, groups, and combinations

### Benefits of This Architecture

1. **No code changes** to add/remove/configure plots
2. **Fast iteration** during development (run single plots)
3. **Self-documenting** via `--list-plots`
4. **Dataset-specific** settings via scenario configs
5. **Extensible** via YAML without touching Python
6. **Backwards compatible** (loader/config are optional parameters)

## Quick Reference

```bash
# Show all available plots and groups
uv run python create_scenario_plots.py --list-plots

# Generate everything (default)
uv run python create_scenario_plots.py --eval-dir EVAL_DIR

# Generate by group
uv run python create_scenario_plots.py --eval-dir EVAL_DIR --plots core
uv run python create_scenario_plots.py --eval-dir EVAL_DIR --plots heatmaps_only
uv run python create_scenario_plots.py --eval-dir EVAL_DIR --plots application

# Generate single plot (fast iteration)
uv run python create_scenario_plots.py --eval-dir EVAL_DIR --plots application.improvement_heatmaps
uv run python create_scenario_plots.py --eval-dir EVAL_DIR --plots metrics.scenario_heatmap

# Generate multiple specific plots
uv run python create_scenario_plots.py --eval-dir EVAL_DIR \
  --plots application.improvement_heatmaps,application.radar_charts

# With high DPI for publication
uv run python create_scenario_plots.py --eval-dir EVAL_DIR --plots core --dpi 600
```

## Migration Notes

### From Old System

The old hardcoded plot registry has been replaced with YAML configuration:

- Old: `--plots application` (hardcoded in Python)
- New: `--plots application` (defined in `scenario_plots.yaml`)

The CLI interface remains **backwards compatible**. All old group names still work:
- `all`, `metrics`, `temporal`, `robustness`, `analysis`, `application`, `heatmaps`, `radar`

### New Capabilities

- **Individual plot selection**: `--plots application.improvement_heatmaps`
- **List all options**: `--list-plots`
- **Custom groups**: Define your own in `scenario_plots.yaml`
- **Metric filtering**: Configure in `scenarios_*.yaml`

