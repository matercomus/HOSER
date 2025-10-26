# Scenario Analysis Visualization Package

Modular plotting system for generating comprehensive scenario analysis figures.

## Structure

```
scenario_plots/
├── __init__.py                   # Package initialization
├── data_loader.py                # Data loading utilities
├── metrics_plots.py              # Plots #1-3, #8 (metric comparisons)
├── temporal_spatial_plots.py     # Plots #4-5 (temporal/spatial)
├── robustness_plots.py           # Plots #6-7 (seed/difficulty)
├── analysis_plots.py             # Plots #9-11 (advanced analysis)
└── application_plots.py          # Plots #12-13 (applications/improvement)
```

## Usage

Generate all 13 scenario analysis plots:

```bash
# For Beijing evaluation
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6

# For Porto evaluation
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-porto-eval-xxx

# With custom DPI
uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6 --dpi 600
```

## Generated Plots

### Metrics Plots (4 plots)
1. **scenario_metrics_heatmap**: 6-panel heatmap showing all metrics across scenarios
2. **train_od_scenario_comparison**: Grouped bar chart for train OD
3. **test_od_scenario_comparison**: Grouped bar chart for test OD
4. **metric_sensitivity_by_scenario**: 3×3 grid showing metric sensitivity

### Temporal/Spatial Plots (2 plots)
5. **temporal_scenarios_comparison**: 3-panel line plot for temporal scenarios
6. **spatial_scenarios_analysis**: 2-panel spatial complexity analysis

### Robustness Plots (2 plots)
7. **seed_robustness_scenarios**: 6-panel bar chart comparing seeds
8. **scenario_difficulty_ranking**: Horizontal bar chart ranking difficulty

### Analysis Plots (3 plots)
9. **duration_ceiling_effect**: Box plots showing duration JSD ceiling
10. **spatial_metrics_differentiation**: Scatter plot with performance zones
11. **scenario_variance_analysis**: Range plot showing variance across scenarios

### Application Plots (2 plots)
12. **application_use_case_radar**: 3 radar charts for different applications
13. **improvement_heatmap**: Percentage improvement matrix

## Output

All plots are saved to `{eval_dir}/figures/scenarios/` in both PDF and PNG formats.

## Requirements

- matplotlib >= 3.5
- seaborn >= 0.12
- numpy
- json (stdlib)
- pathlib (stdlib)

All dependencies are already available in the project environment.

