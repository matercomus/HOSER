# Optuna Hyperparameter Optimization Visualizations

This directory contains interactive visualizations from the Beijing HOSER distillation hyperparameter optimization study.

## Files

- **optimization_history.html** - Convergence trajectory showing validation accuracy over trials
- **param_importance.html** - Hyperparameter sensitivity analysis (fANOVA-based importance scores)
- **parallel_coordinate.html** - Multi-dimensional visualization of hyperparameter-objective relationships
- **slice_plot.html** - 2D projections showing individual parameter effects
- **contour_plot.html** - 2D heatmaps showing parameter interaction effects
- **edf_plot.html** - Empirical distribution function showing convergence characteristics
- **timeline.html** - Trial execution timeline with pruning decisions

## Usage

### Viewing Plots

1. Open any `.html` file in a web browser (Chrome, Firefox, Edge, Safari)
2. Interactive features available:
   - **Zoom:** Scroll wheel or pinch
   - **Pan:** Click and drag
   - **Details:** Hover over data points
   - **Selection:** Click legend items to toggle visibility

### Exporting for Publications

To create PNG images for documentation:

1. Open the HTML file in a browser
2. Use browser's native screenshot tools:
   - **Chrome:** Right-click → "Capture screenshot" (full page)
   - **Firefox:** Right-click → "Take a Screenshot" → "Save full page"
   - **macOS:** Cmd+Shift+4 then select area
   - **Windows:** Windows+Shift+S
3. Or use browser developer tools to save rendered canvas

### Regenerating Plots

If the Optuna database is updated:

```bash
cd /home/matt/Dev/HOSER
uv run python tools/generate_optuna_plots.py
```

This will regenerate all HTML files from the current database state.

## Study Details

- **Study Name:** `hoser_tuning_20251003_162916`
- **Database:** `/mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db`
- **Total Trials:** 12
- **Optimization Objective:** Maximize validation next-step prediction accuracy
- **Search Space:** Lambda (KL weight), Temperature (distribution smoothing), Window (context length)

## Documentation

Complete analysis and interpretation of these visualizations: `../Hyperparameter-Optimization.md`

For implementation details: `../../tune_hoser.py`

