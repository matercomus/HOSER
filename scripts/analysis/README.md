# HOSER Evaluation Analysis Tools

## Overview

This directory contains reusable scripts for aggregating and analyzing HOSER evaluation results across models, seeds, and scenarios.

## Scripts

### `aggregate_eval_scenarios.py`

Aggregates evaluation metrics from HOSER evaluation bundles and generates:
- CSV files with per-run, per-group, and per-scenario metrics
- JSON with structured aggregates for programmatic reuse
- Markdown fragments for inclusion in evaluation documents

**Features:**
- Handles multiple models and seeds automatically
- Computes group-level statistics (distilled vs vanilla)
- Calculates deltas, CV%, and identifies top scenarios
- Defensive parsing with graceful error handling
- Reproducible outputs for CI/batch reporting

## Usage

### Basic Usage

```bash
uv run python scripts/analysis/aggregate_eval_scenarios.py \
  --root /path/to/eval_bundle \
  --dataset porto_hoser \
  --out /path/to/output
```

### Example: Porto Phase 1 Evaluation

```bash
uv run python scripts/analysis/aggregate_eval_scenarios.py \
  --root /home/matt/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --out /home/matt/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--root` | *(required)* | Root directory of evaluation bundle |
| `--dataset` | `porto_hoser` | Dataset identifier |
| `--eval_dir` | `<root>/eval` | Override path to eval runs |
| `--scenarios_dir` | `<root>/scenarios` | Override path to scenarios |
| `--models` | `auto` | Comma-separated model list or 'auto' |
| `--od_sources` | `train,test` | Comma-separated OD sources |
| `--out` | `<root>/analysis` | Output directory |
| `--md` | `True` | Generate Markdown fragments |
| `--fig_prefix` | `figures/` | Prefix for figure paths in markdown |

## Outputs

### CSV Files

- `results_overview.csv` - 12-row table (6 models × 2 OD sources)
- `aggregates_train.csv` - Group-level statistics for train set
- `aggregates_test.csv` - Group-level statistics for test set
- `scenarios_train.csv` - Per-scenario metrics for train set
- `scenarios_test.csv` - Per-scenario metrics for test set
- `top_scenarios_train.csv` - Top-5 scenarios by metric improvement (train)
- `top_scenarios_test.csv` - Top-5 scenarios by metric improvement (test)

### JSON Files

- `aggregates.json` - Complete structured data for programmatic access

### Markdown Fragments

- `md/results_table.md` - Real data baseline + aggregated comparison tables
- `md/scenario_analysis.md` - Per-scenario tables + top scenarios + interpretations

## Input Data Structure

The script expects the following directory structure:

```
eval_bundle_root/
├── eval/
│   ├── <timestamp_1>/
│   │   └── results.json
│   ├── <timestamp_2>/
│   │   └── results.json
│   └── ...
├── scenarios/
│   ├── train/
│   │   ├── distill/
│   │   │   └── scenario_metrics.json
│   │   ├── vanilla/
│   │   │   └── scenario_metrics.json
│   │   └── ...
│   └── test/
│       ├── distill/
│       │   └── scenario_metrics.json
│       ├── vanilla/
│       │   └── scenario_metrics.json
│       └── ...
└── analysis/         (created by script)
    ├── *.csv
    ├── aggregates.json
    └── md/
        ├── results_table.md
        └── scenario_analysis.md
```

### Required Fields in `eval/*/results.json`

```json
{
  "Distance_JSD": 0.0050,
  "Radius_JSD": 0.0092,
  "Duration_JSD": 0.0256,
  "Distance_gen_mean": 3.563,
  "Hausdorff_km": 0.565,
  "DTW_km": 15.57,
  "EDR": 0.476,
  "matched_od_pairs": 4064,
  "total_generated_od_pairs": 4571,
  "metadata": {
    "model_type": "distill",
    "od_source": "test",
    "seed": 42
  }
}
```

### Required Structure in `scenarios/*/model/scenario_metrics.json`

```json
{
  "model": "distill",
  "od_source": "test",
  "dataset": "porto_hoser",
  "individual_scenarios": {
    "city_center": {
      "count": 4571,
      "percentage": 91.42,
      "metrics": {
        "Distance_JSD": 0.0065,
        "Radius_JSD": 0.0118,
        "Distance_gen_mean": 3.552,
        "Hausdorff_km": 0.564,
        "DTW_km": 15.30,
        "EDR": 0.469,
        "matched_od_pairs": 3603,
        "total_generated_od_pairs": 4183
      }
    },
    ...
  }
}
```

## Reusability

This script is designed to be reusable across:
- **Different datasets**: Beijing, Porto Phase 2, future datasets
- **Different evaluation bundles**: Any directory with the expected structure
- **CI/CD pipelines**: Deterministic outputs, exit codes, structured logging

### Example: Beijing Evaluation

```bash
uv run python scripts/analysis/aggregate_eval_scenarios.py \
  --root /home/matt/Dev/HOSER/hoser-distill-optuna-6 \
  --dataset beijing \
  --out /home/matt/Dev/HOSER/hoser-distill-optuna-6/analysis
```

### Example: Porto Phase 2 (Future)

```bash
uv run python scripts/analysis/aggregate_eval_scenarios.py \
  --root /home/matt/Dev/HOSER/hoser-distill-optuna-porto-phase2-eval-HASH \
  --dataset porto_hoser \
  --out /home/matt/Dev/HOSER/hoser-distill-optuna-porto-phase2-eval-HASH/analysis
```

## Metrics Computed

### Overall Metrics (per group, per OD source)
- Match rate (%)
- Distance JSD, Radius JSD, Duration JSD
- Distance generated mean (km)
- Hausdorff (km), DTW (km), EDR
- Mean, min, max, std, CV% across seeds

### Per-Scenario Metrics
- Same metrics as overall, but broken down by scenario
- Delta (Δ) = distilled - vanilla
- Top-5 scenarios by largest positive and negative deltas

### Cross-Seed Consistency
- Coefficient of Variation (CV%) per metric
- Lower CV% indicates more consistent performance across seeds

## Integration with Evaluation Documents

Generated Markdown fragments can be directly included in evaluation documents:

```markdown
<!-- Include aggregated results table -->
<!-- Content from md/results_table.md -->

<!-- Include scenario analysis -->
<!-- Content from md/scenario_analysis.md -->
```

## Dependencies

- Python 3.12+
- `polars` (for CSV/dataframe operations)
- Standard library: `json`, `pathlib`, `argparse`, `collections`

Install dependencies:
```bash
uv add polars
```

## Quality Assurance

- Linted with `ruff`: `uv tool run ruff check scripts/analysis/aggregate_eval_scenarios.py`
- Formatted with `ruff`: `uv tool run ruff format scripts/analysis/aggregate_eval_scenarios.py`
- Defensive parsing: Gracefully handles missing files and fields
- Unit-consistent: km for distances, hours for durations
- Direction-aware: Lower is better for JSD/DTW/Hausdorff/EDR

## Future Enhancements

Potential additions:
- Statistical significance testing (t-tests, bootstrap)
- Automated plot generation (scenario heatmaps, radar charts)
- HTML report generation with embedded figures
- Cross-dataset comparison tables
- Time-series analysis for multi-phase evaluations

