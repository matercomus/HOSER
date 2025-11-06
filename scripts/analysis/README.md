# Analysis Scripts

This directory contains analysis scripts for HOSER evaluation results.

## Scripts

### `cross_seed_analysis.py`

Computes cross-seed statistical analysis for trajectory generation evaluation results.

**Features**:
- Mean ± std for all metrics across seeds
- 95% confidence intervals using t-distribution
- Coefficient of variation (CV%) to identify high-variance metrics
- Automatic detection of evaluation result directories
- Markdown report generation

**Usage**:
```bash
# Analyze Beijing and Porto evaluation results
uv run python scripts/analysis/cross_seed_analysis.py \
    --eval_dirs "hoser-distill-optuna-6/eval" "hoser-distill-optuna-porto-eval-*/eval" \
    --output_dir docs/results \
    --confidence 0.95

# Custom minimum seeds requirement
uv run python scripts/analysis/cross_seed_analysis.py \
    --eval_dirs "hoser-distill-optuna-6/eval" \
    --min_seeds 3 \
    --output_dir docs/results
```

**Output**:
- `docs/results/CROSS_SEED_ANALYSIS.md` - Comprehensive cross-seed statistics report

**Metrics Analyzed**:
- Jensen-Shannon Divergence (Distance, Duration, Radius)
- Hausdorff Distance (km)
- Dynamic Time Warping (km)
- Edit Distance on Real sequence (EDR)
- OD Match Rate
- Real/Generated distribution means

### `aggregate_eval_scenarios.py`

Aggregates evaluation metrics and scenario analysis from HOSER evaluation results.

**Usage**:
```bash
uv run python scripts/analysis/aggregate_eval_scenarios.py \
    --root /path/to/eval_bundle \
    --dataset porto_hoser \
    --out /path/to/output
```

### `extract_wandb_perf.py`

Extracts performance metrics from Weights & Biases runs.

**Usage**:
```bash
uv run python scripts/analysis/extract_wandb_perf.py \
    --project hoser-distill-optuna-6 \
    --output results/performance.csv
```

## Analysis Workflow

1. **Run evaluations** with multiple seeds using `python_pipeline.py` or evaluation scripts
2. **Aggregate scenario analysis** using `aggregate_eval_scenarios.py`  
3. **Compute cross-seed statistics** using `cross_seed_analysis.py`
4. **Extract performance data** using `extract_wandb_perf.py` (optional)

## Output Documentation

Analysis results are saved to:
- `docs/results/CROSS_SEED_ANALYSIS.md` - Cross-seed statistical analysis
- `docs/results/TEACHER_BASELINE_COMPARISON.md` - Teacher model baselines
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` - Wang baseline results

## Statistical Notes

- **Confidence Intervals**: Computed using t-distribution for small sample sizes
- **Coefficient of Variation (CV)**: 
  - Low (<5%): Stable, seed-independent
  - Medium (5-10%): Moderate seed sensitivity
  - High (>10%): High seed sensitivity, interpret with caution
- **Minimum Seeds**: Default minimum of 2 seeds required for variance analysis
