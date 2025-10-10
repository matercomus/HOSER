# Evaluation Analysis Figures

This directory contains publication-quality PDF figures for the HOSER distillation evaluation analysis.

## Figure Descriptions

### 1. `distance_distributions.pdf`
Comparison of generated vs real trajectory distances for train and test OD pairs.
Shows that distilled models generate realistic ~6.4 km trips while vanilla generates unrealistic ~2.4 km trips.

### 2. `od_matching_rates.pdf`
Bar chart showing OD pair coverage rates.
Demonstrates that distilled models achieve 85-89% coverage while vanilla only achieves 12-18%.

### 3. `jsd_comparison.pdf`
Jensen-Shannon Divergence comparison for distance and radius distributions.
Log scale visualization showing distilled models achieve 87-98% improvement over vanilla.

### 4. `metrics_heatmap.pdf`
Comprehensive heatmap of all evaluation metrics across all models and OD sources.
Includes actual values and normalized visualization for easy comparison.

### 5. `train_test_comparison.pdf`
Generalization analysis comparing train vs test performance.
Shows distilled models generalize better (lower JSD on test) while vanilla degrades.

### 6. `seed_robustness.pdf`
Cross-seed consistency analysis for distilled models.
Demonstrates that distillation produces reproducible results across different random seeds.

### 7. `local_metrics.pdf`
Trajectory-level metrics comparison (Hausdorff, DTW, EDR).
Shows local similarity measures with value annotations.

### 8. `performance_radar.pdf`
Overall performance radar chart with 5 key dimensions.
Provides at-a-glance comparison of model strengths across multiple metrics.

## Usage

All figures are referenced in `EVALUATION_ANALYSIS.md` and are suitable for:
- Thesis presentations
- Academic papers
- Technical reports
- Stakeholder presentations

## Regeneration

To regenerate all figures:
```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6
uv run python create_analysis_figures.py
```

## Format

- **Format:** PDF (vector graphics, publication quality)
- **Resolution:** 300 DPI
- **Size:** Optimized for standard paper and presentations
- **Style:** Consistent color scheme and typography

