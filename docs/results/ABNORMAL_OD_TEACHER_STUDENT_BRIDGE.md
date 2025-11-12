# Abnormal OD Analysis: Bridge Between Teacher and Student Evaluation

## Overview

The abnormal OD analysis workflow provides a **common test set** and **evaluation framework** that connects teacher (LM-TAD) and student (HOSER) model evaluation, enabling direct comparison despite different tasks and architectures.

## The Connection

### Teacher Model (LM-TAD)
- **Task**: Outlier detection (classification)
- **Input**: Complete trajectory
- **Output**: Binary classification (normal vs abnormal)
- **Metric**: F1 score on outlier detection
- **Current Performance**: 
  - Beijing: 83.89% F1
  - Porto: 91.10% F1

### Student Model (HOSER)
- **Task**: Trajectory generation
- **Input**: Origin-destination (OD) pair
- **Output**: Generated trajectory
- **Metric**: Abnormality reproduction rate (how well generated trajectories reproduce abnormal patterns)
- **Current Performance**: 
  - Porto: 0.98% mean abnormality rate (vs 9.86% real)

### The Bridge: Abnormal OD Pairs

**Abnormal OD pairs** extracted from real data serve as the common test set:

1. **For Teacher**: Evaluate detection accuracy on trajectories with these OD pairs
2. **For Student**: Generate trajectories for these OD pairs, then evaluate pattern reproduction

## Current Analysis Results (Porto Dataset)

### Real Abnormal Data
- **Total Abnormal Trajectories**: 61,046 (9.86% of 618,891)
- **Unique Abnormal OD Pairs**: 13,348
- **Pattern Distribution**: 100% temporal delays (Abp2_temporal_delay)
- **Test Split Rate**: 10.43% (14,349 / 137,532)
- **Train Split Rate**: 9.70% (46,697 / 481,359)

### Generated Trajectory Performance
- **Mean Abnormality Rate**: 0.98% (dramatically lower than real 9.86%)
- **Best Model**: `vanilla_seed43` (3.54% abnormality rate, 6.89% deviation from real)
- **Pattern**: All models significantly underperform on abnormal pattern reproduction

## Integration Strategy

### Step 1: Extract Abnormal OD Pairs
✅ **Completed** for Porto:
- File: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal_od_pairs_porto_hoser.json`
- Contains 13,348 unique abnormal OD pairs
- Categorized by abnormality type (currently all temporal delays)

### Step 2: Evaluate Teacher on Abnormal OD Pairs
⏳ **Pending**:
- Use LM-TAD teacher model to detect abnormalities in trajectories with these OD pairs
- Calculate teacher's detection accuracy for abnormal OD pairs specifically
- Compare to overall teacher performance (91.10% F1 Porto)

### Step 3: Generate Student Trajectories for Abnormal OD Pairs
⏳ **Pending** (can be enabled):
- Generate trajectories for all 13,348 abnormal OD pairs using student models
- Current workflow skipped generation (analysis-only mode)
- Would produce trajectories that can be evaluated for abnormality reproduction

### Step 4: Compare Teacher Detection vs Student Reproduction
⏳ **Pending**:
- **Teacher Metric**: Detection accuracy on abnormal OD pairs
- **Student Metric**: Abnormality reproduction rate on same OD pairs
- **Target**: Student should achieve 85-95% of teacher's detection capability

## Visualizations Available

The analysis includes comprehensive visualizations in:
`hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/figures/`

1. **abnormal_od_distribution.png** - Top origins and destinations in abnormal trajectories
2. **abnormal_categories_summary.png** - Pattern type distribution
3. **temporal_delay_analysis.png** - Temporal and spatial deviation characteristics
4. **abnormal_od_heatmap.png** - Most frequent abnormal OD pairs
5. **normal_od_heatmap.png** - Normal OD pairs for comparison
6. **od_heatmap_comparison.png** - Side-by-side abnormal vs normal comparison

These visualizations help understand:
- Which OD pairs are most indicative of abnormal patterns
- Spatial patterns that distinguish abnormal from normal behavior
- Characteristics of abnormal trajectories (temporal delays, route deviations)

## Metrics Alignment

### Teacher Evaluation Metrics
- **F1 Score**: Overall outlier detection performance
- **Precision**: How many detected outliers are truly abnormal
- **Recall**: How many true outliers are detected
- **PR-AUC**: Precision-recall area under curve

### Student Evaluation Metrics
- **Abnormality Reproduction Rate**: % of generated trajectories marked as abnormal
- **Pattern Match Rate**: How well generated trajectories match specific abnormal patterns
- **OD Pair Coverage**: How many abnormal OD pairs can student generate for

### Comparison Framework
- **Teacher Detection Rate**: F1 score on abnormal OD pairs
- **Student Reproduction Rate**: Abnormality rate in generated trajectories
- **Target Ratio**: Student should achieve 85-95% of teacher's detection capability

## Next Steps

1. **Enable Generation for Abnormal OD Pairs**:
   ```bash
   # Modify config to disable skip_generation
   # Run workflow to generate trajectories for abnormal OD pairs
   ```

2. **Evaluate Teacher on Abnormal OD Subset**:
   - Filter teacher evaluation to trajectories with abnormal OD pairs
   - Calculate teacher's detection accuracy on this subset
   - Compare to overall teacher performance

3. **Compare Teacher vs Student**:
   - Teacher: Detection accuracy on abnormal OD pairs
   - Student: Abnormality reproduction rate on same OD pairs
   - Calculate performance retention: Student Rate / Teacher Rate

4. **Analyze Failure Modes**:
   - Which abnormal OD pairs does teacher detect but student fails to reproduce?
   - Are there patterns in failures (specific OD pairs, pattern types)?
   - What characteristics distinguish successful vs failed cases?

## References

- **Teacher Baseline Comparison**: `docs/results/TEACHER_BASELINE_COMPARISON.md`
- **Abnormal OD Analysis Report**: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/COMPREHENSIVE_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md`
- **Abnormal OD Workflow Guide**: `docs/ABNORMAL_OD_WORKFLOW_GUIDE.md`
- **Abnormal OD Pairs File**: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal_od_pairs_porto_hoser.json`
