# Abnormal OD Analysis: Bridge Between Teacher and Student Evaluation

## Overview

The abnormal OD analysis workflow provides a **common test set** and **evaluation framework** that connects teacher (LM-TAD) and student (HOSER) model evaluation, enabling direct comparison despite different tasks and architectures.

## Key Components

### Teacher Model (LM-TAD)

**Model Architecture:**
- **Type**: Language Model for Trajectory Anomaly Detection
- **Parameters**: 85M (8 layers × 12 heads × 768d)
- **Input**: Grid-tokenized trajectory sequence
- **Output**: Log perplexity score for abnormality detection
- **Training**: Next-token prediction on normal trajectories

**Evaluation Task:**
- **Primary**: Outlier detection (classification)
- **Input**: Complete trajectory
- **Output**: Binary classification (normal vs abnormal)
- **Metric**: F1 score on outlier detection
- **Current Performance**:
  - Beijing: 83.89% F1 (74.59% precision, 95.83% recall)
  - Porto: 91.10% F1 (83.66% precision, 99.99% recall)

### Student Model (HOSER)

**Model Architecture:**
- **Type**: Hierarchical Origin-destination Spatio-temporal Encoder-decoder
- **Parameters**: 6.7M (6 layers × 8 heads × 384d)
- **Input**: Road network context + OD pair
- **Output**: Road segment sequence
- **Training**: OD completion with distilled teacher guidance

**Evaluation Task:**
- **Primary**: Trajectory generation
- **Input**: Origin-destination (OD) pair
- **Output**: Generated trajectory
- **Metric**: Abnormality reproduction rate
- **Current Performance**:
  - Porto: 0.98% mean abnormality rate (vs 9.86% real)
  - Best Model: 3.54% (vanilla_seed43)
  - Worst Model: 0.92% (distill_phase2_seed44)

## Dual Evaluation Framework

### Core Concept

The dual evaluation framework enables meaningful comparison between teacher and student models despite their different architectures and primary tasks.

**Key Insight**: Use abnormal OD pairs as the bridge between:
1. **Teacher's Detection Ability**: Can it recognize abnormal patterns?
2. **Student's Generation Ability**: Can it reproduce similar patterns?

### Evaluation Process

#### 1. Extract Abnormal OD Pairs
```python
from pathlib import Path
from tools.extract_abnormal_od_pairs import extract_and_save_abnormal_od_pairs

od_pairs = extract_and_save_abnormal_od_pairs(
    detection_results_files=[
        Path("abnormal/porto_hoser/train/real_data/detection_results.json")
    ],
    real_data_files=[Path("data/porto_hoser/train.csv")],
    dataset_name="porto_hoser",
    output_file=Path("abnormal_od_pairs_porto.json")
)
```

Output structure:
```json
{
  "dataset": "porto_hoser",
  "total_abnormal_trajectories": 61046,
  "total_unique_od_pairs": 13348,
  "od_pairs_by_category": {
    "temporal_delay": [[o1,d1], [o2,d2], ...],
    "detour": [[o3,d3], ...],
    "suspicious_stops": [...],
    "circuitous": [...],
    "unusual_duration": [...]
  }
}
```

#### 2. Teacher Evaluation
```python
from tools.evaluate_lmtad_abnormal_od import evaluate_lmtad_abnormal_od

teacher_results = evaluate_lmtad_abnormal_od(
    od_pairs_file=Path("abnormal_od_pairs_porto.json"),
    lmtad_repo=Path("/home/matt/Dev/LMTAD"),
    lmtad_checkpoint=Path("/path/to/weights_only.pt"),
    real_data_file=Path("data/porto_hoser/train.csv"),
    output_dir=Path("eval_abnormal_teacher/porto_hoser"),
    dataset="porto_hoser",
    max_pairs_per_category=20,
    grid_size=0.001
)
```

#### 3. Student Evaluation
```python
from tools.evaluate_abnormal_od import evaluate_abnormal_od

student_results = evaluate_abnormal_od(
    generated_dir=Path("gene_abnormal/porto_hoser/seed42"),
    real_abnormal_file=Path("data/porto_hoser/train.csv"),
    abnormal_od_pairs_file=Path("abnormal_od_pairs_porto.json"),
    output_dir=Path("eval_abnormal/porto_hoser"),
    dataset="porto_hoser"
)
```

#### 4. Combined Analysis
```python
from tools.compare_teacher_student import compare_abnormal_od_performance

comparison = compare_abnormal_od_performance(
    teacher_results=teacher_results,
    student_results_dir=Path("eval_abnormal/porto_hoser"),
    output_dir=Path("analysis_abnormal/porto_hoser")
)
```

### Complementary Metrics

#### Teacher-Side Metrics
1. **Detection Accuracy (F1)**: How well teacher identifies abnormal patterns
2. **Precision**: Confidence in abnormality detection
3. **Recall**: Coverage of abnormal pattern types
4. **PR-AUC**: Overall detection capability

#### Student-Side Metrics
1. **Abnormality Rate**: % of generated trajectories marked abnormal
2. **Pattern Match Rate**: Distribution match with real abnormal patterns
3. **Similarity Metrics**: EDR/DTW/Hausdorff vs real abnormal trajectories
4. **Category Coverage**: Which abnormal patterns are reproduced

### Performance Analysis

#### Current Porto Results

1. **Teacher Performance (LM-TAD)**:
   - F1 Score: 91.10%
   - Precision: 83.66%
   - Recall: 99.99%
   - Clear separation: Normal (0.38 ± 0.14) vs Abnormal (8.41 ± 3.85)

2. **Student Performance (HOSER)**:
   - Real Abnormality Rate: 9.86%
   - Best Model: 3.54% (vanilla_seed43)
   - Mean Performance: ~2.0%
   - Worst Model: 0.92% (distill_phase2_seed44)

3. **Gap Analysis**:
   - Students achieve 15-34% of real abnormality rate
   - Complete failure on training data (0% rate)
   - Significant underperformance vs teacher detection

### Interpretation Framework

#### 1. Detection vs Generation
- **Teacher**: High detection performance = good understanding
- **Student**: Low generation performance = can't reproduce

#### 2. Pattern Types
- **Teacher**: Clear separation between normal/abnormal
- **Student**: Struggles with all abnormality types equally

#### 3. Performance Retention
```python
def calculate_performance_retention(teacher_f1, student_abnormal_rate, real_rate):
    """Calculate how much of teacher's capability student retains"""
    teacher_capability = teacher_f1  # e.g., 0.911 (91.1%)
    student_retention = student_abnormal_rate / real_rate  # e.g., 0.034 (3.4%)
    return (student_retention / teacher_capability) * 100  # As percentage
```

Example (Porto):
```python
retention = calculate_performance_retention(
    teacher_f1=0.911,        # 91.1% F1
    student_abnormal_rate=0.0354,  # 3.54%
    real_rate=0.0986         # 9.86%
)
# Result: 3.73% retention of teacher's capability
```

### Integration Points

#### 1. Training Phase
- Use teacher's detection scores to guide student training
- Focus on areas where teacher has high confidence
- Gradually introduce more complex patterns

#### 2. Evaluation Phase
- Apply both detection and generation metrics
- Compare pattern distributions directly
- Track performance retention across training

#### 3. Analysis Phase
- Identify common failure modes
- Map teacher detection to student generation
- Guide curriculum learning strategies

## Unified Results View

### Performance Summary Table

| Metric | Teacher (LM-TAD) | Student (HOSER) | vs Real | Performance Retention |
|--------|------------------|-----------------|---------|---------------------|
| **Detection F1** | 91.10% | N/A | N/A | N/A |
| **Abnormality Rate** | N/A | 0.98-3.54% | 9.86% | 3.73% (best) |
| **Pattern Coverage** | 100% | ~20% | 100% | 20% |

### Key Findings

1. **Capability Gap**:
   - Teacher excels at detection (91.10% F1)
   - Students struggle with reproduction (3.54% best case)
   - Clear need for improved knowledge transfer

2. **Pattern Understanding**:
   - Teacher: Clear normal/abnormal separation
   - Student: Heavy bias toward normal patterns
   - Missing: Effective abnormal pattern learning

3. **Performance Retention**:
   - Best case: 3.73% of teacher capability
   - Mean case: 2.19% of teacher capability
   - Target: 85-95% retention through improved training

### Research Implications

1. **Training Strategy**:
   - Need specialized abnormal pattern training
   - Consider curriculum learning approach
   - Balance normal vs abnormal examples

2. **Architecture Considerations**:
   - Teacher-student architectural alignment
   - Feature-level knowledge transfer
   - Pattern-specific loss functions

3. **Evaluation Protocol**:
   - Standardized abnormal pattern sets
   - Direct comparison metrics
   - Clear retention targets

## References

1. **LMTAD Documentation**:
   - `docs/LMTAD-Distillation.md`
   - `docs/results/TEACHER_BASELINE_COMPARISON.md`

2. **HOSER Documentation**:
   - `docs/ABNORMAL_OD_WORKFLOW_GUIDE.md`
   - `docs/EVALUATION_PIPELINE_GUIDE.md`

3. **Analysis Reports**:
   - `analysis_abnormal/porto_hoser/COMPREHENSIVE_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md`
   - `eval_abnormal_teacher/porto_hoser/TEACHER_EVALUATION_ANALYSIS.md`

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
