# Abnormal OD Analysis: Bridge Between Teacher and Student Evaluation

## Overview

This document explains how the abnormal OD analysis workflow provides a unified framework for evaluating and comparing LM-TAD teacher and HOSER student models, despite their architectural differences and primary tasks.

## Evaluation Tasks

### Teacher Model (LM-TAD)
- **Primary Task**: Outlier detection via perplexity scoring
- **Input**: Complete trajectory sequence
- **Output**: Binary classification (normal/abnormal)
- **Metrics**: F1 score, precision, recall, PR-AUC

### Student Model (HOSER)
- **Primary Task**: Trajectory generation
- **Input**: Origin-destination pair
- **Output**: Generated trajectory sequence
- **Metrics**: Abnormality reproduction rate

## Common Test Set: Abnormal OD Pairs

The bridge between teacher and student evaluation is built on a shared set of abnormal OD pairs extracted from real data:

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

### Extraction Process
```python
from tools.extract_abnormal_od_pairs import extract_and_save_abnormal_od_pairs

# Extract OD pairs from abnormal trajectories
od_pairs = extract_and_save_abnormal_od_pairs(
    detection_results_files=[Path("abnormal/porto/detection_results.json")],
    real_data_files=[Path("data/porto/train.csv")],
    dataset_name="porto_hoser",
    output_file=Path("abnormal_od_pairs.json")
)
```

## Dual Evaluation Framework

### 1. Teacher Model Evaluation
```python
from tools.evaluate_with_lmtad import evaluate_with_lmtad

# Evaluate real trajectories with these OD pairs
teacher_results = evaluate_with_lmtad(
    trajectory_file="real_trajectories.csv",
    vocab_file="vocab.json",
    lmtad_checkpoint="weights_only.pt",
    lmtad_repo_path="/home/matt/Dev/LMTAD",
    dataset="porto_hoser",
    output_dir="eval_lmtad/porto_hoser/real_data"
)
```

**Output Format**:
```json
{
  "model": "LM-TAD",
  "parameters": 85000000,
  "evaluation": {
    "total_trajectories": 61046,
    "detected_abnormal": 55614,
    "detection_rate": 91.10,
    "metrics": {
      "accuracy": 0.9934,
      "precision": 0.8366,
      "recall": 0.9999,
      "f1": 0.9110,
      "pr_auc": 0.9999
    },
    "perplexity": {
      "mean": 0.38,
      "std": 0.14,
      "abnormal_mean": 8.41,
      "abnormal_std": 3.85
    }
  }
}
```

### 2. Student Model Evaluation
```python
from tools.generate_abnormal_od import generate_abnormal_od_trajectories
from tools.evaluate_abnormal_od import evaluate_abnormal_od

# Generate trajectories for abnormal OD pairs
generate_abnormal_od_trajectories(
    od_pairs_file=od_pairs_file,
    model_dir=model_dir,
    output_dir=gene_dir,
    dataset="porto_hoser",
    num_traj_per_od=50
)

# Evaluate abnormality reproduction
student_results = evaluate_abnormal_od(
    generated_dir=gene_dir,
    real_abnormal_file=real_file,
    abnormal_od_pairs_file=od_pairs_file,
    output_dir=eval_dir
)
```

**Output Format**:
```json
{
  "model": "vanilla_seed43",
  "parameters": 6700000,
  "evaluation": {
    "total_trajectories": 50000,
    "abnormal_trajectories": 1770,
    "abnormality_rate": 0.0354,
    "vs_real_rate": -0.6607,
    "metrics": {
      "edr_distance": 0.342,
      "dtw_distance": 156.7,
      "hausdorff_distance": 89.3
    },
    "pattern_reproduction": {
      "temporal_delay": 0.0354,
      "detour": 0.0,
      "suspicious_stops": 0.0,
      "circuitous": 0.0,
      "unusual_duration": 0.0
    }
  }
}
```

## Performance Comparison

### 1. Classification vs Generation
```python
def calculate_performance_retention(
    teacher_f1: float,
    student_abnormal_rate: float,
    real_abnormal_rate: float
) -> float:
    """Calculate how much of teacher's capability student retains."""
    teacher_capability = teacher_f1  # e.g., 0.911 (91.1%)
    student_retention = student_abnormal_rate / real_abnormal_rate  # e.g., 0.034 (3.4%)
    return (student_retention / teacher_capability) * 100  # As percentage

# Example: Porto dataset
retention = calculate_performance_retention(
    teacher_f1=0.911,        # 91.1% F1
    student_abnormal_rate=0.0354,  # 3.54%
    real_abnormal_rate=0.0986      # 9.86%
)
# Result: 3.73% retention
```

### 2. Pattern Analysis
```python
# Compare pattern distributions
from tools.plot_lmtad_evaluation import plot_pattern_distribution

plot_pattern_distribution(
    teacher_results=teacher_results,
    student_results=student_results,
    output_dir=figures_dir,
    real_distribution=real_distribution
)
```

### 3. Performance Gap Analysis
```python
from tools.analyze_performance_gap import analyze_gap

gap_analysis = analyze_gap(
    teacher_results=teacher_results,
    student_results=student_results,
    real_baseline=real_baseline,
    output_file=analysis_dir / "performance_gap_analysis.json"
)
```

## Current Results (Porto Dataset)

### Teacher Performance
- F1 Score: 91.10%
- Precision: 83.66%
- Recall: 99.99%
- Clear perplexity separation between normal (0.38 ± 0.14) and abnormal (8.41 ± 3.85)

### Student Performance
- Best Model: vanilla_seed43 (3.54% abnormality rate)
- Mean Performance: ~2.0% abnormality rate
- Baseline: 9.86% real abnormality rate
- Pattern Coverage: 100% temporal delays, 0% other patterns

### Performance Gap
1. **Retention Rate**: 3.73% of teacher capability (best case)
2. **Pattern Coverage**: Limited to temporal delays only
3. **Training Data**: Complete failure (0% abnormality rate)

## Research Implications

### 1. Knowledge Transfer
- Need specialized training for abnormal patterns
- Consider curriculum learning approaches
- Balance normal vs abnormal examples

### 2. Architecture Alignment
- Grid tokenization vs road segment representation
- Local vs global context modeling
- Feature-level knowledge transfer strategies

### 3. Evaluation Protocol
- Common test set through abnormal OD pairs
- Complementary metrics (detection vs generation)
- Clear retention targets and baselines

## Action Items

### 1. Immediate Tasks
- [ ] Implement direct outlier detection in student
- [ ] Add pattern-specific loss functions
- [ ] Enhance cross-task evaluation metrics

### 2. Research Questions
- How to improve pattern reproduction?
- Why do students avoid abnormal patterns?
- What architectural changes could help?

### 3. Documentation Needs
- [ ] Document failure modes and patterns
- [ ] Create detailed pattern analysis guide
- [ ] Update training recommendations

## References

1. **Documentation**:
   - `docs/ABNORMAL_OD_WORKFLOW_GUIDE.md`
   - `docs/results/TEACHER_BASELINE_COMPARISON.md`

2. **Implementation**:
   - `tools/evaluate_with_lmtad.py`
   - `tools/evaluate_abnormal_od.py`
   - `tools/analyze_performance_gap.py`

3. **Results**:
   - `analysis_abnormal/porto_hoser/COMPREHENSIVE_ANALYSIS.md`
   - `eval_abnormal_teacher/porto_hoser/TEACHER_EVALUATION.md`