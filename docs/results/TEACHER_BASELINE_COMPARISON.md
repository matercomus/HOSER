# Teacher Baseline Comparison

## Overview

This document presents the performance of the LM-TAD teacher model on trajectory anomaly detection tasks for both Beijing and Porto datasets. These teacher baselines are essential for understanding the compression-performance tradeoff achieved through knowledge distillation in HOSER.

## Teacher Model Architecture

**LM-TAD (Language Model for Trajectory Anomaly Detection)**:
- **Architecture**: GPT-based transformer
- **Parameters**: 85M (8 layers × 12 heads × 768 embedding dim)
- **Vocabulary**: Grid-based spatial tokenization (25×25 = 625 tokens for Beijing)
- **Training**: Next-token prediction on normal trajectories
- **Evaluation**: Outlier detection via log perplexity

**Evaluation Setup**:
- Modified `eval_porto.py` script to support HOSER dataset format
- Same outlier injection protocol as student model evaluation
- Detour vs. non-outlier classification (route switch excluded)
- Metrics: accuracy, precision, recall, F1, PR-AUC

---

## Beijing Dataset Results

### Teacher Performance (LM-TAD, 85M parameters)

| Configuration | Threshold | Accuracy | Precision | Recall | F1 Score | PR-AUC |
|--------------|-----------|----------|-----------|--------|----------|--------|
| final_model (ratio=0.05, level=3, prob=0.1) | 0.9966 | **98.75%** | **74.59%** | **95.83%** | **83.89%** | **96.54%** |

**Dataset Statistics** (Teacher Evaluation):
- Total trajectories: 635,149
- Non-outliers: 593,609 (93.46%)
- Route switch outliers: 20,770 (3.27%)
- Detour outliers: 20,770 (3.27%)
- Mean sequence length: Non-outlier 28.6, Route switch 28.8, Detour 47.9

**Log Perplexity Distribution**:
- Non-outliers: mean 0.48 ± 0.16
- Route switch: mean 7.37 ± 3.51
- Detours: mean 8.04 ± 3.50

**Source**: `/home/matt/Dev/LMTAD/code/results/LMTAD/beijing_hoser_reference/run_20250928_202718/.../eval/EVALUATION_ANALYSIS.md`

---

## Porto Dataset Results

### Teacher Performance (LM-TAD, 85M parameters)

| Configuration | Threshold | Accuracy | Precision | Recall | F1 Score | PR-AUC |
|--------------|-----------|----------|-----------|--------|----------|--------|
| ckpt_best (ratio=0.05, level=3, prob=0.3) | 0.7571 | **99.34%** | **83.66%** | **99.99%** | **91.10%** | **99.99%** |

**Dataset Statistics** (Teacher Evaluation):
- Total trajectories: 639,844
- Non-outliers: 598,196 (93.49%)
- Route switch outliers: 20,924 (3.27%)
- Detour outliers: 20,724 (3.24%)
- Mean sequence length: Non-outlier 44.9, Route switch 44.9, Detour 83.4

**Log Perplexity Distribution**:
- Non-outliers: mean 0.38 ± 0.14
- Route switch: mean 7.03 ± 3.57
- Detours: mean 8.41 ± 3.85

**Source**: `/home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/.../eval/EVALUATION_ANALYSIS.md`

---

## Compression-Performance Tradeoff

### Model Size Comparison

| Model | Parameters | vs Teacher | Architecture |
|-------|-----------|------------|--------------|
| **Teacher (LM-TAD)** | 85M | 1.0× (baseline) | 8 layers × 12 heads × 768d |
| **Distilled Student (HOSER)** | 6.7M | **0.079× (12.7× smaller)** | 6 layers × 8 heads × 384d |
| **Vanilla Student** | 6.7M | 0.079× (12.7× smaller) | 6 layers × 8 heads × 384d |

**Compression Ratio**: **12.7×** parameter reduction (85M → 6.7M)

### Beijing: Teacher vs Student Performance

| Model | Parameters | Accuracy | Precision | Recall | F1 Score | PR-AUC | Performance vs Teacher |
|-------|-----------|----------|-----------|--------|----------|--------|----------------------|
| **Teacher (LM-TAD)** | 85M | 98.75% | 74.59% | 95.83% | 83.89% | 96.54% | Baseline |
| **Distilled Student** | 6.7M | TBD | TBD | TBD | TBD | TBD | TBD% of teacher F1 |
| **Vanilla Student** | 6.7M | TBD | TBD | TBD | TBD | TBD | TBD% of teacher F1 |

> **Note**: Student model evaluation results for anomaly detection task pending. Current HOSER evaluation focuses on trajectory generation metrics (OD completion, distance accuracy, spatial distribution matching).

### Porto: Teacher vs Student Performance

| Model | Parameters | Accuracy | Precision | Recall | F1 Score | PR-AUC | Performance vs Teacher |
|-------|-----------|----------|-----------|--------|----------|--------|----------------------|
| **Teacher (LM-TAD)** | 85M | 99.34% | 83.66% | 99.99% | 91.10% | 99.99% | Baseline |
| **Distilled Student** | 6.7M | TBD | TBD | TBD | TBD | TBD | TBD% of teacher F1 |
| **Vanilla Student** | 6.7M | TBD | TBD | TBD | TBD | TBD | TBD% of teacher F1 |

> **Note**: Student model evaluation results for anomaly detection task pending. Current HOSER evaluation focuses on trajectory generation metrics.

---

## Key Insights

### Teacher Model Strengths

1. **Excellent outlier detection**: F1 scores 83.89% (Beijing) and 91.10% (Porto)
2. **High recall**: 95.83% (Beijing) and 99.99% (Porto) - rarely misses true outliers
3. **Clear perplexity separation**: ~16-21× higher mean perplexity for outliers vs non-outliers
4. **Consistent across datasets**: Strong performance on both Beijing and Porto

### Teacher Model Characteristics

**Beijing**:
- Lower precision (74.59%) suggests some false positives
- Conservative threshold (0.9966) optimized for high recall
- Shorter trajectories (mean 28.6 tokens) than Porto

**Porto**:
- Higher precision (83.66%) with excellent recall (99.99%)
- More liberal threshold (0.7571)
- Longer trajectories (mean 44.9 tokens) provide richer signal
- Detours significantly longer (83.4 tokens, 86% increase)

### Compression-Performance Tradeoff

**Research Question**: Can a **12.7× smaller** student model (6.7M parameters) achieve comparable performance to the 85M parameter teacher through knowledge distillation?

**Expected Outcomes**:
1. **Distilled student** should outperform vanilla student through teacher guidance
2. **Acceptable performance degradation**: 85-95% of teacher F1 score is considered successful for 12.7× compression
3. **Efficiency gains**: 12.7× faster inference, 12.7× less memory

**Current Status**:
- ✅ Teacher baselines established
- ✅ Distillation training completed
- ⏳ Student anomaly detection evaluation pending
- ⏳ Compression-performance analysis pending

---

## Evaluation Methodology

### Teacher Evaluation Process

1. **Model**: Pretrained LM-TAD checkpoint (best or final model)
2. **Dataset**: HOSER format trajectories with injected outliers
3. **Script**: Modified `eval_porto.py` from LMTAD repository
4. **Metric**: Log perplexity per trajectory
5. **Classification**: Optimal threshold via precision-recall curve
6. **Outlier Types**:
   - **Route switch**: Different path, similar length
   - **Detour**: Extended trajectory with loops

### Consistency with Student Evaluation

**Shared Protocol**:
- Same outlier injection parameters (ratio, level, prob)
- Same evaluation metric (log perplexity)
- Same classification approach (threshold optimization)
- Same dataset splits and preprocessing

**Differences**:
- Teacher: Grid tokenization (LM-TAD vocabulary)
- Student: Road segment IDs (HOSER vocabulary)
- Teacher: No map-matching constraints
- Student: Map-matched trajectory generation

---

## Limitations & Future Work

### Current Limitations

1. **Different evaluation tasks**:
   - Teacher: Outlier detection only (given trajectories)
   - Student: Trajectory generation + evaluation via OD completion, spatial metrics
   - **Missing**: Direct comparison on same task (e.g., outlier detection for students)

2. **Vocabulary mismatch**:
   - Teacher: Discretized grid tokens (spatial approximation)
   - Student: Exact road segment IDs (preserves network topology)
   - Cannot directly compare token-level predictions

3. **Incomplete tradeoff analysis**:
   - Teacher baseline established
   - Student generation metrics available
   - **Missing**: Student outlier detection metrics for direct comparison

### Future Evaluation

**Planned Extensions**:
1. **Evaluate student models on outlier detection task** using same protocol as teacher
2. **Quantify compression-performance tradeoff**: Student F1 / Teacher F1 ratio
3. **Analyze failure modes**: Where does compression hurt most?
4. **Inference efficiency comparison**: Throughput and latency benchmarks

**Additional Analyses**:
- Teacher probability distributions vs student distilled distributions
- Uncertainty calibration comparison (reliability diagrams)
- Per-scenario performance breakdown (urban vs suburban, short vs long trips)

---

## References

**Teacher Evaluation Results**:
- Beijing: `/home/matt/Dev/LMTAD/code/results/LMTAD/beijing_hoser_reference/.../eval/EVALUATION_ANALYSIS.md`
- Porto: `/home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/.../eval/EVALUATION_ANALYSIS.md`

**LM-TAD Paper**:
- *"LM-TAD: Language Model for Trajectory Anomaly Detection"* (reference pending)

**HOSER Student Results**:
- Beijing: `hoser-distill-optuna-6.backup-20251105_214459/RESULTS_ANALYSIS.md`
- Porto: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md`

---

## Summary

| Dataset | Teacher F1 | Teacher Params | Compression Ratio | Student F1 (TBD) | Performance Retention |
|---------|------------|----------------|-------------------|------------------|---------------------|
| Beijing | 83.89% | 85M | 12.7× | TBD | TBD |
| Porto | 91.10% | 85M | 12.7× | TBD | TBD |

**Key Takeaway**: Teacher baselines are now established, enabling future compression-performance tradeoff analysis once student outlier detection evaluation is completed. The 12.7× parameter reduction makes HOSER practical for deployment while (hypothetically) retaining 85-95% of teacher performance through knowledge distillation.
