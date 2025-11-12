# Teacher Baseline Comparison

## Overview

This document presents the performance of the LM-TAD teacher model on trajectory anomaly detection tasks for both Beijing and Porto datasets. These teacher baselines are essential for understanding the compression-performance tradeoff achieved through knowledge distillation in HOSER.

## Related Documentation

- **[Cross-Seed Analysis](CROSS_SEED_ANALYSIS.md)** - Statistical analysis of cross-seed variance with confidence intervals
- **[Wang Abnormality Detection](WANG_ABNORMALITY_DETECTION_RESULTS.md)** - Wang baseline anomaly detection results
- **[Abnormal OD Teacher-Student Bridge](ABNORMAL_OD_TEACHER_STUDENT_BRIDGE.md)** - How abnormal OD analysis connects teacher and student evaluation

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

#### Outlier Detection Task (Teacher Evaluation)

| Model | Parameters | Accuracy | Precision | Recall | F1 Score | PR-AUC | Performance vs Teacher |
|-------|-----------|----------|-----------|--------|----------|--------|----------------------|
| **Teacher (LM-TAD)** | 85M | 99.34% | 83.66% | 99.99% | 91.10% | 99.99% | Baseline |

#### Abnormal Pattern Reproduction Task (Student Evaluation)

**Real Data Baseline**:
- **Real Abnormality Rate**: 9.86% (61,046 / 618,891 trajectories)
  - Test split: 10.43% (14,349 / 137,532)
  - Train split: 9.70% (46,697 / 481,359)
- **Unique Abnormal OD Pairs**: 13,348
- **Pattern Distribution**: 100% temporal delays (Abp2_temporal_delay), 0% route deviations

**Student Model Performance** (Abnormality Reproduction Rate on Test Split):

| Model | Parameters | Real Rate | Generated Rate | Absolute Deviation | Relative Deviation | Performance vs Real |
|-------|-----------|-----------|----------------|-------------------|-------------------|-------------------|
| **Teacher (LM-TAD)** | 85M | 10.43% | N/A | N/A | N/A | Detection: 91.10% F1 |
| **vanilla_seed43** | 6.7M | 10.43% | **3.54%** | 6.89% | -66.07% | **33.9%** of real rate |
| **distill_phase1** | 6.7M | 10.43% | **2.60%** | 7.83% | -75.08% | **24.9%** of real rate |
| **vanilla** | 6.7M | 10.43% | **2.32%** | 8.11% | -77.76% | **22.2%** of real rate |
| **distill_phase2** | 6.7M | 10.43% | **2.00%** | 8.43% | -80.83% | **19.2%** of real rate |
| **distill_phase1_seed44** | 6.7M | 10.43% | **1.84%** | 8.59% | -82.36% | **17.6%** of real rate |
| **vanilla_seed44** | 6.7M | 10.43% | **1.60%** | 8.83% | -84.66% | **15.3%** of real rate |
| **distill_phase1_seed43** | 6.7M | 10.43% | **1.50%** | 8.93% | -85.62% | **14.4%** of real rate |
| **distill_phase2_seed43** | 6.7M | 10.43% | **1.30%** | 9.13% | -87.54% | **12.5%** of real rate |
| **distill_phase2_seed44** | 6.7M | 10.43% | **0.92%** | 9.51% | -91.18% | **8.8%** of real rate |

**Key Findings**:
- **Best Student Model**: `vanilla_seed43` achieves 3.54% abnormality rate (33.9% of real 10.43%)
- **Mean Student Performance**: ~2.0% abnormality rate (19.2% of real rate)
- **Performance Gap**: Students reproduce only 15-34% of real abnormal patterns
- **Train Split Performance**: All models show 0% abnormality rate on training data (complete failure)
- **Statistical Significance**: All comparisons show p < 0.001 with medium to large effect sizes

> **Note**: Student models evaluated on abnormal pattern reproduction task (generation), not direct outlier detection. Teacher evaluated on outlier detection task (classification). See [Abnormal OD Teacher-Student Bridge](ABNORMAL_OD_TEACHER_STUDENT_BRIDGE.md) for integration strategy.

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
- ✅ Teacher baselines established (91.10% F1 Porto, 83.89% F1 Beijing)
- ✅ Distillation training completed
- ✅ Student abnormal pattern reproduction evaluated (Porto dataset)
- ⏳ Direct student outlier detection evaluation pending
- ⏳ Compression-performance analysis in progress

**Preliminary Findings (Porto Dataset)**:

**Abnormal Pattern Reproduction**:
- **Real Abnormality Rate**: 9.86% (baseline)
- **Best Student Model**: `vanilla_seed43` achieves 3.54% (33.9% of real)
- **Mean Student Performance**: ~2.0% (19.2% of real)
- **Performance Gap**: Students reproduce only 15-34% of real abnormal patterns

**Key Observations**:
1. **Significant Underperformance**: Students dramatically underperform on abnormal pattern reproduction (19-34% of real rate)
2. **Vanilla vs Distilled**: Mixed results - `vanilla_seed43` performs best, but `distill_phase1` ranks second
3. **Seed Sensitivity**: Performance varies significantly with random seed (0.92% to 3.54%)
4. **Train Split Failure**: All models show 0% abnormality rate on training data, indicating overfitting or training data bias

**Interpretation**:
- Student models excel at generating **normal** trajectories but struggle with **abnormal** patterns
- The 10× gap (9.86% real vs 0.98% mean generated) suggests models are trained to avoid abnormal patterns
- This aligns with teacher's high recall (99.99%) - teacher can detect abnormalities, but students cannot reproduce them
- **Research Direction**: Need specialized training strategies to improve abnormal pattern reproduction

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

### Connection to Abnormal OD Analysis

**Abnormal OD Workflow Integration**:

The abnormal OD analysis workflow provides a bridge between teacher and student evaluation by:

1. **Common Test Set**: Abnormal OD pairs extracted from real data can be used to evaluate both teacher and student models on the same challenging patterns
2. **Pattern Characterization**: The analysis visualizations show:
   - Distribution of abnormal patterns (temporal delays vs route deviations)
   - Most frequent abnormal OD pairs (heatmaps)
   - Normal vs abnormal pattern comparison
   - Temporal and spatial deviation characteristics

3. **Evaluation Strategy**:
   - **Teacher**: Evaluate on real abnormal trajectories (outlier detection task)
   - **Student**: Generate trajectories for abnormal OD pairs, then evaluate how well they reproduce abnormal patterns
   - **Comparison**: Compare teacher's detection accuracy vs student's pattern reproduction rate

**Current Abnormal OD Analysis Results** (Porto Dataset):

- **Real Abnormal Trajectories**: 61,046 (9.86% of total)
  - Test split: 10.43% abnormality rate (14,349 / 137,532)
  - Train split: 9.70% abnormality rate (46,697 / 481,359)
- **Unique Abnormal OD Pairs**: 13,348
- **Pattern Distribution**: 100% temporal delays (Abp2_temporal_delay), 0% route deviations
- **Student Model Performance**:
  - **Best Model**: `vanilla_seed43` achieves 3.54% abnormality rate (33.9% of real 10.43%)
  - **Mean Performance**: ~2.0% abnormality rate (19.2% of real rate)
  - **Worst Model**: `distill_phase2_seed44` achieves 0.92% abnormality rate (8.8% of real rate)
  - **Performance Gap**: Students reproduce only 15-34% of real abnormal patterns
  - **Train Split**: All models show 0% abnormality rate (complete failure on training data)

**Integration Plan**:

1. **Use Abnormal OD Pairs as Test Set**:
   - Extract abnormal OD pairs from Porto/Beijing datasets
   - Evaluate teacher model on these specific OD pairs (outlier detection)
   - Generate student trajectories for same OD pairs
   - Compare teacher detection rate vs student reproduction rate

2. **Cross-Task Evaluation**:
   - **Teacher Task**: Given a trajectory, detect if it's abnormal (classification)
   - **Student Task**: Given an abnormal OD pair, generate a trajectory that reproduces abnormal patterns (generation)
   - **Connection**: Student's ability to reproduce abnormal patterns indicates understanding of what makes trajectories abnormal

3. **Metrics Alignment**:
   - **Teacher**: F1 score on outlier detection (current: 83.89% Beijing, 91.10% Porto)
   - **Student**: Abnormality reproduction rate (current: 0.98% Porto, needs improvement)
   - **Target**: Student should achieve 85-95% of teacher's detection capability when evaluated on same abnormal patterns

**References**:
- **Abnormal OD Analysis Report**: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/COMPREHENSIVE_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md`
- **Abnormal OD Workflow Guide**: `docs/ABNORMAL_OD_WORKFLOW_GUIDE.md`
- **Analysis Visualizations**: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/figures/`

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

**Abnormal OD Analysis**:
- Porto Analysis Report: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/COMPREHENSIVE_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md`
- Porto Visualizations: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/figures/`
- Abnormal OD Pairs: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal_od_pairs_porto_hoser.json`

---

## Summary

### Outlier Detection Task (Teacher Evaluation)

| Dataset | Teacher F1 | Teacher Params | Compression Ratio | Student F1 | Performance Retention |
|---------|------------|----------------|-------------------|------------|---------------------|
| Beijing | 83.89% | 85M | 12.7× | TBD | TBD |
| Porto | 91.10% | 85M | 12.7× | TBD | TBD |

### Abnormal Pattern Reproduction Task (Student Evaluation - Porto)

| Metric | Real Baseline | Best Student | Mean Student | Performance vs Real |
|--------|---------------|--------------|--------------|-------------------|
| **Abnormality Rate** | 9.86% (10.43% test) | 3.54% (vanilla_seed43) | ~2.0% | 19-34% of real |
| **Unique OD Pairs** | 13,348 | N/A | N/A | N/A |
| **Pattern Types** | 100% temporal delays | 100% temporal delays | 100% temporal delays | Pattern match |

**Key Takeaways**:

1. **Teacher Baselines Established**: Excellent outlier detection performance (91.10% F1 Porto, 83.89% F1 Beijing) with 12.7× parameter reduction potential.

2. **Student Abnormal Pattern Reproduction**: Preliminary results show significant underperformance:
   - Best student (`vanilla_seed43`) achieves only 33.9% of real abnormality rate
   - Mean student performance: 19.2% of real rate
   - 10× gap between real (9.86%) and generated (0.98% mean) abnormality rates

3. **Compression-Performance Tradeoff**:
   - **Outlier Detection**: Pending direct student evaluation on same task as teacher
   - **Pattern Reproduction**: Students struggle to reproduce abnormal patterns (19-34% of real)
   - **Research Need**: Specialized training strategies required to improve abnormal pattern handling

4. **Future Work**:
   - Evaluate students on direct outlier detection task (same as teacher)
   - Quantify compression-performance tradeoff: Student F1 / Teacher F1 ratio
   - Develop training strategies to improve abnormal pattern reproduction
   - Analyze failure modes: Why do students avoid abnormal patterns?

**Status**: Teacher baselines established. Student evaluation on abnormal patterns reveals significant challenges in reproducing edge cases, highlighting the need for specialized training approaches to achieve the target 85-95% performance retention.
