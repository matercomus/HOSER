# Teacher Baseline Comparison

## Overview

This document presents the evaluation methodology and results of using the LM-TAD teacher model as a baseline for anomaly detection performance. The analysis enables understanding of the compression-performance tradeoff achieved through knowledge distillation in HOSER.

## Evaluation Methodology

### Teacher Model Architecture

**LM-TAD (Language Model for Trajectory Anomaly Detection)**:
- **Architecture**: GPT-based transformer (8 layers × 12 heads × 768d)
- **Parameters**: 85M
- **Tokenization**: Grid-based spatial encoding
- **Training**: Next-token prediction on normal trajectories
- **Evaluation**: Outlier detection via perplexity scores

### Grid Tokenization

1. **Grid Configuration**:
   - **Beijing**: 0.002° grid size (25×25 = 625 tokens)
   - **Porto**: 0.001° grid size (46×134 = 6,164 tokens)
   - **Special Tokens**: PAD (0), EOT (1), SOT (2)

2. **Conversion Process**:
   ```python
   # Convert road IDs to grid tokens
   from tools.convert_to_lmtad_format import convert_hoser_to_lmtad_format

   convert_hoser_to_lmtad_format(
       trajectory_file="trajectories.csv",  # HOSER format
       roadmap_file="roadmap.geo",         # For centroid mapping
       output_file="lmtad_format.csv",     # Grid token sequences
       vocab_file="vocab.json",            # Token vocabulary
       dataset="beijing_hoser_reference"    # For grid config
   )
   ```

3. **Consistency Requirements**:
   - Grid dimensions must match teacher's training configuration
   - Vocabulary mappings must be identical
   - Special token handling must be preserved

### Perplexity-based Detection

1. **Compute Perplexity**:
   ```python
   from tools.evaluate_with_lmtad import evaluate_with_lmtad

   results = evaluate_with_lmtad(
       trajectory_file="lmtad_format.csv",
       vocab_file="vocab.json",
       lmtad_checkpoint="weights_only.pt",
       lmtad_repo_path="/home/matt/Dev/LMTAD",
       dataset="beijing_hoser_reference",
       output_dir="evaluation_results"
   )
   ```

2. **Threshold Selection**:
   - Auto-computed as μ + 2σ of real data perplexities
   - Can be overridden with explicit threshold
   - Optimized for balanced precision/recall

3. **Memory Efficiency**:
   - Uses automatic mixed precision (AMP)
   - Configurable batch size (default: 128)
   - Supports CPU fallback if needed

### Dataset Structure

1. **Real Data Baseline**:
   - Pre-converted LM-TAD format trajectories
   - Used to establish normal perplexity distribution
   - Provides outlier threshold calibration

2. **Generated Trajectories**:
   - Converted from HOSER road ID sequences
   - Evaluated per model to assess performance
   - Compared against real baseline metrics

## Evaluation Results

### Beijing Dataset

**Teacher Model Performance (85M parameters)**:

| Metric | Value | vs Real Data |
|--------|-------|--------------|
| F1 Score | 83.89% | Baseline |
| Precision | 74.59% | - |
| Recall | 95.83% | - |
| PR-AUC | 96.54% | - |
| Mean Perplexity | 0.48 ± 0.16 | Normal trajectories |
| Outlier Perplexity | 8.04 ± 3.50 | Abnormal patterns |

**Generated Trajectory Metrics**:
- Results pending student evaluation on same test set

### Porto Dataset

**Teacher Model Performance (85M parameters)**:

| Metric | Value | vs Real Data |
|--------|-------|--------------|
| F1 Score | 91.10% | Baseline |
| Precision | 83.66% | - |
| Recall | 99.99% | - |
| PR-AUC | 99.99% | - |
| Mean Perplexity | 0.38 ± 0.14 | Normal trajectories |
| Outlier Perplexity | 8.41 ± 3.85 | Abnormal patterns |

**Generated Trajectory Metrics**:

| Model | Parameters | Abnormality Rate | vs Real (10.43%) |
|-------|------------|------------------|------------------|
| Real Data (Test) | - | 10.43% | Baseline |
| vanilla_seed43 | 6.7M | 3.54% | -66.07% |
| distill_phase1 | 6.7M | 2.60% | -75.08% |
| vanilla | 6.7M | 2.32% | -77.76% |
| distill_phase2 | 6.7M | 2.00% | -80.83% |
| distill_phase1_seed44 | 6.7M | 1.84% | -82.36% |
| vanilla_seed44 | 6.7M | 1.60% | -84.66% |
| distill_phase1_seed43 | 6.7M | 1.50% | -85.62% |
| distill_phase2_seed43 | 6.7M | 1.30% | -87.54% |
| distill_phase2_seed44 | 6.7M | 0.92% | -91.18% |

## Compression-Performance Analysis

### Model Size Comparison

| Model | Parameters | vs Teacher | Architecture |
|-------|------------|------------|--------------|
| Teacher (LM-TAD) | 85M | 1.0× | 8 layers × 12 heads × 768d |
| Student (HOSER) | 6.7M | 0.079× (12.7× smaller) | 6 layers × 8 heads × 384d |

### Performance Retention

**Target Metrics**:
- **Acceptable Range**: 85-95% of teacher F1 score
- **Current Best**: 33.9% of real abnormality rate
- **Mean Performance**: 19.2% of real abnormality rate

**Key Observations**:
1. **Clear Distribution Separation**:
   - Teacher: 16-21× perplexity gap (normal vs abnormal)
   - Enables reliable anomaly detection

2. **Dataset Characteristics**:
   - Porto: Higher precision (83.66% vs 74.59%)
   - Porto: Longer trajectories (44.9 vs 28.6 tokens)
   - Porto: Better contextual signal for detection

3. **Student Performance Gap**:
   - Significant underperformance on abnormal patterns
   - Complete failure on training data (0% rate)
   - Need for improved knowledge transfer

## Analysis Process

### 1. Baseline Establishment

```python
# Get real data baseline metrics
real_results = workflow._evaluate_lmtad_real_baseline(output_dir)
logger.info(f"Real baseline perplexity: {real_results['mean_perplexity']:.4f}")
```

### 2. Generated Trajectory Evaluation

```python
# Evaluate each model's generated trajectories
generated_results = workflow._evaluate_lmtad_generated(output_dir)
for model, results in generated_results.items():
    logger.info(f"{model}: {results['outlier_rate']:.2%} outlier rate")
```

### 3. Performance Comparison

```python
# Compare with teacher baseline
workflow._compare_lmtad_results(
    real_results,
    generated_results,
    output_dir
)
```

### 4. Visualization

```bash
# Generate comparison plots
uv run python tools/plot_lmtad_evaluation.py \\
  --eval-dir eval_lmtad/porto_hoser \\
  --output-dir figures/lmtad \\
  --include-generated
```

## Future Work

1. **Direct Detection Task**:
   - Evaluate students on outlier detection
   - Compare F1 scores with teacher
   - Analyze failure modes

2. **Knowledge Transfer**:
   - Improve abnormal pattern reproduction
   - Develop specialized training strategies
   - Balance normal vs abnormal examples

3. **Comprehensive Analysis**:
   - Per-category performance breakdown
   - Trajectory-level error analysis
   - Feature importance investigation

## References

1. **Implementation**:
   - `tools/evaluate_with_lmtad.py`: Core evaluation logic
   - `tools/convert_to_lmtad_format.py`: Grid tokenization
   - `critics/lmtad_teacher.py`: Model wrapper

2. **Results**:
   - `/home/matt/Dev/LMTAD/code/results/LMTAD/*/EVALUATION_ANALYSIS.md`
   - `hoser-distill-optuna-*/analysis_abnormal/*/workflow_summary.json`

3. **Documentation**:
   - `docs/ABNORMAL_OD_WORKFLOW_GUIDE.md`
   - `docs/results/ABNORMAL_OD_TEACHER_STUDENT_BRIDGE.md`