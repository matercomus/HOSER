# LM-TAD Spatial Abnormality Evaluation - Execution Guide

This document provides instructions for running the LM-TAD spatial abnormality evaluation pipeline, which evaluates how well HOSER models reproduce spatial route patterns using **perplexity-based analysis** instead of traditional classification.

## Overview

The LM-TAD spatial abnormality evaluation complements the Wang temporal abnormality detection by focusing on **spatial route quality** using perplexity metrics from the LM-TAD teacher model. This evaluation:

1. **Extracts** spatial abnormal OD pairs from LM-TAD source evaluation
2. **Generates** trajectories for these OD pairs using HOSER models
3. **Evaluates** generated trajectories with LM-TAD to compute perplexity scores
4. **Analyzes** model performance using perplexity statistics and cross-model OD comparisons
5. **Aggregates** results with statistical tests (KS test, paired t-test, effect sizes)
6. **Visualizes** perplexity distributions, segment-level analysis, and model comparisons

### Key Evaluation Paradigm Shift

**Previous Approach**: Classified trajectories into route switch/detour categories based on source labels or perplexity thresholds.

**New Approach**: Uses **perplexity as a continuous quality metric** to measure how well HOSER models match LM-TAD's expectations for each trajectory segment. Lower perplexity indicates better alignment with the teacher's understanding of valid routes.

## Prerequisites

✅ **LM-TAD Source Evaluation**: Must have completed LM-TAD evaluation on source dataset  
✅ **LM-TAD Checkpoint**: Trained teacher model checkpoint file  
✅ **HOSER Models**: Trained HOSER models in evaluation directory  
✅ **Integration**: Complete in `python_pipeline.py` (phase: `lmtad_spatial_abnormality`)

## Key Concepts

### Perplexity as a Quality Metric

**Log Perplexity** measures how "surprised" the LM-TAD teacher model is by a trajectory. It's computed for each trajectory segment and provides a continuous measure of route quality:

1. **Lower Perplexity = Better Route Alignment**
   - Values near 0-1: Very natural routes, high confidence
   - Values near 1-5: Reasonable routes with minor deviations
   - Values > 7: Unusual routes, possibly abnormal

2. **Segment-Level Analysis**
   - Perplexity is computed for each trajectory segment (road segment)
   - Allows identification of which parts of a route are problematic
   - Enables fine-grained comparison between models

3. **Per-Segment Statistics**
   - Mean perplexity across all trajectories at each position
   - Standard deviation to show variability
   - Distribution analysis (min, max, median) per segment

### Source Labels (Optional Metadata)

If an OD pairs file includes labels, treat them as contextual metadata rather than discrete target classes. Labels can help with stratified sampling and post-hoc analysis, but they do not determine or change how trajectories are evaluated.

- Example metadata values you may see in source OD files:
  - `route_switch`: indicates the source trajectory deviated from the common route for that OD
  - `detour`: indicates a substantially longer route than typical
  - `non_outlier` / `null`: no label or typical route

**Important**: The evaluation does not perform classification of trajectories. Trajectory quality is assessed exclusively via LM‑TAD perplexity scores; labels are optional context used for sampling and analysis only.

### Perplexity-Based Evaluation Method

Instead of classifying trajectories into discrete categories, the evaluation uses perplexity as a continuous metric:

1. **Direct Perplexity Scoring**: Each trajectory receives a log perplexity score from LM-TAD
2. **Statistical Analysis**: Compare perplexity distributions between models
3. **Cross-Model Comparison**: Evaluate multiple models on the same OD pairs
4. **Segment-Level Insights**: Identify which trajectory segments cause high perplexity

## Step 1: Extract Spatial Abnormal OD Pairs

Extract origin-destination pairs from LM-TAD-identified spatial outliers in the source evaluation.

**Prerequisites:**
- LM-TAD source evaluation TSV file (e.g., `ckpt_best_outliers_config_ratio_0.05_level_3_prob_0.3.tsv`)
- Located in: `/home/matt/Dev/LMTAD/code/results/LMTAD/{dataset}/run_*/.../eval/`

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/extract_lmtad_spatial_abnormal_od.py \
  --tsv-file /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval/ckpt_best_outliers_config_ratio_0.05_level_3_prob_0.3.tsv \
  --dataset porto_hoser \
  --source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval \
  --output hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal_od_pairs_lmtad_spatial_porto_hoser.json
```

**Output:**
- JSON file with OD pairs categorized by type (route_switch, detour)
- Metadata including counts and source evaluation directory

**Expected Output:**
```json
{
  "dataset": "porto_hoser",
  "source": "lmtad",
  "total_spatial_abnormal_trajectories": 41851,
  "total_unique_od_pairs": <count>,
  "od_pairs_by_type": {
    "route_switch": [[o1, d1], [o2, d2], ...],
    "detour": [[o3, d3], ...]
  }
}
```

## Step 2: Generate Trajectories (Optional)

Generate trajectories for spatial abnormal OD pairs using HOSER models.

**Note:** This step can be skipped if trajectories are already generated or if you want to reuse existing generation.

### Sampling Strategy

To maintain statistical rigor while keeping trajectory counts manageable (matching other evaluation phases at ~5,000 trajectories per model), the pipeline uses **stratified sampling**:

- **Default: 250 OD pairs** sampled (maintaining route_switch/detour ratio)
- **Default: 20 trajectories per OD pair**
- **Total: ~5,000 trajectories per model** (matches other evaluation phases)

The stratified sampling ensures:
- **Proportional representation**: Route switch and detour OD pairs are sampled proportionally to maintain the original distribution
- **Statistical rigor**: 20 trajectories per OD pair is sufficient for computing mean, std, and basic statistics
- **Consistency**: Matches the trajectory count used in other evaluation phases

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/generate_lmtad_spatial_abnormal_trajectories.py \
  --od-pairs-file hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal_od_pairs_lmtad_spatial_porto_hoser.json \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --seed 42 \
  --num-trajectories-per-od 20 \
  --max-od-pairs 250
```

**Options:**
- `--num-trajectories-per-od`: Number of trajectories per OD pair (default: 20)
- `--max-od-pairs`: Maximum OD pairs to sample (default: 250)
- `--no-stratified-sampling`: Disable stratified sampling (use random sampling instead)

**Output:**
- CSV files per model: `gene_abnormal_lmtad_spatial/{dataset}/seed{seed}/{model}_spatial_abnormal.csv`

**Options:**
- `--models vanilla,distill_phase1`: Generate for specific models only
- `--cuda-device 0`: Specify GPU device
- `--beam-search`: Use beam search (default: False, uses A* search)
- `--beam-width 4`: Beam width for beam search

## Step 3: Evaluate Generated Trajectories

Evaluate generated trajectories with LM-TAD to compute perplexity scores.

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/evaluate_lmtad_spatial_abnormal.py \
  --trajectory-file hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/gene_abnormal_lmtad_spatial/porto_hoser/seed42/vanilla_spatial_abnormal.csv \
  --lmtad-checkpoint /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt \
  --source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval \
  --dataset porto_hoser \
  --output hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/vanilla_spatial_evaluation.json
```

**New Features:**
- **Segment-level perplexity**: Per-trajectory-segment perplexity scores
- **Cross-model comparison flag**: `--cross-model-comparison` to enable OD-pair comparison
- **Per-segment statistics**: Track perplexity per trajectory position

**Output Schema (New):**
```json
{
  "model": "vanilla",
  "dataset": "porto_hoser",
  "trajectories": [
    {
      "trajectory_index": 0,
      "origin": 1234,
      "destination": 5678,
      "log_perplexity": 2.45,
      "segment_log_perplexities": [1.2, 2.1, 3.5, ...],
      "source_label": "route_switch",
      "status": "ok"
    }
  ],
  "summary": {
    "total_trajectories": 5000,
    "valid_trajectories": 4990,
    "failed_trajectories": 10,
    "failed_rate": 0.2,
    "log_perplexity_stats": {
      "mean": 2.45,
      "std": 1.23,
      "median": 2.1,
      "min": 0.05,
      "max": 8.92
    },
    "segment_stats": {
      "max_segment_length": 45,
      "per_index": [...]
    }
  },
  "source_statistics": {
    "non_outlier_mean": 0.3822,
    "non_outlier_std": 0.1249,
    "route_switch_mean": 7.0265,
    "route_switch_std": 1.6068,
    "detour_mean": 8.4132,
    "detour_std": 1.2098
  }
}
```

**Repeat for each model:**
```bash
# Loop through all models
for model in vanilla distill_phase1 distill_phase2 vanilla_seed43 vanilla_seed44 vanilla_seed45 distill_phase1_seed43 distill_phase1_seed44 distill_phase1_seed45; do
  uv run python tools/evaluate_lmtad_spatial_abnormal.py \
    --trajectory-file hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/gene_abnormal_lmtad_spatial/porto_hoser/seed42/${model}_spatial_abnormal.csv \
    --lmtad-checkpoint /path/to/ckpt_best.pt \
    --source-eval-dir /path/to/eval \
    --dataset porto_hoser \
    --output hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/${model}_spatial_evaluation.json
done
```

## Step 4: Aggregate Results

Aggregate evaluation results with statistical comparisons.

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/analyze_lmtad_spatial_results.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval \
  --output hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json
```

**Output:**
- Aggregated JSON with:
  - Summary statistics (real vs generated rates)
  - Statistical tests (chi-square, p-values, effect sizes)
  - Confidence intervals
  - FDR-corrected significance

## Step 5: Generate Visualizations

Create publication-quality visualizations from aggregated results.

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/visualize_lmtad_spatial_results.py \
  --input hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \
  --output-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/figures/lmtad_spatial_abnormality/porto_hoser \
  --dataset porto_hoser
```

**New Perplexity-Focused Generated Plots:**

1. **`perplexity_distribution_{dataset}.png`**
   - Histogram/density plots of log perplexity for each model
   - Shows distribution shape, central tendency, and spread
   - Vertical lines indicate means and medians
   - Lower and more concentrated distributions = better

2. **`segment_level_perplexity_{dataset}.png`**
   - Line plot showing mean perplexity per trajectory segment
   - X-axis: position in trajectory (0 to max length)
   - Y-axis: mean log perplexity
   - Shaded regions: ±1 standard deviation
   - Identifies which positions cause high perplexity

3. **`cross_model_comparison_{dataset}.png`**
   - Box plots comparing perplexity across models
   - Shows median, quartiles, and outliers
   - Ranked by median perplexity (best to worst)
   - Highlight significant differences

4. **`model_rankings_perplexity_{dataset}.png`**
   - Bar chart showing model rankings by mean perplexity
   - Color-coded by performance tier
   - Error bars show standard deviation
   - Lower bars = better performance

5. **`statistical_tests_{dataset}.png`**
   - Heatmap of p-values from paired t-tests
   - Diagonal: N/A (same model)
   - Lower triangle: p-values (blue = significant, red = not significant)
   - Shows which models are statistically different
   - Often includes effect size annotations

All plots are saved in both PNG (300 DPI) and SVG formats for publication quality.

## Statistical Analysis

The perplexity-based evaluation includes comprehensive statistical tests to quantify differences between models.

### Test Suite

#### 1. Kolmogorov-Smirnov Test (KS Test)

**Purpose**: Tests whether perplexity distributions of two models are significantly different.

**What it tests**:
- Null hypothesis (H₀): Two models have the same perplexity distribution
- Alternative hypothesis (H₁): Distributions are different

**Interpretation**:
- p < 0.05: Distributions are significantly different
- p ≥ 0.05: No evidence that distributions differ
- More robust than t-test for non-normal distributions

**Example output**:
```json
{
  "ks_test": {
    "vanilla_vs_distill_phase1": {
      "statistic": 0.12,
      "p_value": 0.003,
      "significant": true
    }
  }
}
```

#### 2. Paired t-Test

**Purpose**: Compares mean perplexity between two models evaluated on the same OD pairs.

**Requirements**:
- Same OD pairs used for both models
- Paired observations (each OD pair has scores for both models)
- Approximately normal distribution of differences

**What it tests**:
- H₀: Mean perplexity difference = 0 (models perform equally)
- H₁: Mean perplexity difference ≠ 0 (models differ)

**Interpretation**:
- p < 0.05: Models have significantly different mean perplexity
- Mean difference: Positive = first model worse, negative = first model better
- 95% CI doesn't include 0 = significant difference

**Example output**:
```json
{
  "paired_t_test": {
    "vanilla_vs_distill_phase1": {
      "mean_difference": 0.45,
      "std_difference": 1.2,
      "t_statistic": 3.21,
      "p_value": 0.001,
      "df": 249,
      "ci_lower": 0.18,
      "ci_upper": 0.72,
      "significant": true
    }
  }
}
```

#### 3. Effect Size (Cohen's d)

**Purpose**: Quantifies the practical significance of differences between models (not just statistical significance).

**Formula**: Cohen's d = (mean₁ - mean₂) / pooled_std

**Interpretation**:
- |d| < 0.2: Negligible effect (no practical difference)
- 0.2 ≤ |d| < 0.5: Small effect
- 0.5 ≤ |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect (very meaningful)

**Example output**:
```json
{
  "cohens_d": {
    "vanilla_vs_distill_phase1": {
      "effect_size": 0.38,
      "magnitude": "small_to_medium",
      "interpretation": "Distill Phase 1 performs moderately better"
    }
  }
}
```

### Statistical Analysis Pipeline

The analysis automatically computes:

1. **Pairwise Comparisons**: All models compared to each other
2. **Multiple Testing Correction**: FDR (False Discovery Rate) or Bonferroni
3. **Summary Statistics**: Mean, median, std for each model
4. **Effect Size Rankings**: Models ranked by effect size

### Understanding Statistical Significance

**Statistical vs Practical Significance**:

Example: Model A (mean=2.1, std=1.0) vs Model B (mean=2.15, std=1.0)
- May be statistically significant (p=0.04) due to large sample size
- But Cohen's d = 0.05 (negligible practical difference)
- **Interpretation**: Models are essentially equivalent in practice

**Recommended Approach**:
1. Check p-value: Is the difference statistically significant?
2. Check effect size: Is the difference practically meaningful?
3. Check confidence interval: What's the range of plausible differences?
4. Check visual plots: Does the distribution make sense?

**Example Interpretation**:
```
Model Comparison: distill_phase1 vs vanilla
- Mean perplexity: 2.45 vs 2.89 (diff = 0.44)
- Paired t-test: p = 0.001 (highly significant)
- Cohen's d = 0.52 (medium effect)
- 95% CI: [0.18, 0.70]
→ distill_phase1 is significantly better with medium practical impact
```

## Step 6: Cross-Model OD Comparison (Optional)

After evaluating all models, you can build a cross-model comparison to analyze performance across the same OD pairs.

**Command:**
```bash
cd /home/matt/Dev/HOSER

# Build cross-model comparison from all evaluation JSON files
uv run python tools/evaluate_lmtad_spatial_abnormal.py \
  --cross-model-comparison \
  --eval-results-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser \
  --output hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/cross_model_od_comparison_porto_hoser.json
```

**Features:**
- Compares all models on identical OD pairs
- Computes per-OD pair statistics and rankings
- Identifies best/worst performing models per OD pair
- Analyzes performance deltas and consistency

**Output:** See Cross-Model OD Pair Comparison section above for detailed schema.

## Step 7: Create Combined Report (Optional)

Combine Wang temporal and LM-TAD spatial results into a comprehensive report.

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/create_combined_abnormal_report.py \
  --wang-results hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/wang_results_aggregated.json \
  --lmtad-spatial-results hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json \
  --output hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/COMBINED_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md \
  --dataset porto_hoser
```

**Output:**
- Markdown report with:
  - Executive summary (temporal + spatial)
  - Temporal abnormality analysis (Wang method)
  - Spatial abnormality analysis (LM-TAD method)
  - Combined model rankings
  - Statistical comparisons
  - Key insights

## Complete Pipeline (All Steps)

Use the pipeline orchestrator to run all steps automatically:

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/run_lmtad_spatial_pipeline.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --lmtad-source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval \
  --lmtad-checkpoint /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt \
  --seed 42 \
  --num-trajectories-per-od 20 \
  --max-od-pairs 250 \
  --cross-model-comparison
```

**Pipeline Steps:**
1. **Extract** OD pairs from LM-TAD source evaluation
2. **Generate** trajectories for each HOSER model
3. **Evaluate** trajectories with LM-TAD (perplexity scores)
4. **Aggregate** results with statistical tests
5. **Generate** visualizations (perplexity distributions, segment analysis)
6. **Build** cross-model OD comparison
7. **Create** combined report (optional)

**Skip Options:**
- `--skip-extraction`: Skip OD pair extraction (use existing file)
- `--skip-generation`: Skip trajectory generation (use existing trajectories)
- `--skip-evaluation`: Skip LM-TAD evaluation (use existing results)
- `--skip-aggregation`: Skip result aggregation (use existing aggregated file)
- `--skip-visualization`: Skip visualization generation
- `--skip-cross-model`: Skip cross-model comparison

**Example (only aggregate and visualize):**
```bash
uv run python tools/run_lmtad_spatial_pipeline.py \
  --eval-dir eval_dir \
  --dataset porto_hoser \
  --lmtad-source-eval-dir /path/to/eval \
  --lmtad-checkpoint /path/to/ckpt_best.pt \
  --skip-extraction \
  --skip-generation \
  --skip-evaluation \
  --skip-cross-model
```

**Example (full evaluation with all features):**
```bash
uv run python tools/run_lmtad_spatial_pipeline.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --lmtad-source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval \
  --lmtad-checkpoint /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt \
  --seed 42 \
  --num-trajectories-per-od 20 \
  --max-od-pairs 250 \
  --cross-model-comparison
```

## Integration with python_pipeline.py

The LM-TAD spatial abnormality evaluation is integrated as a phase in the main pipeline:

**Command:**
```bash
cd /home/matt/Dev/HOSER

uv run python python_pipeline.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --run-lmtad-spatial \
  --only lmtad_spatial_abnormality
```

**With explicit paths (optional):**
```bash
cd /home/matt/Dev/HOSER

uv run python python_pipeline.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --run-lmtad-spatial \
  --lmtad-source-eval-dir /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/eval \
  --only lmtad_spatial_abnormality
```

**Auto-detection:**
- If `--lmtad-source-eval-dir` is not provided, the pipeline will auto-detect the most recent LM-TAD evaluation directory
- Checkpoint is auto-detected from the evaluation directory (checks parent directory for `ckpt_best.pt`)

### CI-friendly LM‑TAD Spatial Abnormality Runs

For CI or quick validation runs you can reduce workload by limiting OD pairs and trajectories per OD, and optionally disable the duplicate-trajectory validator. This is useful to shorten runtime and GPU usage while keeping the pipeline behavior identical to full runs.

- **CI-friendly example** (the command you're running):

```bash
cd /home/matt/Dev/HOSER
uv run python python_pipeline.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --run-lmtad-spatial \
  --only lmtad_spatial_abnormality \
  --lmtad-max-od-pairs 100 \
  --lmtad-num-trajectories-per-od 2 \
  --force \
  --lmtad-max-duplicate-ratio 1.0
```

- **`--lmtad-max-od-pairs`**: maximum number of OD pairs to sample for generation (use a small number like `100` for CI).
- **`--lmtad-num-trajectories-per-od`**: number of generated trajectories per OD pair (set to `2` or small values for CI).
- **`--lmtad-max-duplicate-ratio`**: controls duplicate-segment tolerance. Setting this to `1.0` effectively disables the duplicate check for quick runs; set it lower (e.g., `0.05`) for stricter validation.

- **Notes:**
  - The pipeline maps HOSER road IDs to LM‑TAD token IDs before token-level validation to avoid token-bounds errors.
  - When seeded variants exist (e.g., `vanilla_seed42`, `vanilla_seed43`), the pipeline and aggregation will prefer seeded variants and ignore a plain base model name (`vanilla`) to avoid mixing stale results.


## Cross-Model OD Pair Comparison

The evaluation pipeline now includes **cross-model comparison** that analyzes how different models perform on the same OD pairs. This provides a more robust assessment than per-model statistics alone.

### How It Works

1. **Same OD Pairs**: All models are evaluated on identical origin-destination pairs
2. **Per-OD Statistics**: For each OD pair, compute statistics for each model
3. **Performance Ranking**: Rank models by mean perplexity for each OD pair
4. **Aggregate Analysis**: Summarize rankings across all OD pairs

### Cross-Model OD Output Structure

When using `--cross-model-comparison` flag, the evaluation creates:

```
{eval_dir}/
├── eval_lmtad_spatial/
│   └── {dataset}/
│       ├── {model}_spatial_evaluation.json
│       └── cross_model_od_comparison_{dataset}.json
```

**cross_model_od_comparison.json** contains:
```json
{
  "metadata": {
    "timestamp": "ISO timestamp",
    "model_count": 8,
    "model_names": ["vanilla", "distill_phase1", ...],
    "total_trajectories": 5000
  },
  "models": [
    {
      "name": "vanilla",
      "trajectory_count": 5000,
      "failed_count": 10,
      "failed_rate": 0.2,
      "log_perplexity_stats": {
        "mean": 2.45,
        "std": 1.23,
        "median": 2.1,
        "min": 0.05,
        "max": 8.92
      },
      "segment_stats": {
        "max_segment_length": 45,
        "per_index": [
          {"index": 0, "count": 5000, "mean": 1.2, ...},
          ...
        ]
      },
      "od_pair_label_counts": {
        "route_switch": 1250,
        "detour": 1250,
        null: 2500
      }
    }
  ],
  "od_pairs": {
    "(1234, 5678)": {
      "origin": 1234,
      "destination": 5678,
      "trajectory_count": 20,
      "source_label": "route_switch",
      "per_model_stats": {
        "vanilla": {
          "mean_log_perplexity": 3.2,
          "median_log_perplexity": 3.1,
          "count": 20,
          "best_log_perplexity": 2.1,
          "worst_log_perplexity": 5.2
        },
        "distill_phase1": {...}
      },
      "best_model": "distill_phase1",
      "best_model_mean_log_perplexity": 2.8,
      "worst_model": "vanilla",
      "worst_model_mean_log_perplexity": 3.2,
      "performance_delta": 0.4,
      "ranking": [
        {"model": "distill_phase1", "rank": 1, "mean_log_perplexity": 2.8},
        {"model": "vanilla", "rank": 2, "mean_log_perplexity": 3.2}
      ]
    }
  },
  "od_summary": {
    "total_unique_od_pairs": 250,
    "od_pairs_with_all_models": 250,
    "average_performance_delta": 0.85,
    "std_performance_delta": 0.42,
    "best_performing_models": {
      "distill_phase1": {"best": 145, "worst": 23},
      "vanilla": {"best": 82, "worst": 134}
    },
    "source_label_distribution": {
      "route_switch": 125,
      "detour": 125,
      "unknown": 0
    },
    "statistics_by_source_label": {
      "route_switch": {
        "count": 125,
        "best_models": {"distill_phase1": 78, "vanilla": 47},
        "avg_delta": 0.92,
        "std_delta": 0.45
      },
      "detour": {...}
    }
  }
}
```

### Key Insights from Cross-Model Comparison

1. **Model Performance Consistency**: Some models consistently perform well across OD pairs
2. **OD-Specific Strengths**: Models may excel on certain types of OD pairs
3. **Performance Delta**: Quantifies the spread between best and worst models per OD pair
4. **Label-Based Analysis**: Understand model strengths by source label (route_switch/detour)

## Output Structure

```
{eval_dir}/
├── abnormal_od_pairs_lmtad_spatial_{dataset}.json
├── gene_abnormal_lmtad_spatial/
│   └── {dataset}/
│       └── seed{seed}/
│           ├── {model}_spatial_abnormal.csv
│           └── ...
├── eval_lmtad_spatial/
│   └── {dataset}/
│       ├── {model}_spatial_evaluation.json
│       └── cross_model_od_comparison_{dataset}.json
├── analysis_abnormal/
│   └── {dataset}/
│       ├── lmtad_spatial_results_aggregated.json
│       └── COMBINED_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md
└── figures/
    └── lmtad_spatial_abnormality/
        └── {dataset}/
            ├── perplexity_distribution_{dataset}.png
            ├── segment_level_perplexity_{dataset}.png
            ├── cross_model_comparison_{dataset}.png
            ├── model_rankings_perplexity_{dataset}.png
            └── statistical_tests_{dataset}.png
```

## Results Interpretation

### Key Metrics

1. **Log Perplexity Statistics**:
   - **Mean**: Average perplexity across all trajectories (lower = better)
   - **Standard Deviation**: Variability in route quality
   - **Median**: Robust central tendency, less sensitive to outliers
   - **Min/Max**: Range shows best and worst generated trajectories

   **Interpretation Guide:**
   - Mean < 2.0: Excellent route quality, strong alignment with LM-TAD
   - Mean 2.0-4.0: Good route quality, minor deviations acceptable
   - Mean 4.0-7.0: Moderate quality, some problematic routes
   - Mean > 7.0: Poor quality, frequent route abnormalities

2. **Segment-Level Perplexity**:
   - Identifies which trajectory positions cause high perplexity
   - Patterns: Some models struggle with specific road segment patterns
   - Optimization: Use insights to improve model architecture

3. **Cross-Model Comparison**:
   - **Performance Delta**: Difference between best and worst model per OD pair
   - **Ranking Consistency**: Models that rank consistently well across OD pairs
   - **Best/Worst Counts**: Frequency of each model being best/worst performer

4. **Statistical Tests**:
   - **KS Test**: Tests if perplexity distributions are significantly different
   - **Paired t-test**: Compares mean perplexity between models (same OD pairs)
   - **Effect Size (Cohen's d)**: Quantifies practical significance of differences

### Per-Segment Analysis

The evaluation tracks perplexity **per trajectory segment** (position in route):

```json
{
  "segment_stats": {
    "max_segment_length": 45,
    "per_index": [
      {
        "index": 0,
        "count": 5000,
        "mean": 1.2,
        "std": 0.5,
        "median": 1.1,
        "min": 0.1,
        "max": 4.8
      },
      ...
    ]
  }
}
```

**What it tells you:**
- **Position 0** (start): Usually low perplexity (starting point is well-defined)
- **Middle positions**: May show higher perplexity (multiple valid path choices)
- **Final positions**: Variable perplexity depending on destination clarity

### Expected Results

**Porto Dataset (Typical):**
- Mean perplexity: 2.0-4.0 for well-performing models
- Standard deviation: 1.0-2.0 (indicates consistent quality)
- Failed evaluations: <5% (GPU errors, token bounds, etc.)

**Cross-Model Comparison:**
- Best models show lower **average performance delta**
- Distribution models (distill) typically outperform vanilla
- Differences of 0.5-1.0 perplexity units are practically significant

**Statistical Significance:**
- p < 0.05: Statistically significant difference
- Cohen's d > 0.5: Medium effect size (practically meaningful)
- p < 0.01 with d > 0.8: Strong evidence of model superiority

## Troubleshooting

### Error: LM-TAD checkpoint not found
```
Solution:
1. Check checkpoint path in command
2. Verify checkpoint file exists:
   - Check parent directory: {eval_dir}/../ckpt_best.pt
   - Check checkpoints subdirectory: {eval_dir}/../checkpoints/ckpt_best.pt
   - Check eval directory itself: {eval_dir}/ckpt_best.pt
3. Use --lmtad-checkpoint to specify explicit path
4. The pipeline auto-detection checks all these locations
```

### Error: Source eval directory not found
```
Solution:
1. Verify LM-TAD source evaluation completed
2. Check path to eval directory
3. Ensure TSV files exist in eval directory
4. Use --lmtad-source-eval-dir to specify explicit path
```

### Error: No trajectories found or invalid format
```
Solution:
1. Check CSV file format (should contain road ID sequences)
2. Verify file path is correct
3. Ensure trajectories were generated successfully
4. Check CSV has correct headers (trajectory_id, origin, destination, etc.)
```

### Warning: High perplexity values (>10)
```
Expected behavior: Some routes may naturally have higher perplexity
Interpretation:
- Check if these are route_switch or detour OD pairs (expected)
- Segment-level analysis can identify problematic positions
- Consider checking source statistics for reference
- Very high values (>15) may indicate evaluation errors
```

### Error: CUDA out of memory during evaluation
```
Solutions:
1. Reduce batch size: --batch-size 64 or --batch-size 32
2. Use smaller GPU: --device cuda:0 (instead of cuda:1)
3. Close other GPU processes: nvidia-smi to check
4. Enable gradient checkpointing (if supported by model)
```

### Warning: High failed evaluation rate (>10%)
```
Potential issues:
1. Token bounds errors: Trajectories exceed LM-TAD sequence length
2. Invalid road segments: Some road IDs don't exist in grid
3. Grid mapping issues: Check GridMapper configuration
4. Check failed trajectories for common patterns
```

### Error: Cross-model comparison missing data
```
Solution:
1. Ensure all models have been evaluated before running comparison
2. Check that all models used the same OD pairs
3. Verify trajectory counts match across models
4. Check for missing evaluation JSON files
```

### Unexpected perplexity distribution (multiple peaks)
```
Possible causes:
1. Mixed OD pair types (route_switch, detour, non-outlier)
2. Models generating different route types
3. Check segment-level plots to identify specific positions
4. Verify source statistics match dataset expectations
```

### Statistical tests failing or returning NaN
```
Solutions:
1. Ensure sufficient sample size (≥30 trajectories per model)
2. Check for infinite or NaN values in perplexity data
3. Verify paired data exists (same OD pairs for both models)
4. Use non-parametric tests (Mann-Whitney U) if t-test assumptions violated
```

### Visualizations not generating
```
Solution:
1. Install required packages: matplotlib, seaborn, plotly
2. Check output directory permissions
3. Verify aggregated JSON file is valid
4. Check for large file sizes that may timeout
```

### Segment-level analysis shows all zeros
```
Solution:
1. Verify segment_log_perplexities field exists in trajectory data
2. Check that trajectories have sufficient length (>5 segments)
3. Verify LM-TAD evaluation is computing per-segment scores
4. Check for evaluation failures that skip segment computation
```

### Model rankings don't match expectations
```
Interpretation tips:
1. Lower mean perplexity = better (don't confuse with classification rates)
2. Check median, not just mean (outliers can skew results)
3. Consider effect size, not just statistical significance
4. Cross-model comparison provides more robust rankings
5. Segment-level analysis explains where models differ
```

## Verification Commands

### Check OD pairs extracted:
```bash
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal_od_pairs_lmtad_spatial_porto_hoser.json | jq '.total_unique_od_pairs'
```

### Check evaluation results for a specific model:
```bash
# List all evaluation JSON files
ls -lh hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/

# Check mean perplexity for a model
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/vanilla_spatial_evaluation.json | \
  jq '.summary.log_perplexity_stats.mean'

# Check segment statistics
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/vanilla_spatial_evaluation.json | \
  jq '.summary.segment_stats.per_index[0:3]'
```

### Check cross-model comparison:
```bash
# Verify cross-model comparison was created
ls -lh hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/cross_model_od_comparison_*.json

# Check summary statistics
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/cross_model_od_comparison_porto_hoser.json | \
  jq '.od_summary.average_performance_delta'

# Check which model performs best
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/cross_model_od_comparison_porto_hoser.json | \
  jq '.od_summary.best_performing_models'
```

### Check aggregated results:
```bash
# View summary statistics
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json | \
  jq '.summary_statistics'

# Check statistical test results
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json | \
  jq '.statistical_tests.paired_t_test | to_entries[] | {model_comparison: .key, p_value: .value.p_value, significant: .value.significant}'
```

### Check visualizations:
```bash
# List all generated plots
ls -lh hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/figures/lmtad_spatial_abnormality/porto_hoser/

# Verify specific plot types exist
ls hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/figures/lmtad_spatial_abnormality/porto_hoser/ | \
  grep -E "(perplexity_distribution|segment_level_perplexity|cross_model_comparison|model_rankings_perplexity|statistical_tests)"
```

## Best Practices

### 1. **Standalone Operation**
   - Pipeline supports running with existing results
   - Use skip flags to avoid re-running expensive steps
   - Check for existing files before running
   - Useful for iterative analysis and debugging

### 2. **Dataset-Agnostic**
   - Works with any dataset (Porto, Beijing, etc.)
   - Ensure LM-TAD source evaluation exists for target dataset
   - Adjust grid size if needed (Porto: 0.001, Beijing: 0.002)
   - Perplexity interpretation may vary by dataset

### 3. **Resource Management**
   - Trajectory generation can be time-consuming
   - Evaluation requires GPU (CUDA) for LM-TAD teacher model
   - Consider running generation overnight
   - Use smaller batch sizes if GPU memory is limited
   - Segment-level analysis adds minimal overhead

### 4. **Result Validation**
   - Compare perplexity statistics with source LM-TAD evaluation
   - Check that distributions match expected ranges
   - Validate statistical test assumptions (normality, paired data)
   - Cross-model comparison provides more robust insights than per-model stats

### 5. **Interpreting Perplexity Results**
   - **Lower = Better**: Always optimize for lower perplexity
   - **Check Medians**: Less sensitive to outliers than means
   - **Segment Analysis**: Identifies where models struggle
   - **Effect Size Matters**: Statistical significance ≠ practical significance
   - **Consistency**: Look for models that perform well across many OD pairs

### 6. **Cross-Model Comparison**
   - Ensure all models evaluated on identical OD pairs
   - Use stratified sampling for representative results
   - Analyze performance deltas to understand model differences
   - Check rankings are consistent across different metrics

## Next Steps After Completion

1. **Review aggregated results** in `analysis_abnormal/{dataset}/lmtad_spatial_results_aggregated.json`
2. **Examine visualizations** in `figures/lmtad_spatial_abnormality/{dataset}/`
3. **Compare with Wang temporal results** using combined report
4. **Analyze model performance** on spatial vs temporal abnormalities
5. **Document findings** in results document

## Related Documentation

- **[ABNORMAL_OD_WORKFLOW_GUIDE.md](../ABNORMAL_OD_WORKFLOW_GUIDE.md)** - Complete abnormal OD workflow
- **[RUN_WANG_ABNORMALITY_ANALYSIS.md](./RUN_WANG_ABNORMALITY_ANALYSIS.md)** - Wang temporal abnormality detection
- **[ABNORMAL_OD_TEACHER_STUDENT_BRIDGE.md](../results/ABNORMAL_OD_TEACHER_STUDENT_BRIDGE.md)** - Teacher-student evaluation bridge
- **[TEACHER_BASELINE_COMPARISON.md](../results/TEACHER_BASELINE_COMPARISON.md)** - LM-TAD teacher performance

