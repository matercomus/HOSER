# LM-TAD Spatial Abnormality Detection - Execution Guide

This document provides instructions for running the LM-TAD spatial abnormality evaluation pipeline, which evaluates how well HOSER models reproduce spatial abnormalities (route switches and detours) identified by the LM-TAD teacher model.

## Overview

The LM-TAD spatial abnormality evaluation complements the Wang temporal abnormality detection by focusing on **spatial** route deviations rather than temporal delays. This evaluation:

1. **Extracts** spatial abnormal OD pairs from LM-TAD source evaluation (route switches and detours)
2. **Generates** trajectories for these OD pairs using HOSER models
3. **Evaluates** generated trajectories with LM-TAD to classify spatial abnormality types
4. **Aggregates** results with statistical comparisons
5. **Visualizes** spatial abnormality rates, model rankings, and statistical significance

## Prerequisites

✅ **LM-TAD Source Evaluation**: Must have completed LM-TAD evaluation on source dataset  
✅ **LM-TAD Checkpoint**: Trained teacher model checkpoint file  
✅ **HOSER Models**: Trained HOSER models in evaluation directory  
✅ **Integration**: Complete in `python_pipeline.py` (phase: `lmtad_spatial_abnormality`)

## Key Concepts

### Spatial Abnormality Types

1. **Route Switch** (~3.27% in Porto):
   - Trajectory takes a different route than expected
   - Log perplexity: ~7.03 (mean)
   - Indicates spatial deviation from normal routes

2. **Detour** (~3.27% in Porto):
   - Trajectory takes a longer, circuitous route
   - Log perplexity: ~8.41 (mean)
   - Indicates significant spatial deviation

3. **Non-Outlier** (~93.46% in Porto):
   - Normal trajectories following expected routes
   - Log perplexity: ~0.38 (mean)
   - Baseline for comparison

### Classification Method

Spatial abnormality types are classified based on **log perplexity thresholds** derived from the LM-TAD source evaluation:

- **Non-outlier**: log_perplexity < 6.0
- **Route switch**: 6.0 ≤ log_perplexity < 8.0
- **Detour**: log_perplexity ≥ 8.0

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

Evaluate generated trajectories with LM-TAD and classify spatial abnormality types.

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

**Output:**
- JSON file per model with:
  - Spatial abnormality rates (overall, route switch, detour)
  - Log perplexity statistics
  - Per-trajectory classifications
  - Source statistics used for classification

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

**Generated Plots:**
1. `spatial_abnormality_rates_{dataset}.png` - Real vs generated rates with confidence intervals
2. `route_switch_vs_detour_{dataset}.png` - Stacked bar chart breakdown
3. `model_rankings_spatial_{dataset}.png` - Models ranked by deviation from real rate
4. `statistical_significance_spatial_{dataset}.png` - Significance markers with CIs
5. `perplexity_distribution_spatial_{dataset}.png` - Log perplexity distribution comparison

All plots are saved in both PNG (300 DPI) and SVG formats.

## Step 6: Create Combined Report (Optional)

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
  --max-od-pairs 250
```

**Skip Options:**
- `--skip-extraction`: Skip OD pair extraction (use existing file)
- `--skip-generation`: Skip trajectory generation (use existing trajectories)
- `--skip-evaluation`: Skip LM-TAD evaluation (use existing results)
- `--skip-aggregation`: Skip result aggregation (use existing aggregated file)
- `--skip-visualization`: Skip visualization generation

**Example (only aggregate and visualize):**
```bash
uv run python tools/run_lmtad_spatial_pipeline.py \
  --eval-dir eval_dir \
  --dataset porto_hoser \
  --lmtad-source-eval-dir /path/to/eval \
  --lmtad-checkpoint /path/to/ckpt_best.pt \
  --skip-extraction \
  --skip-generation \
  --skip-evaluation
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
│       └── ...
├── analysis_abnormal/
│   └── {dataset}/
│       ├── lmtad_spatial_results_aggregated.json
│       └── COMBINED_ABNORMAL_TRAJECTORY_ANALYSIS_REPORT.md
└── figures/
    └── lmtad_spatial_abnormality/
        └── {dataset}/
            ├── spatial_abnormality_rates_{dataset}.png
            ├── route_switch_vs_detour_{dataset}.png
            ├── model_rankings_spatial_{dataset}.png
            ├── statistical_significance_spatial_{dataset}.png
            └── perplexity_distribution_spatial_{dataset}.png
```

## Results Interpretation

### Key Metrics

1. **Spatial Abnormality Rate**:
   - Overall rate of spatial abnormalities (route switch + detour)
   - Compare generated vs real data rate
   - Lower rates indicate models generate fewer spatial deviations

2. **Route Switch vs Detour Breakdown**:
   - Route switch: Different route, moderate deviation
   - Detour: Longer route, significant deviation
   - Models may perform differently on each type

3. **Statistical Significance**:
   - Chi-square tests compare generated vs real rates
   - FDR correction for multiple comparisons
   - Effect sizes (Cohen's h) indicate magnitude of difference

4. **Model Rankings**:
   - Ranked by absolute deviation from real spatial abnormality rate
   - Lower deviation = better reproduction of spatial patterns

### Expected Results

**Porto Dataset:**
- Real spatial abnormality rate: ~6.54% (3.27% route switch + 3.27% detour)
- Generated rates typically lower (models generate fewer spatial deviations)
- Best models: Closer to 6.54% indicates better spatial pattern reproduction

**Beijing Dataset:**
- Rates may differ based on dataset characteristics
- Compare with source LM-TAD evaluation for baseline

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

### Error: No spatial outliers found in TSV
```
Solution:
1. Check TSV file contains "route switch" or "detour" in outlier column
2. Verify TSV file format (tab-separated)
3. Check source evaluation configuration
```

### Warning: Roadmap file not found
```
Solution:
1. Ensure data/{dataset}/roadmap.geo exists
2. Check dataset name matches directory structure
3. Verify roadmap.geo file format
```

### Low spatial abnormality rates in generated trajectories
```
Expected behavior: Models typically generate fewer spatial abnormalities than real data
Interpretation: 
- Very low rates (<1%) may indicate models are too conservative
- Moderate rates (2-4%) may indicate good balance
- Compare with Wang temporal rates for full picture
```

## Verification Commands

Check OD pairs extracted:
```bash
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal_od_pairs_lmtad_spatial_porto_hoser.json | jq '.total_unique_od_pairs'
```

Check evaluation results:
```bash
ls -lh hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval_lmtad_spatial/porto_hoser/
```

Check aggregated results:
```bash
cat hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/lmtad_spatial_results_aggregated.json | jq '.summary_statistics'
```

Check visualizations:
```bash
ls -lh hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/figures/lmtad_spatial_abnormality/porto_hoser/
```

## Best Practices

1. **Standalone Operation**:
   - Pipeline supports running with existing results
   - Use skip flags to avoid re-running expensive steps
   - Check for existing files before running

2. **Dataset-Agnostic**:
   - Works with any dataset (Porto, Beijing, etc.)
   - Ensure LM-TAD source evaluation exists for target dataset
   - Adjust grid size if needed (Porto: 0.001, Beijing: 0.002)

3. **Resource Management**:
   - Trajectory generation can be time-consuming
   - Evaluation requires GPU (CUDA)
   - Consider running generation overnight

4. **Result Validation**:
   - Compare with source LM-TAD evaluation statistics
   - Check perplexity distributions match expected ranges
   - Validate statistical test results

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

