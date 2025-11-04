# Wang Statistical Abnormality Detection - Execution Guide

This document provides the commands to run the Wang et al. 2018 statistical abnormality detection on both Beijing and Porto evaluation directories, comparing against the old threshold-based method.

## Prerequisites

✅ **Beijing Baselines**: Computed (baselines/baselines_beijing.json)  
✅ **BJUT_Beijing Baselines**: Computed (baselines/baselines_bjut_beijing.json)  
❌ **Porto Baselines**: Need to compute first  
✅ **Configs**: Created in both eval dirs (abnormal_detection_statistical.yaml)  
✅ **Integration**: Complete in python_pipeline.py

## ⚠️ Important: Z-Score Results Not Usable

The existing z-score abnormality detection results show **0% abnormalities** for all datasets and models. This is because:
- The z-score method expects GPS coordinates, but datasets use road ID sequences
- No OD-pair baselines were computed when z-score ran
- Method couldn't execute properly on the data format

**Decision**: Run ONLY Wang statistical method (configs already updated). See `Z_SCORE_RESULTS_ANALYSIS.md` for full analysis.

## Step 1: Compute Porto Baselines (REQUIRED FIRST)

Porto baselines need to be computed before running abnormality detection. This takes ~15-20 minutes:

```bash
cd /home/matt/Dev/HOSER
uv run python tools/compute_trajectory_baselines.py --dataset porto_hoser
```

**Output**: Creates `baselines/baselines_porto_hoser.json`

**Expected**: ~1.5M trajectories, ~1M OD pairs

## Step 2: Run Abnormality Analysis on Beijing Eval Dir

This will run BOTH the old threshold-based method AND the new Wang statistical method, creating comparison results.

```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6

uv run python ../python_pipeline.py \
  --eval-dir . \
  --only abnormal \
  --run-abnormal \
  --abnormal-config config/abnormal_detection_statistical.yaml \
  2>&1 | tee abnormal_wang_comparison_$(date +%Y%m%d_%H%M%S).log
```

**What it does**:
- Analyzes real data (train/test) for Beijing dataset
- Analyzes real data (train/test) for BJUT_Beijing cross-dataset  
- Analyzes all generated trajectories from all models
- Runs Wang statistical method ONLY (z-score results already exist but unusable)
- Creates comparison reports showing real vs generated abnormality rates

**Output locations**:
```
abnormal/
├── Beijing/
│   ├── train/
│   │   ├── real_data/
│   │   │   ├── detection_results.json          # OLD z-score (0% abnormal - keep for reference)
│   │   │   └── detection_results_wang.json     # NEW Wang statistical results
│   │   └── generated/{model}/
│   │       └── detection_results_wang.json     # NEW Wang results per model
│   ├── test/ (same structure)
│   └── comparison_report.json  # Real vs Generated comparison (using Wang results)
└── BJUT_Beijing/ (same structure)
```

**Expected runtime**: 2-4 hours (depends on number of models and trajectories)

## Step 3: Run Abnormality Analysis on Porto Eval Dir

Same as Beijing but for Porto dataset:

```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732

uv run python ../python_pipeline.py \
  --eval-dir . \
  --only abnormal \
  --run-abnormal \
  --abnormal-config config/abnormal_detection_statistical.yaml \
  2>&1 | tee abnormal_wang_comparison_$(date +%Y%m%d_%H%M%S).log
```

**Output locations**:
```
abnormal/
└── porto_hoser/
    ├── train/
    │   ├── real_data/ (both methods + comparison)
    │   └── generated/{model}/ (both methods + comparison)
    ├── test/ (same structure)
    └── comparison_report.json
```

**Expected runtime**: 2-4 hours

## Combined Overnight Command (Run Both Sequentially)

To run both analyses overnight in sequence:

```bash
cd /home/matt/Dev/HOSER

# Step 1: Compute Porto baselines (if not done yet)
uv run python tools/compute_trajectory_baselines.py --dataset porto_hoser 2>&1 | tee baselines_porto_$(date +%Y%m%d_%H%M%S).log

# Step 2: Beijing analysis
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6
uv run python ../python_pipeline.py \
  --eval-dir . \
  --only abnormal \
  --run-abnormal \
  --abnormal-config config/abnormal_detection_statistical.yaml \
  2>&1 | tee abnormal_wang_beijing_$(date +%Y%m%d_%H%M%S).log

# Step 3: Porto analysis
cd /home/matt/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732
uv run python ../python_pipeline.py \
  --eval-dir . \
  --only abnormal \
  --run-abnormal \
  --abnormal-config config/abnormal_detection_statistical.yaml \
  2>&1 | tee abnormal_wang_porto_$(date +%Y%m%d_%H%M%S).log

# Summary
echo "✅ All Wang statistical abnormality analyses complete!"
echo "📊 Results in:"
echo "   - hoser-distill-optuna-6/abnormal/"
echo "   - hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal/"
```

## Results Interpretation

### Method Comparison Files

Each `method_comparison.json` contains:
- **Threshold-based results**: Old z_score method with category counts
- **Wang statistical results**: New method with Abp1-4 pattern counts
- **Agreement metrics**: How often both methods agree/disagree

### Key Metrics to Compare

1. **Detection Rates**:
   - Threshold method: % trajectories flagged as abnormal
   - Wang method: % trajectories in Abp2-4 (abnormal patterns)

2. **Pattern Distribution** (Wang method):
   - Abp1 (Normal): Baseline compliant trajectories
   - Abp2 (Temporal delay): Long duration, normal length
   - Abp3 (Route deviation): Long route, normal time (speeding/detour)
   - Abp4 (Both deviations): Major anomalies

3. **Real vs Generated**:
   - Compare abnormality rates between real data and each model
   - Evaluate if models replicate realistic abnormality patterns

## Verification Commands

Check if baselines exist:
```bash
ls -lh baselines/baselines_*.json
```

Check abnormality results:
```bash
# Beijing
find hoser-distill-optuna-6/abnormal -name "method_comparison.json"

# Porto
find hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/abnormal -name "method_comparison.json"
```

## Troubleshooting

### Error: Baseline file not found
```
Solution: Run compute_trajectory_baselines.py for the missing dataset
```

### Error: Config file not found
```
Solution: Configs are already created at:
  - hoser-distill-optuna-6/config/abnormal_detection_statistical.yaml
  - hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/config/abnormal_detection_statistical.yaml
```

### Warning: OD pair not found in baselines
```
Expected: Some OD pairs will fall back to global statistics (normal behavior)
```

## Next Steps After Completion

1. **Review comparison reports** in each dataset's abnormal/ directory
2. **Analyze pattern distributions** (Abp1-4) across models
3. **Compare detection rates** between methods
4. **Update WANG_STATISTICAL_DETECTION_PLAN.md** with Phase 5 results
5. **Create summary document** with key findings

## Technical Notes

- **Baseline Statistics**: Pre-computed OD-pair means/stds from real trajectory data
- **Hybrid Threshold Strategy**: Uses minimum of fixed (5km/5min) and statistical (2.5σ) thresholds
- **Comparison Mode**: Runs both methods independently, saves separate results + comparison
- **Cross-Dataset**: Beijing eval dir also analyzes BJUT_Beijing with road ID translation

