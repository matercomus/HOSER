# Z-Score Abnormality Detection Results - Analysis

## Executive Summary

The existing z-score based abnormality detection results show **ZERO abnormalities found** across **all categories** for **both datasets** (Beijing and Porto). This is **NOT useful** for research purposes.

## Current Results

### Beijing Dataset (hoser-distill-optuna-6)
```
Real data (train): 629,380 trajectories → 0 abnormal (0.0%)
Real data (test):  179,823 trajectories → 0 abnormal (0.0%)
All generated models: 5,000 trajectories each → 0 abnormal (0.0%)
```

### Porto Dataset (hoser-distill-optuna-porto-eval-*)
```
Real data (train): 372,068 trajectories → 0 abnormal (0.0%)
Real data (test):  137,532 trajectories → 0 abnormal (0.0%)
All generated models: 5,000 trajectories each → 0 abnormal (0.0%)
```

### Categories Analyzed (All showing 0%)
- Speeding
- Detour
- Suspicious stops  
- Unusual duration
- Circuitous routes

## Why Zero Abnormalities?

### Root Causes

1. **Data Format Mismatch**:
   - The z_score detector expects trajectories with GPS coordinates for speed calculations
   - Beijing/Porto datasets use road ID sequences without GPS coordinates
   - Speed profile calculation returns empty: `if not speed_profile["speeds"]` → returns False

2. **Missing Baseline Statistics**:
   - Detour and unusual_duration detection require OD-pair baseline statistics  
   - These were not computed when z_score method was run
   - All OD-pair comparisons fail → no detections

3. **Threshold Configuration**:
   - Config shows `dataset: BJUT_Beijing` but was applied to Beijing dataset
   - Potential mismatch between expected data format and actual format

## Are These Results Useful?

### ❌ **NO - Not Useful for Research**

**Reasons**:
1. **No discriminative power**: 0% abnormality rate across 1M+ trajectories is unrealistic
2. **Cannot compare models**: All models show identical 0% rate
3. **Cannot validate generated trajectories**: No baseline for realistic abnormality patterns
4. **Method didn't execute properly**: Zero samples saved, empty detection lists

### What We Need Instead

✅ **Wang Statistical Method** provides:
- **OD-pair specific baselines** (already computed)
- **Theoretically grounded thresholds** (5km/5min + 2.5σ)
- **Four behavior patterns** (Abp1-4) instead of binary normal/abnormal
- **Works with road ID sequences** (no GPS coords needed)
- **Published methodology** (Wang et al. 2018, ISPRS)

## Recommendation

### ⚠️ **DO NOT RE-RUN Z-SCORE METHOD**

1. **Keep existing z_score results** as they are (for documentation/reference)
2. **Run ONLY Wang statistical method** to get meaningful abnormality analysis
3. **Update configs** to use `method: "wang_statistical"` (already done)
4. **Focus Phase 5 comparison** on Wang results vs baseline expectations

### Benefits of This Approach

- **Saves ~2-3 hours** of computation time (no need to re-run old method)
- **Uses proven methodology** from published research  
- **Provides actionable insights** (pattern distributions, OD-specific analysis)
- **Enables meaningful comparison** between real and generated trajectories

## Technical Details

### Z-Score Method Limitations (For This Dataset)

```python
# From detect_abnormal_trajectories.py:208-211
speed_profile = self.calculate_speed_profile(traj)
if not speed_profile["speeds"]:
    return False, {}  # ← Returns False for all trajectories
```

**Problem**: `calculate_speed_profile()` requires GPS coordinates to compute distances and speeds. Beijing/Porto datasets only have road ID sequences.

### Wang Method Advantages

```python
# From detect_abnormal_statistical.py:424-426
actual_length_m = self.metrics.compute_route_length(road_ids)  # ← Uses geo file
actual_time_sec = self.metrics.compute_travel_time(timestamps)
od_baseline = self.baselines.get_od_baseline(origin, destination)  # ← Pre-computed
```

**Solution**: Uses road network geometry (`.geo` file) to compute distances, and pre-computed OD-pair baselines for comparison.

## Action Plan

### What to Run Tonight

**Step 1**: Compute Porto baselines (~15-20 min)
```bash
uv run python tools/compute_trajectory_baselines.py --dataset porto_hoser
```

**Step 2**: Run Wang statistical detection on Beijing (~2-3 hours)
```bash
cd hoser-distill-optuna-6
uv run python ../python_pipeline.py \
  --eval-dir . \
  --only abnormal \
  --run-abnormal \
  --abnormal-config config/abnormal_detection_statistical.yaml
```

**Step 3**: Run Wang statistical detection on Porto (~2-3 hours)
```bash
cd hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732
uv run python ../python_pipeline.py \
  --eval-dir . \
  --only abnormal \
  --run-abnormal \
  --abnormal-config config/abnormal_detection_statistical.yaml
```

### What You'll Get

**For each dataset** (Beijing, BJUT_Beijing, Porto):
```
abnormal/{dataset}/{split}/real_data/
├── detection_results_wang.json  # NEW: Wang method results
│   └── Contains:
│       - Abp1 (Normal): X trajectories
│       - Abp2 (Temporal delay): Y trajectories  
│       - Abp3 (Route deviation): Z trajectories
│       - Abp4 (Both deviations): W trajectories
├── detection_results.json       # OLD: Z-score results (keep for reference)
└── comparison_report.json       # Will show model vs real comparisons
```

## Conclusion

The existing z-score results are **not useful** because:
1. They found 0 abnormalities (unrealistic)
2. The method couldn't execute properly on road ID sequences
3. No OD-pair baselines were available

**Solution**: Run Wang statistical method only, which is:
- Theoretically sound (published methodology)
- Properly implemented (tested and working)
- Compatible with road ID data format
- Will provide meaningful research insights

**Status**: Configs updated to run `wang_statistical` method only. Ready to execute overnight.

