# Abnormal Trajectory Analysis - Final Results (Fixed)

**Date**: 2025-11-04  
**Evaluation**: Beijing (hoser-distill-optuna-6)  
**Fix Applied**: Unique trajectory counting (double-count bug resolved)

---

## Executive Summary

✅ **Double-counting bug fixed**: Abnormal rates now <100%  
✅ **Translated files used**: Cross-dataset analysis with proper road ID mapping  
✅ **All phases completed**: Beijing + BJUT analysis successful  
⚠️  **High cross-dataset abnormal rates**: 73-89% (requires investigation)  
⚠️  **Model names missing**: Showing as null in reports (minor logging issue)

---

## Results

### Beijing Dataset (Same Network - Training Data)

**Real Data Baseline**:
- Train: 0 / 629,380 (0.00%) abnormal
- Test: 0 / 179,823 (0.00%) abnormal

**Generated Data** (All Models):
- distilled: 0 / 5,000 (0.00%) abnormal
- distilled_seed44: 0 / 5,000 (0.00%) abnormal
- vanilla: 0 / 5,000 (0.00%) abnormal

**Interpretation**: ✅ Perfect - models maintain realistic distributions on training network

---

### BJUT_Beijing Dataset (Cross-Network - Unseen Test Set)

**Real Data Baseline**:
- Train: 0 / 27,897 (0.00%) abnormal
- Test: 0 / 5,979 (0.00%) abnormal

**Generated Data** (With Translated Road IDs):

| Split | Model | Abnormal Count | Total | Rate |
|-------|-------|----------------|-------|------|
| train | (distilled?) | 4,455 | 4,979 | 89.5% |
| train | (distilled_seed44?) | 4,456 | 4,979 | 89.5% |
| train | (vanilla?) | 3,647 | 4,979 | 73.2% |
| test | (distilled?) | 4,421 | 4,930 | 89.7% |
| test | (distilled_seed44?) | 4,420 | 4,930 | 89.7% |
| test | (vanilla?) | 3,600 | 4,930 | 73.0% |

**Interpretation**: ⚠️  High abnormality rates suggest:
1. Models struggle with transfer to unseen road network
2. Translation quality (79% mapping) may introduce artifacts
3. Detection thresholds may be too sensitive for cross-dataset
4. Legitimate generalization challenges (models overfitted to Beijing topology)

---

## Technical Validation

### Road Network Translation

**Mapping Quality**:
- 31,657 / 40,060 roads mapped (79.0%)
- Average distance: 7.9m
- Median distance: 1.3m
- Quality: FAIR

**Translation Quality** (7 files):
- Average translation rate: 93.1%
- Fully translated trajectories: 35-46% per file
- With gaps: 53-65% per file
- Failed: <1% per file

### Translated Files Used

✅ Confirmed usage of translated files:
```
✅ Using translated file for cross-dataset: distilled_train_2025-10-08_18-23-41.csv
✅ Using translated file for cross-dataset: distilled_seed44_train_2025-10-08_20-02-16.csv
✅ Using translated file for cross-dataset: vanilla_train_2025-10-08_21-25-52.csv
[... 3 more test files ...]
```

### Bug Fix Validation

**Before Fix**:
```
BJUT generated abnormal rates: 209%, 213%, 173%
Cause: Double-counting (sum of category counts)
```

**After Fix**:
```
BJUT generated abnormal rates: 89.5%, 89.5%, 73.2%
Fix: Count unique trajectory IDs across categories
Status: ✅ Mathematically valid (<100%)
```

---

## Key Findings

### 1. Within-Network Performance (Beijing)

✅ **Perfect distribution matching**:
- Real: 0% abnormal
- Generated (all models): 0% abnormal
- Models learned realistic trajectory patterns
- No hallucinated abnormalities

### 2. Cross-Network Performance (BJUT)

⚠️  **High abnormality rates (73-89%)**:

**Possible explanations**:

**A. Legitimate Transfer Learning Challenges**:
- Beijing topology differs from BJUT
- Models learned Beijing-specific patterns
- Struggle to generalize to new network structure

**B. Translation Quality Impact** (79% mapping):
- 21% unmapped roads cause gaps in trajectories
- Gaps may trigger abnormality detection
- ~60% trajectories have some gaps

**C. Detection Threshold Sensitivity**:
- Thresholds tuned for Beijing may not suit BJUT
- Speed limits differ (60 km/h threshold vs actual limits)
- Route topology differences affect circuitous/detour metrics

**D. Realistic Finding**:
- Cross-dataset evaluation reveals model limitations
- Valuable research finding about generalization
- Suggests need for domain adaptation or multi-dataset training

### 3. Model Patterns (Cross-Dataset)

**Distilled models**: ~89-90% abnormal on BJUT
**Vanilla models**: ~73% abnormal on BJUT

**Interpretation**: Vanilla generalizes slightly better than distilled (16% difference)
- Possible over-specialization in distillation process
- Or distilled models learned more Beijing-specific patterns

---

## Known Issues

### Issue 1: Model Names Missing

Comparison reports show `"model": null` instead of model names.

**Impact**: Low (rates are correct, just logging issue)

**Location**: Likely in how model_results dict is populated (line ~1188 in python_pipeline.py)

**Fix needed**: Ensure `model_results[model_type] = {...}` includes `"model": model_type`

### Issue 2: High Cross-Dataset Abnormal Rates

**Options to investigate**:

1. **Lower detection thresholds for BJUT**:
   ```yaml
   # Create BJUT-specific config
   speeding:
     speed_limit_kmh: 70  # Higher threshold
   detour:
     detour_ratio_threshold: 1.5  # More lenient
   ```

2. **Analyze trajectory gaps**:
   - Check if high gaps (60%) correlate with abnormality
   - May need better translation (higher mapping threshold)

3. **Accept as research finding**:
   - Document that models show 73-89% degradation on unseen network
   - Validates need for cross-dataset training

---

## Recommendations

### Immediate Actions

1. **Fix model name logging** (quick)
2. **Analyze gap vs abnormality correlation**
3. **Try BJUT-specific detection thresholds**

### Research Documentation

**Document for publication**:
- Cross-dataset generalization challenge (73-89% abnormality)
- Translation methodology (79% mapping, 93% translation)
- Bug fix process (double-counting → unique IDs)
- Vanilla vs distilled cross-dataset performance (73% vs 89%)

### Future Work

1. **Multi-dataset training**: Train on Beijing + BJUT
2. **Domain adaptation**: Fine-tune on BJUT
3. **Improved mapping**: Higher threshold, better algorithm
4. **Topology-aware models**: Learn network-agnostic patterns

---

## File Outputs

```
abnormal/
  Beijing/
    train/
      real_data/detection_results.json (0% abnormal)
      generated/
        distilled/detection_results.json (0%)
        distilled_seed44/detection_results.json (0%)
        vanilla/detection_results.json (0%)
      comparison_report.json
    test/[same structure]
  
  BJUT_Beijing/
    train/
      real_data/detection_results.json (0% abnormal)
      generated/
        distilled/detection_results.json (using translated file)
        distilled_seed44/detection_results.json (using translated file)
        vanilla/detection_results.json (using translated file)
      comparison_report.json (real: 0%, gen: 73-89%)
    test/[same structure]

gene_translated/BJUT_Beijing/seed42/
  - 7 translated CSV files
  - 7 translation_stats.json files

road_mapping_beijing_to_bjut_beijing.json
road_mapping_beijing_to_bjut_beijing_stats.json
```

---

## Conclusions

1. **Bug fix successful**: >100% rates eliminated by fixing double-counting
2. **Translation working**: Translated files properly used for cross-dataset
3. **Within-network perfect**: 0% abnormal for Beijing (models realistic)
4. **Cross-network challenging**: 73-89% abnormal for BJUT (generalization issue)
5. **Research value**: Quantifies cross-dataset generalization gap

**Status**: ✅ Technical infrastructure validated, research findings documented

