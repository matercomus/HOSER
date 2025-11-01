# Hardcoded Configuration Audit

## Summary

Found **extensive hardcoded configurations** across all plotting modules except `application_plots.py` (which was just fixed). These modules were not updated during the initial dynamic fixes.

---

## 🔴 Critical Issues

### 1. **Hardcoded Models** (5 modules)
All modules hardcode: `['vanilla', 'distilled', 'distilled_seed44']`

**Affected files:**
- `metrics_plots.py` (lines 63, 102, 158)
- `temporal_spatial_plots.py` (lines 52, 96)
- `analysis_plots.py` (lines 58, 120, 179)
- `robustness_plots.py` (indirectly via hardcoded model names)
- ~~`application_plots.py`~~ ✅ **FIXED**

**Impact**: Won't display Porto's 3 vanilla + 3 distilled models properly

---

### 2. **Hardcoded Scenarios** (4 modules)

**Affected files:**

#### `metrics_plots.py`
- Line 61-62: `['off_peak', 'peak', 'weekday', 'weekend', 'city_center', 'suburban']`
- Line 100-101: Same list (different function)
- Line 146-148: Separate temporal/spatial/trip_type lists

#### `temporal_spatial_plots.py`
- Line 49: `['off_peak', 'peak', 'weekday', 'weekend']`
- Line 95: `['city_center', 'suburban']`

#### `robustness_plots.py`
- Line 38-39: `['off_peak', 'peak', 'city_center', 'suburban', 'weekday', 'weekend']`
- Line 139: `['off_peak', 'peak', 'weekday', 'weekend']`

#### `analysis_plots.py`
- Line 53: `['off_peak', 'peak', 'weekday', 'weekend']` (in dict)
- Line 124: Scenario marker mapping

**Impact**: Ignores scenarios like `'to_airport'`, `'from_center'`, `'within_center'` that exist in data

---

### 3. **Hardcoded Metrics** (3 modules)

#### `metrics_plots.py`
- Line 54-55: `['Distance_JSD', 'Duration_JSD', 'Radius_JSD', 'Hausdorff_km', 'DTW_km', 'EDR']`
- Line 150: `['Distance_JSD', 'Duration_JSD', 'Radius_JSD']` (subset)

#### `robustness_plots.py`
- Line 41-42: `['Distance_JSD', 'Duration_JSD', 'Radius_JSD', 'Hausdorff_km', 'DTW_km', 'EDR']`

#### `analysis_plots.py`
- Line 175-176: `['Distance_JSD', 'Duration_JSD', 'Radius_JSD', 'Hausdorff_km', 'DTW_km', 'EDR']`

**Impact**: Missing metrics like `Distance_mean`, `Radius_mean`, etc. won't be plotted

---

### 4. **Hardcoded Color/Label Mappings** (3 modules)

All three modules have:
```python
COLORS = {
    'vanilla': '#e74c3c',
    'distilled': '#3498db',
    'distilled_seed44': '#2ecc71'
}

MODEL_LABELS = {
    'vanilla': 'Vanilla',
    'distilled': 'Distilled (seed 42)',
    'distilled_seed44': 'Distilled (seed 44)'
}
```

**Affected files:**
- `metrics_plots.py` (lines 28-36)
- `temporal_spatial_plots.py` (lines 25-33)
- `analysis_plots.py` (lines 26-34)

**Impact**: 
- Porto models (distill, distill_seed43, distill_seed44, vanilla_seed43, vanilla_seed44) won't have colors
- Will cause KeyError when trying to plot

---

## 🟡 Secondary Issues

### 5. **Filtering Instead of Using All Data**

Many places fetch scenarios then filter them:
```python
scenarios = get_scenario_list(data, 'train', 'vanilla')
scenarios = [s for s in scenarios if s in ['off_peak', 'peak', ...]]  # WHY?
```

This defeats the purpose of dynamic extraction!

**Locations:**
- `metrics_plots.py:61`, `metrics_plots.py:100`, `metrics_plots.py:169`
- `robustness_plots.py:38`

---

## ✅ Fixed

- ~~`application_plots.py`~~ - Fully dynamic (committed: ca26c8c)

---

## 📝 Recommended Fix Strategy

### Option A: Full Refactor (Recommended)
Replace hardcoded lists with dynamic extraction everywhere:

```python
# Before
models = ['vanilla', 'distilled', 'distilled_seed44']
scenarios = ['off_peak', 'peak', ...]

# After
vanilla_models, distilled_models = classify_models(data, 'train')
models = sorted(vanilla_models + distilled_models)
scenarios = get_available_scenarios(data, 'train')
```

### Option B: Gradual Migration
1. Fix color/label mappings first (will break Porto immediately)
2. Fix model lists
3. Fix scenario lists
4. Fix metric lists

---

## Impact Assessment

### Current State
- ✅ **Beijing**: Works (happens to have exactly the hardcoded models)
- ❌ **Porto**: **BROKEN** - Missing models, colors cause KeyError
- ❌ **Future datasets**: Won't work without code changes

### After Fix
- ✅ All datasets work automatically
- ✅ No code changes needed for new datasets
- ✅ Consistent with `application_plots.py` approach

---

## Estimated Effort

- **Lines to change**: ~40-50
- **Files to modify**: 4 (metrics, temporal, analysis, robustness)
- **Testing needed**: Run on both Beijing and Porto
- **Risk**: Medium (extensive changes to visualization code)

