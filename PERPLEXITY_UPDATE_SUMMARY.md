# LM-TAD Spatial Results Analysis - Perplexity-Based Update

## Overview

This document summarizes the changes made to `tools/analyze_lmtad_spatial_results.py` to remove route_switch/detour classification logic and focus on perplexity-based metrics for trajectory evaluation.

## Changes Implemented

### 1. Removed Route_Switch/Detour Aggregation (TODO #1)

**Removed:**
- `route_switch_count` and `route_switch_rate` metrics
- `detour_count` and `detour_rate` metrics  
- `SpatialEvaluationMetrics` dataclass with route_switch/detour fields
- `load_source_real_rates()` function that loaded real abnormality rates from TSV files
- `compute_statistical_test()` function that performed chi-square/Fisher's exact tests

**Rationale:** The new approach focuses on perplexity as the primary metric for evaluating trajectory quality, removing binary classification of routes.

### 2. Added Per-OD-Pair Comparison (TODO #2)

**New Functions:**

#### `build_od_pair_data(evaluation_results)`
- Groups trajectories by Origin-Destination (OD) pairs
- OD pair defined as: `{first_road_id}-{last_road_id}`
- Structure: `od_pair -> {model: {log_perplexity, segment_log_perplexities, trajectory}}`
- Supports multiple models evaluated on same OD pairs

#### `compute_per_od_pair_statistics(od_pair_data, models)`
- Computes statistics per OD pair across multiple models
- Calculates: mean, std, min, max log perplexity per OD pair
- Only processes OD pairs with data from 2+ models
- Returns dict mapping OD pair key to statistics

### 3. Added Perplexity Distribution Analysis (TODO #3)

**New Functions:**

#### `compare_perplexity_distributions()`
- Compares perplexity distributions between two models using:
  - Kolmogorov-Smirnov test (tests if distributions are different)
  - Mann-Whitney U test (tests if distributions have different medians)
- Returns test statistics and p-values
- Handles empty arrays gracefully with NaN values
- Supports FDR correction for multiple comparisons

#### `paired_perplexity_test()`
- Performs paired t-test on log-perplexity per OD pair
- Compares models on shared OD pairs only
- Computes Cohen's d for effect size
- Requires minimum number of paired observations (configurable)
- Returns mean difference, standard deviation, t-statistic, p-value

### 4. Updated Data Structures

**Old:** `SpatialEvaluationMetrics`
```python
@dataclass
class SpatialEvaluationMetrics:
    route_switch_count: int
    route_switch_rate: float
    detour_count: int
    detour_rate: float
```

**New:** `PerplexityEvaluationMetrics`
```python
@dataclass
class PerplexityEvaluationMetrics:
    log_perplexity_stats: Dict[str, float]
    segment_log_perplexities: Optional[List[List[float]]] = None
    od_pair_data: Optional[Dict[str, Dict[str, Union[float, List[float]]]]] = None
```

**New:** `PerplexityStatisticalComparison`
```python
@dataclass
class PerplexityStatisticalComparison:
    dataset: str
    model1: str
    model2: str
    mean_perplexity_1: float
    mean_perplexity_2: float
    ks_statistic: float
    ks_p_value: float
    mannwhitney_u_statistic: float
    mannwhitney_u_p_value: float
```

### 5. Main Aggregation Function

**Replaced:** `aggregate_lmtad_spatial_results()` 
**With:** `aggregate_lmtad_perplexity_results()`

**Key Features:**

1. **Data Loading:**
   - Loads evaluation results from JSON files
   - Extracts log perplexity statistics per model
   - Collects trajectory-level perplexity data when available

2. **OD Pair Analysis:**
   - Builds OD pair comparison structure
   - Computes per-OD-pair statistics for multi-model comparisons
   - Tracks shared OD pairs across models

3. **Statistical Tests:**
   - **Distribution Comparisons:** KS test and Mann-Whitney U test for perplexity distributions
   - **Paired Tests:** Paired t-test on shared OD pairs
   - **Effect Size:** Cohen's d for paired tests
   - **Multiple Testing Correction:** FDR (Benjamini-Hochberg) correction for p-values

4. **Output Structure:**
   ```python
   {
     "summary_statistics": {
       "total_models": int,
       "model_names": List[str],
       "total_od_pairs": int,
       "compared_od_pairs": int,
       "per_model_perplexity": Dict[str, Dict]
     },
     "generated_data": {...},
     "od_pair_data": {...},
     "per_od_pair_statistics": {...},
     "statistical_analysis": {
       "perplexity_comparisons": [...],    # Simple mean comparisons
       "distribution_tests": [...],        # KS and Mann-Whitney U tests
       "paired_tests": [...],              # Paired t-tests on OD pairs
       "correction_method": "FDR (Benjamini-Hochberg)",
       "alpha": 0.05
     }
   }
   ```

### 6. Statistical Tests Implemented

#### Kolmogorov-Smirnov Test (KS Test)
- **Purpose:** Tests whether two distributions are significantly different
- **Null Hypothesis:** The distributions are identical
- **Returns:** KS statistic and p-value
- **Interpretation:** p < 0.05 indicates distributions are significantly different

#### Mann-Whitney U Test  
- **Purpose:** Tests whether two samples come from the same distribution
- **Null Hypothesis:** The distributions have equal medians
- **Returns:** U statistic and p-value
- **Interpretation:** p < 0.05 indicates distributions differ significantly

#### Paired t-test
- **Purpose:** Tests whether mean perplexity differs between models for same OD pairs
- **Null Hypothesis:** Mean difference in perplexity is zero
- **Returns:** t-statistic, p-value, Cohen's d (effect size)
- **Interpretation:** p < 0.05 indicates significant difference in mean perplexity

### 7. Error Handling

All statistical functions include:
- Input validation (check for empty arrays)
- Exception handling for test failures
- Graceful degradation with NaN values
- Warning messages for debugging

### 8. Usage Examples

```bash
# Aggregate perplexity-based results
uv run python tools/analyze_lmtad_spatial_results.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --source-eval-dir /home/matt/Dev/LMTAD/.../eval \
  --output analysis_abnormal/porto_hoser/lmtad_perplexity_results_aggregated.json
```

### 9. Key Benefits

1. **More Granular Analysis:** Perplexity provides continuous measure instead of binary classification
2. **Cross-Model Comparison:** Enables comparison of multiple models on same OD pairs
3. **Statistical Rigor:** Multiple statistical tests with proper multiple testing correction
4. **Effect Size:** Quantifies practical significance, not just statistical significance
5. **Flexibility:** Handles both trajectory-level and aggregate-level perplexity data
6. **Extensibility:** Easy to add new statistical tests or comparison methods

### 10. Backward Compatibility

The script maintains the same CLI interface but:
- Input JSON files should contain `log_perplexity_stats` field
- Optional `trajectories_with_perplexity` for trajectory-level analysis
- Output format is different (focused on perplexity instead of abnormality rates)

### 11. Future Enhancements

1. **Real Data Comparison:** Implement loading of real data perplexity when available
2. **Visualization:** Add plotting functions for perplexity distributions
3. **Segmentation Analysis:** Extend segment-level perplexity analysis
4. **Temporal Analysis:** Add time-based perplexity trends
5. **Confidence Intervals:** Add bootstrap confidence intervals for perplexity statistics

## Testing

The script includes comprehensive error handling for:
- Missing or malformed input files
- Empty perplexity arrays
- Statistical test failures
- Multiple testing correction edge cases

All numpy types are converted to Python types for JSON serialization.
