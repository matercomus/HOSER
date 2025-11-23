<!-- 4f1afb6c-ae72-4137-bf2f-938a0f4d6bf8 c0cff0da-d5cc-45be-91bf-c3208f84d042 -->
# Refactor LM-TAD Spatial Evaluation to Perplexity-Based Analysis

## Problem Statement

The current approach incorrectly classifies generated trajectories based on source labels (route_switch/detour from OD pairs file). This is flawed because:

- Source trajectory abnormality doesn't imply generated trajectory abnormality
- Models generate routes independently given only OD pairs
- We should evaluate model performance via LM-TAD perplexity, not match source labels

## Solution Overview

Refactor to focus on perplexity-based evaluation:

1. Remove source-label-based classification entirely
2. Evaluate all generated trajectories using LM-TAD perplexity
3. Compute per-road-segment perplexity (each road gets its own perplexity)
4. Compare models on the same OD pairs
5. Track OD pair source metadata (route_switch/detour) for grouping/analysis only
6. Aggregate results showing perplexity distributions and model comparisons (both per-OD-pair and aggregate statistics)

## Implementation Plan

### Phase 1: Add Segment-Level Perplexity

**File: `simple_evaluate_with_lmtad.py`**

1. **Extend `evaluate_trajectories_direct`**

- Add optional parameter `return_segment_perplexity: bool = False`
- When enabled, return per-road-segment log perplexities alongside overall perplexity
- Segment perplexity = -log_prob for each road position in trajectory
- Store as list of lists: one list per trajectory, each containing perplexity per road segment

2. **Update return type**

- Return `Tuple[np.ndarray, np.ndarray, Optional[List[List[float]]]]`
- Third element is list of segment perplexities per trajectory (None if disabled)
- Each inner list has length = trajectory_length (one perplexity per road segment)

### Phase 2: Update Core Evaluation Function

**File: `tools/evaluate_lmtad_spatial_abnormal.py`**

1. **Remove classification logic entirely**

- Remove `od_pairs_file` parameter usage for classification
- Remove `classify_spatial_abnormality_type` function and all calls
- Keep `od_pairs_file` parameter for metadata tracking only (optional)

2. **Add segment-level perplexity support**

- Call `evaluate_trajectories_direct` with `return_segment_perplexity=True`
- Store segment perplexities in result structure
- Compute segment-level statistics per trajectory (mean/std per position)

3. **Update result structure**

- Remove `by_type.route_switch`, `by_type.detour`, `by_type.non_outlier` entirely
- Remove `spatial_abnormal_count`, `spatial_abnormality_rate`
- Keep `log_perplexity_stats` (mean, std, median, min, max)
- Add `per_trajectory_perplexities`: list of log perplexity values (one per trajectory)
- Add `per_trajectory_segment_perplexities`: list of lists (one list per trajectory, each containing perplexity per road segment)
- Add `segment_perplexity_stats`: aggregate statistics across all segments (mean/std per position index)
- Add optional `od_pair_source_metadata`: track which OD pairs came from route_switch vs detour sources (for grouping only)

### Phase 3: Update Aggregation and Analysis

**File: `tools/analyze_lmtad_spatial_results.py`**

1. **Remove route_switch/detour aggregation**

- Remove `route_switch_count`, `route_switch_rate`, `detour_count`, `detour_rate` from metrics
- Focus on perplexity-based metrics only

2. **Add per-OD-pair comparison**

- Group trajectories by OD pair (first and last road ID)
- Compare perplexity across models for same OD pairs
- Compute statistics: mean/std perplexity per OD pair per model

3. **Update statistical comparisons**

- Compare perplexity distributions between models
- Compare perplexity distributions to real data (if available)
- Use appropriate statistical tests (t-test, Mann-Whitney U, etc.)

### Phase 4: Update Visualizations

**File: `tools/visualize_lmtad_spatial_results.py`**

1. **Remove route_switch/detour visualizations**

- Remove `plot_route_switch_vs_detour` function
- Remove route_switch/detour bars from other plots

2. **Add perplexity-focused visualizations**

- Perplexity distribution comparison (box plots, histograms)
- Per-OD-pair perplexity comparison (heatmap or grouped bar chart)
- Model ranking by mean perplexity
- Segment-level perplexity visualization (if implemented)

3. **Update existing plots**

- Replace "spatial abnormality rates" with "perplexity statistics"
- Focus on perplexity distributions rather than classification counts

### Phase 5: Update Pipeline Integration

**File: `tools/run_lmtad_spatial_pipeline.py`**

1. **Remove OD pairs file passing for classification**

- Keep OD pairs file for trajectory generation only
- Remove `od_pairs_file` parameter from `evaluate_spatial_abnormal_trajectories` call

2. **Update logging and output**

- Update log messages to reflect perplexity-based evaluation
- Remove references to route_switch/detour classification

### Phase 6: Update Documentation

**File: `docs/guides/RUN_LMTAD_SPATIAL_ABNORMALITY_ANALYSIS.md`**

1. **Update classification method section**

- Explain perplexity-based evaluation approach
- Remove references to source-label-based classification
- Explain how models are compared on same OD pairs

2. **Update result interpretation**

- Explain what perplexity means (lower = better match to LM-TAD expectations)
- Explain how to interpret model comparisons
- Add segment-level analysis section (if implemented)

## Key Design Decisions

1. **Classification**: Keep simple perplexity-threshold-based classification (abnormal vs normal) OR remove classification entirely and just report statistics
2. **Segment-level**: Implement per-road-segment perplexity for detailed analysis
3. **OD pair tracking**: Track OD pairs for comparison but don't use source labels for classification
4. **Backward compatibility**: Consider keeping old result format for existing analyses, or migrate fully

## Testing Strategy

1. **Unit tests**: Update tests in `tests/test_evaluate_lmtad_spatial_abnormal.py`

- Remove tests for source-label-based classification
- Add tests for perplexity-based evaluation
- Add tests for segment-level perplexity (if implemented)

2. **Integration tests**: Verify pipeline runs end-to-end

- Generate trajectories for OD pairs
- Evaluate with new perplexity-based approach
- Aggregate and visualize results

3. **Validation**: Compare results with previous approach to ensure correctness

## Completed Implementation Status

### ✅ Phase 1: Add Segment-Level Perplexity - COMPLETED

**File: `simple_evaluate_with_lmtad.py`**

1. **`evaluate_trajectories_direct` function** (lines ~280-480)
   - ✅ Added `return_segment_perplexity: bool = False` parameter
   - ✅ Returns per-road-segment log perplexities alongside overall perplexity
   - ✅ Segment perplexity = -log_prob for each road position in trajectory
   - ✅ Returns: `Tuple[np.ndarray, np.ndarray, Optional[List[List[float]]]]`
   - ✅ Third element is list of segment perplexities per trajectory (None if disabled)
   - ✅ Each inner list has length = trajectory_length (one perplexity per road segment)
   - ✅ Comprehensive bounds checking and validation (lines ~350-403)

**File: `tests/test_simple_evaluate_with_lmtad.py`**

- ✅ Tests for `return_segment_perplexity` parameter (class `TestSegmentPerplexityCapture`)
- ✅ Test: `test_segment_perplexity_enabled_returns_values` - verifies segment logs are captured
- ✅ Test: `test_segment_perplexity_disabled_returns_none` - verifies None when disabled
- ✅ Tests for bounds checking (class `TestBoundsChecking`)
- ✅ Multiple edge cases tested: out of bounds IDs, negative IDs, mixed valid/invalid

### ✅ Phase 2: Update Core Evaluation Function - COMPLETED

**File: `tools/evaluate_lmtad_spatial_abnormal.py`**

1. **Classification logic removal** - COMPLETED
   - ✅ Function `classify_spatial_abnormality_type` REMOVED (line ~2193 in old version)
   - ✅ Source statistics loading retained for reference but NOT used for classification
   - ✅ OD pairs file parameter kept for metadata tracking only (optional)

2. **Segment-level perplexity support** - COMPLETED
   - ✅ Calls `evaluate_trajectories_direct` with `return_segment_perplexity=True` (line ~440)
   - ✅ Stores segment perplexities in result structure (line ~512)
   - ✅ Helper functions added:
     - `_compute_log_perplexity_stats` - computes aggregate statistics
     - `_compute_segment_stats` - computes per-segment statistics across trajectories

3. **Result structure updated** - COMPLETED
   - ✅ Removed: `by_type.route_switch`, `by_type.detour`, `by_type.non_outlier`
   - ✅ Removed: `spatial_abnormal_count`, `spatial_abnormality_rate`
   - ✅ Kept: `log_perplexity_stats` (mean, std, median, min, max)
   - ✅ Added: `per_trajectory_segment_perplexities` - list of lists per trajectory
   - ✅ Added: `segment_stats` - aggregate statistics per segment position
   - ✅ Added: `failed_trajectory_count`, `failed_trajectory_rate`
   - ✅ Added: `trajectories` - full per-trajectory records with status, log_perplexity, segment_log_perplexities
   - ✅ Added: `od_pair_label_counts` - metadata tracking for route_switch/detour sources
   - ✅ All outputs are JSON-serializable primitives

**File: `tests/test_evaluate_lmtad_spatial_abnormal.py`**

- ✅ Updated to match new behavior (lines 400+)
- ✅ Removed legacy classification tests
- ✅ Added tests for new schema: status flags, per-segment arrays, OD-label metadata
- ✅ Added JSON import setup for OD-pair metadata test

## Remaining Work - To Be Implemented

### ❌ Phase 3: Update Aggregation and Analysis

**File: `tools/analyze_lmtad_spatial_results.py`**

**TODO #1: Remove route_switch/detour aggregation**
- [ ] Remove `route_switch_count`, `route_switch_rate`, `detour_count`, `detour_rate` from metrics
- [ ] Focus on perplexity-based metrics only

**TODO #2: Add per-OD-pair comparison**
- [ ] Group trajectories by OD pair (first and last road ID)
- [ ] Compare perplexity across models for same OD pairs
- [ ] Compute statistics: mean/std perplexity per OD pair per model
- [ ] Support multiple models evaluated on same OD pairs
- [ ] Build shared structure: `od_pair -> {model_name: {log_perplexity, segment_log_perplexities, trajectory}}`

**TODO #3: Update statistical comparisons**
- [ ] Replace chi-square tests on abnormal counts with perplexity-based tests
- [ ] Compare perplexity distributions between models (Kolmogorov-Smirnov test, Mann-Whitney U)
- [ ] Compare perplexity distributions to real data (if available)
- [ ] Implement paired t-test on log-perplexity per OD pair when multiple models evaluated
- [ ] Add effect size calculations

### ❌ Phase 4: Update Visualizations

**File: `tools/visualize_lmtad_spatial_results.py`**

**TODO #4: Remove route_switch/detour visualizations**
- [ ] Remove `plot_route_switch_vs_detour` function (if exists)
- [ ] Remove route_switch/detour bars from other plots

**TODO #5: Add perplexity-focused visualizations**
- [ ] Perplexity distribution comparison (box plots, histograms, violin plots)
- [ ] Per-OD-pair perplexity comparison (heatmap or grouped bar chart)
- [ ] Model ranking by mean perplexity
- [ ] Segment-level perplexity visualization:
  - [ ] Color-coded trajectory visualization (per-segment perplexity)
  - [ ] Tooltip showing perplexity values
  - [ ] Aggregate segment statistics heatmap

**TODO #6: Update existing plots**
- [ ] Replace "spatial abnormality rates" with "perplexity statistics"
- [ ] Focus on perplexity distributions rather than classification counts
- [ ] Add cross-model comparison plots

### ❌ Phase 5: Update Pipeline Integration

**File: `tools/run_lmtad_spatial_pipeline.py`**

**TODO #7: Remove OD pairs file passing for classification**
- [ ] Keep OD pairs file for trajectory generation only
- [ ] Remove `od_pairs_file` parameter from `evaluate_spatial_abnormal_trajectories` call
- [ ] OR keep it for metadata tracking

**TODO #8: Update logging and output**
- [ ] Update log messages to reflect perplexity-based evaluation
- [ ] Remove references to route_switch/detour classification
- [ ] Add segment perplexity reporting to logs

### ❌ Phase 6: Cross-Model OD Comparison Data Structure

**File: `tools/evaluate_lmtad_spatial_abnormal.py` (extension)**

**TODO #9: Build shared OD model comparison data**

Design cross-model aggregation structure:

```python
# Desired schema for cross-model comparison:
{
  "metadata": {
    "num_models": int,
    "dataset": str,
    "models": [model_name_1, model_name_2, ...],
    "num_od_pairs": int,
  },
  "per_od_pair": {
    "od_pair_key": {
      "origin_road": int,
      "destination_road": int,
      "source_label": "route_switch" | "detour" | None,
      "per_model": {
        "model_name_1": {
          "log_perplexity": float,
          "segment_log_perplexities": [float, ...],
          "trajectory": [int, ...],
          "status": "ok" | "evaluation_failed"
        },
        "model_name_2": { ... },
        ...
      },
      "model_comparison": {
        "best_model": model_name,  # lowest mean perplexity
        "perplexity_deltas": {
          "model_name_2": float,  # delta from best
          ...
        }
      }
    },
    ...
  },
  "od_summary": {
    "per_od_pair_stats": {
      "model_ranking_count": {
        "model_name_1": int,  # how many ODs where this model is best
        ...
      },
      "mean_perplexity_per_model": {
        "model_name_1": float,
        ...
      }
    }
  }
}
```

**TODO #10: Implement `_build_cross_model_od_comparison` helper**
- [ ] Input: List of evaluation results (one per model)
- [ ] Group by OD pair (origin, destination)
- [ ] Align trajectories by OD pair across models
- [ ] Compute rankings and deltas per OD pair
- [ ] Output structured data for visualization

### ❌ Phase 7: Update Documentation

**File: `docs/guides/RUN_LMTAD_SPATIAL_ABNORMALITY_ANALYSIS.md`**

**TODO #11: Update classification method section**
- [ ] Explain perplexity-based evaluation approach (no more route_switch/detour classification)
- [ ] Remove references to source-label-based classification
- [ ] Explain how models are compared on same OD pairs
- [ ] Document segment-level perplexity interpretation

**TODO #12: Update result interpretation**
- [ ] Explain what perplexity means (lower = better match to LM-TAD expectations)
- [ ] Explain how to interpret model comparisons
- [ ] Add segment-level analysis section
- [ ] Document new result schema fields
- [ ] Add examples of cross-model comparison output

**TODO #13: Update CLI examples**
- [ ] Update command-line examples
- [ ] Document new parameters (if any)
- [ ] Show example outputs with new schema

### ❌ Phase 8: Integration with visualize_trajectories.py

**File: `tools/visualize_trajectories.py`**

**TODO #14: Modify cross-model visualization**
- [ ] Study existing `plot_cross_model_comparison` and `plot_cross_model_abnormal` functions
- [ ] Extend to show per-segment perplexities
- [ ] Implement color coding for segment perplexity levels
- [ ] Add tooltips with perplexity values
- [ ] Create heatmap visualization for segment statistics

**TODO #15: Implement segment perplexity visualization**
- [ ] Color-coded trajectory visualization (low perplexity = green, high = red)
- [ ] Legend mapping perplexity values to colors
- [ ] Numeric overlays for exact perplexity values
- [ ] Interactive plots (Plotly) with hover information

### ❌ Phase 9: Testing

**File: `tests/test_analyze_lmtad_spatial_results.py` (update)**

**TODO #16: Update tests for new aggregation logic**
- [ ] Test per-OD-pair comparison functionality
- [ ] Test cross-model data structure building
- [ ] Test statistical comparisons (KS test, t-test)
- [ ] Validate JSON schema compliance

**File: `tests/test_visualize_lmtad_spatial_results.py` (update)**

**TODO #17: Update visualization tests**
- [ ] Test new perplexity-based plots
- [ ] Test segment perplexity visualization
- [ ] Validate plot outputs

## Implementation Priorities

### High Priority
1. **TODO #9-10**: Cross-model OD comparison data structure (blocks downstream work)
2. **TODO #1-3**: Aggregation and analysis updates (core functionality)
3. **TODO #4-6**: Visualization updates (user-facing output)

### Medium Priority
4. **TODO #7-8**: Pipeline integration (workflow polish)
5. **TODO #14-15**: Trajectory visualization integration (enhances understanding)
6. **TODO #11-13**: Documentation (user guidance)

### Low Priority
7. **TODO #16-17**: Comprehensive test coverage (quality assurance)

## Key Design Decisions Confirmed

1. **Classification**: No classification - report raw perplexity and segment perplexities only
2. **Segment-level**: Per road segment (each road gets its own perplexity)
3. **OD pair tracking**: Track OD pairs for comparison but not use source labels for classification
4. **Backward compatibility**: Full migration to new schema (no legacy support needed)

## Dependencies

- **Phase 1-2**: Already complete, no dependencies
- **Phase 3**: Depends on Phase 2 completion (✅ done)
- **Phase 4**: Depends on Phase 3 completion
- **Phase 5**: Depends on Phase 2 completion
- **Phase 6**: Depends on Phase 2 completion
- **Phase 7**: Depends on Phase 3-4 completion
- **Phase 8**: Depends on Phase 6 completion
- **Phase 9**: Depends on all implementation phases

## Files Modified/To Modify

### Modified (✅)
- `simple_evaluate_with_lmtad.py` - Added segment perplexity support
- `tests/test_simple_evaluate_with_lmtad.py` - Added comprehensive tests
- `tools/evaluate_lmtad_spatial_abnormal.py` - Refactored to perplexity-based approach
- `tests/test_evaluate_lmtad_spatial_abnormal.py` - Updated for new schema

### To Modify (❌)
- `tools/analyze_lmtad_spatial_results.py` - TODO #1-3
- `tools/visualize_lmtad_spatial_results.py` - TODO #4-6
- `tools/run_lmtad_spatial_pipeline.py` - TODO #7-8
- `tools/visualize_trajectories.py` - TODO #14-15
- `docs/guides/RUN_LMTAD_SPATIAL_ABNORMALITY_ANALYSIS.md` - TODO #11-13
- `tests/test_analyze_lmtad_spatial_results.py` - TODO #16
- `tests/test_visualize_lmtad_spatial_results.py` - TODO #17