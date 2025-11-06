# Beam Ablation Study: Using Normalized Metrics

## Current Situation

**Problem**: The currently running beam ablation study started BEFORE normalized metrics were implemented.

**Timeline**:
- Ablation started: Nov 5, 21:44 CST (using OLD evaluation.py)
- Normalized metrics added: Nov 6, 05:04 UTC (commit 241b4cf, Issue #14)
- Current status: A* generation 99% complete (990/1000 vanilla test trajectories)

**Impact**: The A* search evaluation will use **raw metrics only** (Hausdorff_km, DTW_km) without the new normalized versions (Hausdorff_norm, DTW_norm).

## Why Normalized Metrics Matter for Beam Ablation

The ablation study compares:
- **A* search**: Greedy, generates trajectories of varying lengths
- **Beam search**: Explores multiple hypotheses, may generate different-length trajectories

**Without normalization**: 
- Longer trajectories naturally have larger DTW/Hausdorff distances
- Cannot fairly compare trajectory quality across different generation methods
- Metric values confound length with similarity

**With normalization**:
- DTW_norm and Hausdorff_norm are trajectory-length independent
- Fair comparison between A* and Beam search
- Clear separation of trajectory quality vs. trajectory length

## Solution: Single Script to Complete Study

**When**: After current vanilla test generation finishes (should be within 30 minutes)

**Command**:
```bash
./complete_beam_ablation_with_normalized_metrics.sh
```

**What it does**:
1. **Verifies normalized metrics** are in evaluation.py (commit 241b4cf)
2. **Re-evaluates A* trajectories** with normalized metrics (1-2 hours)
   - Uses existing A* generation from seed42
   - Adds Hausdorff_norm and DTW_norm to results
3. **Generates beam search trajectories** with width=4 (3-4 hours)
4. **Evaluates beam search** with normalized metrics (1-2 hours)

**Total estimated time**: 5-7 hours

**Result**: Complete A* vs Beam comparison with normalized, trajectory-length independent metrics

### Compare A* vs Beam

**Metrics to compare**:
```python
# Old metrics (confounded by length)
- Hausdorff_km
- DTW_km

# NEW metrics (length-independent) ⭐
- Hausdorff_norm  (km/point)
- DTW_norm        (km/point)
```

**Fair comparison**:
- Use `*_norm` metrics to compare trajectory quality
- Use `*_km` metrics to understand absolute distances
- Compare distilled vs vanilla improvement for each method

## Execution Plan

```bash
# 1. Wait for current A* generation to complete (currently at 99%)
#    Expected completion: ~30 minutes

# 2. Run single consolidated script (handles everything)
./complete_beam_ablation_with_normalized_metrics.sh  # 5-7 hours
#    - Re-evaluates A* with normalized metrics (1-2 hours)
#    - Generates and evaluates Beam search (4-5 hours)

# 3. Compare results using normalized metrics
cd hoser-distill-optuna-6/eval
ls -lt | head -20  # Find latest evaluation files
```

## Why Not Restart from Scratch?

**Generation already complete for A***:
- Distilled (train): ✅ 1000 trajectories (59 min)
- Distilled (test): ✅ 1000 trajectories (57 min)
- Distilled_seed44 (train): ✅ 1000 trajectories (52 min)
- Distilled_seed44 (test): ✅ 1000 trajectories (51 min)
- Vanilla (train): ✅ 1000 trajectories (5h 57m)
- Vanilla (test): 🔄 990/1000 (5h 42m, ~30min remaining)

**Total generation time invested**: ~13 hours

**Re-running evaluation only**: 1-2 hours (saves 11+ hours)

## Benefits of This Approach

1. **No wasted computation**: Reuse expensive A* generation
2. **Fair comparison**: Both A* and Beam evaluated with same normalized metrics
3. **Backward compatible**: Old raw metrics still available
4. **Scientific rigor**: Trajectory-length independent comparison
5. **Addresses Issue #14**: Uses newly implemented normalized metrics

## Expected Outcome

**Hypothesis**: Distillation provides similar speedup regardless of search method

**Test with normalized metrics**:
```
Metric          | A* (distilled) | Beam (distilled) | A* (vanilla) | Beam (vanilla)
----------------|----------------|------------------|--------------|---------------
DTW_norm        | ?              | ?                | ?            | ?
Hausdorff_norm  | ?              | ?                | ?            | ?
Traj/sec        | 0.28           | ?                | 0.05         | ?
```

**Key question**: Does beam search maintain the ~6x speedup advantage that A* shows?

## References

- **Issue #8**: Beam Search Evaluation Dependence
- **Issue #14**: Local Metrics Not Normalized (FIXED, commit 241b4cf)
- **PR #40**: Added normalized metrics to evaluation.py

---

**Status**: Waiting for vanilla test generation to complete (~30 min)  
**Next action**: Run `./complete_beam_ablation_with_normalized_metrics.sh`

**Scripts**:
- `run_beam_ablation.sh` - Original script (started A* generation, now running)
- `complete_beam_ablation_with_normalized_metrics.sh` - NEW consolidated script for completion
