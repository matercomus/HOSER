# Beam Search Ablation Study - COMPLETED

**Status**: ✅ COMPLETE  
**Date Completed**: 2025-11-06 15:07 CST  
**GitHub Issue**: #8 (Closed)  
**Total Runtime**: 26 hours

---

## Summary

This document archives the completed beam search ablation study for the Beijing dataset. The study compared A* search vs Beam search (width=4) across all models to determine if distillation benefits are search-method dependent.

**Key Finding**: **YES**, distillation benefits ARE search-method dependent!

---

## Results

### Generation Speed Comparison

| Search Method | Distilled | Vanilla | Distilled Advantage |
|---------------|-----------|---------|---------------------|
| **A* Search** | 0.30 traj/s | 0.05 traj/s | ✅ **6.0x faster** |
| **Beam Search (width=4)** | 1.79 traj/s | 2.46 traj/s | ❌ **1.4x slower** |

### Trajectory Quality (Normalized Metrics)

**A* Search**:
- Distilled: DTW_norm=0.41, Hausdorff_norm=0.024, EDR=0.46
- Vanilla: DTW_norm=0.31, Hausdorff_norm=0.026, EDR=0.48

**Beam Search**:
- Distilled: DTW_norm=0.57, Hausdorff_norm=0.026, EDR=0.51
- Vanilla: DTW_norm=0.31, Hausdorff_norm=0.032, EDR=0.54

---

## Key Insights

1. **Search Method Determines Benefit**: 
   - A* search leverages distillation for 6x speedup
   - Beam search shows opposite pattern (vanilla faster)

2. **Normalized Metrics Critical**:
   - First study to use trajectory-length independent metrics
   - Fair comparison across different trajectory lengths
   - DTW_norm, Hausdorff_norm enable proper evaluation

3. **OD Matching vs Local Quality**:
   - Distilled excels at OD matching (86-98%)
   - Vanilla shows better local trajectory quality (lower DTW_norm)
   - Trade-offs depend on application requirements

---

## Documentation

**Comprehensive Analysis**: 
- See Issue #8 comments for detailed analysis
- 3 detailed comments with tables, interpretation, recommendations
- Publication impact assessment included

**Data Preserved**:
- 24 trajectory files (A* + Beam × 6 models × 2 OD)
- 24 evaluation directories with normalized metrics
- All performance JSON files

**Location**: `/home/matt/Dev/HOSER/hoser-distill-optuna-6/`
- Trajectories: `gene/Beijing/seed42/`
- Evaluations: `eval/2025-11-06_*/`

---

## Future Work

**Porto Cross-Dataset Validation**: Tracked in Issue #44
- Different baseline (vanilla succeeds on Porto)
- Tests if interaction is Beijing-specific
- Optional: selective (2 models, 10h) or full (9 models, 40h)

---

## Recommendations

**For Deployment**:

1. **Real-Time Applications** (Speed Priority):
   - Use: Vanilla + Beam Search (width=4)
   - Speed: 2.46 traj/s (fastest)
   - Trade-off: Lower OD match rate (19%)

2. **High-Quality Generation** (OD Matching Priority):
   - Use: Distilled + Beam Search (width=4)
   - OD Match: 86-98% (best)
   - Trade-off: Slower (1.79 traj/s)

3. **Speed with Distilled Models**:
   - Use: Distilled + A* Search
   - Speed: 0.30 traj/s (6x faster than vanilla A*)
   - OD Match: ~98%

---

## Archived Files

This completion document replaces:
- `BEAM_ABLATION_NORMALIZED_METRICS_PLAN.md` (execution plan - no longer needed)
- `complete_beam_ablation_with_normalized_metrics.sh` (execution script - completed)

**Status**: Study complete, findings documented, Issue #8 closed.

**References**:
- Issue #8: https://github.com/matercomus/HOSER/issues/8
- Issue #44: https://github.com/matercomus/HOSER/issues/44 (Porto validation)
- Issue #14: Normalized metrics implementation
- Issue #22: CV handling for normalized metrics

