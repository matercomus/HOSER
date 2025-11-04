# Porto Evaluation Results - 9 Models

**Date**: 2025-11-04  
**Evaluation Directory**: `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732`  
**Dataset**: Porto (porto_hoser)  
**Models**: 9 (3 vanilla + 3 distilled phase1 + 3 distilled phase2)

---

## Summary

✅ **Generation**: 18 trajectory files created (9 models × 2 splits × 5000 traj)  
✅ **Evaluation**: Metrics computed and synced to WandB  
✅ **Abnormal Detection**: Completed on Porto dataset  
❌ **Scenarios**: Failed due to ScenarioConfig error (fixed, needs re-run)

---

## Generated Trajectories

**Total**: 18 CSV files (112MB)  
**Location**: `gene/porto_hoser/seed42/`

### Files Created:
- 6 Phase 1 distilled (distill_phase1 × 3 seeds × 2 splits)
- 6 Phase 2 distilled (distill_phase2 × 3 seeds × 2 splits)
- 6 Vanilla (vanilla × 3 seeds × 2 splits)

**Generation time**: ~13 hours (10:41 PM → 11:31 AM next day)

---

## Evaluation Results

**Status**: Partial (eval directory only 148KB - may be incomplete)  
**WandB**: 36 files synced successfully

**Next Step**: Verify evaluation metrics were saved correctly

---

## Abnormal Detection Results

**Status**: Completed (352KB of results)  
**Real Data**: 0% abnormal (Porto dataset is clean)  
**Generated Data**: Needs investigation (model names now fixed)

**Files**:
- `abnormal/porto_hoser/train/comparison_report.json`
- `abnormal/porto_hoser/test/comparison_report.json`
- Detection results per model

---

## Known Issues (Fixed in Latest Code)

1. ✅ **Model names missing**: Fixed by adding "model": model_type
2. ✅ **Scenario config error**: Fixed by adding plotting field to ScenarioConfig
3. ⏳ **Scenarios incomplete**: Needs re-run with fixed config

---

## Next Steps

1. Re-run scenarios phase: `--only scenarios`
2. Verify all 18 evaluations completed
3. Analyze abnormal detection with corrected model names
4. Generate comparison analysis vs Beijing results

