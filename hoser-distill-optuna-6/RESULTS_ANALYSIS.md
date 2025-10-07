# Evaluation Results Comparison: Vanilla vs. Distilled (Seed 42)

## Results Summary

| Metric | Vanilla | Distilled | Difference | Interpretation |
|--------|---------|-----------|------------|----------------|
| **Distance (JSD)** | 0.0663 | 0.0663 | 0.0000 | ✅ Identical - same route lengths |
| **Duration (JSD)** | 0.1896 | 0.2481 | +0.0585 | ⚠️ Different - timing predictions differ |
| **Radius (JSD)** | 0.0709 | 0.0709 | 0.0000 | ✅ Identical - same spatial spread |
| **Hausdorff (km)** | 0.4407 | 0.4407 | 0.0000 | ✅ Identical - same route shapes |
| **DTW (km)** | 6.6540 | 6.6540 | 0.0000 | ✅ Nearly identical - routes + timing |

## Why Results Are Nearly Identical

### NetworkX A* Mode Behavior

The pipeline used `--nx_astar` mode, which means:

1. **Routing**: NetworkX A* algorithm determines the path (graph shortest path)
2. **Model Role**: HOSER models ONLY predict travel times along the predetermined path

```
┌─────────────────────────────────────────────────────────┐
│  NetworkX A* Mode (What We Ran)                         │
├─────────────────────────────────────────────────────────┤
│  1. Use graph algorithm to find optimal route           │
│     → Both models get SAME routes (graph-determined)    │
│                                                          │
│  2. Use HOSER model ONLY for timing predictions         │
│     → Small differences in travel time estimates        │
└─────────────────────────────────────────────────────────┘
```

### Result Interpretation

**Identical Metrics** (Distance, Radius, Hausdorff):
- These measure **route geometry and structure**
- Since NetworkX A* picks routes, both models follow identical paths
- **This is expected behavior in `--nx_astar` mode**

**Different Metric** (Duration JSD: 0.1896 vs 0.2481):
- Measures **timing prediction quality**
- Vanilla: Better timing predictions (lower JSD = closer to real distribution)
- Distilled: Slightly worse timing predictions
- **This shows the actual model difference!**

## What This Tells Us

### Good News ✅
- Both models produce **valid, high-quality routes** (low Hausdorff, low DTW)
- Routes are **spatially consistent** with real trajectories (low Distance/Radius JSD)
- NetworkX A* provides optimal shortest paths

### The Catch ⚠️
- We're **not testing route prediction ability** - that's what distillation was meant to improve!
- We're only seeing **timing prediction differences**
- NetworkX A* masks the distillation benefits

## To See Real Distillation Impact

### Option 1: Use Model-Based A* Search (Recommended)

Remove `--nx_astar` to let the model make routing decisions:

```bash
# Edit run_gene_eval_pipeline.sh:
# Remove or comment out the --nx_astar flag (line 179)

# Then force re-run with model-based search:
./run_gene_eval_pipeline.sh --force
```

**Expected changes**:
- Distance/Radius/Hausdorff will differ between models
- Shows distillation impact on route choices
- Much slower (~2-4 hours per 5000 trajectories instead of ~20 minutes)

### Option 2: Use Beam Search for Better Exploration

```bash
# Edit run_gene_eval_pipeline.sh:
# Replace --nx_astar with --beam_search --beam_width 8

./run_gene_eval_pipeline.sh --force
```

**Trade-off**:
- Slower than A* but faster than full model search
- Better route diversity
- Shows how well models explore candidate routes

### Option 3: Accept NetworkX A* Results

If **timing accuracy** is sufficient:
- Keep NetworkX A* for fast, optimal routes
- Duration JSD shows vanilla has better timing (0.1896 vs 0.2481)
- This may be acceptable if route quality matters more than prediction ability

## Recommendation

**For Your Thesis/Paper**:

1. **Keep these results** as "NetworkX A* mode" (optimal routes + learned timing)
   - Shows both models can guide timing along optimal paths
   - Fast generation for large-scale evaluation

2. **Add a model-based search comparison** (at least for seed 42):
   - Remove `--nx_astar` flag
   - Re-run with `--force` flag
   - This shows distillation impact on **route prediction** (the main contribution)

3. **Report both**:
   - "In NetworkX A* mode (optimal routing), both models achieve similar route quality..."
   - "In model-based search mode, distilled model shows X% improvement in [metrics]..."

## Current Results Validity

These results are **valid and publishable** for:
- ✅ Showing both models can predict reasonable timing
- ✅ Demonstrating pipeline infrastructure works
- ✅ Baseline for optimal routing + learned timing
- ❌ **NOT** showing distillation impact on route prediction (main contribution)

## Next Steps

1. **Decision Point**: Do you want to see distillation impact on routing?
   - Yes → Remove `--nx_astar`, re-run with `--force`
   - No → Accept timing-only comparison, document this clearly

2. **For Seeds 43 & 44**:
   - Decide on search mode before running
   - Use same mode for fair comparison
   - If using model-based search, budget ~4-6 hours per seed

## Quick Comparison Table

| Mode | Runtime (5k traj) | What It Tests | Shows Distillation? |
|------|-------------------|---------------|---------------------|
| **NetworkX A*** | ~20 min | Timing only | Minimal (Duration JSD) |
| **Beam Search** | ~1-2 hours | Routes + timing | Yes (all metrics) |
| **Model A*** | ~2-4 hours | Full prediction | Yes (all metrics, slow) |

---

**Current Status**: NetworkX A* mode completed successfully for seed 42 (both models).

**Action Required**: Decide if you want model-based search results for fair distillation comparison.

