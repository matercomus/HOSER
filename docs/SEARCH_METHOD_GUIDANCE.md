# Search Method Selection Guide

**Purpose:** Guide for choosing between A* and Beam Search for trajectory generation  
**Based on:** Beijing Dataset Beam Search Ablation Study (Issue #8, Nov 2025)  
**Audience:** Researchers, practitioners deploying HOSER models

---

## Quick Decision Matrix

| Priority | Recommended Method | Model | Expected Performance |
|----------|-------------------|-------|---------------------|
| **Speed (real-time)** | Beam Search (width=4) | Vanilla | 2.46 traj/s, 19% OD match |
| **OD Matching** | Beam Search (width=4) | Distilled | 1.79 traj/s, 85-86% OD match |
| **Distilled Speed** | A* Search | Distilled | 0.30 traj/s, 98% OD match |
| **Exploration** | Beam Search (any width) | Either | Diverse trajectories |

---

## Search Methods Overview

### A* Search (Original HOSER)

**Algorithm:** Greedy best-first search with learned heuristic

**How it works:**
1. Explores single path at a time
2. Uses model predictions to guide next step
3. Backtracks when dead-ends encountered
4. Relies heavily on heuristic quality

**Advantages:**
- ✅ Leverages distilled model's improved guidance (6x speedup)
- ✅ Highest OD matching with distilled models (98%)
- ✅ Memory efficient (single path)

**Disadvantages:**
- ❌ Slow with vanilla models (0.05 traj/s)
- ❌ Many forward passes with weak heuristics (3100 vs 500)
- ❌ No trajectory diversity (deterministic for given seed)

**When to use:**
- You have a well-trained distilled model
- OD matching is critical
- You want to leverage distillation benefits
- Memory is constrained

### Beam Search (Current Default)

**Algorithm:** Parallel exploration of top-k candidates

**How it works:**
1. Maintains beam_width active paths simultaneously
2. Evaluates all candidates at each step
3. Keeps top-k paths based on combined score
4. Less dependent on heuristic quality

**Advantages:**
- ✅ Fast with vanilla models (2.46 traj/s)
- ✅ Fewer forward passes overall (32-43)
- ✅ Trajectory diversity (explores multiple paths)
- ✅ Robust to poor predictions

**Disadvantages:**
- ❌ Slower with distilled models (1.79 traj/s)
- ❌ Higher memory usage (width × model size)
- ❌ Can generate longer trajectories

**When to use:**
- Speed is the primary concern
- Using vanilla (non-distilled) models
- Want trajectory diversity
- Memory is not constrained

---

## Performance Comparison (Beijing Dataset)

### Generation Speed (trajectories/second)

```
Search Method     │ Distilled │ Vanilla  │ Winner
──────────────────┼───────────┼──────────┼─────────
A* Search         │  0.30     │  0.05    │ Distilled (6.0x faster)
Beam Search (w=4) │  1.79     │  2.46    │ Vanilla (1.4x faster)
```

**Key Insight:** The faster model **flips** depending on search method!

### OD Destination Matching

```
Configuration              │ OD Match Rate │ Use Case
───────────────────────────┼───────────────┼──────────
Distilled + A* Search      │     98%       │ High precision
Distilled + Beam Search    │   85-86%      │ Balance
Vanilla + Any Search       │     19%       │ Not recommended
```

**Key Insight:** Distillation provides 4-5x improvement in reaching correct destinations.

### Trajectory Quality (per point)

```
Configuration              │ DTW_norm (km/pt) │ Hausdorff_norm (km/pt)
───────────────────────────┼──────────────────┼────────────────────────
Vanilla + Beam Search      │      0.358       │         0.033
Vanilla + A* Search        │      0.347       │         0.029
Distilled + Beam Search    │      0.559       │         0.025
Distilled + A* Search      │      0.363       │         0.023
```

**Key Insight:** Vanilla has lower DTW_norm (better local quality) despite poor OD matching.

---

## Use Case Recommendations

### 1. Real-Time Navigation (Speed Priority)

**Recommended:** Vanilla + Beam Search (width=4)

**Performance:**
- Speed: 2.46 traj/s (fastest)
- OD Match: 19% (low but fast)
- Memory: Moderate

**Trade-offs:**
- ✅ Fastest generation
- ✅ Good local trajectory quality (DTW_norm)
- ❌ Poor destination matching
- ❌ Not suitable for OD-based tasks

**Example:** Real-time map visualization where approximate paths matter more than exact destinations.

### 2. High-Quality Generation (OD Matching Priority)

**Recommended:** Distilled + Beam Search (width=4)

**Performance:**
- Speed: 1.79 traj/s (moderate)
- OD Match: 85-86% (high)
- Memory: Moderate

**Trade-offs:**
- ✅ High destination accuracy
- ✅ Realistic trajectory lengths
- ✅ Trajectory diversity from beam search
- ❌ Slower than vanilla
- ❌ Higher DTW_norm than vanilla

**Example:** Generating trajectories for OD-specific analysis, validation datasets, or when destination accuracy is critical.

### 3. Distilled Model Deployment (Leveraging Distillation)

**Recommended:** Distilled + A* Search

**Performance:**
- Speed: 0.30 traj/s (moderate)
- OD Match: 98% (highest)
- Memory: Low

**Trade-offs:**
- ✅ Highest OD matching
- ✅ 6x faster than vanilla A*
- ✅ Memory efficient
- ❌ Slower than beam search
- ❌ No trajectory diversity

**Example:** When you've invested in distillation training and want to maximize its benefits for precise OD-based generation.

### 4. Trajectory Diversity (Exploration)

**Recommended:** Any Model + Beam Search (width=4-10)

**Performance:**
- Depends on beam width
- Wider beam = more diversity
- Trade-off with speed/memory

**Trade-offs:**
- ✅ Multiple plausible paths
- ✅ Useful for analysis and validation
- ❌ Slower and more memory intensive

**Example:** Generating multiple candidate trajectories for the same OD pair, uncertainty quantification, or ensemble methods.

---

## Configuration Guide

### Enabling A* Search

```bash
# In python_pipeline.py or gene.py
uv run python python_pipeline.py --use-astar

# Or modify config
beam_search = False  # Uses A* by default
```

### Enabling Beam Search (Default)

```bash
# Default configuration (no flag needed)
uv run python python_pipeline.py

# Custom beam width
beam_width = 4  # Default, can adjust to 1-20
```

### Beam Width Selection

**Tested:** width=4 (default)  
**Not tested:** Other widths (1, 5, 10, 20)

**General guidance:**
- `width=1`: Equivalent to greedy search (faster, less diverse)
- `width=4`: Current default (good balance)
- `width=10`: Original beam search setting (more diverse, slower)
- `width=20`: Maximum exploration (very slow)

**Ablation note:** Only width=4 has been systematically tested. Other widths may show different performance characteristics.

---

## Cross-Dataset Considerations

### Beijing Dataset (Tested)

**Vanilla baseline:** Fails dramatically (19% OD match)  
**Distillation impact:** Massive (4-5x improvement)  
**Search method matters:** Yes, 6x speedup flips to 1.4x slowdown

**Recommendation:** Use distilled models for Beijing due to vanilla failure.

### Porto Dataset (Not Yet Tested)

**Vanilla baseline:** Succeeds well (88% OD match)  
**Distillation impact:** Minimal with Phase 1 hyperparameters  
**Search method matters:** Unknown (ablation not yet performed)

**Hypothesis:** Search method impact may be less pronounced when vanilla already succeeds.

**Future work:** Porto beam search ablation tracked in Issue #44

---

## Research and Publication Considerations

### When to Include Ablation Studies

**Include when:**
- Making speed or efficiency claims
- Comparing across search methods
- Arguing distillation provides computational benefits

**Document:**
- Which search method was used
- Why that method was chosen
- Whether results generalize to other methods

### Honest Reporting

**✅ Do say:**
- "Distillation improves OD matching from 19% to 85-98%"
- "A* search with distilled models is 6x faster than vanilla A*"
- "Beam search with vanilla is fastest overall (2.46 traj/s)"

**❌ Don't say:**
- "Distillation improves speed" (depends on search method!)
- "Our method is faster" (without specifying configuration)
- "Distillation improves trajectory quality" (DTW_norm is worse)

### Limitations Section

**Acknowledge:**
- Search method dependency of benefits
- Beam search ablation only on Beijing
- Only width=4 tested systematically
- Trade-offs between speed, OD matching, and trajectory quality

---

## Future Work

### Recommended Studies

1. **Porto Beam Search Ablation** (Issue #44)
   - Tests generalizability when vanilla succeeds
   - Validates with Phase 2 optimized hyperparameters

2. **Beam Width Ablation**
   - Test widths: 1, 4, 10, 20
   - Understand diversity-speed trade-off

3. **Multi-Dataset Meta-Analysis**
   - Identify when distillation helps most
   - Dataset characteristics that predict benefit

4. **Hybrid Search Methods**
   - Combine A* and Beam Search
   - Adaptive width based on confidence

---

## Summary

**Main Takeaway:** There is no universal "best" search method - it depends on your priorities.

**Key Decision Factors:**
1. **Speed vs OD Matching** - Vanilla+Beam fastest, Distilled+A* most accurate
2. **Model Type** - Distilled benefits from A*, Vanilla benefits from Beam
3. **Use Case** - Real-time vs high-quality generation
4. **Resources** - Memory, compute, training budget

**When in doubt:** Use **Distilled + Beam Search (width=4)** for balanced performance (moderate speed, high OD matching).

---

**References:**
- Beijing Beam Search Ablation: Issue #8 (Nov 2025)
- Normalized Metrics: Issue #14, `docs/NORMALIZED_METRICS_IMPACT_SUMMARY.md`
- Evaluation Comparison: `docs/EVALUATION_COMPARISON.md`

**Related Documentation:**
- `docs/EVALUATION_PIPELINE_GUIDE.md` - How to run evaluations
- `docs/SETUP_EVALUATION_GUIDE.md` - Setup instructions
- `README.md` - Project overview
