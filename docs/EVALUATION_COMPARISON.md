# Cross-Dataset Evaluation Analysis Comparison

**Date:** October 31, 2025  
**Purpose:** Compare Beijing and Porto evaluation analyses for completeness and consistency

---

## Document Structure Comparison

| Section | Beijing | Porto | Notes |
|---------|---------|-------|-------|
| **Executive Summary** | ✅ | ✅ | Both present |
| **1. Experimental Setup** | ✅ | ✅ | Both present |
| **2. Results Overview** | ✅ | ✅ | Both present |
| **3. Key Findings** | ✅ | ✅ | Both present |
| **3.1 Path Completion** | ✅ | ✅ | Both present |
| **3.2 Trip Length Realism** | ✅ | ✅ | Both present |
| **3.3 Spatial Distribution** | ✅ | ✅ | Both present |
| **3.4 Generalization** | ✅ | ✅ | Both present |
| **3.5 Scenario-Level Analysis** | ✅ | ✅ | **Both added (Oct 31, 2025)** |
| **3.5.1 Per-Scenario Tables** | ✅ | ✅ | Identical structure |
| **3.5.2 Key Scenario Findings** | ✅ | ✅ | Dataset-specific insights |
| **3.5.3 Notable Scenarios** | ✅ | ✅ | Top-5 lists |
| **4. Dataset-Specific Section** | "Why Vanilla Fails" | "Porto vs Beijing" | Appropriate for each |
| **5. What Distillation Transferred** | ✅ | ✅ (Phase 1) | Both present |
| **6. Trajectory-Level Analysis** | ✅ | ✅ | Both present |
| **7. Statistical Summary** | ✅ | ✅ | Both present |
| **8. Conclusions** | ✅ | ✅ | Both present |
| **9. Appendix** | ✅ | ✅ | Both present |

**Total Lines:**
- Beijing: 882 lines
- Porto: 975 lines (includes Phase 2 context)

---

## Scenario Analysis Coverage

### Subsection Structure

| Subsection | Beijing | Porto | Status |
|------------|---------|-------|--------|
| **Scenario Taxonomy** | ✅ 9 scenarios | ✅ 9 scenarios | ✅ **MATCHED** |
| **Scenario Distribution Plots** | ✅ 2 figures | ✅ 2 figures | ✅ **MATCHED** |
| **Per-Scenario Table (Train)** | ❌ Not shown | ❌ Not shown | Both omit train (focus on test) |
| **Per-Scenario Table (Test)** | ✅ 9 rows | ✅ 9 rows | ✅ **MATCHED** |
| **Metric Comparison Plots** | ✅ 2 figures | ✅ 2 figures | ✅ **MATCHED** |
| **Hierarchical Breakdowns** | ✅ 2 figures | ✅ 2 figures | ✅ **MATCHED** |
| **Scenario-Specific Insights** | ✅ Detailed | ✅ Detailed | ✅ **MATCHED** |
| **Notable Scenarios (Top-5)** | ✅ Lists | ✅ Lists | ✅ **MATCHED** |
| **Cross-Dataset Comparison** | ✅ Brief | ✅ Detailed | Porto has more context |

### Scenarios Analyzed (Both Datasets)

**Temporal:**
- `weekday` (70-71%)
- `weekend` (29-30%)
- `peak` (8-11%)
- `off_peak` (88-92%)

**Spatial:**
- `city_center` (88-91%)
- `suburban` (9-12%)
- `within_center` (60-62%)
- `to_center` (14-16%)
- `from_center` (10-17%)

**Status:** ✅ **IDENTICAL TAXONOMY**

### Aggregated Analysis Outputs

| Output | Beijing | Porto | Status |
|--------|---------|-------|--------|
| **scenarios_train.csv** | ✅ 9 scenarios | ✅ 9 scenarios | ✅ **MATCHED** |
| **scenarios_test.csv** | ✅ 9 scenarios | ✅ 9 scenarios | ✅ **MATCHED** |
| **top_scenarios_train.csv** | ✅ 30 entries | ✅ 30 entries | ✅ **MATCHED** |
| **top_scenarios_test.csv** | ✅ 30 entries | ✅ 30 entries | ✅ **MATCHED** |
| **aggregates.json** | ✅ Present | ✅ Present | ✅ **MATCHED** |
| **md/scenario_analysis.md** | ✅ Generated | ✅ Generated | ✅ **MATCHED** |

---

## Key Findings Comparison

### Beijing Findings

1. **Distillation dramatically improves** (85-89% OD vs 12-18%)
2. **Vanilla catastrophically fails** across all scenarios
3. **Distance JSD reduced 87%** (0.145 → 0.018)
4. **Radius JSD reduced 98%** (0.198 → 0.003)
5. **Universal scenario benefits** (all Δ large and negative)
6. **Long-distance navigation** benefits most (Δ = -0.24)

### Porto Findings

1. **Both models perform well** (87-92% OD for both)
2. **Minimal distillation benefit** with Phase 1 hyperparameters
3. **Distance JSD similar** (distilled 0.006 vs vanilla 0.0055)
4. **Radius JSD similar** (distilled 0.011 vs vanilla 0.011)
5. **Scenario-dependent benefits** (±Δ mixed, average near-zero)
6. **Dense urban scenarios** show marginal distilled advantage

**Interpretation:** Task complexity determines distillation value.

---

## Missing Analyses

### ❌ Inference Speed / Computational Performance

**Neither document includes:**
- Trajectory generation time (seconds per trajectory)
- Throughput metrics (trajectories per second)
- Beam search timing breakdown
- Model inference latency
- GPU vs CPU performance comparison
- Memory usage during generation
- Batch generation efficiency

**What's present:**
- Beijing: "Caching for efficiency", "GPU for generation"
- Porto: "GPU-accelerated beam search", "CPU-based evaluation"

**Status:** ⚠️ **GAP IN BOTH DOCUMENTS**

**Impact:** Cannot assess:
- Whether distillation adds computational overhead
- Real-time generation feasibility
- Scalability to large-scale generation
- Hardware requirements for deployment

### ❌ Training Time / Convergence Analysis

**Neither document includes:**
- Training time per epoch
- Total training wall-clock time
- Convergence curves (train vs val loss over epochs)
- Early stopping analysis
- GPU utilization during training
- Memory footprint during training

**What's present:**
- Both: "25 epochs", training hyperparameters
- Neither: Actual timing or convergence behavior

**Status:** ⚠️ **GAP IN BOTH DOCUMENTS**

### ❌ Ablation Studies

**Neither document includes:**
- Effect of varying distillation hyperparameters (λ, τ, w)
- Teacher vs student architecture comparison
- Alternative teacher models
- Distillation window size sensitivity

**What's present:**
- Porto: References to Phase 1 vs Phase 2 hyperparameters
- Beijing: References to Optuna tuning (but not detailed ablations)

**Status:** ⚠️ **GAP IN BOTH DOCUMENTS** (though Porto has more context via Hyperparameter-Optimization-Porto.md)

### ❌ Error Analysis / Failure Case Studies

**Neither document includes:**
- Specific failure case examples with trajectory visualizations
- Categorization of failure modes (stuck, loops, wrong direction)
- Spatial distribution of failures (where do models fail?)
- OD-pair difficulty analysis (which OD pairs are hardest?)

**What's present:**
- Beijing: General description of vanilla failures ("gets stuck", "stops early")
- Porto: Brief mention of vanilla success
- Both: Multi-scenario trajectory grids (but no detailed failure analysis)

**Status:** ⚠️ **PARTIAL COVERAGE** (qualitative descriptions, no systematic analysis)

### ❌ Model Size / Parameter Count Comparison

**Neither document includes:**
- Number of parameters (vanilla vs distilled)
- Model size on disk (MB)
- Architecture details (layer counts, hidden dimensions)

**What's present:**
- Both: "Identical architecture" for vanilla vs distilled
- Neither: Actual parameter counts or model sizes

**Status:** ⚠️ **GAP IN BOTH DOCUMENTS**

---

## Visualizations Comparison

| Visualization Type | Beijing | Porto | Status |
|--------------------|---------|-------|--------|
| **Distance Distribution** | ✅ train/test | ✅ train/test | ✅ **MATCHED** |
| **Radius Distribution** | ✅ train/test | ✅ train/test | ✅ **MATCHED** |
| **OD Matching Rates** | ✅ Bar chart | ✅ In table | Beijing has dedicated figure |
| **JSD Comparison** | ✅ Figure | ✅ In table | Beijing has dedicated figure |
| **Metrics Heatmap** | ✅ Figure | ❌ Missing | Beijing more comprehensive |
| **Performance Radar** | ✅ Figure | ❌ Missing | Beijing more comprehensive |
| **Scenario Distribution** | ✅ 2 models | ✅ 2 models | ✅ **MATCHED** |
| **Metric Comparison** | ✅ 2 models | ✅ 2 models | ✅ **MATCHED** |
| **Hierarchical Plots** | ✅ 2 models | ✅ 2 models | ✅ **MATCHED** |
| **Multi-Scenario Grids** | ❌ Not referenced | ✅ 3 featured + 4 listed | Porto more detailed |
| **Train vs Test** | ✅ Figure | ✅ In table | Beijing has dedicated figure |
| **Seed Robustness** | ✅ Figure | ✅ In table | Beijing has dedicated figure |

**Summary:**
- Beijing: More standalone figures (8 primary + 4 distributions = 12)
- Porto: More trajectory visualizations (7 multi-scenario grids)
- Both: Complete scenario analysis visualizations (6 files each)

---

## Methodology Details Comparison

| Detail | Beijing | Porto | Status |
|--------|---------|-------|--------|
| **OD Matching Algorithm** | ✅ Code snippet | ✅ Code snippet | ✅ **MATCHED** |
| **JSD Calculation** | ✅ Formula + code | ✅ Formula only | Beijing more detailed |
| **Metrics Formulas** | ✅ All metrics | ✅ All metrics | ✅ **MATCHED** |
| **Grid Resolution** | ✅ 0.001° (~111m) | ✅ 0.001° (~111m) | ✅ **MATCHED** |
| **Beam Search Width** | ✅ Width 4 | ✅ Width 4 | ✅ **MATCHED** |
| **EDR Threshold** | ✅ 100m | ✅ 100m | ✅ **MATCHED** |
| **Evaluation Pipeline** | ✅ Brief | ✅ More detailed | Porto lists specific scripts |
| **Hardware** | ✅ Generic | ✅ Generic | Both lack specifics |
| **Reproducibility** | ✅ Seed 42 | ✅ Seeds 42/43/44 | Porto has more seeds |
| **Scenario Aggregation** | ✅ Command | ✅ Command | ✅ **MATCHED** |

---

## Appendix Completeness

| Item | Beijing | Porto | Status |
|------|---------|-------|--------|
| **Figure List** | ✅ 12 figures | ✅ Comprehensive | Porto more detailed |
| **Scenario Assets** | ✅ Added (Oct 31) | ✅ Added (Oct 31) | ✅ **MATCHED** |
| **Aggregation Script** | ✅ Command | ✅ Command | ✅ **MATCHED** |
| **OD Matching Code** | ✅ Python snippet | ✅ Python snippet | ✅ **MATCHED** |
| **JSD Calculation** | ✅ Formula + bins | ✅ Formula only | Beijing more detailed |
| **Data Sources** | ✅ Counts | ✅ Counts | ✅ **MATCHED** |
| **Computational Details** | ✅ Brief | ✅ More detailed | Porto lists software |

---

## Recommendations

### Priority 1: Add Inference Speed Analysis (Both Documents)

Add a new subsection: **"6.X Inference Performance"** or **"9.X Computational Performance"**

**Metrics to include:**
- Generation time per trajectory (mean ± std)
- Throughput (trajectories/second)
- Beam search breakdown (time per step)
- Memory usage (GPU/CPU)
- Batch vs single trajectory efficiency
- Vanilla vs distilled comparison

**Data sources:**
- Profile `gene.py` with timing instrumentation
- Use `torch.cuda.Event()` for GPU timing
- Log memory with `torch.cuda.max_memory_allocated()`
- Measure on standardized hardware (specify GPU model)

### Priority 2: Harmonize Visualizations

**Beijing gaps:**
- Add multi-scenario trajectory grid references (if available)
- More detailed trajectory visualization examples

**Porto gaps:**
- Consider adding dedicated OD matching rate figure
- Consider adding performance radar chart
- Consider adding metrics heatmap

**Both:**
- Ensure all referenced figures exist and are accessible
- Use consistent naming conventions
- Include figure captions with interpretation

### Priority 3: Add Training Convergence Analysis

Add to Appendix or Section 5:
- Training loss curves (train vs val over 25 epochs)
- Wall-clock training time
- GPU utilization during training
- Memory footprint comparison

### Priority 4: Enhance Error Analysis

Add subsection under Section 6 or dedicated section:
- Failure case taxonomy (categorize by type)
- Spatial distribution of failures (heat map)
- Difficult OD pair analysis (which pairs fail most?)
- Trajectory visualizations of specific failures

### Priority 5: Document Model Specifications

Add to Section 1 or Appendix:
- Parameter count (total, by layer)
- Model size on disk
- Architecture diagram (if not already in LMTAD-Distillation.md)
- Inference memory requirements

---

## Summary

### ✅ Well-Matched Aspects

1. **Scenario-level analysis**: Both documents now have comprehensive, identically-structured scenario analyses
2. **Core evaluation metrics**: All key metrics (JSD, OD coverage, DTW, etc.) present in both
3. **Methodology**: Grid resolution, beam width, evaluation protocol all consistent
4. **Aggregation tooling**: Both use the same reusable script with identical outputs

### ⚠️ Gaps Present in BOTH Documents

1. **Inference speed**: No timing, throughput, or latency analysis
2. **Training performance**: No convergence curves or training time data
3. **Ablation studies**: Limited analysis of hyperparameter sensitivity
4. **Systematic error analysis**: No categorization or spatial analysis of failures
5. **Model specifications**: No parameter counts or size information

### 📊 Dataset-Specific Differences (Appropriate)

1. **Beijing "Why Vanilla Fails"** vs **Porto "Porto vs Beijing"** - makes sense given findings
2. **Beijing has more standalone figures** (heatmaps, radar charts) - appropriate for dramatic differences
3. **Porto has more trajectory grids** - appropriate for nuanced comparisons
4. **Porto includes Phase 2 context** - dataset-specific phased approach

### 🎯 Action Items

1. **Add inference speed analysis** to both documents (Priority 1)
2. **Harmonize visualization coverage** (Priority 2)
3. **Document training convergence** (Priority 3)
4. **Enhance error analysis** (Priority 4)
5. **Add model specifications** (Priority 5)

---

**Generated:** October 31, 2025  
**Comparison Version:** 1.0  
**Documents Compared:**
- Beijing: `/home/matt/Dev/HOSER/hoser-distill-optuna-6/EVALUATION_ANALYSIS.md` (882 lines)
- Porto: `/home/matt/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md` (975 lines)

---

## ✅ Search Method Ablation Study (November 2025)

**Status:** **COMPLETED** - Addresses computational performance and search method dependency gaps

**Study:** Beijing dataset beam search ablation (Issue #8)  
**Date:** November 5-6, 2025  
**Computation:** 26 hours (12 trajectory generation runs + 12 evaluation runs)

### Motivation

**Research Question:** Are distillation benefits dependent on search method (A* vs beam search)?

**Why this matters:**
- Original HOSER paper uses A* search (greedy heuristic-guided)
- Current implementation defaults to beam search (width=4)
- Cannot separate model quality improvements from search effectiveness without ablation

### Experimental Design

**Models Tested:**
- `distilled` (seed 42)
- `distilled_seed44` (independent seed)
- `vanilla` (baseline)

**Search Methods:**
1. **A* Search** - Original HOSER greedy algorithm (single-path exploration)
2. **Beam Search (width=4)** - Parallel exploration (current default)

**Configuration:**
- Dataset: Beijing
- Trajectories: 1,000 per model/OD-source/search-method
- Seed: 42 (reproducible)
- OD Sources: train, test
- Metrics: Speed (traj/s), forward passes, DTW_norm, Hausdorff_norm, EDR

**Total Runs:** 24 evaluations (3 models × 2 OD sources × 2 search methods)

### Key Findings: Search-Method Interaction Effect

#### 🔬 Performance Results

| Search Method | Distilled Speed | Vanilla Speed | Speedup Ratio |
|---------------|-----------------|---------------|---------------|
| **A* Search** | 0.30 traj/s | 0.05 traj/s | **6.0x faster** ⬆️ |
| **Beam Search (width=4)** | 1.79 traj/s | 2.46 traj/s | **0.73x (1.4x slower!)** ⬇️ |

**Critical Finding:** Distillation benefit **flips** depending on search method!

#### Forward Passes per Trajectory

| Search Method | Distilled | Vanilla | Exploration Efficiency |
|---------------|-----------|---------|------------------------|
| **A* Search** | ~500 passes | ~3100 passes | **6.2x fewer** ⬆️ |
| **Beam Search (width=4)** | ~43 passes | ~32 passes | **1.3x more** ⬇️ |

**Interpretation:**
- **A* benefits from distillation**: Better heuristic guidance → fewer backtracks → faster
- **Beam negates distillation advantage**: Parallel exploration reduces dependency on prediction quality

#### Trajectory Quality (Normalized Metrics)

**A* Search:**

| Model | DTW_norm (km/point) | Hausdorff_norm (km/point) | EDR |
|-------|---------------------|---------------------------|-----|
| Distilled (test) | 0.363 | 0.023 | 0.439 |
| Vanilla (test) | 0.347 | 0.029 | 0.483 |

**Beam Search:**

| Model | DTW_norm (km/point) | Hausdorff_norm (km/point) | EDR |
|-------|---------------------|---------------------------|-----|
| Distilled (test) | 0.559 | 0.025 | 0.492 |
| Vanilla (test) | 0.358 | 0.033 | 0.528 |

**Pattern:** Vanilla shows lower DTW_norm (better local trajectory quality) across both search methods

#### OD Destination Matching

| Model + Search | OD Match Rate |
|----------------|---------------|
| Distilled + A* | ~98% |
| Distilled + Beam | 85-86% |
| Vanilla + A* | 19% |
| Vanilla + Beam | 19% |

**Key Insight:** Distillation dramatically improves OD matching (4-5x better) regardless of search method

### Interpretation: Context-Dependent Benefits

#### ✅ Distillation Improves:
1. **OD Destination Matching** - 85-98% vs 19% (4-5x improvement)
2. **Realistic Trajectory Lengths** - Distilled generates appropriate-length paths
3. **A* Search Speed** - 6x faster with better heuristic guidance

#### ❌ Distillation Does NOT Improve:
1. **Beam Search Speed** - 1.4x slower with distilled model
2. **Local Trajectory Quality** - Vanilla has lower DTW_norm per point
3. **Universal Search Benefits** - Effect reverses depending on search method

#### 🎯 Recommendation Matrix

| Use Case | Recommended Configuration | Rationale |
|----------|--------------------------|-----------|
| **Real-time generation (speed priority)** | Vanilla + Beam (width=4) | Fastest (2.46 traj/s) |
| **High OD matching (quality priority)** | Distilled + Beam (width=4) | 85-86% OD match, realistic lengths |
| **Distilled model deployment** | Distilled + A* | Leverages distillation advantage (6x speedup) |
| **Trajectory diversity** | Any + Beam | Parallel exploration provides variety |

### Research Implications

**Strengthens Contribution:**
- ✅ Shows distillation has clear, measurable benefits (OD matching)
- ✅ Provides nuanced analysis (not universal improvement)
- ✅ Uses trajectory-length independent metrics (fair comparison)
- ✅ Addresses search method dependence concerns

**Honest Limitations:**
- ⚠️ Distillation benefit is context-dependent
- ⚠️ Search method choice significantly impacts performance
- ⚠️ Speed-quality tradeoffs must be considered for deployment

**Publication Value:**
- Demonstrates thorough experimental validation
- Shows understanding of method limitations
- Provides actionable deployment guidance
- Uses rigorous statistical methodology

### Cross-Dataset Validation

**Beijing Study:** Completed (documented here)

**Porto Study:** Tracked in Issue #44 (optional future work)
- Different baseline: Porto vanilla succeeds (88% OD match) vs Beijing vanilla fails (19%)
- Tests whether interaction persists when vanilla performs well
- Validates with Phase 2 optimized hyperparameters

**Decision:** Beijing ablation sufficient for publication, Porto validation strengthens generalizability claims

### Files Generated

**Trajectory Data:** (47MB total)
- 12 CSV files: `hoser-distill-optuna-6/gene/Beijing/seed42/*.csv`
- 12 performance JSON: `*_perf.json` with timing metrics

**Evaluation Results:**
- 12 evaluation directories with normalized metrics
- `results.json` with DTW_norm, Hausdorff_norm, EDR

**Scripts:**
- `run_beam_ablation.sh` - Automated ablation execution
- `complete_beam_ablation_with_normalized_metrics.sh` - Full study with normalization

### Integration with Other Improvements

**Synchronized with Issue #14 (Normalized Metrics):**
- Ablation uses trajectory-length independent metrics throughout
- Fair comparison between A* (shorter paths) and Beam (longer paths)
- Enables scientific rigor in search method comparison

**Complements Issue #16 (Paired Statistical Tests):**
- Provides performance data for future paired comparisons
- Documents variance sources (search method + model)
- Supports multi-factor statistical analysis

### Conclusion

**Resolution:** ✅ **Can now separate model quality from search effectiveness**

**Finding:** Distillation benefit is **search-method dependent**:
- A* leverages distillation for 6x speedup
- Beam search shows opposite pattern (vanilla 1.4x faster)
- OD matching improves with distillation regardless of search method

**Impact:** Provides deployment guidance based on use case priorities (speed vs OD matching vs trajectory quality)

