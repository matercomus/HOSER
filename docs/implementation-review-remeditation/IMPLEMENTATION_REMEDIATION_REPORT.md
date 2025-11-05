# Implementation Remediation Report

**Repository**: `/home/matt/Dev/HOSER` (HOSER Knowledge Distillation Research)  
**Date**: January 2026  
**Source**: Comprehensive Peer Review + Thesis Response Analysis  
**Purpose**: Address all repository-specific implementation issues identified in peer review

---

## Executive Summary

This report documents all implementation issues in the HOSER knowledge distillation research repository that require remediation. These are issues that either:
1. Were identified as "NOT APPLICABLE TO THESIS" (repository documentation/code only)
2. Require implementation changes even if documented in thesis
3. Are present in code but not in thesis manuscript

### Issue Distribution

**Total Repository Issues**: 35 (out of 41 total peer review issues)

**By Severity**:
- **CRITICAL**: 8 issues (must fix - threaten validity)
- **MAJOR**: 14 issues (seriously weaken conclusions)
- **MODERATE**: 8 issues (limit generalizability)
- **MINOR**: 5 issues (enhancements)

**By Feasibility**:
- **FIX**: 15 issues (can be directly fixed in code/docs)
- **MITIGATE**: 8 issues (cannot fully fix but can reduce impact)
- **DOCUMENT**: 7 issues (document limitation/clarify)
- **DEFER**: 5 issues (require new experiments/data)

**Estimated Total Effort**: 80-120 hours (2-3 weeks full-time)

### Critical Path

**Phase 1 (P0 - Blocking)**: 4 critical fixes (~20 hours)
1. Fix hyperparameter optimization confound (re-run experiments)
2. Fix Bonferroni correction implementation
3. Fix/document JSD binning documentation mismatch
4. Add vocabulary mapping validation

**Phase 2 (P1 - High Priority)**: 10 major fixes (~40 hours)
- Teacher baseline evaluation
- Ablation studies implementation
- Statistical test corrections
- Documentation updates

**Phase 3 (P2 - Enhancement)**: 13 moderate/minor fixes (~30 hours)
- Additional validations
- Documentation completeness
- Code refactoring

### Key Findings

**Most Critical Discovery**: Issue 1.1 (Hyperparameter Optimization Confound)
- Peer review correctly identified that vanilla baseline does NOT participate in Optuna hyperparameter search
- Thesis claims "Trial 0 = vanilla" but this appears to be a misunderstanding
- **Code evidence** (`tune_hoser.py` lines 215, 271, 279) shows all Optuna trials are distilled
- **Impact**: Comparison is unfair (optimized distilled vs non-optimized vanilla)
- **Required**: Re-run experiments with fair comparison

**Documentation-Code Divergence**:
- Multiple instances where documentation claims differ from code implementation
- JSD bins: docs say 50, code uses 100
- Bonferroni: docs claim correction, code doesn't implement
- Calibration: docs claim "calibrated uncertainty", no metrics computed

---

## GitHub Issue Creation Guide

**For AI Agents**: When creating GitHub issues for each of the 35 repository issues below, use this template structure:

```markdown
Title: Issue X.X: [Brief Title]

## Problem Statement
[Description of the issue from report]

## Severity & Feasibility
- **Severity**: {CRITICAL|MAJOR|MODERATE|MINOR}
- **Feasibility**: {FIX|MITIGATE|DOCUMENT|DEFER}
- **Effort**: {X} hours
- **Priority**: {P0|P1|P2|P3}

## Evidence
[File paths and line numbers from report]

## Required Changes
[Specific actions from report]

## Validation Steps
[Validation checklist from report]

## Dependencies
[Dependencies from report or "None"]

## Files to Modify
[List of files from report]

---
**Reference**: `docs/implementation-review-remeditation/IMPLEMENTATION_REMEDIATION_REPORT.md` - Issue X.X

Labels: [priority], [feasibility], [category]
Milestone: [Phase name]
```

**Example**: See Issue 1.1 below for a complete example with all fields populated.

---

## Section 1: CRITICAL Issues (Must Fix Before Publication)

### Issue 1.1: Hyperparameter Optimization Confound ⚠️ CRITICAL

**GitHub Issue**: #1 (created via GitHub MCP) | [View on GitHub](link)

**Severity**: CRITICAL  
**Feasibility**: DEFER (requires re-running experiments)  
**Complexity**: Complex (>40 hours)  
**Priority**: P0 (blocking publication)

#### Problem Statement

**What is wrong**: Vanilla baseline does NOT receive hyperparameter optimization, while distilled model gets 12 trials of intelligent CMA-ES search. This creates an unfair comparison where performance differences may stem from hyperparameter optimization rather than distillation.

**Why it matters**: This is the most fundamental flaw in the experimental design. Without a fair comparison, cannot attribute improvements to distillation vs better hyperparameters.

**Thesis vs Repository Divergence**:
- **Thesis claims** (lines 540, 785, 809): "Trial 0 = vanilla (λ=0)"
- **Code reality** (`tune_hoser.py`): All Optuna trials (0-11) are distilled; vanilla runs separately in Phase 0/3

#### Evidence from Code

**File**: `tune_hoser.py`

```python
# Line 215
# All trials are distillation trials (vanilla baseline runs separately in Phase 0)

# Line 271
# All trials use distillation (vanilla baseline runs separately in Phase 0)

# Line 279
config["wandb"]["run_name"] = f"trial_{trial.number:03d}_distilled"

# Line 303
base_seed + trial.number  # Seeds differ per trial
```

**Function**: `_run_vanilla_baseline()` (lines 758-865)
- Vanilla runs with FIXED hyperparameters
- Not part of Optuna study
- No hyperparameter search applied

**Configuration**: `config/Beijing.yaml`
- Optuna configuration only for distilled models
- No vanilla hyperparameter search space defined

#### Current Implementation

**What happens now**:
1. Vanilla baseline runs with default hyperparameters (learning rate, batch size, etc.)
2. Optuna launches 12 trials with CMA-ES sampler
3. All 12 trials use distillation with varying λ, τ, w
4. Best distilled trial is selected
5. Comparison: vanilla_default vs distilled_optimal

**The unfair advantage**: Distilled model gets 12 attempts with intelligent search; vanilla gets 1 attempt with defaults.

#### Required Changes

**Option A: Re-run Vanilla with Hyperparameter Search** (Recommended)

**Files to modify**:
- `tune_hoser.py`: Add vanilla baseline to Optuna study
- `config/Beijing.yaml`: Define vanilla search space (learning rate, batch size, weight decay, dropout)
- `config/Porto.yaml`: Same for Porto

**Steps**:
1. Create `vanilla_hyperparameter_search()` function
2. Define search space: learning rate [1e-5, 1e-3], batch size [32, 64, 128], weight decay [1e-5, 1e-3]
3. Run Optuna study for vanilla with same budget (12 trials)
4. Compare: vanilla_optimal vs distilled_optimal
5. Re-run all evaluations with fair baselines

**Pseudocode**:
```python
def study_vanilla_hyperparameters(config, optuna_config):
    """Run Optuna study for vanilla baseline."""
    
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
        
        # Force lambda = 0 (no distillation)
        config['distillation']['lambda'] = 0.0
        config['training']['learning_rate'] = lr
        config['training']['batch_size'] = batch_size
        config['training']['weight_decay'] = weight_decay
        
        # Train and evaluate
        val_acc = train_and_evaluate(config)
        return val_acc
    
    study = optuna.create_study(
        sampler=CmaEsSampler(),
        pruner=HyperbandPruner()
    )
    study.optimize(objective, n_trials=12)
    return study.best_params
```

**Option B: Document as Limitation** (If re-running infeasible)

If re-running experiments is not feasible:
1. Add limitation section to all documentation
2. Clearly state vanilla did not receive hyperparameter search
3. Acknowledge performance differences may be confounded
4. Reduce strength of claims about distillation effectiveness

**Files to modify**:
- `docs/EVALUATION_ANALYSIS.md`: Add prominent limitation section
- `docs/LMTAD-Distillation.md`: Clarify experimental design
- `README.md`: Add caveat about results interpretation

#### Validation Steps

**After fixing**:
1. Verify vanilla Optuna study completes successfully
2. Compare vanilla_optimal vs vanilla_default to quantify hyperparameter impact
3. Compare vanilla_optimal vs distilled_optimal for fair assessment
4. Re-compute all metrics with new baselines
5. Update all figures and tables

**Expected outcomes**:
- Vanilla performance may improve significantly
- Distillation benefit may be smaller than originally reported
- More accurate assessment of distillation contribution

#### Dependencies

**Blocks**:
- All evaluation comparisons (depend on fair baseline)
- All claims about distillation effectiveness
- Porto and Beijing results interpretation

**Requires**:
- Computational resources for re-training (~12 trials × 25 epochs × 2 datasets)
- Time: ~40-60 hours of compute + analysis

#### Mitigation Strategy

**If cannot re-run experiments**:

1. **Partial mitigation**: Run ablation on learning rate only
   - Fix batch size, weight decay at defaults
   - Sweep learning rate [1e-5, 5e-4, 1e-3, 5e-3]
   - Assess if vanilla is learning-rate limited

2. **Statistical control**: Report effect size of hyperparameter choices
   - Use Optuna's parameter importance analysis
   - Estimate how much variance is explained by λ vs other hyperparameters

3. **Transparent documentation**: 
   - State clearly: "Vanilla baseline did not receive hyperparameter optimization"
   - Add to limitations: "Performance differences may be partially due to hyperparameter optimization advantage"
   - Reduce claims: "Distillation with optimized hyperparameters outperforms vanilla with default hyperparameters"

#### Estimated Effort

**Full fix (Option A)**: 50-60 hours
- Code modification: 4 hours
- Experiment execution: 40 hours compute time
- Analysis and documentation: 10 hours

**Mitigation (Option B)**: 3-5 hours
- Documentation updates only

#### GitHub Issue Template

**AI Agent**: Create GitHub issue with this content:

```markdown
Title: Issue 1.1: Hyperparameter Optimization Confound

## Problem Statement
Vanilla baseline does NOT receive hyperparameter optimization, while distilled model gets 12 trials of intelligent CMA-ES search. This creates an unfair comparison where performance differences may stem from hyperparameter optimization rather than distillation.

## Severity & Feasibility
- **Severity**: CRITICAL
- **Feasibility**: DEFER
- **Effort**: 50-60 hours (full fix) or 3-5 hours (documentation)
- **Priority**: P0

## Evidence
- File: `tune_hoser.py` lines 215, 271, 279, 303
- Trial 0 is NOT vanilla - it's a distilled trial
- Vanilla runs separately in Phase 0/3, not in Optuna study

## Required Changes

**Option A: Re-run with Fair Comparison** (50-60 hours)
- Modify `tune_hoser.py` to include vanilla in Optuna study
- Define vanilla search space (learning rate, batch size, weight decay)
- Run 12 trials for vanilla baseline
- Compare: vanilla_optimal vs distilled_optimal

**Option B: Document Limitation** (3-5 hours)
- Add prominent limitation section
- Acknowledge unfair comparison
- Reduce strength of claims

## Validation Steps
- [ ] Vanilla Optuna study completes
- [ ] Compare vanilla_optimal vs vanilla_default
- [ ] Re-evaluate all metrics with fair baselines
- [ ] Update documentation

## Dependencies
None (but blocks all other result interpretations)

## Files to Modify
- `tune_hoser.py`
- `config/Beijing.yaml`
- `config/Porto.yaml`
- All evaluation documentation

---
**Reference**: `docs/implementation-review-remeditation/IMPLEMENTATION_REMEDIATION_REPORT.md` - Issue 1.1

Labels: `P0-Critical`, `defer`, `experimental`
Milestone: Phase 1: Critical Fixes
```

**GitHub MCP Command**:
```python
create_issue(
    title="Issue 1.1: Hyperparameter Optimization Confound",
    body=<template_above>,
    labels=["P0-Critical", "defer", "experimental"],
    milestone="Phase 1: Critical Fixes"
)
```

---

### Issue 1.2: Bonferroni Correction Misrepresentation ⚠️ CRITICAL

**Severity**: CRITICAL  
**Feasibility**: FIX  
**Complexity**: Medium (4-6 hours)  
**Priority**: P0

#### Problem Statement

**What is wrong**: Documentation claims Bonferroni correction (α = 0.001) but code uses uncorrected α = 0.05.

**Why it matters**: Misrepresents statistical rigor. Multiple testing without correction inflates false discovery rate. Readers are misled about significance thresholds.

**Thesis status**: NOT APPLICABLE (Wang analysis not reported in thesis)

#### Evidence from Code

**File**: `tools/analyze_wang_results.py`

```python
# Line 558
chi2, p_value = stats.chi2_contingency(contingency)[:2]

# Line 568
"significant": p_value < 0.05  # Uses 0.05, NOT corrected threshold
```

**File**: `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md`

```markdown
# Line 126
Statistical significance determined with α = 0.001 (Bonferroni correction)
```

**CONTRADICTION**: Documentation claims α = 0.001 with Bonferroni, code uses α = 0.05 without correction.

#### Current Implementation

**What code does**:
1. Runs chi-square test for each comparison
2. Compares p-value to 0.05 threshold
3. No adjustment for multiple comparisons
4. Labels results as "significant" or "not significant"

**Scale of problem**:
- Wang analysis: 13 statistical tests
- Expected false discoveries: 13 × 0.05 = 0.65
- With Bonferroni: α_adjusted = 0.05 / 13 = 0.00385

#### Required Changes

**File**: `tools/analyze_wang_results.py`

**Step 1: Count total comparisons**

```python
# After line 525 (inside analyze_abnormality_patterns function)
num_comparisons = 0
# Count all chi-square tests that will be performed
for dataset in datasets:
    for split in ['train_od', 'test_od']:
        num_comparisons += 1  # One test per dataset-split combo
```

**Step 2: Apply Bonferroni correction**

```python
# Replace line 568
# OLD:
"significant": p_value < 0.05

# NEW:
alpha_bonferroni = 0.05 / num_comparisons
"significant": p_value < alpha_bonferroni,
"alpha_bonferroni": alpha_bonferroni,
"raw_p_value": p_value,
"adjusted_p_value": min(p_value * num_comparisons, 1.0)  # Bonferroni adjustment
```

**Step 3: Update output JSON schema**

Add fields to `wang_results_aggregated.json`:
- `num_comparisons`: Total tests performed
- `alpha_bonferroni`: Corrected threshold
- `raw_p_value`: Original p-value
- `adjusted_p_value`: Bonferroni-adjusted p-value

**Step 4: Update documentation**

**File**: `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md`

```markdown
# Replace line 126
Statistical significance determined with Bonferroni correction:
- Total comparisons: 13
- Uncorrected α: 0.05
- Bonferroni-corrected α: 0.05/13 = 0.00385
- Reported p-values are Bonferroni-adjusted
```

#### Alternative: FDR Correction (Recommended)

Bonferroni is conservative. Consider False Discovery Rate (FDR) correction instead:

```python
from statsmodels.stats.multitest import multipletests

# Collect all p-values
p_values = [result['p_value'] for result in all_results]

# Apply Benjamini-Hochberg FDR correction
reject, pvals_corrected, _, _ = multipletests(
    p_values, 
    alpha=0.05, 
    method='fdr_bh'
)

# Update results with FDR-corrected p-values
for result, pval_corrected, is_significant in zip(all_results, pvals_corrected, reject):
    result['adjusted_p_value'] = pval_corrected
    result['significant'] = is_significant
```

#### Validation Steps

1. Re-run `python tools/analyze_wang_results.py`
2. Verify output includes:
   - `num_comparisons` field
   - `raw_p_value` and `adjusted_p_value`
   - Correct significance determination
3. Check how many results change from significant to not significant
4. Update all result tables with adjusted p-values

#### Dependencies

**Blocks**: Wang analysis result interpretation

**Requires**: None (standalone fix)

#### Estimated Effort

**Implementation**: 3 hours
- Code changes: 1 hour
- Testing: 1 hour
- Documentation updates: 1 hour

**Re-analysis**: 1 hour
- Re-run analysis script
- Update result tables

**Total**: 4 hours

---

### Issue 1.3: Translation Quality Confound ⚠️ CRITICAL

**Severity**: CRITICAL  
**Feasibility**: MITIGATE  
**Complexity**: Medium (6-8 hours)  
**Priority**: P1

#### Problem Statement

**What is wrong**: Cross-dataset evaluation (Beijing → BJUT) uses 79% translation quality without filtering low-quality translations.

**Why it matters**: Low translation quality introduces noise that confounds model comparison. Cannot determine if performance differences are due to models or translation artifacts.

**Thesis status**: NOT APPLICABLE (cross-dataset evaluation marked [TO BE COMPLETED])

#### Evidence from Code

**Likely files** (need to verify):
- Translation/cross-network evaluation scripts
- BJUT dataset preprocessing
- Translation quality metrics

**Problem**: Trajectories translated from Beijing road network to BJUT road network have only 79% average translation quality, meaning 21% of translated trajectories have significant errors.

#### Current Implementation

**Assumed current behavior**:
1. Train model on Beijing dataset
2. Translate Beijing trajectories to BJUT network (79% quality)
3. Evaluate on translated trajectories without filtering
4. Poor performance attributed to model vs translation artifacts

#### Required Changes

**Option A: Filter by Translation Quality** (Recommended)

**Step 1: Add quality filtering**

```python
def filter_by_translation_quality(trajectories, quality_scores, threshold=0.95):
    """Keep only high-quality translations."""
    filtered = [
        (traj, score) 
        for traj, score in zip(trajectories, quality_scores) 
        if score >= threshold
    ]
    print(f"Kept {len(filtered)}/{len(trajectories)} trajectories (>= {threshold} quality)")
    return [t[0] for t in filtered]
```

**Step 2: Report quality distribution**

```python
def analyze_translation_quality(quality_scores):
    """Report translation quality statistics."""
    print(f"Translation Quality Statistics:")
    print(f"  Mean: {np.mean(quality_scores):.3f}")
    print(f"  Median: {np.median(quality_scores):.3f}")
    print(f"  Min: {np.min(quality_scores):.3f}")
    print(f"  Max: {np.max(quality_scores):.3f}")
    print(f"  % >= 0.95: {100 * np.mean(quality_scores >= 0.95):.1f}%")
    print(f"  % >= 0.90: {100 * np.mean(quality_scores >= 0.90):.1f}%")
```

**Step 3: Ablation by quality threshold**

Evaluate at multiple thresholds to assess impact:
- No filtering (79% average)
- ≥ 0.90 quality
- ≥ 0.95 quality
- ≥ 0.99 quality

Plot performance vs quality threshold to show confound.

**Option B: Document as Limitation**

If filtering reduces sample size too much:
1. Report translation quality distribution
2. Analyze correlation between quality and performance
3. Document as limitation
4. Recommend future work with higher-quality translation

#### Validation Steps

1. Verify translation quality scores are available
2. Compute quality distribution statistics
3. Re-run evaluation with filtered dataset
4. Compare results: all data vs high-quality only
5. Document sensitivity to translation quality

#### Dependencies

**Requires**: Access to translation quality scores per trajectory

**Blocks**: Cross-dataset generalization claims

#### Mitigation Strategy

**If quality scores unavailable**:
1. Clearly document 79% average quality
2. State: "Results may be confounded by translation artifacts"
3. Recommend future work with improved translation or native evaluation
4. Reduce strength of cross-dataset claims

#### Estimated Effort

**With quality scores available**: 6 hours
- Code modification: 2 hours
- Re-run evaluation: 2 hours
- Analysis and documentation: 2 hours

**Without quality scores**: 2 hours
- Documentation updates only

---

### Issue 1.4: JSD Binning Documentation Mismatch ⚠️ CRITICAL

**Severity**: CRITICAL  
**Feasibility**: FIX  
**Complexity**: Quick (1 hour)  
**Priority**: P0

#### Problem Statement

**What is wrong**: Documentation claims 50 bins for JSD computation, but code uses 100 bins.

**Why it matters**: Misleads readers about implementation details. Bin count affects JSD values and comparison fairness.

**Thesis status**: Thesis is CORRECT (documents 100 bins). Repository documentation is WRONG.

#### Evidence

**Code**: `evaluation.py` line 578
```python
bins = np.linspace(0, real_max, 100).tolist()  # 100 bins
```

**Thesis**: Lines 1669, 1684, 1716
> "We create histograms with **100 bins** spanning [0, max(Dreal)]"

**Repository docs**: `docs/EVALUATION_ANALYSIS.md` line 143
> "50 equal-width bins"

**VERDICT**: Code and thesis are consistent (100 bins). Repository doc is wrong (says 50).

#### Required Changes

**File**: `docs/EVALUATION_ANALYSIS.md`

**Find and replace**:
```markdown
# OLD (line 143):
50 equal-width bins

# NEW:
100 equal-width bins
```

Search for all mentions of "50 bins" and update to "100 bins".

**Files to check**:
- `docs/EVALUATION_ANALYSIS.md`
- `docs/LMTAD-Distillation.md`
- `README.md`
- Any evaluation documentation files

#### Validation Steps

1. Search entire repository for "50 bins"
2. Replace all instances with "100 bins"
3. Verify code still uses 100 bins (no accidental changes)
4. Check that documentation matches code

#### Estimated Effort

**Total**: 1 hour
- Search and replace: 15 minutes
- Verification: 15 minutes
- Documentation review: 30 minutes

---

### Issue 1.5: Data Leakage in Wang Baseline Computation ⚠️ CRITICAL

**Severity**: CRITICAL  
**Feasibility**: FIX  
**Complexity**: Medium (4-6 hours)  
**Priority**: P0

#### Problem Statement

**What is wrong**: Wang abnormality baseline computation pools train+test data when computing OD-pair-specific baselines, causing data leakage.

**Why it matters**: Test set information leaks into baseline computation, artificially inflating detection accuracy. Results are invalid.

**Thesis status**: NOT APPLICABLE (Wang analysis not in thesis)

#### Evidence from Code

**File**: `tools/compute_trajectory_baselines.py`

**Suspected issue**:
```python
# Pools all trajectories (train + test) to compute OD baselines
all_trajectories = load_train_trajectories() + load_test_trajectories()

for od_pair in unique_od_pairs:
    od_trajectories = filter_by_od(all_trajectories, od_pair)
    baseline_stats = compute_stats(od_trajectories)  # Length, time mean/std
```

**The problem**: When computing "normal" length/time for an OD pair, test trajectories are included. Then test trajectories are compared to this baseline. This is circular.

#### Current Implementation

**What happens**:
1. Load all trajectories (train + test)
2. Group by OD pair
3. Compute mean length, std length, mean time, std time per OD
4. Use these as "normal" baselines
5. Flag trajectories as abnormal if: length > mean + 5km OR time > mean + 5min
6. Evaluate on test set using baselines computed from train+test

**The leak**: Test trajectory statistics influence the baseline, then test trajectories are evaluated against that baseline.

#### Required Changes

**Step 1: Separate train/test baseline computation**

```python
def compute_od_baselines_train_only(train_trajectories):
    """Compute baselines using ONLY training data."""
    baselines = {}
    
    for od_pair in get_unique_od_pairs(train_trajectories):
        od_trajs = filter_by_od(train_trajectories, od_pair)
        
        if len(od_trajs) < 5:  # Minimum sample size
            continue
            
        baselines[od_pair] = {
            'length_mean': np.mean([t.length for t in od_trajs]),
            'length_std': np.std([t.length for t in od_trajs]),
            'time_mean': np.mean([t.duration for t in od_trajs]),
            'time_std': np.std([t.duration for t in od_trajs]),
            'n_samples': len(od_trajs)
        }
    
    return baselines
```

**Step 2: Handle unseen OD pairs in test set**

```python
def detect_abnormality_with_fallback(trajectory, baselines, global_baseline):
    """Detect abnormality, fallback to global baseline for unseen OD pairs."""
    od_pair = (trajectory.origin, trajectory.dest)
    
    if od_pair in baselines:
        # Use OD-specific baseline
        baseline = baselines[od_pair]
    else:
        # Fallback to global baseline (computed from train only)
        baseline = global_baseline
        
    # Apply Wang thresholds
    length_abnormal = trajectory.length > baseline['length_mean'] + 5.0  # 5km
    time_abnormal = trajectory.duration > baseline['time_mean'] + 5.0  # 5min
    
    return length_abnormal or time_abnormal
```

**Step 3: Compute global fallback baseline**

```python
def compute_global_baseline(train_trajectories):
    """Global baseline for OD pairs not seen in training."""
    return {
        'length_mean': np.mean([t.length for t in train_trajectories]),
        'length_std': np.std([t.length for t in train_trajectories]),
        'time_mean': np.mean([t.duration for t in train_trajectories]),
        'time_std': np.std([t.duration for t in train_trajectories])
    }
```

#### Validation Steps

1. Re-compute baselines using train-only data
2. Evaluate on test set with train-derived baselines
3. Compare old (leaked) vs new (clean) results
4. Report how many test OD pairs were unseen in training
5. Quantify impact of data leakage on accuracy

**Expected changes**:
- Detection accuracy will likely decrease
- More test trajectories will use global baseline
- Results will be more conservative but valid

#### Dependencies

**Blocks**: Wang abnormality detection results

**Requires**: Access to train/test split information

#### Estimated Effort

**Implementation**: 4 hours
- Code refactoring: 2 hours
- Re-run analysis: 1 hour
- Documentation: 1 hour

**Validation**: 2 hours
- Compare old vs new results
- Quantify leakage impact

**Total**: 6 hours

---

### Issue 1.6: Vocabulary Mapping Unvalidated ⚠️ CRITICAL

**Severity**: CRITICAL  
**Feasibility**: DOCUMENT + Partial FIX  
**Complexity**: Medium (6-10 hours)  
**Priority**: P1

#### Problem Statement

**What is wrong**: Many-to-one road→grid token mapping introduces systematic artifacts that are not validated or controlled.

**Why it matters**: Cannot determine if improvements come from knowledge transfer or mapping artifacts.

**Thesis status**: ACKNOWLEDGED (requires validation paragraph)

#### Evidence from Code

**Files**:
- `critics/grid_mapper.py`: line 98 - mapping implementation
- `tools/map_road_networks.py`: lines 173-194 - tracks many-to-one but not reported
- `docs/LMTAD-Distillation.md`: lines 679-685, 798-801 - acknowledges mapping but no validation

**Problem**: 
- Beijing: 40,060 roads → 51,663 grid cells (many-to-one mapping)
- Multiple roads share same grid token
- Teacher probabilities for grid token distributed across multiple candidate roads
- No validation that this doesn't create false distillation signals

#### Current Implementation

**Mapping algorithm**:
```python
def map_road_to_grid(road_centroid, grid_resolution):
    """Map road centroid to grid cell."""
    grid_x = int(road_centroid.x / grid_resolution)
    grid_y = int(road_centroid.y / grid_resolution)
    grid_token_id = grid_y * num_cols + grid_x
    return grid_token_id
```

**Result**: Multiple nearby roads get same grid token.

#### Required Changes

**Step 1: Compute and report mapping statistics**

```python
def analyze_vocabulary_mapping(road_network, grid_resolution):
    """Analyze many-to-one mapping statistics."""
    mapping = {}  # grid_token -> list of roads
    
    for road_id, road in road_network.items():
        grid_token = map_road_to_grid(road.centroid, grid_resolution)
        if grid_token not in mapping:
            mapping[grid_token] = []
        mapping[grid_token].append(road_id)
    
    # Statistics
    roads_per_cell = [len(roads) for roads in mapping.values()]
    
    print(f"Vocabulary Mapping Statistics:")
    print(f"  Total roads: {len(road_network)}")
    print(f"  Total grid cells: {len(mapping)}")
    print(f"  Roads per cell - mean: {np.mean(roads_per_cell):.2f}")
    print(f"  Roads per cell - median: {np.median(roads_per_cell):.0f}")
    print(f"  Roads per cell - max: {np.max(roads_per_cell)}")
    print(f"  % cells with 1 road: {100 * np.mean(np.array(roads_per_cell) == 1):.1f}%")
    print(f"  % cells with >5 roads: {100 * np.mean(np.array(roads_per_cell) > 5):.1f}%")
    
    # Spatial distribution
    many_to_one_cells = {k: v for k, v in mapping.items() if len(v) > 1}
    print(f"  Many-to-one cells: {len(many_to_one_cells)} ({100*len(many_to_one_cells)/len(mapping):.1f}%)")
    
    return mapping, roads_per_cell
```

**Step 2: Correlation analysis**

Test if many-to-one cases benefit more from distillation:
```python
def analyze_distillation_benefit_by_mapping(results, mapping):
    """Check if distillation benefit correlates with many-to-one ratio."""
    benefits = []
    roads_per_cell_list = []
    
    for road_id in results.keys():
        grid_token = get_grid_token(road_id)
        roads_in_cell = len(mapping[grid_token])
        
        vanilla_acc = results[road_id]['vanilla_accuracy']
        distilled_acc = results[road_id]['distilled_accuracy']
        benefit = distilled_acc - vanilla_acc
        
        benefits.append(benefit)
        roads_per_cell_list.append(roads_in_cell)
    
    correlation = np.corrcoef(benefits, roads_per_cell_list)[0, 1]
    print(f"Correlation between distillation benefit and roads-per-cell: {correlation:.3f}")
    
    if abs(correlation) > 0.3:
        print("WARNING: Strong correlation suggests mapping artifacts!")
    
    return correlation
```

**Step 3: Ablation with randomized mapping**

```python
def ablation_randomized_mapping(model, data, mapping):
    """Test with shuffled grid mappings to control for artifacts."""
    # Shuffle road→grid assignments randomly
    roads = list(mapping.keys())
    grid_tokens = list(mapping.values())
    np.random.shuffle(grid_tokens)
    randomized_mapping = dict(zip(roads, grid_tokens))
    
    # Train with randomized mapping
    model_random = train_with_mapping(data, randomized_mapping)
    
    # Compare: real mapping vs random mapping
    print("Real mapping accuracy:", model.val_accuracy)
    print("Random mapping accuracy:", model_random.val_accuracy)
    
    if model_random.val_accuracy > model.val_accuracy * 0.9:
        print("WARNING: Random mapping works almost as well - suggests artifacts!")
```

#### Validation Steps

1. Run mapping analysis on Beijing and Porto datasets
2. Report statistics in documentation
3. Compute distillation benefit correlation
4. If feasible, run randomized mapping ablation
5. Document findings

#### Mitigation Strategy

**If ablations are infeasible**:
1. Report mapping statistics prominently
2. Acknowledge as limitation: "Many-to-one mapping may introduce artifacts"
3. Recommend future work with 1:1 mapping or validation
4. Be conservative in knowledge transfer claims

#### Estimated Effort

**Analysis only**: 4 hours
- Compute statistics: 2 hours
- Correlation analysis: 1 hour
- Documentation: 1 hour

**With ablation**: 10 hours
- Above + randomized mapping experiment: 6 hours

---

### Issue 1.7: Missing Calibration Metrics ⚠️ CRITICAL

**Severity**: CRITICAL  
**Feasibility**: FIX or REMOVE CLAIM  
**Complexity**: Medium (6-8 hours to fix, 1 hour to remove claim)  
**Priority**: P1

#### Problem Statement

**What is wrong**: Documentation claims "calibrated uncertainty" but no calibration metrics (ECE, Brier score) are computed.

**Why it matters**: Unverified claim misleads readers about model capabilities.

**Thesis status**: NOT APPLICABLE (no calibration claims in thesis)

#### Evidence

**Documentation**: `docs/LMTAD-Distillation.md` line 84
> "calibrated uncertainty"

**Code**: No ECE, Brier score, or reliability diagram implementation found

**Evaluation**: `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md` lines 833-910
- Reports accuracy, MAPE, but no calibration metrics

#### Required Changes

**Option A: Implement Calibration Metrics** (Recommended)

```python
def compute_calibration_metrics(predictions, labels, n_bins=10):
    """Compute ECE and Brier score."""
    probs = predictions.softmax(dim=-1).max(dim=-1)[0]  # Confidence scores
    correct = (predictions.argmax(dim=-1) == labels).float()
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if bin_mask.sum() > 0:
            bin_accuracy = correct[bin_mask].mean().item()
            bin_confidence = probs[bin_mask].mean().item()
            bin_size = bin_mask.sum().item()
            ece += (bin_size / len(probs)) * abs(bin_accuracy - bin_confidence)
    
    # Brier Score
    one_hot_labels = F.one_hot(labels, num_classes=predictions.size(-1))
    brier_score = ((predictions.softmax(dim=-1) - one_hot_labels) ** 2).sum(dim=-1).mean().item()
    
    return {
        'ece': ece,
        'brier_score': brier_score,
        'mean_confidence': probs.mean().item(),
        'accuracy': correct.mean().item()
    }
```

**Add to evaluation pipeline**:
```python
# In evaluation loop
calibration_vanilla = compute_calibration_metrics(vanilla_predictions, labels)
calibration_distilled = compute_calibration_metrics(distilled_predictions, labels)

print(f"Vanilla - ECE: {calibration_vanilla['ece']:.4f}, Brier: {calibration_vanilla['brier_score']:.4f}")
print(f"Distilled - ECE: {calibration_distilled['ece']:.4f}, Brier: {calibration_distilled['brier_score']:.4f}")
```

**Reliability diagram**:
```python
def plot_reliability_diagram(predictions, labels, model_name, save_path):
    """Plot predicted confidence vs actual accuracy."""
    probs = predictions.softmax(dim=-1).max(dim=-1)[0]
    correct = (predictions.argmax(dim=-1) == labels).float()
    
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    
    for i in range(n_bins):
        bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if bin_mask.sum() > 0:
            bin_accuracies.append(correct[bin_mask].mean().item())
            bin_confidences.append(probs[bin_mask].mean().item())
            bin_sizes.append(bin_mask.sum().item())
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(bin_confidences, bin_accuracies, s=bin_sizes, alpha=0.6, label=model_name)
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Actual Accuracy')
    plt.title(f'Reliability Diagram - {model_name}')
    plt.legend()
    plt.savefig(save_path)
```

**Option B: Remove Claim** (If fixing not feasible)

```markdown
# Remove from docs/LMTAD-Distillation.md line 84
# DELETE: "calibrated uncertainty"

# Or replace with:
"improved prediction confidence (calibration not measured)"
```

#### Validation Steps

1. Implement ECE and Brier score computation
2. Add to evaluation pipeline
3. Generate reliability diagrams
4. Compare vanilla vs distilled calibration
5. Report metrics in evaluation documentation

#### Estimated Effort

**Implement metrics**: 6 hours
- Code implementation: 3 hours
- Integration: 2 hours
- Documentation: 1 hour

**Remove claim**: 1 hour

---

### Issue 1.8: Beam Search Evaluation Dependence ⚠️ CRITICAL

**Severity**: CRITICAL  
**Feasibility**: DOCUMENT + Partial FIX  
**Complexity**: Medium (10-15 hours for full ablation, 2 hours for documentation)  
**Priority**: P1

#### Problem Statement

**What is wrong**: All evaluation uses beam width b=4 without justification or ablation. Results may depend on search parameters rather than model quality.

**Why it matters**: Cannot distinguish between model quality and search quality. Different models may have different optimal beam widths.

**Thesis status**: ACKNOWLEDGED (requires justification)

#### Evidence

**Code**: Generation uses beam search with b=4
**Problem**: No ablation testing b=1,2,4,8,16

**Issue**: 
- Greedy decoding (b=1) would show per-step model quality
- Larger beams may help worse models more than better models
- Metrics (especially OD match rate) sensitive to beam width

#### Required Changes

**Step 1: Beam width ablation**

```python
def evaluate_beam_width_sensitivity(model, test_data, beam_widths=[1, 2, 4, 8, 16]):
    """Test model performance across beam widths."""
    results = {}
    
    for beam_width in beam_widths:
        print(f"Evaluating with beam width = {beam_width}")
        
        metrics = generate_and_evaluate(
            model=model,
            data=test_data,
            beam_width=beam_width
        )
        
        results[beam_width] = {
            'od_match_rate': metrics['od_match_rate'],
            'jsd_distance': metrics['jsd_distance'],
            'generation_time': metrics['generation_time']
        }
    
    # Plot sensitivity
    plot_beam_width_sensitivity(results, model_name=model.name)
    
    return results
```

**Step 2: Compare vanilla vs distilled sensitivity**

```python
def compare_beam_sensitivity(vanilla_model, distilled_model, test_data):
    """Check if distillation benefit varies with beam width."""
    beam_widths = [1, 2, 4, 8, 16]
    
    vanilla_results = evaluate_beam_width_sensitivity(vanilla_model, test_data, beam_widths)
    distilled_results = evaluate_beam_width_sensitivity(distilled_model, test_data, beam_widths)
    
    # Compute distillation benefit per beam width
    benefits = {}
    for bw in beam_widths:
        od_benefit = distilled_results[bw]['od_match_rate'] - vanilla_results[bw]['od_match_rate']
        jsd_benefit = vanilla_results[bw]['jsd_distance'] - distilled_results[bw]['jsd_distance']
        benefits[bw] = {'od_benefit': od_benefit, 'jsd_benefit': jsd_benefit}
    
    print("Distillation benefit by beam width:")
    for bw, b in benefits.items():
        print(f"  b={bw}: OD +{b['od_benefit']:.1%}, JSD -{b['jsd_benefit']:.3f}")
    
    # Check if benefit is beam-dependent
    od_benefits = [b['od_benefit'] for b in benefits.values()]
    if max(od_benefits) / min(od_benefits) > 2.0:
        print("WARNING: Distillation benefit strongly depends on beam width!")
```

**Step 3: Documentation**

Add to `docs/EVALUATION_ANALYSIS.md`:
```markdown
## Beam Search Configuration

All evaluations use beam search with width b=4 for fair comparison between models.

### Beam Width Sensitivity Analysis

| Beam Width | Vanilla OD Match | Distilled OD Match | Benefit |
|------------|------------------|-------------------|---------|
| b=1 (greedy) | X.X% | Y.Y% | +Z.Z% |
| b=2 | X.X% | Y.Y% | +Z.Z% |
| b=4 | X.X% | Y.Y% | +Z.Z% |
| b=8 | X.X% | Y.Y% | +Z.Z% |

**Finding**: Distillation benefit is [consistent/varies] across beam widths, suggesting improvements stem from [model quality/search efficiency].
```

#### Mitigation Strategy

**If full ablation infeasible**:
1. Document that b=4 is used for all models
2. State: "Both vanilla and distilled use identical beam width for fair comparison"
3. Acknowledge limitation: "Metrics may depend on beam search configuration"
4. Note as future work: "Systematic beam width ablation"

#### Estimated Effort

**Full ablation**: 12 hours
- Implementation: 3 hours
- Experiments: 6 hours
- Analysis and plots: 3 hours

**Documentation only**: 2 hours

---

## Section 2: MAJOR Issues (Seriously Weaken Conclusions)

### Issue 2.1: No Teacher Baseline ⚠️ MAJOR

**Severity**: MAJOR  
**Feasibility**: DEFER (requires significant development)  
**Complexity**: Complex (20-30 hours)  
**Priority**: P2

#### Problem Statement

**What is wrong**: LM-TAD teacher is never evaluated as trajectory generator. Core claim is "teacher transfers spatial knowledge" but teacher quality is unknown.

**Why it matters**: Cannot validate fundamental premise. Teacher may be worse than student for trajectory prediction.

**Thesis status**: ACKNOWLEDGED (architectural incompatibility noted)

#### Evidence from Code

**Files**:
- `critics/lmtad_teacher.py`: Wrapper only, provides probabilities but no standalone evaluation
- No teacher evaluation scripts found

**Missing evaluations**:
1. Teacher next-road prediction accuracy
2. Teacher OD completion rate
3. Teacher trajectory generation capability
4. Three-way comparison: vanilla < teacher < distilled?

#### Current Implementation

**What code does**:
- Teacher provides soft probability distributions during training
- Teacher is frozen (no updates)
- Teacher used only for KL divergence computation
- Never evaluated on trajectory prediction task

#### Required Changes

**Challenge**: LM-TAD uses different vocabulary (grid tokens) and architecture (transformer).

**Option A: Adapt Teacher for Prediction Task** (Complex)

Would require:
1. Modify LM-TAD to use road vocabulary
2. Adapt architecture for autoregressive generation
3. Handle different input formats
4. Significant engineering effort

**Option B: Evaluate Teacher in Grid Space** (Easier)

```python
def evaluate_teacher_grid_space(teacher_model, test_data_grid):
    """Evaluate teacher on grid-tokenized trajectories."""
    metrics = {
        'next_token_accuracy': 0.0,
        'perplexity': 0.0,
        'completion_rate': 0.0
    }
    
    for trajectory in test_data_grid:
        # Teacher predicts next grid token
        predictions = teacher_model(trajectory[:-1])
        
        # Compute metrics
        accuracy = (predictions.argmax(-1) == trajectory[1:]).float().mean()
        metrics['next_token_accuracy'] += accuracy.item()
    
    metrics['next_token_accuracy'] /= len(test_data_grid)
    
    return metrics
```

**Option C: Document as Limitation** (Recommended)

Add to documentation:
```markdown
## Teacher Model Evaluation

LM-TAD teacher model uses different architecture (transformer) and vocabulary (grid tokens) compared to HOSER student (GNN, road segments). Direct comparison is non-trivial due to:

1. **Vocabulary mismatch**: 51K grid cells vs 40K road segments
2. **Architecture difference**: Causal transformer vs hierarchical graph model
3. **Task mismatch**: Trained for anomaly detection vs trajectory prediction

**Limitation**: Teacher standalone performance on trajectory prediction task is not evaluated. Teacher quality is inferred from source task (anomaly detection) performance reported in LM-TAD paper.

**Future work**: Adapt LM-TAD for autoregressive trajectory generation or evaluate in grid space.
```

#### Mitigation Strategy

1. Document architectural incompatibility clearly
2. Reference LM-TAD paper for teacher quality on source task
3. Note as limitation
4. Recommend future work for fair teacher evaluation

#### Estimated Effort

**Full evaluation**: 25+ hours (requires significant development)

**Documentation**: 2 hours

---


