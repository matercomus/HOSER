# Implementation Fixes Checklist

**Quick Reference for AI Agents**  
**Repository**: `/home/matt/Dev/HOSER`  
**Purpose**: Actionable checklist for fixing all 35 repository-specific issues

---

## GitHub Issue Tracking

**AI Agents**: Use GitHub MCP to create and track all checklist items as GitHub issues.

### Quick Setup
```bash
# AI agents should run these GitHub MCP commands:

# 1. Create milestones
create_milestone("Phase 1: Critical Fixes", "Week 1 - Must fix before publication", "2026-02-07")
create_milestone("Phase 2: Major Issues", "Week 2 - High priority", "2026-02-14")
create_milestone("Phase 3: Moderate/Minor", "Week 3 - Enhancements", "2026-02-21")

# 2. Create priority labels
create_label("P0-Critical", "d73a4a", "Must fix before publication")
create_label("P1-Major", "ff9900", "Seriously weakens conclusions")
create_label("P2-Moderate", "ffcc00", "Limits generalizability")
create_label("P3-Minor", "66ff66", "Enhancement opportunity")

# 3. Create feasibility labels
create_label("fix", "0366d6", "Can be directly fixed")
create_label("mitigate", "8b60d6", "Can reduce impact")
create_label("document", "cccccc", "Document limitation")
create_label("defer", "8b4513", "Requires new experiments")

# 4. Create category labels
create_label("statistical", "006b75", "Statistical rigor")
create_label("experimental", "7057ff", "Needs experiments")
create_label("documentation", "0e8a16", "Documentation only")
create_label("code-fix", "1d76db", "Code implementation")
```

### Creating Issues from Checklist

**AI Agent Workflow**:
For each checklist item below, create a GitHub issue using this template:

```python
# Pseudocode for AI agents
for issue in checklist:
    github_mcp.create_issue(
        title=f"Issue {issue.number}: {issue.title}",
        body=generate_issue_body(issue),  # From IMPLEMENTATION_REMEDIATION_REPORT.md
        labels=[issue.priority_label, issue.feasibility_label, issue.category_label],
        milestone=issue.phase_milestone,
        assignees=["ai-agent"]  # or specific developer
    )
```

**Linking Issues to Checklist**:
Once issues are created, update checklist items with GitHub issue numbers:
- [ ] **Issue 1.1** → GitHub Issue #X
- [ ] **Issue 1.2** → GitHub Issue #Y

---

## CRITICAL Issues (P0 - Must Fix)

### Issue 1.1: Hyperparameter Optimization Confound
- [ ] **DEFER**: Requires re-running experiments (50-60 hours)
- [ ] **Files**: `tune_hoser.py`, `config/Beijing.yaml`, `config/Porto.yaml`
- [ ] **Action**: Add vanilla baseline to Optuna study with same search budget
- [ ] **Alternative**: Document as limitation (3-5 hours)
- [ ] **Validation**: Compare vanilla_optimal vs distilled_optimal

### Issue 1.2: Bonferroni Correction Misrepresentation  
- [ ] **FIX**: Implement correction (4 hours)
- [ ] **File**: `tools/analyze_wang_results.py` lines 558, 568
- [ ] **Action**: Add Bonferroni or FDR correction
- [ ] **Change**: `alpha_bonferroni = 0.05 / num_comparisons`
- [ ] **Update**: `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` line 126
- [ ] **Validation**: Re-run analysis, update result tables

### Issue 1.3: Translation Quality Confound
- [ ] **MITIGATE**: Filter by quality (6-8 hours)
- [ ] **Action**: Filter trajectories with quality >= 0.95
- [ ] **Report**: Translation quality distribution statistics
- [ ] **Alternative**: Document limitation (2 hours)
- [ ] **Validation**: Compare results with/without filtering

### Issue 1.4: JSD Binning Documentation Mismatch
- [ ] **FIX**: Update documentation (1 hour)
- [ ] **File**: `docs/EVALUATION_ANALYSIS.md` line 143
- [ ] **Change**: Replace "50 bins" with "100 bins"
- [ ] **Search**: Find all instances of "50 bins" in repo
- [ ] **Validation**: Verify documentation matches code (`evaluation.py` line 578)

### Issue 1.5: Data Leakage in Wang Baseline
- [ ] **FIX**: Compute baselines from train-only data (6 hours)
- [ ] **File**: `tools/compute_trajectory_baselines.py`
- [ ] **Action**: Separate train/test baseline computation
- [ ] **Add**: Global fallback baseline for unseen OD pairs
- [ ] **Validation**: Re-compute baselines, compare old vs new accuracy

### Issue 1.6: Vocabulary Mapping Unvalidated
- [ ] **DOCUMENT + Partial FIX**: Add validation analysis (4-10 hours)
- [ ] **Files**: `critics/grid_mapper.py`, `tools/map_road_networks.py`
- [ ] **Action**: Compute and report roads-per-cell statistics
- [ ] **Analysis**: Correlation between distillation benefit and many-to-one ratio
- [ ] **Optional**: Randomized mapping ablation (adds 6 hours)
- [ ] **Documentation**: Add mapping statistics to all docs

### Issue 1.7: Missing Calibration Metrics
- [ ] **FIX or REMOVE**: Implement ECE/Brier or remove claim (6-8 hours to fix, 1 hour to remove)
- [ ] **File**: `docs/LMTAD-Distillation.md` line 84
- [ ] **Option A**: Implement `compute_calibration_metrics()` function
- [ ] **Option B**: Remove "calibrated uncertainty" claim
- [ ] **Add**: Reliability diagrams for vanilla vs distilled
- [ ] **Validation**: Compare calibration metrics

### Issue 1.8: Beam Search Evaluation Dependence
- [ ] **DOCUMENT + Partial FIX**: Ablation or document (10-15 hours full, 2 hours doc only)
- [ ] **Action**: Test beam widths [1, 2, 4, 8, 16]
- [ ] **Analysis**: Compare vanilla vs distilled sensitivity to beam width
- [ ] **Alternative**: Document b=4 used for fair comparison
- [ ] **Validation**: Plot performance vs beam width

---

## MAJOR Issues (P1 - High Priority)

### Issue 2.1: No Teacher Baseline
- [ ] **DEFER**: Complex development (20-30 hours)
- [ ] **File**: `critics/lmtad_teacher.py`
- [ ] **Alternative**: Document architectural incompatibility (2 hours)
- [ ] **Action**: Add limitation section explaining vocabulary/architecture mismatch
- [ ] **Note**: Reference LM-TAD paper for teacher quality on source task

### Issue 2.2: No Ablation Studies
- [ ] **DEFER**: Requires new experiments (15-20 hours per ablation)
- [ ] **Missing**: Temperature τ sweep [1.0, 2.0, 3.0, 4.0, 5.0]
- [ ] **Missing**: Lambda λ sweep [0.0, 0.001, 0.01, 0.1]
- [ ] **Missing**: Window size sweep [1, 2, 4, 7, 10]
- [ ] **Alternative**: Document as limitation (2 hours)
- [ ] **Action**: Add future work section for systematic ablations

### Issue 2.3: No Regularization Baselines
- [ ] **DEFER**: Requires experiments (10-15 hours)
- [ ] **Missing**: Label smoothing baseline
- [ ] **Missing**: Self-distillation baseline
- [ ] **Missing**: Random teacher baseline
- [ ] **Alternative**: Document limitation (2 hours)
- [ ] **Action**: Acknowledge cannot rule out regularization mechanism

### Issue 2.4: KL Divergence Direction Unjustified
- [ ] **DOCUMENT**: Add justification (2-3 hours)
- [ ] **File**: `docs/LMTAD-Distillation.md` lines 810-814
- [ ] **Action**: Explain forward KL choice (mode-seeking behavior)
- [ ] **Optional**: Run reverse KL ablation (adds 8 hours)
- [ ] **Validation**: Compare forward vs reverse KL if ablated

### Issue 2.5: Candidate Top-K Filtering Bias
- [ ] **DOCUMENT**: Add analysis (4-6 hours)
- [ ] **Files**: `docs/LMTAD-Distillation.md` lines 1613-1620, 1778-1782
- [ ] **Action**: Report teacher-student candidate overlap statistics
- [ ] **Analysis**: Fraction of timesteps where teacher top-1 in student top-k
- [ ] **Optional**: Ablate k values [32, 64, 128, no filtering]
- [ ] **Documentation**: Clarify evaluation uses same k as training

### Issue 2.6: DTW/Hausdorff Not Normalized
- [ ] **FIX**: Normalize by trajectory length (3-4 hours)
- [ ] **File**: Evaluation code (likely `evaluation.py`)
- [ ] **Action**: Divide DTW/Hausdorff by trajectory length
- [ ] **Alternative**: Report length-normalized EDR prominently
- [ ] **Validation**: Re-compute normalized metrics

### Issue 2.7: OD Match Definition Confusion
- [ ] **CLARIFY**: Separate metrics (4-6 hours)
- [ ] **Action**: Report separately:
  - [ ] Destination reach rate (% where last road = target)
  - [ ] Endpoint realism (% where actual endpoints ∈ real ODs)
  - [ ] Combined OD match rate (current metric)
- [ ] **Documentation**: Clarify what OD match measures
- [ ] **Validation**: Verify separation provides insight

### Issue 2.8: Missing Paired Statistical Tests
- [ ] **FIX**: Add paired tests (4-6 hours)
- [ ] **Tests**: McNemar for OD match (binary), Wilcoxon for DTW/Hausdorff
- [ ] **Action**: Implement paired test functions
- [ ] **Report**: p-values and effect sizes
- [ ] **Validation**: Verify tests are properly paired

### Issue 2.9: No Cross-Seed Statistical Analysis
- [ ] **FIX**: Add variance tests (3-4 hours)
- [ ] **Tests**: Compare distilled vs vanilla variance (Levene's test)
- [ ] **Action**: Test if distilled has lower variance than vanilla
- [ ] **Caveat**: Note n=3 limits statistical power
- [ ] **Report**: Variance comparison with interpretation

### Issue 2.10: Multiple Testing Without Proper Correction
- [ ] **FIX**: Same as Issue 1.2 (Bonferroni correction)
- [ ] **See**: Issue 1.2 checklist above

### Issue 2.11: Missing Effect Sizes and Confidence Intervals
- [ ] **FIX**: Add effect sizes (4-6 hours)
- [ ] **File**: `tools/analyze_wang_results.py`
- [ ] **Action**: Compute Cohen's h, relative risk, odds ratio
- [ ] **Bootstrap**: 95% CIs using bootstrap (1000 resamples)
- [ ] **Update**: All result tables with effect size columns
- [ ] **Interpretation**: Classify as small/medium/large

---

## MODERATE Issues (P2 - Enhancement)

### Issue 3.1: Sample Size Not Justified
- [ ] **DOCUMENT**: Add justification (1-2 hours)
- [ ] **File**: `gene.py` line 2207
- [ ] **Action**: Add comment explaining 5,000 trajectory choice
- [ ] **Optional**: Run power analysis (adds 2 hours)
- [ ] **Justification**: "5,000 balances statistical power with computational cost"

### Issue 3.2: OD Matching Arbitrariness
- [ ] **DOCUMENT**: Justify grid size (2 hours)
- [ ] **Action**: Explain 0.001° ≈ 111m choice
- [ ] **Optional**: Sensitivity analysis [0.0005°, 0.001°, 0.002°]
- [ ] **Note**: Both models use identical grid size (fair comparison)

### Issue 3.3: Threshold Justification Missing (Wang)
- [ ] **DOCUMENT**: Reference Wang paper (1 hour)
- [ ] **File**: `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md`
- [ ] **Action**: Add citation for 5km/5min thresholds
- [ ] **Note**: Thresholds from Wang et al. 2018 paper

### Issue 3.4: CV Misuse (n=2-3)
- [ ] **DOCUMENT**: Add caveat (1 hour)
- [ ] **Action**: Note "CV estimates based on n=3 seeds; limited precision"
- [ ] **Clarify**: CV is descriptive, not for statistical inference

### Issue 3.5: Missing Environment Information
- [ ] **FIX**: Add software versions (1 hour)
- [ ] **Files**: `README.md`, `docs/EVALUATION_ANALYSIS.md`
- [ ] **Action**: Document PyTorch, CUDA, Python versions
- [ ] **Add**: Hardware specs (GPU model, RAM, CPU)
- [ ] **Format**: Create "Environment" section in docs

### Issue 3.6: Data Preprocessing Pipeline Incomplete
- [ ] **DOCUMENT**: Add preprocessing details (2-3 hours)
- [ ] **Action**: Document end-to-end pipeline
- [ ] **Include**: Map matching, zone partitioning, vocabulary mapping
- [ ] **Diagram**: Flow chart of data preprocessing steps

### Issue 3.7: Inconsistent Seed Usage
- [ ] **CLARIFY**: Document seed usage (1 hour)
- [ ] **Action**: Clarify training uses seeds {42, 43, 44}
- [ ] **Note**: Evaluation uses same models (no new seeds)
- [ ] **Verify**: Confirm seeds are consistent

### Issue 3.8: Hybrid Threshold Sensitivity Not Tested
- [ ] **DOCUMENT**: Note fixed thresholds (1 hour)
- [ ] **Action**: Document Wang thresholds as fixed
- [ ] **Future work**: Recommend sensitivity analysis

---

## MINOR Issues (P3 - Nice to Have)

### Issue 4.1: Inconsistent Defaults (Beam Width)
- [ ] **FIX**: Standardize defaults (1 hour)
- [ ] **Action**: Ensure all configs use b=4 consistently
- [ ] **Files**: Check all config files and scripts
- [ ] **Documentation**: Document standard beam width

### Issue 4.2: Missing Architecture Spec
- [ ] **DOCUMENT**: Add architecture table (2-3 hours)
- [ ] **Action**: Document hidden dims, attention heads, dropout, layers
- [ ] **Table**: Create model architecture specification table
- [ ] **Reference**: Link to HOSER paper for full details

### Issue 4.3: Data Split Methodology
- [ ] **DOCUMENT**: Clarify split process (1-2 hours)
- [ ] **Action**: Document train/val/test split ratios and methodology
- [ ] **Include**: How OD pairs are stratified/split
- [ ] **Note**: Random seed for reproducibility

### Issue 4.4: Selective Reporting (Porto)
- [ ] **DOCUMENT**: Strengthen Porto discussion (2 hours)
- [ ] **Action**: Emphasize dataset-dependent effectiveness
- [ ] **Balance**: Give equal weight to Porto results
- [ ] **Interpretation**: Discuss why distillation works differently

### Issue 4.5: No Failure Analysis
- [ ] **ADD**: Characterize failure modes (4-6 hours)
- [ ] **Action**: Analyze when/why models fail
- [ ] **Categories**: Failure taxonomy (stuck loops, wrong turns, etc.)
- [ ] **Report**: Failure mode distribution

---

## Quick Statistics

**Total Issues**: 35  
**P0 (Critical)**: 8 issues - 90-120 hours (with full fixes) or 15-25 hours (with mitigation)  
**P1 (Major)**: 11 issues - 40-60 hours (with experiments) or 15-20 hours (documentation only)  
**P2 (Moderate)**: 8 issues - 10-15 hours  
**P3 (Minor)**: 5 issues - 8-12 hours  

**Minimum Viable Fix** (documentation + quick fixes only): **40-50 hours**  
**Comprehensive Fix** (all experiments + fixes): **150-200 hours**

---

## Priority Recommendations

### Week 1 (P0 Critical - Must Fix)
1. Issue 1.4: JSD binning docs (1 hour) ✅ **Quick win**
2. Issue 1.2: Bonferroni correction (4 hours)
3. Issue 1.5: Data leakage fix (6 hours)
4. Issue 1.7: Remove calibration claim OR implement (1-8 hours)
5. Issue 1.6: Vocabulary mapping analysis (4 hours)

**Total**: 16-23 hours

### Week 2 (P1 Major - Documentation)
6. Issue 2.1: Document teacher limitation (2 hours)
7. Issue 2.4: KL direction justification (2 hours)
8. Issue 2.5: Candidate filtering docs (2 hours)
9. Issue 3.5: Environment info (1 hour)
10. Issue 2.11: Effect sizes (6 hours)

**Total**: 13 hours

### Week 3 (Remaining P2/P3 + Select P1)
11. Issue 1.8: Beam width documentation (2 hours)
12. Issue 2.6: DTW/Hausdorff normalization (4 hours)
13. Issue 2.8: Paired tests (5 hours)
14. Moderate/Minor issues as time permits

**Total**: 11+ hours

### Long-term (Deferred - Requires Experiments)
- Issue 1.1: Vanilla hyperparameter optimization (50-60 hours)
- Issue 1.3: Translation quality filtering (if data available)
- Issue 2.2: Systematic ablation studies (15-20 hours per ablation)
- Issue 2.3: Regularization baselines (10-15 hours)
- Issue 1.8: Full beam width ablation (12 hours)

---

## Files Requiring Changes

**Code Files**:
- `tune_hoser.py` - Issue 1.1 (hyperparameter confound)
- `tools/analyze_wang_results.py` - Issues 1.2, 2.11 (statistics)
- `tools/compute_trajectory_baselines.py` - Issue 1.5 (data leakage)
- `evaluation.py` - Issues 1.7 (calibration), 2.6 (normalization), 2.8 (tests)
- `critics/grid_mapper.py` - Issue 1.6 (vocabulary mapping)
- `gene.py` - Issue 3.1 (sample size)

**Documentation Files**:
- `docs/EVALUATION_ANALYSIS.md` - Issues 1.4 (bins), 1.8 (beam width), 3.5 (environment)
- `docs/LMTAD-Distillation.md` - Issues 1.7 (calibration claim), 2.4 (KL direction)
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` - Issues 1.2 (Bonferroni), 3.3 (thresholds)
- `README.md` - Issue 3.5 (environment), general clarity

**Configuration Files**:
- `config/Beijing.yaml` - Issues 1.1 (Optuna), 2.5 (candidate_top_k)
- `config/Porto.yaml` - Same as Beijing

---

## Validation Checklist

After each fix, verify:
- [ ] Code changes compile/run without errors
- [ ] Tests pass (if applicable)
- [ ] Documentation updated to match code
- [ ] Results regenerated if needed
- [ ] Git commit with clear message
- [ ] No regressions introduced

---

## GitHub Progress Tracking

### AI Agent Commands for Progress Monitoring

**View Phase 1 Status**:
```
list_issues(milestone="Phase 1: Critical Fixes", state="all")
# Count: open, closed, in_progress
```

**View Issues by Priority**:
```
search_issues("is:open label:P0-Critical")
search_issues("is:open label:P1-Major")
```

**Generate Progress Report**:
```
For each milestone:
  - Total issues
  - Completed (closed)
  - In Progress (labeled "in-progress")
  - Blocked (labeled "blocked")
  - Completion percentage
```

**Check Dependencies**:
```
# Before starting Issue 2.11, verify Issue 1.2 is closed
get_issue(2)  # Issue 1.2
if issue.state == "closed":
    # Safe to start Issue 2.11
    update_issue(11, labels=["in-progress"])
```

### Automated Status Updates

AI agents should use GitHub MCP to:
1. **On start**: Move issue to "In Progress", add comment with start timestamp
2. **During work**: Add comments with code changes, challenges encountered
3. **On validation**: Add comment with test results, move to "Validation"
4. **On completion**: Add final summary comment, close issue, update checklist

Example comment format:
```markdown
## Implementation Progress

**Started**: 2026-01-15 10:30 UTC
**Status**: In Progress

### Changes Made
- Modified `tools/analyze_wang_results.py` lines 558, 568
- Added Bonferroni correction: `alpha = 0.05 / 13 = 0.00385`
- Updated documentation in WANG_ABNORMALITY_DETECTION_RESULTS.md

### Next Steps
- [ ] Re-run analysis script
- [ ] Verify results
- [ ] Update result tables

### Validation
- [ ] Code compiles without errors
- [ ] Tests pass
- [ ] Documentation matches code
```

---

**End of Checklist**

Use this document to systematically address all implementation issues. Mark items as completed and track progress. Use GitHub MCP for automated issue tracking and project management.


