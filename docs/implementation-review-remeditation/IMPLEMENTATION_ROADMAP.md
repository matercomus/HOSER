# Implementation Roadmap

**Repository**: `/home/matt/Dev/HOSER`  
**Purpose**: Timeline and dependency graph for fixing all repository issues  
**Created**: January 2026

---

## Overview

This roadmap organizes the 35 repository-specific issues into 3 implementation phases based on priority, dependencies, and feasibility.

**Total Estimated Effort**:
- **Minimum Viable** (documentation + quick fixes): 40-50 hours (1-2 weeks)
- **Comprehensive** (all experiments + fixes): 150-200 hours (4-6 weeks)

**Recommended Approach**: Start with Minimum Viable path to address critical issues quickly, then pursue experimental work in parallel.

---

## GitHub Project Board Organization

### Project Board Structure

**Board Name**: "HOSER Peer Review Remediation"

**Columns**:
1. **To Do** - Issues not yet started
2. **In Progress** - Currently being worked on
3. **Validation** - Implementation complete, testing in progress
4. **Blocked** - Waiting on dependencies or decisions
5. **Complete** - Validated and closed

**AI Agent Setup**:
```python
# Create project board
project = create_project(
    name="HOSER Peer Review Remediation",
    body="Tracking all 35 repository-specific issues from peer review"
)

# Add all issues to board
for issue in all_issues:
    add_issue_to_project(project.id, issue.id)
    update_project_item_status(item.id, "To Do")
```

### Milestone-Based Views

**Milestones correspond to phases**:
- **Phase 1**: "Critical Fixes (Week 1)" - 8 issues
- **Phase 2**: "Major Issues (Week 2)" - 11 issues  
- **Phase 3**: "Moderate/Minor (Week 3)" - 13 issues
- **Deferred**: "Experimental Work" - 5 issues

**AI Agent Queries**:
```python
# View current phase status
list_issues(milestone="Phase 1: Critical Fixes", state="open")

# Check phase completion
phase1_total = count_issues(milestone="Phase 1")
phase1_closed = count_issues(milestone="Phase 1", state="closed")
completion = (phase1_closed / phase1_total) * 100
```

### Dependency Tracking with Issue References

Issues with dependencies should reference blocking issues:

Example Issue 2.11 body:
```markdown
...
**Dependencies**: 
- Blocked by #2 (Issue 1.2: Bonferroni correction)
- Must be completed before starting

**Status Check**:
```python
# AI agent checks before starting
issue_1_2 = get_issue(2)
if issue_1_2.state != "closed":
    update_issue(11, labels=["blocked"])
    add_issue_comment(11, "Blocked by #2 - waiting for Bonferroni correction")
else:
    update_issue(11, labels=["in-progress"])
```
...
```

---

## Phase 1: Critical Fixes (Week 1) - P0 Blocking Issues

**Goal**: Fix or mitigate all CRITICAL issues that threaten validity  
**Duration**: 16-23 hours  
**Priority**: P0 (must complete before publication)

### Week 1 Tasks

| Issue | Task | Effort | Dependencies | GitHub | Status |
|-------|------|--------|--------------|--------|--------|
| 1.4 | Fix JSD binning documentation | 1 hour | None | #4 | ⬜ Not started |
| 1.2 | Implement Bonferroni correction | 4 hours | None | #2 | ⬜ Not started |
| 1.5 | Fix data leakage in Wang baselines | 6 hours | None | #5 | ⬜ Not started |
| 1.7 | Remove calibration claim OR implement metrics | 1-8 hours | None | #7 | ⬜ Not started |
| 1.6 | Add vocabulary mapping validation | 4 hours | None | #6 | ⬜ Not started |

### Quick Wins (High Impact, Low Effort)

**Day 1** (2 hours):
1. Issue 1.4: Update "50 bins" to "100 bins" in docs (1 hour)
2. Issue 1.7: Remove "calibrated uncertainty" claim (1 hour)

**Day 2** (4 hours):
3. Issue 1.2: Implement Bonferroni correction (4 hours)

**Day 3-4** (10 hours):
4. Issue 1.5: Fix data leakage in baselines (6 hours)
5. Issue 1.6: Vocabulary mapping analysis (4 hours)

### Dependencies

```
Phase 1 Issues (No internal dependencies - can be parallelized)
├── Issue 1.4 (docs) ✓ Independent
├── Issue 1.2 (stats) ✓ Independent
├── Issue 1.5 (baselines) ✓ Independent
├── Issue 1.7 (calibration) ✓ Independent
└── Issue 1.6 (mapping) ✓ Independent
```

### Phase 1 Deliverables

- [ ] Updated documentation with correct bin count
- [ ] Bonferroni-corrected statistical results
- [ ] Clean Wang baseline computation (no data leakage)
- [ ] Calibration claims removed or validated
- [ ] Vocabulary mapping statistics reported

### Phase 1 GitHub Automation

**AI Agent Workflow**:
```python
# 1. Query phase issues
phase_issues = list_issues(milestone="Phase 1: Critical Fixes", state="open", sort="priority")

# 2. Process sequentially
for issue in phase_issues:
    # Check dependencies
    if has_blocking_dependencies(issue):
        update_issue(issue.number, labels=["blocked"])
        continue
    
    # Start work
    update_issue(issue.number, labels=["in-progress"])
    add_issue_comment(issue.number, "Started implementation")
    
    # Implement fix (from IMPLEMENTATION_REMEDIATION_REPORT.md)
    implement_fix(issue)
    
    # Validate
    validation_results = run_validation(issue)
    add_issue_comment(issue.number, f"Validation: {validation_results}")
    
    # Complete
    if validation_results.passed:
        close_issue(issue.number)
        add_issue_comment(issue.number, "✅ Validated and complete")
    else:
        update_issue(issue.number, labels=["needs-revision"])
```

---

## Phase 2: Major Issues - Documentation & Quick Fixes (Week 2)

**Goal**: Address MAJOR issues through documentation and feasible code fixes  
**Duration**: 15-20 hours  
**Priority**: P1 (high priority, strengthens validity)

### Week 2 Tasks

| Issue | Task | Effort | Dependencies | GitHub | Status |
|-------|------|--------|--------------|--------|--------|
| 2.1 | Document teacher limitation | 2 hours | None | #9 | ⬜ Not started |
| 2.4 | Add KL direction justification | 2 hours | None | #12 | ⬜ Not started |
| 2.5 | Document candidate filtering | 2 hours | None | #13 | ⬜ Not started |
| 3.5 | Add environment information | 1 hour | None | #24 | ⬜ Not started |
| 2.11 | Implement effect sizes & CIs | 6 hours | Issue 1.2 (#2) | #19 | ⬜ Not started |
| 1.8 | Document beam width choice | 2 hours | None | #8 | ⬜ Not started |
| 2.6 | Normalize DTW/Hausdorff | 4 hours | None | #14 | ⬜ Not started |
| 2.8 | Add paired statistical tests | 5 hours | Issue 1.2 (#2) | #16 | ⬜ Not started |

### Day-by-Day Plan

**Day 1** (5 hours):
1. Issue 2.1: Teacher limitation documentation (2 hours)
2. Issue 2.4: KL direction justification (2 hours)
3. Issue 3.5: Environment info (1 hour)

**Day 2** (6 hours):
4. Issue 2.11: Effect sizes and CIs (6 hours) - **Requires Issue 1.2 complete**

**Day 3** (9 hours):
5. Issue 2.5: Candidate filtering docs (2 hours)
6. Issue 1.8: Beam width documentation (2 hours)
7. Issue 2.8: Paired tests (5 hours) - **Requires Issue 1.2 complete**

**Optional Day 4** (4 hours):
8. Issue 2.6: DTW/Hausdorff normalization (4 hours)

### Dependencies

```
Phase 2 Issues
├── Independent Group (can parallelize)
│   ├── Issue 2.1 (teacher docs)
│   ├── Issue 2.4 (KL justification)
│   ├── Issue 2.5 (candidate filtering)
│   ├── Issue 3.5 (environment)
│   ├── Issue 1.8 (beam width)
│   └── Issue 2.6 (normalization)
└── Statistical Group (depends on Issue 1.2)
    ├── Issue 2.11 (effect sizes) → requires Bonferroni
    └── Issue 2.8 (paired tests) → requires Bonferroni
```

### Phase 2 Deliverables

- [ ] Teacher architectural incompatibility documented
- [ ] KL direction choice justified
- [ ] Candidate filtering process clarified
- [ ] Software/hardware environment documented
- [ ] Effect sizes and CIs added to all statistical tests
- [ ] Beam width choice justified
- [ ] Optional: Normalized trajectory metrics

### Phase 2 GitHub Automation

**AI Agent Workflow**:
```python
# 1. Query phase issues
phase_issues = list_issues(milestone="Phase 2: Major Issues", state="open", sort="priority")

# 2. Process sequentially
for issue in phase_issues:
    # Check dependencies
    if has_blocking_dependencies(issue):
        update_issue(issue.number, labels=["blocked"])
        continue
    
    # Start work
    update_issue(issue.number, labels=["in-progress"])
    add_issue_comment(issue.number, "Started implementation")
    
    # Implement fix (from IMPLEMENTATION_REMEDIATION_REPORT.md)
    implement_fix(issue)
    
    # Validate
    validation_results = run_validation(issue)
    add_issue_comment(issue.number, f"Validation: {validation_results}")
    
    # Complete
    if validation_results.passed:
        close_issue(issue.number)
        add_issue_comment(issue.number, "✅ Validated and complete")
    else:
        update_issue(issue.number, labels=["needs-revision"])
```

---

## Phase 3: Moderate/Minor Issues & Remaining Fixes (Week 3)

**Goal**: Complete remaining documentation and feasible enhancements  
**Duration**: 10-15 hours  
**Priority**: P2-P3 (enhancements, nice to have)

### Week 3 Tasks

| Issue | Task | Effort | Dependencies | GitHub | Status |
|-------|------|--------|--------------|--------|--------|
| 2.9 | Cross-seed variance analysis | 3 hours | None | #17 | ⬜ Not started |
| 2.7 | Separate OD match metrics | 4 hours | None | #15 | ⬜ Not started |
| 3.1 | Document sample size | 1 hour | None | #20 | ⬜ Not started |
| 3.2 | Document grid size choice | 1 hour | None | #21 | ⬜ Not started |
| 3.3 | Reference Wang thresholds | 1 hour | None | #22 | ⬜ Not started |
| 3.4 | Add CV caveat | 1 hour | None | #23 | ⬜ Not started |
| 3.6 | Document preprocessing pipeline | 2 hours | None | #25 | ⬜ Not started |
| 3.7 | Clarify seed usage | 1 hour | None | #26 | ⬜ Not started |
| 3.8 | Note threshold sensitivity | 1 hour | None | #27 | ⬜ Not started |
| 4.1 | Standardize beam width defaults | 1 hour | None | #28 | ⬜ Not started |
| 4.2 | Add architecture specification | 2 hours | None | #29 | ⬜ Not started |
| 4.3 | Document data split methodology | 1 hour | None | #30 | ⬜ Not started |
| 4.4 | Strengthen Porto discussion | 2 hours | None | #31 | ⬜ Not started |

### Week 3 Plan (Flexible)

**Days 1-2** (6 hours): Priority P2 issues
- Issue 2.9: Cross-seed analysis (3 hours)
- Issue 2.7: OD match separation (4 hours) - **if time permits**

**Days 3-5** (9 hours): Documentation sweep
- All remaining P3 issues (9 hours total)

### Dependencies

All Phase 3 issues are independent and can be completed in any order.

### Phase 3 Deliverables

- [ ] Cross-seed variance comparison
- [ ] Optional: Separated OD match metrics
- [ ] Complete documentation for all methodological choices
- [ ] Preprocessing pipeline documented
- [ ] Architecture specification added
- [ ] Porto results discussion strengthened

### Phase 3 GitHub Automation

**AI Agent Workflow**:
```python
# 1. Query phase issues
phase_issues = list_issues(milestone="Phase 3: Moderate/Minor", state="open", sort="priority")

# 2. Process sequentially
for issue in phase_issues:
    # Check dependencies (all Phase 3 issues are independent)
    if has_blocking_dependencies(issue):
        update_issue(issue.number, labels=["blocked"])
        continue
    
    # Start work
    update_issue(issue.number, labels=["in-progress"])
    add_issue_comment(issue.number, "Started implementation")
    
    # Implement fix (from IMPLEMENTATION_REMEDIATION_REPORT.md)
    implement_fix(issue)
    
    # Validate
    validation_results = run_validation(issue)
    add_issue_comment(issue.number, f"Validation: {validation_results}")
    
    # Complete
    if validation_results.passed:
        close_issue(issue.number)
        add_issue_comment(issue.number, "✅ Validated and complete")
    else:
        update_issue(issue.number, labels=["needs-revision"])
```

---

## Deferred Issues (Long-term - Requires Experiments)

**These issues require significant experimental work and are deferred to future work.**

### Deferred Experimental Work

| Issue | Task | Effort | Feasibility | Timeline |
|-------|------|--------|-------------|----------|
| 1.1 | Vanilla hyperparameter optimization | 50-60 hours | HIGH | 2-3 weeks |
| 2.2 | Systematic ablation studies | 15-20 hours each | MEDIUM | 4-6 weeks |
| 2.3 | Regularization baselines | 10-15 hours | MEDIUM | 1-2 weeks |
| 1.3 | Translation quality filtering | 6-8 hours | DEPENDS | If data available |
| 1.8 | Full beam width ablation | 12 hours | HIGH | 1 week |
| 2.4 | Reverse KL ablation | 8 hours | HIGH | 1 week |
| 1.6 | Randomized mapping ablation | 6 hours | MEDIUM | 1 week |

### Recommended Priority Order

**If pursuing experimental work**:

1. **Issue 1.1**: Vanilla hyperparameter optimization
   - **Impact**: Resolves most fundamental confound
   - **Effort**: 50-60 hours
   - **Result**: Fair baseline comparison
   - **Decision**: Required if claiming distillation effectiveness

2. **Issue 1.8**: Beam width ablation
   - **Impact**: Validates results are not search artifacts
   - **Effort**: 12 hours
   - **Result**: Confidence that model quality matters

3. **Issue 2.3**: Regularization baselines
   - **Impact**: Distinguishes knowledge transfer from regularization
   - **Effort**: 10-15 hours
   - **Result**: Mechanism understanding

4. **Issue 2.2**: Systematic ablations
   - **Impact**: Component contribution understanding
   - **Effort**: 15-20 hours per ablation
   - **Result**: Design principles for distillation

### Experimental Work Timeline

If all deferred work is pursued:

```
Month 1-2: Issue 1.1 (Vanilla hyperparameter search)
  └── Week 1-2: Beijing experiments
  └── Week 3-4: Porto experiments
  └── Week 5: Analysis and re-evaluation

Month 2-3: Other ablations (in parallel if resources permit)
  ├── Issue 1.8: Beam width (Week 6)
  ├── Issue 2.3: Regularization baselines (Week 7-8)
  └── Issue 2.2: Systematic ablations (Week 9-12)
```

**Total experimental timeline**: 3-4 months

---

## Parallel Work Streams

To maximize efficiency, work can be parallelized into 3 streams:

### Stream A: Documentation (Low computational cost)
- All documentation-only issues
- Can be done by one person
- No experimental dependencies
- **Timeline**: 2-3 weeks

### Stream B: Code Fixes (Moderate computational cost)
- Statistical corrections
- Metric normalization
- Data leakage fixes
- **Timeline**: 1-2 weeks

### Stream C: Experimental Work (High computational cost)
- Hyperparameter searches
- Ablation studies
- New baseline training
- **Timeline**: 3-4 months
- **Can start after Phase 1-2 complete**

### Gantt Chart

```
Week 1:  [===== Phase 1: Critical Fixes =====]
         [A: Quick docs] [B: Code fixes]

Week 2:  [========= Phase 2: Major Issues =========]
         [A: Documentation] [B: Statistical fixes]

Week 3:  [==== Phase 3: Moderate/Minor ====]
         [A: Documentation sweep]

Week 4+: [============== Experimental Work (Stream C) ==============]
         (If pursuing deferred issues)
```

---

## Decision Points

### Critical Decision: Issue 1.1 (Hyperparameter Confound)

**Option A**: Re-run experiments with fair comparison (50-60 hours)
- **Pros**: Scientifically rigorous, resolves fundamental confound
- **Cons**: Time-consuming, may reduce reported improvements
- **Recommendation**: **Required** if claiming distillation is effective

**Option B**: Document as limitation only (3-5 hours)
- **Pros**: Fast, acknowledges issue
- **Cons**: Undermines all conclusions, weakens paper significantly
- **Recommendation**: Only if Option A is infeasible

**Decision criteria**: 
- Publication timeline urgency?
- Computational resources available?
- Strength of claims desired?

### Secondary Decision: Experimental Ablations

**Pursue if**:
- Computational resources available
- Want to strengthen mechanism understanding
- Aiming for top-tier publication

**Skip if**:
- Tight publication deadline
- Limited computational budget
- Current results sufficient for target venue

---

## Milestones & Checkpoints

### Milestone 1: Critical Fixes Complete (End of Week 1)
- [ ] All P0 issues addressed (fixed or mitigated)
- [ ] Documentation updated
- [ ] Statistical corrections implemented
- [ ] Ready for internal review

### Milestone 2: Major Issues Resolved (End of Week 2)
- [ ] All P1 issues addressed
- [ ] Comprehensive documentation in place
- [ ] Statistical rigor improved
- [ ] Ready for external review

### Milestone 3: Repository Clean (End of Week 3)
- [ ] All feasible issues resolved
- [ ] Deferred issues documented as future work
- [ ] Code and docs consistent
- [ ] Ready for publication

### Milestone 4: Experimental Validation (End of Month 3-4)
- [ ] Fair baseline comparison complete
- [ ] Ablation studies finished
- [ ] Mechanism understood
- [ ] Ready for resubmission if needed

---

## Risk Management

### High Risk Items

**Issue 1.1** (Hyperparameter confound):
- **Risk**: Vanilla performance improves significantly, reducing distillation benefit
- **Mitigation**: Run quick learning rate sweep first to estimate impact
- **Contingency**: Pivot to "optimized distillation vs optimized vanilla" comparison

**Issue 2.3** (Regularization baselines):
- **Risk**: Label smoothing performs as well as distillation
- **Mitigation**: Test early to understand mechanism
- **Contingency**: Reframe as "distillation as effective regularization" contribution

### Medium Risk Items

**Statistical corrections**:
- **Risk**: Significance disappears after Bonferroni correction
- **Mitigation**: Report effect sizes prominently, focus on practical significance
- **Contingency**: Emphasize magnitude of improvements, not p-values

**Translation quality**:
- **Risk**: High-quality filter reduces sample size dramatically
- **Mitigation**: Report results at multiple quality thresholds
- **Contingency**: Document limitation, recommend native evaluation

---

## Success Criteria

### Minimum Viable Success (Phase 1-2 complete)
- [ ] All critical validity threats addressed
- [ ] Statistical rigor improved
- [ ] Documentation accurate and complete
- [ ] Limitations honestly acknowledged
- [ ] Ready for publication with caveats

### Comprehensive Success (All phases including experimental)
- [ ] Fair baseline comparison
- [ ] Mechanism understood
- [ ] All ablations complete
- [ ] Strong scientific rigor
- [ ] Publication-ready without major caveats

---

## Resource Requirements

### Computational Resources

**Phase 1-2** (Minimum Viable):
- GPU time: ~10 hours (re-running analyses)
- Storage: ~10 GB (new result files)
- RAM: 32 GB minimum

**Phase 3** (Experimental Work):
- GPU time: ~500 hours (all experiments)
- Storage: ~100 GB (models + results)
- RAM: 64 GB recommended

### Human Resources

**Single developer**:
- Phase 1-2: 3-4 weeks
- Phase 3 (experimental): Additional 3-4 months

**Team of 2**:
- Phase 1-2: 2 weeks (parallel streams)
- Phase 3 (experimental): 2 months (parallel experiments)

---

## Next Steps

1. **Review this roadmap** with team/supervisor
2. **Decide on experimental work**: Pursue Issue 1.1 or document limitation?
3. **Allocate resources**: GPU time, developer time
4. **Start Phase 1**: Begin with quick wins (Issue 1.4, 1.7)
5. **Track progress**: Update checklist and roadmap as work completes
6. **Regular checkpoints**: Weekly progress reviews
7. **Adjust timeline**: Re-estimate as needed based on actual effort

---

**End of Roadmap**

This roadmap provides a clear path from current state to publication-ready implementation. Follow the phases systematically and adjust based on resources and priorities.


