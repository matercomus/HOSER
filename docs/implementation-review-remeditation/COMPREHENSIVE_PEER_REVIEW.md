# Comprehensive Peer Review: HOSER Knowledge Distillation Research

**Review Date**: November 2025  
**Reviewers**: AI Peer Review System (Multiple Review Phases)  
**Scope**: Complete evaluation of experimental design, statistical rigor, methodology, implementation, and scientific claims  
**Sources**: 
- CRITICAL_REVIEW_FINDINGS.md (Code investigation findings)
- PEER_REVIEW_CRITICAL_ANALYSIS.md (Comprehensive analysis)
- PEER_REVIEW_CRITICAL_ISSUES.md (Methodology review)

---

## Executive Summary

This comprehensive peer review synthesizes findings from three detailed investigations of the HOSER knowledge distillation research. The work presents an interesting application of knowledge distillation to trajectory generation with promising results on the Beijing dataset. However, **fundamental experimental design flaws, statistical inadequacies, and methodological inconsistencies** prevent drawing strong causal conclusions about distillation effectiveness.

While the work demonstrates technical sophistication and good engineering practices (well-structured code, comprehensive evaluation metrics, multiple datasets), **critical issues threaten the validity of conclusions** and require immediate remediation before publication.

**Overall Assessment**: **Major revisions required** before publication.

### Key Findings Summary

**CRITICAL FLAWS (Threaten Validity)**:
1. **Hyperparameter Optimization Confound**: Vanilla baseline receives NO hyperparameter search while distilled gets 12 trials of optimization
2. **Bonferroni Correction Misrepresentation**: Documentation claims correction but code doesn't implement it
3. **Translation Quality Confound**: Cross-dataset evaluation uses 79% translation quality without filtering
4. **JSD Binning Misrepresentation**: Documentation claims 50 bins, code uses 100 bins
5. **Data Leakage in Wang Baseline Computation**: Train+test pooling inflates detection accuracy
6. **Missing Calibration Metrics**: "Calibrated uncertainty" claimed but never measured
7. **Vocabulary Mapping Unvalidated**: Many-to-one road→grid artifacts not controlled
8. **Beam Search Evaluation Dependence**: Metrics depend on search parameters, not model quality

**MAJOR ISSUES (Seriously Weaken Conclusions)**:
9. **No Teacher Baseline**: LM-TAD teacher never evaluated as trajectory generator
10. **No Ablation Studies**: Cannot isolate which distillation components matter
11. **No Regularization Baselines**: Cannot distinguish knowledge transfer from regularization
12. **Local Metrics Contradiction**: Vanilla has lower Hausdorff/DTW (closer to real) but dismissed
13. **Multiple Testing Without Proper Correction**: Dozens of comparisons inflate false discoveries
14. **Beam Search Confound**: No ablation of beam width, results may be search artifacts
15. **DTW/Hausdorff Normalization**: Not normalized by trajectory length, unfair comparisons
16. **OD Match Definition Confusion**: Conflates path completion with endpoint realism
17. **Missing Paired Statistical Tests**: Wrong test types for matched data
18. **No Cross-Seed Analysis**: Multiple seeds used but no statistical tests across seeds
19. **KL Divergence Direction Unjustified**: Forward KL used without justification
20. **Candidate Top-K Filtering Bias**: k=64 may favor certain model types
21. **Temperature/Lambda Sensitivity Missing**: No systematic sensitivity analysis
22. **Statistical vs Practical Significance Confusion**: No clear interpretation framework

**MODERATE ISSUES (Limit Generalizability)**:
23. **Sample Size Not Justified**: 5,000 trajectories with no power analysis
24. **OD Matching Arbitrariness**: Grid size 0.001° with no sensitivity analysis
25. **Threshold Justification Missing**: 5km/5min thresholds from paper but no validation
26. **OD Coverage Misleading**: "OD-specific" actually 99% global baseline
27. **CV Misuse**: Coefficient of variation computed on n=2-3 seeds
28. **Hybrid Threshold Sensitivity Not Tested**: No analysis of threshold robustness
29. **Missing Environment Information**: No software versions documented
30. **Data Preprocessing Pipeline Incomplete**: End-to-end pipeline not documented

**MINOR ISSUES (Should Be Addressed)**:
31. **Inconsistent Defaults**: Beam width defaults differ (4 vs 8)
32. **Missing Architecture Spec**: Complete model dimensions not fully documented
33. **Data Split Methodology**: Train/test split process not clearly documented
34. **Selective Reporting**: Porto failure explained away rather than highlighted
35. **Missing Related Work**: Minimal citations to distillation literature
36. **Inconsistent Seed Usage**: Evaluation uses different seeds than training
37. **Contradictory Statements**: Multiple contradictions across documentation
38. **No Failure Analysis**: Failure modes not characterized
39. **No Intermediate Checkpoints**: Only final models reported, no training dynamics
40. **No Computational Cost Analysis**: Cost-benefit not analyzed
41. **No Cross-Model Comparisons**: No external baselines

---

## Issue Classification System

Throughout this review, issues are classified by severity:

- **CRITICAL**: Must fix before publication (threatens validity of conclusions)
- **MAJOR**: Significant methodological weakness (seriously weakens conclusions)
- **MODERATE**: Limits generalizability or applicability
- **MINOR**: Enhancement opportunity (improves rigor but doesn't threaten validity)

---

## Section 1: Experimental Design Flaws

### 1.1 Hyperparameter Optimization Confound ⚠️ CRITICAL

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 1.1, PEER_REVIEW_CRITICAL_ANALYSIS Section 1.1]

**Issue**: The vanilla baseline is treated unfairly in comparison - it receives NO hyperparameter search while the distilled model gets 12 trials of intelligent optimization.

**Evidence from Code Investigation** (`tune_hoser.py`):
- Line 215: `# All trials are distillation trials (vanilla baseline runs separately in Phase 0)`
- Line 271: `# All trials use distillation (vanilla baseline runs separately in Phase 0)`
- Line 279: `config["wandb"]["run_name"] = f"trial_{trial.number:03d}_distilled"`
- Line 303: Seeds are different: `base_seed + trial.number`, so Trial 0 gets seed 43, Trial 1 gets seed 44, etc.

**Key Discovery**:
- **Trial 0 IS NOT vanilla** - it's a distilled trial with hyperparameters suggested by CMA-ES
- Vanilla baseline runs in **Phase 0** (before tuning) and **Phase 3** (after tuning) as separate runs
- All Optuna trials (0-12) are distilled with different hyperparameters
- Vanilla baseline is NOT part of the Optuna study - it's a completely separate run
- This means vanilla doesn't get any hyperparameter search at all

**The Confound is WORSE than Initially Thought**:
The comparison is: **single vanilla run with fixed hyperparameters** vs **optimized distilled model (best of 12 trials with intelligent search)**

**Documentation Misrepresentation**:
- `EVALUATION_ANALYSIS.md` line 87-95 claims "Trial 0" is vanilla
- But Trial 0 is actually a distilled trial with hyperparameters suggested by CMA-ES
- This is misleading to readers

**Impact**:
- This is NOT a fair comparison
- Performance difference may come from hyperparameter optimization, not distillation
- Cannot isolate distillation effect from optimization effect
- The comparison breaks scientific validity - most critical issue found

**Files Examined**:
- `tune_hoser.py` lines 215, 271, 279, 303
- `config/Beijing.yaml` - Optuna configuration
- `_run_vanilla_baseline()` function (lines 758-865)

**Required Fix**:
1. Run Optuna study for vanilla baseline with same search budget (12 trials)
2. Optimize vanilla's hyperparameters (learning rate, batch size, weight decay, etc.)
3. Compare: vanilla_optimal vs distilled_optimal (both with 12 trials)
4. Update documentation to clarify Trial 0 is NOT vanilla
5. Re-interpret all results with fair comparison

**Code References**:
- `tune_hoser.py` lines 215, 271, 279, 303
- `tune_hoser.py` lines 758-865: `_run_vanilla_baseline()` function
- `config/Beijing.yaml`: Optuna configuration

---

### 1.2 Missing Ablation Studies ⚠️ MAJOR

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 1.2, PEER_REVIEW_CRITICAL_ANALYSIS Section 1.2, PEER_REVIEW_CRITICAL_ISSUES Section 1.3]

**Issue**: No systematic ablation experiments to isolate which distillation components contribute to improvement.

**Evidence from Code Investigation**:
- No ablation experiment directories found
- No parameter sweep code beyond Optuna trials
- Documentation mentions "Set lambda=0 to disable distillation" but no systematic sweep
- Documentation mentions "sweep temperature" (`docs/LMTAD-Distillation.md` line 1827) but no results provided
- Optuna trials provide SOME implicit ablation (different λ, τ, window) but not systematic

**What's Missing**:
1. **Temperature τ ablation**: No systematic sweep [1.0, 2.0, 3.0, 4.0, 5.0] with fixed λ
2. **Lambda λ ablation**: No systematic sweep [0.0, 0.001, 0.01, 0.1] with fixed τ
3. **Window size ablation**: Systematic sweep [1, 2, 4, 7, 10] with fixed λ, τ (partially covered in Optuna but not systematic)
4. **Component isolation**: Which matters more - temperature, lambda, or window?
5. **Teacher quality ablation**: No evaluation of teacher itself

**Dataset-Specific Hyperparameter Differences** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:59-62`: Beijing hyperparameters (λ=0.0014, τ=4.37, w=7) differ dramatically from Porto (λ=0.00644, τ=2.802, w=4)
- `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md:94-97`: Phase 1 Porto parameters differ from Phase 2
- No analysis of why Beijing needs different τ/window than Porto

**Optuna Limitations**:
- `docs/LMTAD-Distillation.md:821-829`: Parameter ranges provided but no sensitivity curves
- No sensitivity plots to determine if results are robust to small hyperparameter changes
- No evidence that Optuna found global optimum vs local optimum
- Interaction effects: λ and τ interact (higher τ may require different λ), but this is not explored

**Impact**: 
- Cannot determine which aspects of distillation contribute to improvement
- The improvement could be entirely from temperature-scaled regularization, not teacher knowledge
- Cannot isolate distillation effect from regularization effect
- Cannot assess robustness to hyperparameter choices

**Files Examined**:
- `tune_hoser.py` - only Optuna-based hyperparameter search
- `docs/LMTAD-Distillation.md` line 1827: mentions "sweep temperature" but no results
- `docs/LMTAD-Distillation.md:821-829`: Parameter ranges
- No ablation analysis scripts found

**Required Fix**:
1. Run systematic ablation: fix 2 parameters, sweep 1
2. Report which component has largest impact
3. Test if λ=0 (no distillation) with temperature-scaling still helps (would prove it's regularization)
4. **Sensitivity analysis**: Fix two hyperparameters, vary third, plot validation accuracy surface
5. **Interaction plots**: 2D heatmaps showing λ×τ, λ×window, τ×window interactions
6. **Stability analysis**: Re-run best hyperparameters with different seeds, report variance
7. **Cross-dataset validation**: Test Beijing hyperparameters on Porto (and vice versa) to assess transferability

**References**:
- `docs/LMTAD-Distillation.md` line 1827
- `docs/LMTAD-Distillation.md:2489-2608`
- `tune_hoser.py` - only Optuna-based search, not systematic ablations
- `tune_hoser.py:1-94`

---

### 1.3 Teacher Model Baseline ⚠️ MAJOR

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 1.3, PEER_REVIEW_CRITICAL_ANALYSIS Section 1.3]

**Issue**: LM-TAD teacher is NEVER evaluated as trajectory generator, yet the core claim is "teacher transfers spatial knowledge."

**Evidence from Code**:
- `critics/lmtad_teacher.py` only provides probability distributions
- No evaluation code for teacher standalone
- No teacher accuracy metrics logged
- Teacher is only used to provide soft targets during training
- No teacher evaluation scripts found

**What's Missing**:
1. Teacher next-road prediction accuracy on validation set
2. Teacher OD completion rate if used for generation
3. Teacher trajectory generation capability
4. Comparison: vanilla vs teacher vs distilled (all evaluated same way)

**Impact**: 
Core claim "teacher transfers spatial knowledge" has **ZERO evidence**. The teacher may be:
- Worse than student for trajectory prediction
- Only good at anomaly detection (original task)
- Providing no useful spatial knowledge

Without evaluating the teacher, the fundamental premise of the research cannot be validated.

**Files Examined**:
- `critics/lmtad_teacher.py` - wrapper only, no evaluation
- `tools/export_lmtad_weights.py` - export only
- No teacher evaluation scripts found

**Required Fix**:
1. Evaluate LM-TAD on trajectory prediction task (next-road accuracy)
2. Generate trajectories using teacher alone (if possible)
3. Compare all three: vanilla < teacher < distilled? Or teacher < vanilla < distilled?
4. If teacher is worse than vanilla, the distillation explanation fails
5. Test spatial reasoning directly (not just trajectory generation)

**References**:
- `critics/lmtad_teacher.py` - wrapper only
- `tools/export_lmtad_weights.py` - export only

---

### 1.4 Vocabulary Mapping Validation ⚠️ CRITICAL

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 1.1]

**Issue**: The many-to-one road→grid token mapping introduces systematic artifacts that are not validated or controlled.

**Evidence**:
- `docs/LMTAD-Distillation.md:679-685`: Mapping formula uses road centroids → grid cells
- `docs/LMTAD-Distillation.md:798-801`: Acknowledges "many-to-one mapping is acceptable" but provides no validation
- `tools/map_road_networks.py:173-194`: Tracks many-to-one mappings but this analysis is not reported for distillation

**Problems**:
1. **Information loss**: Multiple roads (40,060) map to fewer grid cells (51,663), but many roads share the same grid token
2. **Artifact introduction**: Teacher probabilities for a grid token are distributed across multiple candidate roads, potentially creating false distillation signals
3. **No ablation**: No experiment testing whether distillation benefit persists with shuffled/random grid mappings
4. **No validation**: No report of how many roads map to each grid cell, distribution of many-to-one ratios

**Impact**:
Cannot determine if improvements come from actual knowledge transfer or mapping artifacts. If many-to-one cases benefit more from distillation, this suggests the improvement is an artifact of the mapping, not genuine spatial knowledge transfer.

**Required Actions**:
1. **Report mapping statistics**: Distribution of roads-per-grid-cell, maximum roads per cell, spatial clustering of many-to-one cases
2. **Ablation study**: Compare distillation with:
   - Random road→grid permutation (controls for mapping artifacts)
   - Identity mapping (if possible) or 1:1 approximation
   - Shuffled teacher outputs (controls for actual knowledge transfer)
3. **Validation metric**: Compute correlation between distillation benefit and roads-per-cell ratio (if many-to-one cases benefit more, suggests artifact)

**Citation**: 
- `docs/LMTAD-Distillation.md:679-685`
- `docs/LMTAD-Distillation.md:798-801`
- `docs/LMTAD-Distillation.md:1825`
- `critics/grid_mapper.py:98`
- `tools/map_road_networks.py:173-194`

---

### 1.5 KL Divergence Direction and Justification ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 1.2]

**Issue**: Forward KL (teacher→student) is used without justification, and alternative directions are not explored.

**Evidence**:
- `docs/LMTAD-Distillation.md:810-814`: States forward KL "matches our goal" but provides no theoretical or empirical justification
- No ablation comparing forward vs reverse KL
- No discussion of mode-seeking vs mean-seeking behavior

**Problems**:
1. **Direction choice**: 
   - Forward KL `KL(q||p)` penalizes student for being uncertain where teacher is confident (mode-seeking)
   - Reverse KL `KL(p||q)` would penalize student for being overconfident (mean-seeking)
   - The choice fundamentally changes what knowledge is transferred
2. **No empirical validation**: No comparison showing forward KL is better than reverse KL
3. **Temperature interaction**: Temperature scaling interacts differently with forward vs reverse KL, but this interaction is not analyzed

**Impact**:
The choice of KL direction has fundamental implications for what is learned. Without justification and ablation, cannot determine if forward KL is optimal or even appropriate for this task.

**Required Actions**:
1. **Ablation**: Train models with reverse KL, compare validation accuracy and trajectory metrics
2. **Theoretical justification**: Provide mathematical argument for why mode-seeking is preferable for trajectory prediction
3. **Report**: Document why forward KL was chosen and whether reverse KL was tested

**Citation**: 
- `docs/LMTAD-Distillation.md:760-775`
- `docs/LMTAD-Distillation.md:810-814`

---

### 1.6 Candidate Top-K Filtering Bias ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 1.4]

**Issue**: Top-k candidate filtering (k=64) may introduce systematic bias that favors certain model types.

**Evidence**:
- `docs/LMTAD-Distillation.md:1778-1782`: Candidate filtering enabled with k=64
- `docs/LMTAD-Distillation.md:1613-1620`: Top-k filtering prevents "pathological traces with 1000+ candidates"
- No ablation testing different k values or no filtering

**Problems**:
1. **Information loss**: Distillation operates only on filtered candidates, so teacher knowledge outside top-k is ignored
2. **Selection bias**: If teacher's preferred roads are filtered out by top-k (by distance), distillation cannot help
3. **Evaluation mismatch**: Training uses top-k, but evaluation uses full candidate set (or different k?)
4. **No analysis**: No report of how often teacher's top-1 candidate is outside student's top-64

**Impact**:
If the teacher's best suggestions are systematically filtered out, the distillation cannot work as intended. The value k=64 is arbitrary and may be sub-optimal.

**Required Actions**:
1. **Overlap analysis**: Report fraction of timesteps where teacher's top-1 candidate is in student's top-k
2. **Ablation**: Train/evaluate with k=32, 64, 128, no filtering, compare results
3. **Teacher-aware filtering**: Test filtering candidates by teacher probability instead of distance
4. **Documentation**: Clarify whether evaluation also uses top-k or full candidate set

**Citation**: 
- `docs/LMTAD-Distillation.md:1613-1620`
- `docs/LMTAD-Distillation.md:1778-1782`
- `config/Beijing.yaml` (data.candidate_top_k)

---

### 1.7 Missing Calibration Metrics ⚠️ CRITICAL

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 1.5]

**Issue**: No calibration metrics (ECE, Brier score) are reported, so we cannot assess whether distillation improves uncertainty quantification.

**Evidence**:
- `docs/LMTAD-Distillation.md:84`: Claims "calibrated uncertainty" but no metrics provided
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:833-910`: Validation metrics include accuracy and MAPE, but no calibration
- No reliability diagrams or confidence calibration plots

**Problems**:
1. **Unverified claim**: "Calibrated uncertainty" is stated but never measured
2. **Missing evaluation**: Calibration is critical for deployment (knowing when model is uncertain)
3. **Temperature interaction**: Temperature affects calibration, but this is not analyzed
4. **No post-hoc calibration**: No attempt to calibrate predictions using temperature scaling or Platt scaling

**Impact**:
A key claimed benefit of distillation is better calibrated uncertainty, but this is never verified. The claim should either be removed or validated with proper metrics.

**Required Actions**:
1. **Compute ECE**: Expected Calibration Error on validation set (bin predictions by confidence, compare to accuracy)
2. **Brier score**: Measure probability calibration (lower is better)
3. **Reliability diagrams**: Plot predicted confidence vs actual accuracy
4. **Temperature sweep**: Test post-hoc temperature scaling to improve calibration
5. **Remove or verify**: Either remove "calibrated uncertainty" claim or provide evidence

**Citation**: 
- `docs/LMTAD-Distillation.md:84`
- `docs/LMTAD-Distillation.md:831-910`
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:833-910`

---

## Section 2: Statistical Rigor Issues

### 2.1 Multiple Testing Without Correction ⚠️ CRITICAL

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 2.2, PEER_REVIEW_CRITICAL_ANALYSIS Section 2.1, PEER_REVIEW_CRITICAL_ISSUES Section 3.2]

**Issue**: Bonferroni correction is CLAIMED in documentation but NOT IMPLEMENTED in code.

**Evidence from Code** (`tools/analyze_wang_results.py`):
- Line 568: `"significant": p_value < 0.05` - uses 0.05, not 0.001
- Line 558: `chi2, p_value = stats.chi2_contingency(contingency)[:2]` - no correction applied
- No Bonferroni, FDR, or FWER correction code found

**Evidence from Documentation**:
- `WANG_ABNORMALITY_DETECTION_RESULTS.md` line 126: Claims "α = 0.001 (Bonferroni correction)"
- But code uses `p < 0.05` threshold

**CONTRADICTION**: Documentation claims Bonferroni correction, but code doesn't implement it.

**Scale of Problem**:
- Wang analysis: 12 models × 2 splits = 24 comparisons
- With α=0.05 and 24 tests, expect ~1.2 false positives by chance
- Porto Phase 1: 6 models × 9 scenarios × 6 metrics = 324 potential comparisons
- With 324 comparisons at α=0.05, expect ~16 false positives
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:344-380`: 6 tests for Beijing, 4 for Porto, 3 for BJUT (13 total tests)
- Bonferroni with 13 tests would require α = 0.001/13 ≈ 0.000077 per test

**Impact**: 
"Statistically significant" findings may be noise. Documentation misleads readers. With huge sample sizes and multiple comparisons, some "significant" results are likely false discoveries.

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **False discovery inflation**: With 13 tests at α=0.001, expect 0.013 false discoveries, but with no correction, false discovery rate is much higher
2. **Inconsistent threshold**: Documentation claims Bonferroni but code uses 0.05, results use 0.001
3. **No correction applied**: No evidence that p-values are adjusted for multiple comparisons
4. **Interpretation risk**: Some "significant" results may be false discoveries

**Files Examined**:
- `tools/analyze_wang_results.py` lines 530-588, specifically lines 558, 568
- `wang_results_aggregated.json` - p-values stored but no correction applied
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` line 126

**Required Fix**:
1. Implement Bonferroni correction: `alpha_adjusted = 0.05 / num_comparisons`
2. Or implement FDR (False Discovery Rate) correction using Benjamini-Hochberg
3. Report adjusted p-values alongside raw p-values
4. Update documentation to match implementation
5. Re-interpret results with corrected thresholds
6. **Report adjusted p-values**: Include both raw and adjusted p-values in tables
7. **Reinterpret results**: Re-assess significance after correction

**References**:
- `tools/analyze_wang_results.py` lines 530-588, 558, 568
- `tools/analyze_wang_results.py:558-571`
- `wang_results_aggregated.json`
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` line 126
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:344-380`

---

### 2.2 Missing Effect Sizes and Confidence Intervals ⚠️ CRITICAL

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 3.1]

**Issue**: Statistical significance tests (chi-square) are reported without effect sizes or confidence intervals, making it impossible to assess practical significance.

**Evidence**:
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:344-380`: Chi-square tests with p-values, but no effect sizes
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:383-390`: "Practical significance" section acknowledges issue but doesn't provide CIs
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md`: No statistical tests reported at all
- No bootstrap confidence intervals or standard errors

**Problems**:
1. **Statistical vs practical**: p < 0.001 tells us distributions differ, but not by how much
2. **No effect sizes**: Cohen's h, relative risk, or odds ratios not reported
3. **No uncertainty quantification**: Point estimates without confidence intervals
4. **Cannot compare improvements**: 3.5% vs 50% deviation are both "significant" but very different magnitudes

**Impact**:
With huge sample sizes (179,823 test trajectories), even trivial 0.1% differences would be "statistically significant" but meaningless. Cannot distinguish between small technical differences and large practical improvements.

**Required Actions**:
1. **Report effect sizes**: 
   - Cohen's h for proportion differences: `h = 2 * (arcsin(√p1) - arcsin(√p2))`
   - Relative risk: `RR = p_generated / p_real`
   - Odds ratio: `OR = (p_gen/(1-p_gen)) / (p_real/(1-p_real))`
2. **Bootstrap CIs**: Compute 95% confidence intervals using bootstrap (1000 resamples)
3. **Update tables**: Add effect size and CI columns to all statistical test results
4. **Interpretation**: Classify effect sizes (small/medium/large) using standard thresholds

**Citation**: 
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:344-390`
- `tools/analyze_wang_results.py:525-588`
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md`

---

### 2.3 Sample Size Not Justified ⚠️ MODERATE

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 2.1, PEER_REVIEW_CRITICAL_ANALYSIS Section 2.2]

**Issue**: 5,000 generated trajectories with no power analysis or justification.

**Evidence**:
- `gene.py` line 2207: `default=5000` with no comment or justification
- No power analysis code found
- No sample size calculation documentation
- Appears to be arbitrary choice

**Questions**:
- What effect size are you powered to detect? (1%, 5%, 10% improvement?)
- Why 5,000 and not 1,000 or 10,000?
- Given real data has 629K trajectories, is 5K representative?

**Impact**: 
May be over- or under-powered for effect sizes of interest. May miss small but meaningful differences, or waste computation on unnecessarily large samples.

**Files Examined**:
- `gene.py` line 2207 - default parameter only
- No documentation justifying sample size

**Required Fix**:
1. Perform power analysis: calculate required sample size for detecting 1%, 5%, 10% improvements
2. Justify choice of 5,000 with effect size target
3. Report confidence intervals, not just point estimates
4. Consider that trajectory-level metrics may need different sample sizes than distribution-level metrics

**References**:
- `gene.py` line 2207

---

### 2.4 Coefficient of Variation Misuse ⚠️ MODERATE

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 2.3, PEER_REVIEW_CRITICAL_ANALYSIS Section 2.3]

**Issue**: CV% reported for metrics where it's meaningless or misleading.

**Evidence**:
- `hoser-distill-optuna-6/create_analysis_figures.py` line 595: CV calculation
- `hoser-distill-optuna-6/create_analysis_figures.py` lines 591-599
- Only 2-3 seeds used per experiment
- CV computed on near-zero metrics (JSD ~0.016-0.022)
- Documentation reports CV% extensively

**Problems**:
1. With only 2 seeds (42, 44), CV has huge sampling error
2. JSD near zero makes relative measures unstable
3. Should report absolute differences, not CV
4. With only 3 seeds, statistical power is very limited

**Example**: Distance JSD has CV = 8.9% with only 2 seeds means the variation between seed 42 and seed 44 is 8.9% of the average value, but with n=2 this is not statistically meaningful.

**Status** [Source: CRITICAL_REVIEW_FINDINGS]:
Investigation ongoing...

**Required Fix**:
1. Report range [min, max] instead of CV for n=2-3
2. Report mean ± std or [min, max] format
3. Acknowledge limited statistical power with small n
4. Only use CV for metrics that are meaningfully positive (>0.1)

**References**:
- `hoser-distill-optuna-6/create_analysis_figures.py` lines 591-599, line 595
- Documentation reports CV% extensively

---

### 2.5 Missing Paired Statistical Tests ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 3.3]

**Issue**: Trajectory-level metrics (DTW, Hausdorff, EDR) compare individual trajectories but use unpaired statistical tests (if any), missing the paired nature of the comparison.

**Evidence**:
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:629-683`: Reports mean DTW, Hausdorff, EDR but no statistical tests
- Trajectories are paired (same OD pair, same real trajectory), but tests don't account for pairing
- No McNemar test for binary outcomes (OD match)
- No paired t-test or Wilcoxon signed-rank for continuous metrics

**Problems**:
1. **Reduced statistical power**: Paired tests are more powerful than unpaired for matched data
2. **Wrong test type**: Unpaired tests assume independence, but trajectories are matched by OD pair
3. **Missing tests**: No statistical tests reported for trajectory-level metrics at all
4. **Cannot assess significance**: We don't know if 28 km vs 8 km DTW difference is statistically significant

**Impact**:
Without proper paired tests, cannot determine if trajectory-level differences are statistically meaningful.

**Required Actions**:
1. **Paired tests**: For each OD pair, compare vanilla vs distilled trajectory metrics
2. **McNemar test**: For binary OD match outcomes, use McNemar's test (paired chi-square)
3. **Wilcoxon signed-rank**: For continuous metrics (DTW, Hausdorff), use paired non-parametric test
4. **Report results**: Include p-values, effect sizes, and CIs for all trajectory-level comparisons
5. **Acknowledge pairing**: Document that trajectories are matched by OD pair

**Citation**: 
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:629-683`

---

### 2.6 No Cross-Seed Statistical Analysis ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 3.4]

**Issue**: Multiple seeds (42, 43, 44) are used but no statistical analysis across seeds to assess robustness.

**Evidence**:
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:604-623`: Reports CV (coefficient of variation) but no statistical tests
- `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md:580-593`: Reports CV% but no tests
- No ANOVA or mixed-effects model to test if seed effect is significant
- No report of whether seed differences are statistically significant

**Problems**:
1. **No significance test**: CV < 15% is "low variability" but we don't know if it's significantly different from high variability
2. **Missing analysis**: No test of whether distilled models have different seed variance than vanilla
3. **No interaction**: No test of seed × model type interaction
4. **Interpretation risk**: Low CV might be due to small sample size (3 seeds), not actual stability

**Impact**:
Cannot determine if observed seed-to-seed variation is statistically meaningful or if distilled models are truly more stable than vanilla.

**Required Actions**:
1. **ANOVA**: Test if seed effect is significant (F-test)
2. **Levene's test**: Test if variance differs between model types (distilled vs vanilla)
3. **Mixed-effects model**: If applicable, model seed as random effect, test fixed effects
4. **Bootstrap**: If only 3 seeds, use bootstrap to estimate CI for CV
5. **Report**: Include statistical tests for seed effects in results

**Citation**: 
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:604-623`
- `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md:580-593`

---

## Section 3: Evaluation Methodology Issues

### 3.1 Beam Search Configuration ⚠️ CRITICAL/MAJOR

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 3.1, PEER_REVIEW_CRITICAL_ANALYSIS Section 3.1, PEER_REVIEW_CRITICAL_ISSUES Section 2.1]

**Issue**: Evaluation metrics depend on beam search parameters (width, termination), creating evaluation-dependent results that may not reflect model quality.

**Evidence**:
- `gene.py` line 1829: `beam_width: int = 4` (programmatic default)
- `gene.py` line 2220: `default=8` (CLI default) - **INCONSISTENT!**
- Documentation claims "optimal for 11GB VRAM" - hardware constraint, not scientific choice
- No ablation of beam width reported
- `docs/LMTAD-Distillation.md:1857-1904`: Beam search width 4 chosen for "optimal evaluation"
- `docs/LMTAD-Distillation.md:1899-1903`: Claims "95-99% correlation with optimal A*" but no evidence provided
- `gene.py:455-505`: Beam search termination when destination reached or max_search_step=5000
- No comparison of metrics across different beam widths

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **Evaluation-dependent results**: Metrics change with beam width, so improvements may be artifacts of search, not model quality
2. **Unverified claim**: "95-99% correlation with A*" is not supported by data or citations
3. **Termination bias**: Early termination (max_search_step=5000) may favor models that reach destinations faster, not necessarily better models
4. **No ablation**: No report of how metrics change with beam width 1, 2, 4, 8, 16

**Missing Controls**:
1. Beam width ablation: test width=1,2,4,8,16
2. Search budget normalization: give vanilla more search steps
3. Greedy decoding baseline: test both models without beam search

**Impact**: 
- Beam width choice may affect results differently for vanilla vs distilled
- No ablation = unknown if results are confounded by search algorithm parameters
- Distilled model might just be better at pruning during beam search, not generating better trajectories
- The comparison is unfair if beam search benefits distilled models differently

**Files Examined**:
- `gene.py` lines 1829, 2220, 455-505
- `docs/LMTAD-Distillation.md` lines 1894-1903, 1857-1904

**Required Fix**:
1. Ablate beam width: test both models at width=1,2,4,8
2. Report results across beam widths
3. Normalize by search budget if needed
4. Test greedy decoding (width=1) as baseline
5. **Beam width sweep**: Generate trajectories with beam width 1, 2, 4, 8, report all metrics for each
6. **Verify A* claim**: Either provide data showing 95-99% correlation or remove claim
7. **Termination analysis**: Report fraction of trajectories that hit max_search_step vs reach destination
8. **Fair comparison**: Ensure both vanilla and distilled models use identical beam search parameters and termination criteria
9. **Alternative metrics**: Report metrics that are less dependent on search (e.g., per-step accuracy on fixed trajectories)

**References**:
- `gene.py` lines 1829, 2220, 455-505
- `docs/LMTAD-Distillation.md` lines 1894-1903, 1857-1904

---

### 3.2 OD Matching Methodology ⚠️ MODERATE/MAJOR

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 3.2, PEER_REVIEW_CRITICAL_ANALYSIS Section 3.2]

**Issue**: Grid-based OD matching (0.001° = 111m) is arbitrary and dataset-dependent, with no sensitivity analysis.

**Evidence**:
- `evaluation.py` line 951: `grid_size: float = 0.001` - default parameter
- `python_pipeline.py` line 136: `self.grid_size = 0.001` - hardcoded default
- No sensitivity analysis code found
- Documentation claims "consistent with Beijing" - circular reasoning

**Problems**:
1. 111m in city center covers multiple roads, in suburbs covers fewer
2. Different grid sizes would give different match rates
3. Endpoints within 110m = "match", 112m = "fail" - discontinuous and arbitrary
4. No sensitivity analysis to grid size

**Impact**: 
- Match rate metric is fragile to grid size choice. 110m vs 112m = match vs fail.
- A model that consistently ends 115m away from target looks worse than one that randomly ends 0-220m away (50% within 111m)
- Match rate metric may not reflect actual navigation quality

**Files Examined**:
- `evaluation.py` line 951 - grid_size parameter
- `python_pipeline.py` line 136
- No grid size sensitivity analysis found

**Required Fix**:
1. Test grid sizes: [0.0005, 0.001, 0.002] degrees
2. Report how match rates change with grid size
3. Use multiple metrics, not just one arbitrary grid size
4. Consider distance-based matching instead of grid-based

**References**:
- `evaluation.py` line 951
- `python_pipeline.py` line 136

---

### 3.3 JSD Binning Inconsistency ⚠️ CRITICAL

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 3.3, PEER_REVIEW_CRITICAL_ANALYSIS Section 3.3, PEER_REVIEW_CRITICAL_ISSUES Section 2.3]

**Issue**: Documentation states 50 bins for JSD, but code uses 100 bins, creating inconsistency and potential confusion.

**Evidence from Code** (`evaluation.py` line 578):
```python
bins = np.linspace(0, real_max, 100).tolist()
```

**Evidence from Documentation**:
- `EVALUATION_ANALYSIS.md` line 143: "Bins: 50 equal-width bins"
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:142`: "Bins: 50 equal-width bins across distance range"
- `docs/LMTAD-Distillation.md:2142`: States JSD calculation but doesn't specify bins

**CONTRADICTION**: Code uses 100 bins, documentation claims 50.

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **Documentation error**: Documentation and code don't match
2. **Reproducibility risk**: Readers may implement 50 bins based on docs, get different results
3. **No justification**: Neither 50 nor 100 bins is justified (why not 20, 200, adaptive binning?)
4. **Sensitivity unknown**: No analysis of how JSD changes with different bin counts

**Impact**: 
JSD values depend on bin count. Documentation misrepresents actual implementation. Different bin counts would give different JSD values. Your "87% JSD improvement" claims depend critically on an arbitrary binning choice.

**Files Examined**:
- `evaluation.py` line 578 - actual implementation (100 bins)
- `evaluation.py:576-580`
- Documentation claims 50 bins
- `EVALUATION_ANALYSIS.md` line 143
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:142`

**Required Fix**:
1. Fix documentation to match code (100 bins)
2. Or fix code to match documentation (50 bins)
3. Test sensitivity to bin count: [25, 50, 100, 200]
4. Report how JSD changes with bin count
5. Justify bin count choice (not just "50 bins")
6. **Justify binning**: Provide rationale for chosen bin count (e.g., Freedman-Diaconis rule, Sturges' rule, or empirical sensitivity)
7. **Sensitivity analysis**: Test JSD with 20, 50, 100, 200 bins, report if results are robust
8. **Code comment**: Add comment in evaluation.py explaining why 100 bins

**References**:
- `evaluation.py` line 578, lines 576-580
- `EVALUATION_ANALYSIS.md` line 143
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:142`
- `docs/LMTAD-Distillation.md:2142`

---

### 3.4 Local Metrics Normalization ⚠️ MAJOR

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 3.4, PEER_REVIEW_CRITICAL_ANALYSIS Section 3.4, PEER_REVIEW_CRITICAL_ISSUES Section 2.4]

**Issue**: DTW and Hausdorff distances are not consistently normalized by trajectory length, making comparisons across different trip lengths difficult.

**Evidence from Code** (`evaluation.py`):
- Line 751: `return max(h_u_v, h_v_u)` - Hausdorff NOT normalized
- Line 765: `return dist` - DTW NOT normalized (returns raw km)
- Line 818: `return float(C[n0][n1]) / max([n0, n1])` - EDR IS normalized

**Evidence from Documentation**:
- `EVALUATION_ANALYSIS.md` line 233: "Not normalized by trajectory length"
- But then line 639: "Lower Hausdorff/DTW for vanilla is not better - it reflects shorter trips"
- Documentation acknowledges issue but doesn't normalize in code
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:636-653`: Acknowledges DTW scales with length, provides "DTW per km" calculation
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:200-206`: Hausdorff caveat notes it "scales with trajectory length" but no normalization applied
- Results tables report raw DTW/Hausdorff, not normalized versions

**Evidence from Results**:
- Beijing: Vanilla Hausdorff 0.51km vs Distilled 0.96km (vanilla is CLOSER)
- Beijing: Vanilla DTW 7.7km vs Distilled 28.0km (vanilla is CLOSER)

**Documentation Hand-Waving**:
- Documentation acknowledges normalization is needed, then doesn't normalize
- `EVALUATION_ANALYSIS.md` line 639: "Lower Hausdorff/DTW for vanilla is not better - it reflects shorter trips"

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **Length bias**: Longer trajectories naturally have higher DTW/Hausdorff, so comparisons are unfair
2. **Inconsistent normalization**: DTW per km is mentioned but not consistently reported in tables
3. **Hausdorff not normalized**: No per-km normalization for Hausdorff, even though it also scales with length
4. **Interpretation difficulty**: Raw values are hard to interpret (is 28 km DTW good for a 6.4 km trip?)

**Impact**: 
Vanilla's lower Hausdorff (0.51km) vs distilled (0.96km) means vanilla trajectories are CLOSER to real paths. This contradicts the "distilled is better" narrative. Documentation hand-waves this away instead of addressing it directly. Your own trajectory-level metrics suggest vanilla produces paths more similar to ground truth, contradicting your distribution-level conclusions.

**Files Examined**:
- `evaluation.py` lines 736-766 (Hausdorff, DTW implementations)
- `evaluation.py` lines 751, 765, 818
- `evaluation.py:598-750`
- Documentation interpretation
- `EVALUATION_ANALYSIS.md` lines 629-653

**Required Fix**:
1. Normalize Hausdorff and DTW by trajectory length: `H_norm = H / trajectory_length`
2. Report normalized metrics alongside raw metrics
3. Address the contradiction: if vanilla is closer to real paths, why is distilled "better"?
4. Consider that shorter trajectories may be more realistic for some OD pairs
5. Don't dismiss metrics that contradict your narrative
6. **Normalize all metrics**: Report DTW/distance and Hausdorff/distance ratios in addition to raw values
7. **Update tables**: Include normalized columns in all results tables
8. **Interpretation guide**: Provide thresholds for normalized metrics (e.g., DTW/distance < 5.0 is good)
9. **Statistical tests**: Perform significance tests on normalized metrics, not raw metrics

**References**:
- `evaluation.py` lines 736-766, 751, 765, 818
- `evaluation.py:598-750`
- `EVALUATION_ANALYSIS.md` lines 629-653, line 233, line 639
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:636-653`
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:200-206`

---

### 3.5 OD Match Definition Confusion ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 2.2]

**Issue**: OD match rate conflates path completion success with endpoint realism, making it unclear what is being measured.

**Evidence**:
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:272-293`: OD match uses "actual trajectory endpoints" not "input request"
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:277-287`: Explains that trajectories may end at intermediate locations, creating (A,Y) instead of (A,Z)
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:289-292`: Interpretation conflates "path completion" with "realistic endpoints"

**Problems**:
1. **Metric confusion**: OD match rate measures two things: (a) whether model reached intended destination, (b) whether endpoint exists in real data. These should be separate metrics.
2. **Interpretation difficulty**: 85% match rate could mean: (a) 85% reached destination, (b) 85% ended at realistic locations (even if wrong destination), or (c) combination
3. **No breakdown**: No report of: destination reached rate, endpoint realism rate, both achieved rate

**Impact**:
Cannot interpret what 85% match rate actually means. The metric conflates two distinct aspects of trajectory quality.

**Required Actions**:
1. **Separate metrics**: 
   - Path completion rate: % trajectories that reached intended destination
   - Endpoint realism rate: % trajectories whose actual endpoint exists in real OD pairs (regardless of intended destination)
   - Combined match rate: % trajectories that both reached destination AND endpoint is realistic
2. **Report all three**: Provide clear breakdown in results tables
3. **Clarify interpretation**: Update documentation to distinguish these metrics

**Citation**: 
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:272-293`
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:277-287`
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:289-292`

---

## Section 4: Wang Abnormality Detection Issues

### 4.1 Data Leakage in Baseline Computation ⚠️ CRITICAL

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 4.1]

**Issue**: Training and test sets are pooled to compute OD-pair baselines, creating data leakage that inflates detection accuracy.

**Evidence**:
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:75-76`: "Load all real trajectories (train + test combined)"
- `docs/reference/BASELINE_STATISTICS.md:13`: "Step 1: Load all real trajectories (train + test combined)"
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:166-167`: Evaluation uses "train split" and "test split" but baselines computed from both

**Problems**:
1. **Data leakage**: Test set information (OD patterns) leaks into baseline computation
2. **Inflated performance**: Baselines are more accurate because they include test data
3. **Unrealistic evaluation**: In real deployment, baselines would be computed only from training data
4. **No ablation**: No comparison of detection rates with train-only vs train+test baselines

**Impact**:
This is a fundamental violation of machine learning evaluation principles. Results cannot be trusted because the evaluation uses information from the test set. The abnormality detection rates are likely artificially lower (better) than they would be in a real deployment scenario.

**Required Actions**:
1. **Recompute baselines**: Use only training data for baseline computation
2. **Re-evaluate**: Re-run all abnormality detection with train-only baselines
3. **Compare results**: Report difference in abnormality rates between train-only and train+test baselines
4. **Acknowledge**: Document that original results used train+test baselines (if keeping for comparison)
5. **Update methodology**: Fix baseline computation to use only training data going forward

**Citation**: 
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:75-91`
- `docs/reference/BASELINE_STATISTICS.md:13`
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:166-167`

---

### 4.2 Threshold Justification ⚠️ MODERATE

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 6.1, PEER_REVIEW_CRITICAL_ANALYSIS Section 5.1, PEER_REVIEW_CRITICAL_ISSUES Section 4.3]

**Issue**: Hybrid threshold strategy uses 5km/5min fixed thresholds with no justification or sensitivity analysis.

**Evidence**:
- `tools/detect_abnormal_statistical.py` line 51: "default: 5000m from paper"
- `config/abnormal_detection_statistical.yaml` line 26: "# Fixed thresholds from Wang et al. 2018 paper"
- No sensitivity analysis code found
- No dataset-specific tuning
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:117-120`: Hybrid strategy uses minimum of fixed and statistical thresholds
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:172-179`: Configuration shows fixed thresholds (5km, 5min) and sigma (2.5)
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:504`: Lists "adaptive thresholds" as future work but doesn't test current thresholds

**Questions**:
- Why 5km and not 3km or 10km?
- Are these derived from data (e.g., 95th percentile) or arbitrary?
- Different cities have different "normal" deviations
- Beijing average trip: 5.16km, so 5km threshold is ~100% of average (very strict!)

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **Arbitrary thresholds**: 5km, 5min, 2.5σ are chosen without justification
2. **No sensitivity**: No analysis of how abnormality rates change with different thresholds
3. **Dataset invariance**: Fixed thresholds (5km/5min) may not be appropriate for all datasets (BJUT has 2.6km avg, Porto 4.0km)
4. **Interaction unknown**: How does min(5km, 2.5σ) behave when distributions have different variances?

**Impact**: 
Abnormality rates heavily depend on threshold choice. Different cities may need different thresholds. No evidence thresholds are appropriate for Beijing/Porto. "22% abnormal in Beijing" may be artifact of threshold, not reality.

**Files Examined**:
- `tools/detect_abnormal_statistical.py` lines 51-52
- `config/abnormal_detection_statistical.yaml` lines 26-27
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:117-120`
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:172-179`

**Required Fix**:
1. Test sensitivity to thresholds: [3km, 5km, 7km, 10km]
2. Derive thresholds from data: use 95th percentile of deviations
3. Dataset-specific thresholds: Beijing may need different values than Porto
4. Report how abnormality rates change with thresholds
5. **Sensitivity sweep**: Test abnormality detection with:
   - Fixed thresholds: 3km, 5km, 7km (distance) and 3min, 5min, 7min (time)
   - Sigma multipliers: 2.0, 2.5, 3.0
   - Hybrid vs fixed-only vs statistical-only
6. **Report results**: Create table showing abnormality rates across threshold combinations
7. **Justify choice**: Explain why 5km/5min/2.5σ were chosen (e.g., based on data distribution, literature)
8. **Dataset-specific**: Consider whether fixed thresholds should scale with dataset characteristics (e.g., 5km for Beijing but 3km for BJUT)

**References**:
- `tools/detect_abnormal_statistical.py` lines 51-52
- `config/abnormal_detection_statistical.yaml` lines 26-27
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:117-120, 172-179, 504`
- `config/abnormal_detection_statistical.yaml`

---

### 4.3 OD Pair Coverage ⚠️ MODERATE/CRITICAL

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 6.2, PEER_REVIEW_CRITICAL_ANALYSIS Section 5.2, PEER_REVIEW_CRITICAL_ISSUES Section 4.2]

**Issue**: Only 0.5-0.6% of OD pairs have ≥5 samples, meaning 99.4% of trajectories use global baselines, undermining the "OD-pair-specific" methodology.

**Evidence**:
- `tools/compute_trajectory_baselines.py` line 229: `coverage_pct` calculation
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` line 87: "Beijing: 712,435 OD pairs (0.6% with ≥5 samples)"
- `min_samples_per_od: 5` is configurable but set to 5
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:86-91`: Beijing 0.6%, BJUT 0.5%, Porto 5.1% coverage
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:91`: "Most trajectories use global baselines due to extreme OD diversity"
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:496`: Lists this as "limitation" but doesn't address it

**Documentation Claim is CORRECT** [Source: CRITICAL_REVIEW_FINDINGS]:
Only 0.5-0.6% coverage confirmed.

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **Methodology mismatch**: Claims "OD-pair-specific baselines" but 99% of data uses global baselines
2. **Sensitivity unknown**: No analysis of how results change if we use stricter thresholds (≥10, ≥20 samples)
3. **No stratification**: Results don't report separate abnormality rates for OD-specific vs global baseline trajectories
4. **Misleading name**: "OD-pair-specific" is misleading when most data uses global baselines

**Impact**: 
- 99%+ of trajectories use GLOBAL baseline, not OD-specific
- The "OD-pair-specific" methodology is misleading - it's actually "global baseline with rare OD-specific exceptions."
- Your "OD-pair-specific baselines" suggests per-OD adaptation, but it's global for almost all trajectories
- Results would be nearly identical if you just used global mean ± 2.5σ

**Files Examined**:
- `tools/compute_trajectory_baselines.py` lines 110-275, 225-241, line 229
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` lines 86-91, line 87, line 91, line 496
- Baseline computation logic
- `docs/reference/BASELINE_STATISTICS.md:40-47`

**Required Fix**:
1. Rename methodology: "Global baseline with OD-specific exceptions"
2. Report what percentage actually uses OD-specific vs global
3. Consider if OD-specific is even necessary (maybe global is sufficient)
4. Test if results differ if you force all trajectories to use global baseline
5. **Stratified analysis**: Report abnormality rates separately for:
   - Trajectories using OD-specific baselines (≥5 samples)
   - Trajectories using global baselines (<5 samples)
6. **Sensitivity analysis**: Test different minimum sample thresholds (5, 10, 20, 50) and report how results change
7. **Rename methodology**: Call it "hybrid baseline" or "OD-adaptive baseline" instead of "OD-pair-specific"
8. **Quantify impact**: Report how many trajectories would switch baseline type with different thresholds
9. **Porto analysis**: Porto has 5.1% coverage - analyze if results differ more between OD-specific and global subsets

**References**:
- `tools/compute_trajectory_baselines.py` lines 110-275, 225-241, line 229
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` lines 86-91
- `docs/reference/BASELINE_STATISTICS.md:40-47`

---

### 4.4 Translation Quality Confound ⚠️ CRITICAL

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 6.3, PEER_REVIEW_CRITICAL_ANALYSIS Section 5.3, PEER_REVIEW_CRITICAL_ISSUES Section 4.4]

**Issue**: Cross-dataset evaluation (Beijing→BJUT) uses all translated trajectories without quality filtering, despite documentation recommending >95% translation rate.

**Evidence**:
- `tools/filter_translated_by_quality.py` EXISTS - can filter by translation rate
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` line 185: "Quality filtering: Not applied"
- Documentation says 79% mapping rate (below "Fair" threshold of 85%)
- Filtering tool allows min_translation_rate=95% but wasn't used
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:181-185`: Translation quality: 79% mapping, 93% trajectory translation, "quality filtering: Not applied"
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:502`: Lists "translation quality filtering" as future work
- `docs/reference/ROAD_NETWORK_MAPPING.md:231-234`: Recommends >95% translation rate for quality
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:452`: Acknowledges "translation artifacts likely contribute to false positives"

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **Known artifacts**: 21% of roads unmapped, 7% of trajectory points untranslated, but all trajectories used
2. **False positives**: High abnormality rates (52-66%) may be due to translation errors, not model failure
3. **Uncontrolled confound**: Cannot distinguish between model failure and translation artifacts
4. **Ignored recommendation**: Documentation recommends filtering but it wasn't applied

**Impact**: 
- Cross-dataset results (52-66% abnormal) are likely translation artifacts, not model failure
- Should only evaluate trajectories with >95% translation success
- Your "cross-dataset failure" conclusions are INVALID
- You're measuring translation quality, not model quality
- 21% of roads cannot be mapped → trajectories through these roads are invalid

**Files Examined**:
- `tools/filter_translated_by_quality.py` - filtering capability exists
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` - confirms filtering disabled, lines 181-185, 185, line 502, line 452
- `docs/reference/ROAD_NETWORK_MAPPING.md` lines 298-314, 231-234 - quality impact discussion

**Required Fix**:
1. Re-run cross-dataset evaluation with quality filtering (>95% translation rate)
2. Report abnormality rates on clean subset only
3. Compare: all trajectories vs clean trajectories
4. Acknowledge that 79% translation quality is too low for reliable evaluation
5. Update conclusions: cross-dataset results are confounded by translation artifacts
6. **Apply filtering**: Re-run BJUT evaluation with >95% translation rate filter
7. **Compare results**: Report abnormality rates with and without filtering
8. **Stratified analysis**: Analyze separately:
   - Trajectories with >95% translation (high quality)
   - Trajectories with 80-95% translation (medium quality)
   - Trajectories with <80% translation (low quality, exclude)
9. **Update interpretation**: Revise conclusions about cross-network transfer based on filtered results
10. **Acknowledge limitation**: Document that original results include translation artifacts

**References**:
- `tools/filter_translated_by_quality.py` - filtering capability exists
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` lines 181-185, line 185, line 502, line 452
- `docs/reference/ROAD_NETWORK_MAPPING.md` lines 231-234, 298-314

---

### 4.5 Statistical vs Practical Significance Confusion ⚠️ MAJOR

[Sources: PEER_REVIEW_CRITICAL_ANALYSIS Section 5.4, PEER_REVIEW_CRITICAL_ISSUES Section 4.5]

**Issue**: Discussion conflates statistical significance (p < 0.001) with practical significance (magnitude of difference), but doesn't provide clear framework for interpretation.

**Evidence**:
- With 179,823 test trajectories, you have HUGE statistical power
- Even 0.1% difference would be "significant" but meaningless
- Documentation presents significance as positive finding
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:383-390`: "Practical Significance" section acknowledges issue
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md:388-389`: Classifies differences as "Acceptable", "Moderate gap", "Severe failure" but no quantitative thresholds
- All comparisons are statistically significant, so p-values don't help distinguish

**Problem**: Statistical significance just means "not due to chance," not "meaningful" or "correct methodology."

**Correct Interpretation**: Statistical significance is irrelevant with huge sample sizes. Focus on PRACTICAL significance (effect sizes).

**Problems** [Source: PEER_REVIEW_CRITICAL_ISSUES]:
1. **No quantitative thresholds**: "Acceptable" vs "Moderate" vs "Severe" are subjective
2. **No effect size framework**: No standardized way to interpret 3.5% vs 50% differences
3. **Inconsistent interpretation**: Beijing 18.5% vs 22% (3.5% diff) is "acceptable" but Porto 3.64% vs 9.44% (5.8% diff) is "moderate gap" - why?
4. **Missing context**: No comparison to baseline variation or measurement error

**Impact**:
Cannot distinguish between trivial statistical differences and meaningful practical improvements. The interpretation framework is arbitrary and inconsistent.

**Required Fix**:
1. Report effect sizes (Cohen's d, relative improvement %)
2. Don't just say "significant," say "significant with d=0.3 (small effect)"
3. Distinguish statistical vs practical significance
4. With huge n, almost everything is "significant" - focus on magnitude
5. **Define thresholds**: Establish quantitative criteria:
   - Excellent: |difference| < 2% or < 10% relative
   - Good: |difference| < 5% or < 25% relative
   - Acceptable: |difference| < 10% or < 50% relative
   - Poor: |difference| ≥ 10% or ≥ 50% relative
6. **Report both**: Always report both absolute and relative differences
7. **Contextualize**: Compare differences to:
   - Real data variation (train vs test split differences)
   - Measurement uncertainty (bootstrap CI width)
   - Baseline variation (different seed/model variance)
8. **Update tables**: Include "Practical Significance" column in all comparison tables

**References**:
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` lines 376-379, 383-390, 388-389
- Statistical test interpretation

---

## Section 5: Unsupported Claims

### 5.1 "Distillation Transfers Spatial Understanding" ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 4.1]

**Claim**: `EVALUATION_ANALYSIS.md` Section 5: "Distillation transferred three critical spatial capabilities"

**Evidence Against**:
1. Porto results show vanilla = distilled, contradicting "spatial understanding" transfer
2. No direct test of spatial reasoning (e.g., distance estimation, direction prediction)
3. Beijing vanilla may just be undertrained, not lacking spatial knowledge
4. Teacher (LM-TAD) never proven to have better spatial understanding

**Missing Tests**:
1. Probe both models for spatial embeddings (e.g., road distance predictions)
2. Compare learned zone representations vanilla vs distilled
3. Test on held-out spatial reasoning tasks (not just trajectory generation)

**Alternative Explanation**: Improvement may come from:
- Better training dynamics (temperature-scaled loss)
- Regularization effects (KL divergence as entropy regularization)
- Hyperparameter optimization (distilled gets 12 trials, vanilla gets 1)

**Impact**:
The core narrative of the research rests on this claim, but it has no direct evidence. Without testing the teacher's spatial understanding or probing the learned representations, the claim is speculation.

**Required Fix**:
1. Add direct spatial reasoning probes
2. Compare learned representations (embeddings, attention patterns)
3. Test on spatial reasoning tasks (distance prediction, direction estimation)
4. Acknowledge alternative explanations

**References**:
- `EVALUATION_ANALYSIS.md` Section 5.1
- Porto results contradict Beijing claims

---

### 5.2 "Knowledge Distillation" vs "Regularization" ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 4.2]

**Issue**: All evidence is consistent with distillation acting as a REGULARIZER, not knowledge transfer.

**Alternative Explanations for Beijing Improvement**:
1. **Temperature-scaled softmax** = label smoothing = known regularization technique
2. **KL divergence loss** = entropy regularization = prevents overconfidence
3. **Window-based teacher** = sequence regularization

**Missing Experiments**:
1. **Label smoothing baseline**: Smooth ground truth labels with temperature
2. **Self-distillation baseline**: Distill model into itself
3. **Entropy regularization baseline**: Add entropy penalty to vanilla loss
4. **Teacher-free distillation**: Test if improvement comes from soft targets vs teacher knowledge

**Impact**: Your core claim may be false. You may have discovered that "regularization helps" not "LM-TAD transfers spatial knowledge."

**Required Fix**:
1. Run label smoothing baseline (smooth labels with temperature, no teacher)
2. Run entropy regularization baseline (add -Σ p log p to loss)
3. Compare: vanilla < label_smoothing < distilled? Or label_smoothing = distilled?
4. If label smoothing matches distilled, then it's regularization, not knowledge transfer

**References**:
- No regularization baseline code found
- No ablation isolating knowledge transfer from regularization

---

### 5.3 "Generalization" Claims ⚠️ MODERATE

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 4.3]

**Issue**: You claim models "generalize" based on train vs test OD pairs, but this is weak evidence.

**Problems**:
1. Test OD pairs still come from same geographic region, same road network, same vehicle types
2. True generalization would be: different city, different time period, different road network
3. Small difference in performance (85.8% train vs 85.7% test) is within noise

**Evidence**:
- `EVALUATION_ANALYSIS.md`: "Distilled models generalize" based on 0.1% difference in match rate
- Test uses same road network as train (just different OD pairs)

**Impact**: Overstated generalization claims. Your models may just be interpolating, not generalizing.

**Required Fix**:
1. Change language: "test set performance" not "generalization"
2. Test true generalization: different city, different time period
3. Acknowledge that test OD pairs are same-distribution interpolation
4. Distinguish interpolation from generalization

**References**:
- `EVALUATION_ANALYSIS.md` Section 3.4
- Train/test split uses same road network

---

## Section 6: Reproducibility and Documentation Issues

### 6.1 Missing Hyperparameter Details ⚠️ MODERATE

[Sources: CRITICAL_REVIEW_FINDINGS Investigation 4.1, PEER_REVIEW_CRITICAL_ANALYSIS Section 6.1]

**Status**: Config files are well-structured, but some defaults may be in code.

**Evidence**:
- `config/Beijing.yaml` - comprehensive but may not include all defaults
- `train_with_distill.py` - may set defaults not in config
- Need to check all hyperparameters are documented

**Beijing Config Extract** [Source: CRITICAL_REVIEW_FINDINGS]:
```yaml
road_network_encoder_config:
  road_id_emb_dim: 64
  len_emb_dim: 16
  type_emb_dim: 16
  lon_emb_dim: 16
  lat_emb_dim: 16
  intersection_emb_dim: 128
  zone_id_emb_dim: 128
  zone_id_num_embeddings: 300

trajectory_encoder_config:
  hidden_dim: 128
  num_heads: 2
  num_layers: 2
  dropout: 0.0
  max_len: 1024
  grad_checkpoint: false

optimizer_config:
  max_epoch: 25
  batch_size: 128
  accum_steps: 8
  learning_rate: 0.001
  weight_decay: 0.1
  warmup_ratio: 0.1
  max_norm: 1.0

distill:
  enable: true
  window: 4
  lambda: 0.01
  temperature: 2.0
  grid_size: 0.001
  downsample: 1

dataloader:
  num_workers: 6
  pin_memory: false
  prefetch_factor: 16
  persistent_workers: false

data:
  candidate_top_k: 64

training:
  seed: 43
  allow_tf32: true
  cudnn_benchmark: true
  torch_compile: true
  torch_compile_mode: max-autotune
  disable_cudagraphs: true
```

**Note**: Need to extract ALL hyperparameters, including defaults from code...

**Required Fix**:
1. Extract ALL hyperparameters from code (including defaults)
2. Document in config files or appendix
3. Make configs executable (can load and run)

**References**:
- `config/Beijing.yaml`
- `train_with_distill.py`

---

### 6.2 Model Architecture Specification ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 6.3]

**Status**: Architecture code exists but not fully documented.

**Evidence from Code**:
- `models/hoser.py`: Main model structure
- `models/road_network_encoder.py`: 
  - Road embeddings: 64-dim road_id + 16-dim len + 16-dim type + 16-dim lon + 16-dim lat = 128-dim total
  - Zone embeddings: 128-dim, 300 zones
  - Road GAT: 2-layer GAT with 128-dim hidden
  - Zone GCN: 2-layer GCN with 128-dim hidden
- `models/trajectory_encoder.py`:
  - Hidden dim: 128
  - Num heads: 2
  - Num layers: 2
  - Dropout: 0.0
  - Max len: 1024
- `models/navigator.py`:
  - Hidden dim: 128
  - Attention-based scoring
  - Time head: 3-layer MLP (128*3 → 64 → 1)

**Required Fix**:
1. Create architecture diagram
2. Document all dimensions in one place
3. Provide layer-by-layer breakdown

**References**:
- `models/hoser.py`
- `models/road_network_encoder.py` lines 1-102
- `models/trajectory_encoder.py` lines 1-192
- `models/navigator.py` lines 15-79

---

### 6.3 Missing Environment Information ⚠️ MAJOR

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 5.1]

**Issue**: No documentation of software versions, CUDA version, or hardware specifications, making reproduction difficult.

**Evidence**:
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:710-716`: Hardware listed (RTX 4090, Ryzen 9) but no software versions
- `docs/LMTAD-Distillation.md`: No mention of PyTorch version, CUDA version, or dependency versions
- No `requirements.txt` or `environment.yml` in documentation
- No commit hashes for code versions used

**Problems**:
1. **Reproducibility risk**: Different PyTorch/CUDA versions may produce different results
2. **Dependency drift**: Without version pins, future readers may get different results
3. **Debugging difficulty**: If reproduction fails, unclear if it's due to version mismatch
4. **No provenance**: Cannot verify which code version produced reported results

**Required Actions**:
1. **Document versions**: Create `REPRODUCIBILITY.md` with:
   - Python version
   - PyTorch version
   - CUDA version
   - Key dependency versions (numpy, pandas, polars, etc.)
   - OS version
2. **Commit hashes**: Document git commit hashes for code used in each experiment
3. **Environment file**: Provide `requirements.txt` or `environment.yml` with pinned versions
4. **Hardware details**: Document GPU model, driver version, CPU model, RAM

**Citation**: 
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:710-716`

---

### 6.4 Inconsistent Seed Usage ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ISSUES Section 5.2]

**Issue**: Seeds are used inconsistently across experiments, and some evaluations use fixed seed 42 while others use multiple seeds.

**Evidence**:
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:944`: "Fixed seed (42) for all evaluations"
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:50-68`: Training uses seeds 42, 43, 44
- `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md:79-90`: Porto uses seeds 42, 43, 44
- No explanation for why evaluation uses seed 42 but training uses multiple

**Problems**:
1. **Inconsistent**: Training robustness assessed with multiple seeds, but evaluation uses single seed
2. **Missing evaluation variance**: Don't know if evaluation metrics vary with different seeds
3. **No justification**: No explanation for why single seed is sufficient for evaluation

**Required Actions**:
1. **Document rationale**: Explain why evaluation uses single seed (if it's intentional)
2. **Or use multiple**: If evaluation should be robust, use multiple seeds and report variance
3. **Consistency**: Align seed usage between training and evaluation (both multiple or both single)

**Citation**: 
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:944`
- `hoser-distill-optuna-6/EVALUATION_ANALYSIS.md:50-68`
- `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md:79-90`

---

### 6.5 Data Preprocessing Pipeline ⚠️ MODERATE

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 6.2]

**Status**: Need to document complete preprocessing steps.

**Evidence**:
- Dataset loading code exists
- Preprocessing scripts exist but pipeline not documented end-to-end
- Grid token computation needs documentation

**Required Fix**:
1. Document all preprocessing steps
2. Provide pseudocode for complex steps
3. Report data statistics at each stage

---

## Section 7: Missing Experiments

### 7.1 No Failure Analysis ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 7.1]

**Status**: Failure rates discussed but failure modes not analyzed.

**Missing**:
1. What do failure trajectories look like? (visualizations)
2. Which OD pairs cause failures? (characterize hard cases)
3. Error modes: stuck in loops? Wrong direction? Dead ends?
4. Model confidence on failures vs successes

**Required Fix**: Add failure mode analysis to evaluation pipeline.

---

### 7.2 No Intermediate Checkpoints ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 7.2]

**Status**: Only final 25-epoch models reported.

**Missing**:
1. Training curves (loss, accuracy vs epoch)
2. When does distilled model diverge from vanilla?
3. Early stopping analysis
4. Overtraining analysis

**Required Fix**: Report training dynamics, not just final results.

---

### 7.3 No Computational Cost Analysis ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 7.3]

**Status**: Training speed mentioned but total cost not analyzed.

**Missing**:
1. Total training time: vanilla vs distilled (wall-clock hours)
2. GPU memory: peak memory usage
3. FLOPs comparison
4. Cost-benefit: is 1% improvement worth 5× training time?

**Required Fix**: Add cost-benefit analysis to documentation.

---

### 7.4 No Cross-Model Comparisons ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 7.4]

**Status**: Only vanilla HOSER vs distilled HOSER compared.

**Missing**:
1. Pure LM-TAD (teacher alone)
2. RNN/LSTM baselines
3. Transformer baselines
4. Classical methods (A*, Dijkstra with learned costs)

**Impact**: Cannot assess absolute performance, only relative improvement. Maybe BOTH models are terrible compared to other methods.

**Required Fix**: Add external baselines for absolute performance assessment.

---

## Section 8: Documentation Issues

### 8.1 Contradictory Statements ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 8.1]

**Examples**:
1. Porto Phase 1 claims "minimal distillation benefit" but then says distilled is "2.6% faster at inference"
2. Wang doc says "cross-network transfer is difficult" but uses 79% translation quality as if acceptable
3. Beijing claims "generalization" based on test OD, but Porto shows vanilla generalizes just as well

**Required Fix**: Reconcile contradictions across documents.

---

### 8.2 Selective Reporting ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 8.2]

**Issue**: Porto Phase 1 shows distillation FAILS, but this is buried and explained away.

**Evidence**:
- Porto results contradict Beijing results
- Instead of "distillation doesn't work on Porto," you say "Phase 1 hyperparameters not optimal"
- Phase 2 is presented as likely to fix it, but this is speculation

**Honest Reporting Needed**: "Distillation provided dramatic benefits on Beijing but NO benefits on Porto with Phase 1 hyperparameters. This suggests distillation effectiveness is dataset-dependent and hyperparameter-sensitive. Further investigation needed."

**Required Fix**: Highlight negative results, don't explain them away.

---

### 8.3 Missing Related Work ⚠️ MINOR

[Source: PEER_REVIEW_CRITICAL_ANALYSIS Section 8.3]

**Issue**: No comparison to existing knowledge distillation literature.

**Missing**:
1. Hinton et al. 2015 (original distillation paper)
2. Recent trajectory prediction papers with distillation
3. Spatial knowledge transfer in other domains
4. Alternative knowledge distillation methods

**Required Fix**: Add related work section with citations and comparisons.

---

## Prioritized Recommendations

### Must Fix (Critical Flaws - Publication Blockers)

[Sources: PEER_REVIEW_CRITICAL_ANALYSIS, PEER_REVIEW_CRITICAL_ISSUES]

1. **Run matched hyperparameter search for vanilla baseline**
   - This is the most critical issue - breaks scientific validity
   - Compare vanilla_optimal vs distilled_optimal (both with 12 trials)
   - Optimize vanilla's hyperparameters (learning rate, batch size, weight decay, etc.)

2. **Implement Bonferroni correction or fix documentation**
   - Either implement correction or remove claim from documentation
   - Re-interpret results with corrected thresholds
   - Report adjusted p-values

3. **Re-evaluate Wang cross-dataset with quality filtering**
   - Only use trajectories with >95% translation success
   - Update conclusions about cross-dataset performance
   - Re-run BJUT evaluation with proper filtering

4. **Fix JSD binning documentation**
   - Either change code to 50 bins or update docs to 100 bins
   - Add sensitivity analysis to bin count
   - Justify binning choice

5. **Fix data leakage in Wang baseline computation**
   - Recompute baselines using only training data
   - Re-run all abnormality detection with train-only baselines
   - Compare results and update conclusions

6. **Validate vocabulary mapping or add ablations**
   - Report mapping statistics (roads-per-grid-cell distribution)
   - Ablation study with random permutation controls
   - Validate that improvements aren't mapping artifacts

7. **Add calibration metrics or remove claims**
   - Compute ECE and Brier scores
   - Or remove "calibrated uncertainty" claim
   - Provide reliability diagrams

8. **Report effect sizes and confidence intervals**
   - Add Cohen's h, relative risk, odds ratios
   - Bootstrap 95% CIs for all comparisons
   - Distinguish statistical vs practical significance

### Should Fix (Major Issues - Seriously Weaken Conclusions)

9. **Add teacher baseline evaluation**
   - Evaluate LM-TAD on trajectory prediction task
   - Compare vanilla vs teacher vs distilled
   - Validate core claim about spatial knowledge transfer

10. **Run systematic ablation studies**
    - Temperature τ ablation: [1.0, 2.0, 3.0, 4.0, 5.0]
    - Lambda λ ablation: [0.0, 0.001, 0.01, 0.1]
    - Window size ablation: [1, 2, 4, 7, 10]
    - Interaction plots (λ×τ, λ×window, τ×window)

11. **Add regularization baselines**
    - Label smoothing baseline
    - Entropy regularization baseline
    - Self-distillation baseline
    - Prove it's knowledge transfer, not just regularization

12. **Normalize local metrics**
    - Report Hausdorff/DTW normalized by trajectory length
    - Address contradiction with vanilla having lower (better) metrics
    - Provide normalized values in all tables

13. **Ablate beam search width**
    - Test both models at width=1,2,4,8,16
    - Ensure results aren't confounded by search algorithm
    - Report metrics across all beam widths

14. **Add paired statistical tests**
    - McNemar test for binary OD match outcomes
    - Wilcoxon signed-rank for continuous metrics
    - Account for pairing in trajectory comparisons

15. **Add cross-seed statistical analysis**
    - ANOVA to test if seed effect is significant
    - Levene's test for variance differences
    - Bootstrap CIs for CV estimates

16. **Separate OD match metrics**
    - Path completion rate (reached intended destination)
    - Endpoint realism rate (endpoint exists in real OD pairs)
    - Combined match rate (both conditions met)

17. **Justify KL divergence direction**
    - Ablation with reverse KL
    - Theoretical justification for forward KL
    - Compare mode-seeking vs mean-seeking behavior

### Nice to Have (Improvements - Enhance Rigor)

18. **Justify sample size**
    - Power analysis for 5,000 trajectories
    - Report confidence intervals
    - Justify with effect size targets

19. **Sensitivity analyses**
    - Grid size for OD matching: [0.0005, 0.001, 0.002]
    - Threshold values for Wang detection: [3km, 5km, 7km]
    - Bin count for JSD: [25, 50, 100, 200]
    - Report robustness to parameter choices

20. **Add external baselines**
    - Compare to other trajectory generation methods
    - Assess absolute performance
    - Test LM-TAD teacher standalone

21. **Failure analysis**
    - Characterize failure modes
    - Visualize failed trajectories
    - Identify hard OD pairs

22. **Complete documentation**
    - Model architecture specification with diagrams
    - Data preprocessing pipeline end-to-end
    - Training dynamics curves
    - Environment and version information

23. **Document candidate filtering**
    - Report teacher-student overlap analysis
    - Test different k values: [32, 64, 128, no filtering]
    - Clarify evaluation vs training filtering

24. **Stratified Wang analysis**
    - Separate results for OD-specific vs global baselines
    - Test different minimum sample thresholds
    - Analyze translation quality tiers

---

## Positive Aspects (To Be Fair)

[Sources: PEER_REVIEW_CRITICAL_ANALYSIS, PEER_REVIEW_CRITICAL_ISSUES]

Despite the criticisms above, the research has notable strengths:

1. **Comprehensive evaluation**: Multiple metrics, multiple scenarios, multiple datasets
2. **Reproducibility effort**: Detailed documentation, WandB logging, code organization
3. **Honest reporting of Porto failure**: Many papers would hide negative results
4. **Thorough scenario analysis**: 9 scenarios provide detailed breakdown
5. **Visualization quality**: Good use of figures to illustrate findings
6. **Multiple seeds**: 3 seeds per model shows awareness of reproducibility
7. **Code quality**: Well-structured, modular codebase
8. **Config-first design**: Hyperparameters centralized in YAML files
9. **Clear documentation**: Detailed methodology descriptions and worked examples
10. **Reproducible pipeline**: Code and scripts are provided
11. **Statistical methodology**: Wang abnormality detection is a sophisticated approach
12. **Cross-dataset analysis**: Evaluation on multiple datasets strengthens external validity
13. **Technical sophistication**: Complex models and evaluation frameworks

**The issues identified are fixable and do not invalidate the core contributions.** However, addressing them is essential for scientific rigor and publication readiness.

These strengths provide a foundation for improvement. With the recommended fixes, this research could make a solid contribution to trajectory generation and knowledge distillation literature.

---

## Conclusion

This comprehensive peer review synthesizes findings from three detailed investigations. The research demonstrates interesting preliminary results and strong engineering practices, but suffers from **fundamental experimental design flaws, statistical inadequacies, and methodological inconsistencies** that prevent drawing strong causal conclusions about knowledge distillation effectiveness.

### Key Findings:

**Most Critical Issue**: The hyperparameter optimization confound (vanilla gets no search, distilled gets 12 trials) breaks the scientific validity of the comparison. This alone invalidates the main conclusions.

**Second Critical Issue**: Data leakage in Wang baseline computation (train+test pooling) violates fundamental ML evaluation principles.

**Third Critical Issue**: Multiple testing without correction and missing effect sizes/CIs mean statistical claims cannot be trusted.

**Core Scientific Question Unresolved**: Is the improvement from knowledge transfer or regularization? Without ablations testing label smoothing and entropy regularization, this question remains unanswered.

**Teacher Never Evaluated**: The core claim "teacher transfers spatial knowledge" has zero direct evidence. The teacher has never been evaluated on the trajectory prediction task.

### The Beijing Results:

The Beijing results are promising but confounded by:
- Hyperparameter optimization (distilled gets 12 trials, vanilla gets 1)
- Unknown contribution from regularization vs knowledge transfer
- Potential vocabulary mapping artifacts
- Beam search evaluation dependence

### The Porto Results:

Porto Phase 1 shows vanilla = distilled, contradicting the Beijing findings. This is explained away rather than investigated, but it directly challenges the "spatial knowledge transfer" narrative.

### The Wang Results:

The Wang abnormality detection evaluation is confounded by:
- Data leakage (train+test pooling in baselines)
- Translation quality issues (79% quality, no filtering)
- Misleading "OD-specific" methodology (99% use global baseline)
- Missing effect sizes and multiple testing correction

### Overall Assessment:

**Recommendation**: **Major revisions required** before publication.

The authors must address at minimum the critical issues, particularly:

1. Run matched hyperparameter search for vanilla baseline
2. Fix data leakage in Wang baselines  
3. Implement proper statistical corrections for multiple testing
4. Report effect sizes and confidence intervals
5. Re-evaluate Wang cross-dataset results with quality filtering
6. Add ablation studies to isolate distillation components
7. Provide regularization baselines to rule out alternative explanations
8. Evaluate teacher model on trajectory prediction task

**Current Status**: The work reads as exploratory analysis with promising directions, not rigorous scientific evaluation. The engineering is sound, but the experimental design has fundamental flaws that must be corrected.

**Potential**: With these corrections, this could become a strong contribution to trajectory generation and knowledge distillation literature. The comprehensive evaluation framework, multiple datasets, and thorough documentation provide an excellent foundation. The issues are serious but fixable.

---

## Appendix: Evidence Index by File

### Code Files

**`tune_hoser.py`**:
- Lines 215, 271, 279, 303: Hyperparameter confound evidence
- Lines 758-865: `_run_vanilla_baseline()` function
- Lines 1-94: Optuna configuration

**`evaluation.py`**:
- Line 578: JSD binning (100 bins)
- Lines 576-580: JSD implementation
- Line 751: Hausdorff not normalized
- Line 765: DTW not normalized
- Line 818: EDR IS normalized
- Lines 736-766: Local metrics implementations
- Lines 598-750: Trajectory metrics
- Line 951: Grid size parameter (0.001)

**`gene.py`**:
- Line 2207: Sample size (5000) - no justification
- Line 1829: Beam width programmatic default (4)
- Line 2220: Beam width CLI default (8) - inconsistent!
- Lines 455-505: Beam search termination logic

**`tools/analyze_wang_results.py`**:
- Line 558: Chi-square test without correction
- Line 568: Uses p < 0.05, not 0.001
- Lines 530-588: Statistical testing code
- Lines 525-588: Wang analysis

**`tools/detect_abnormal_statistical.py`**:
- Line 51: Default threshold (5000m from paper)
- Lines 51-52: Threshold configuration

**`tools/compute_trajectory_baselines.py`**:
- Line 229: Coverage percentage calculation
- Lines 110-275: Baseline computation logic
- Lines 225-241: OD coverage analysis

**`tools/filter_translated_by_quality.py`**:
- Filtering capability exists but not used

**`tools/map_road_networks.py`**:
- Lines 173-194: Many-to-one mapping tracking

**`critics/lmtad_teacher.py`**:
- Wrapper only, no evaluation

**`critics/grid_mapper.py`**:
- Line 98: Grid mapping implementation

**`python_pipeline.py`**:
- Line 136: Grid size hardcoded (0.001)

**Model Files**:
- `models/hoser.py`: Main structure
- `models/road_network_encoder.py` lines 1-102: Architecture details
- `models/trajectory_encoder.py` lines 1-192: Encoder specification
- `models/navigator.py` lines 15-79: Navigator architecture

### Configuration Files

**`config/Beijing.yaml`**:
- Complete hyperparameter specification
- Optuna configuration
- Distillation parameters

**`config/abnormal_detection_statistical.yaml`**:
- Lines 26-27: Fixed thresholds from Wang paper

### Documentation Files

**`docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md`**:
- Line 126: Claims Bonferroni correction (not implemented)
- Lines 75-76: Train+test pooling documented
- Line 87: OD coverage (0.6%)
- Lines 86-91: Coverage analysis
- Line 91: "Most trajectories use global baselines"
- Line 185: "Quality filtering: Not applied"
- Lines 181-185: Translation quality analysis
- Lines 344-380: Chi-square tests
- Lines 383-390: Practical significance discussion
- Lines 388-389: Subjective classification
- Lines 376-379: Statistical significance claims
- Line 452: Acknowledges translation artifacts
- Line 496: Lists OD coverage as limitation
- Line 502: Lists filtering as future work
- Line 504: Lists adaptive thresholds as future work
- Lines 117-120: Hybrid threshold strategy
- Lines 172-179: Configuration details
- Lines 166-167: Evaluation splits

**`docs/reference/BASELINE_STATISTICS.md`**:
- Line 13: Train+test pooling documented
- Lines 40-47: OD coverage discussion

**`docs/reference/ROAD_NETWORK_MAPPING.md`**:
- Lines 231-234: Recommends >95% translation rate
- Lines 298-314: Quality impact discussion

**`docs/LMTAD-Distillation.md`**:
- Line 1827: Mentions ablation but no results
- Lines 821-829: Parameter ranges
- Lines 2489-2608: Optuna discussion
- Lines 1894-1903: Hardware constraint explanation
- Lines 1857-1904: Beam search configuration
- Lines 1899-1903: Claims 95-99% A* correlation (unverified)
- Line 2142: JSD calculation (bins not specified)
- Lines 679-685: Vocabulary mapping formula
- Lines 798-801: Acknowledges many-to-one mapping
- Line 1825: Grid mapping reference
- Lines 760-775: KL divergence discussion
- Lines 810-814: Forward KL justification (weak)
- Lines 1778-1782: Candidate filtering (k=64)
- Lines 1613-1620: Top-k filtering rationale
- Line 84: Claims "calibrated uncertainty"
- Lines 831-910: Validation metrics (no calibration)

**`EVALUATION_ANALYSIS.md`** and **`hoser-distill-optuna-6/EVALUATION_ANALYSIS.md`**:
- Line 143: "50 equal-width bins" (contradicts code)
- Line 142: "50 bins" specification
- Line 233: Acknowledges not normalized
- Line 639: Dismisses Hausdorff/DTW contradiction
- Lines 629-653: Local metrics interpretation
- Lines 636-653: DTW scaling acknowledgment
- Lines 200-206: Hausdorff scaling caveat
- Lines 272-293: OD match definition
- Lines 277-287: Endpoint confusion explanation
- Lines 289-292: Conflated interpretation
- Lines 604-623: CV reporting
- Lines 629-683: Trajectory metrics (no tests)
- Line 944: Fixed seed (42) for evaluation
- Lines 50-68: Training uses seeds 42, 43, 44
- Lines 710-716: Hardware listed (no software versions)
- Lines 833-910: Validation metrics
- Section 5: Claims spatial understanding transfer
- Section 5.1: Spatial capabilities claim
- Section 3.4: Generalization claims
- Lines 87-95: Trial 0 misrepresentation

**`hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/EVALUATION_ANALYSIS_PHASE1.md`**:
- Lines 94-97: Phase 1 Porto parameters
- Lines 79-90: Porto seed usage
- Lines 580-593: CV reporting

**`hoser-distill-optuna-6/create_analysis_figures.py`**:
- Line 595: CV calculation
- Lines 591-599: CV computation

---

**Review Completed**: November 2025  
**Document Version**: Comprehensive Synthesis v1.0  
**Next Steps**: Authors should prioritize Critical issues, then Major, then Moderate/Minor enhancements

---


