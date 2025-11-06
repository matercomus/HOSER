# HOSER Evaluation Documentation

**Last Updated**: 2025-11-06  
**Purpose**: Comprehensive guide to trajectory evaluation methodology, metrics, and analysis workflows

---

## 📚 Documentation Structure

This directory organizes all evaluation-related documentation for the HOSER Knowledge Distillation project.

---

## 🎯 Core Evaluation Documentation

### Essential Reading

1. **[Evaluation Pipeline Guide](../EVALUATION_PIPELINE_GUIDE.md)**
   - Overview of evaluation workflow
   - How to run evaluations programmatically
   - Integration with generation pipeline
   - **Start here** for basic evaluation understanding

2. **[Setup Evaluation Guide](../SETUP_EVALUATION_GUIDE.md)**
   - Environment setup
   - Required datasets and file structure
   - Configuration options
   - Troubleshooting common issues

3. **[Evaluation Comparison](../EVALUATION_COMPARISON.md)** ⭐
   - Cross-dataset analysis (Beijing vs Porto)
   - Comprehensive coverage comparison
   - Scenario analysis methodology
   - **NEW**: Beam search ablation study findings
   - Gap analysis and recommendations

---

## 📊 Metrics & Methodology

### Statistical Methods

4. **[Paired Statistical Tests Guide](../PAIRED_STATISTICAL_TESTS_GUIDE.md)** ⭐
   - Paired vs unpaired test selection
   - Paired t-test and Wilcoxon signed-rank
   - Effect size (Cohen's d) interpretation
   - Complete usage examples
   - **Implementation**: `tools/paired_statistical_tests.py`

5. **[Effect Size Interpretation](../EFFECT_SIZE_INTERPRETATION.md)**
   - Cohen's d, Cohen's h, Cramér's V
   - Practical vs statistical significance
   - Context-specific interpretation
   - Common pitfalls and best practices

6. **[Normalized Metrics Impact Summary](../NORMALIZED_METRICS_IMPACT_SUMMARY.md)** ⭐
   - Trajectory-length independent metrics
   - `Hausdorff_norm`, `DTW_norm` implementation
   - Beam search ablation application
   - Cross-seed analysis enhancement
   - CV handling for interval scale metrics

### Evaluation Methodologies

7. **[Scenario Analysis Guide](../SCENARIO_ANALYSIS_GUIDE.md)**
   - 9 scenario taxonomy (temporal + spatial)
   - OD-specific aggregation
   - Hierarchical analysis
   - Per-scenario metric computation

8. **[Wang Abnormality Detection](../results/WANG_ABNORMALITY_DETECTION_RESULTS.md)** ⭐
   - Statistical abnormality detection (Wang et al. 2018)
   - OD-pair-specific baselines
   - 4 behavior patterns (Normal, Temporal, Route, Both)
   - Comprehensive results across 809k trajectories
   - Cross-dataset transfer analysis

9. **[Search Method Guidance](../SEARCH_METHOD_GUIDANCE.md)** 🆕
   - A* vs Beam search comparison
   - Decision matrix (speed vs quality vs OD matching)
   - Configuration recommendations
   - Performance benchmarks from ablation study

10. **[Beam Ablation Study](../BEAM_ABLATION_STUDY.md)** 🆕
    - 26-hour experimental study
    - Search-method dependency findings
    - Performance tables and interpretation
    - Critical discovery: Distillation benefit flips with search method

---

## 📈 Results & Analysis

### Published Results

11. **[Cross-Seed Analysis](../results/CROSS_SEED_ANALYSIS.md)**
    - Statistical variance across seeds (42, 43, 44)
    - Mean, SD, 95% CI, CV%
    - All metrics: Global, local, normalized
    - Confidence intervals for reproducibility

12. **[Teacher Baseline Comparison](../results/TEACHER_BASELINE_COMPARISON.md)**
    - Vanilla vs Distilled performance
    - LM-TAD teacher comparison
    - Dataset-specific findings

13. **[Wang Abnormality Detection Results](../results/WANG_ABNORMALITY_DETECTION_RESULTS.md)**
    - 12 models across 3 datasets
    - Real vs generated abnormality rates
    - Statistical significance with FDR correction
    - Pattern distribution analysis

---

## 🛠️ Guides & Workflows

### Practical Implementation

14. **[Abnormal OD Workflow](../guides/ABNORMAL_OD_WORKFLOW.md)**
    - End-to-end workflow for abnormality analysis
    - From generation to statistical detection
    - Step-by-step execution

15. **[Run Wang Abnormality Analysis](../guides/RUN_WANG_ABNORMALITY_ANALYSIS.md)**
    - Command-line execution guide
    - Configuration options
    - Output interpretation

---

## 📖 Reference & Technical Details

### Implementation Details

16. **[Baseline Statistics](../reference/BASELINE_STATISTICS.md)**
    - OD-pair-specific baseline computation
    - Statistical thresholds
    - Coverage analysis

17. **[Road Network Mapping](../reference/ROAD_NETWORK_MAPPING.md)**
    - Vocabulary mapping validation
    - Cross-dataset translation
    - Road ID mapping methodology

18. **[Model Locations](../reference/MODEL_LOCATIONS.md)**
    - Trained model checkpoints
    - Versioning and organization
    - Reproduction details

---

## 🔍 Advanced Topics

### Specialized Analyses

19. **[Vocabulary Mapping Validation](../VOCABULARY_MAPPING_VALIDATION.md)**
    - Beijing/Porto vocabulary mapping
    - Coverage verification
    - Distribution analysis

20. **[Performance Profiling Summary](../PERFORMANCE_PROFILING_SUMMARY.md)**
    - Generation speed benchmarks
    - Computational efficiency
    - Bottleneck analysis

21. **[Visualization Guide](../VISUALIZATION_GUIDE.md)**
    - Trajectory visualization tools
    - Heatmap generation
    - Statistical plot creation

22. **[Dynamic Heatmap Implementation](../DYNAMIC_HEATMAP_IMPLEMENTATION.md)**
    - Temporal-spatial visualization
    - Implementation details

---

## 📦 Archive

Historical documentation (for reference):

23. **[Archive: Wang Implementation Summary](../archive/WANG_IMPLEMENTATION_SUMMARY.md)**
24. **[Archive: Abnormal Analysis Results Beijing](../archive/ABNORMAL_ANALYSIS_RESULTS_BEIJING.md)**
25. **[Archive: Z-Score Results Analysis](../archive/Z_SCORE_RESULTS_ANALYSIS.md)**
26. **[Archive: Completed Plans](../archive/completed_plans/)**

---

## 🚀 Quick Start Paths

### For Researchers

**Path 1: Understanding Evaluation Metrics**
1. [Evaluation Pipeline Guide](../EVALUATION_PIPELINE_GUIDE.md)
2. [Normalized Metrics Impact](../NORMALIZED_METRICS_IMPACT_SUMMARY.md)
3. [Effect Size Interpretation](../EFFECT_SIZE_INTERPRETATION.md)

**Path 2: Running Evaluations**
1. [Setup Evaluation Guide](../SETUP_EVALUATION_GUIDE.md)
2. [Evaluation Pipeline Guide](../EVALUATION_PIPELINE_GUIDE.md)
3. [Scenario Analysis Guide](../SCENARIO_ANALYSIS_GUIDE.md)

**Path 3: Statistical Analysis**
1. [Paired Statistical Tests Guide](../PAIRED_STATISTICAL_TESTS_GUIDE.md)
2. [Cross-Seed Analysis](../results/CROSS_SEED_ANALYSIS.md)
3. [Wang Abnormality Detection](../results/WANG_ABNORMALITY_DETECTION_RESULTS.md)

### For Implementation

**Understanding Current System**:
1. Review [Evaluation Comparison](../EVALUATION_COMPARISON.md) for gaps
2. Check [Normalized Metrics Impact](../NORMALIZED_METRICS_IMPACT_SUMMARY.md) for recent changes
3. Read [Paired Statistical Tests Guide](../PAIRED_STATISTICAL_TESTS_GUIDE.md) for methodology

**Running Experiments**:
1. [Setup Evaluation Guide](../SETUP_EVALUATION_GUIDE.md) - Configuration
2. [Abnormal OD Workflow](../guides/ABNORMAL_OD_WORKFLOW.md) - Execution
3. [Search Method Guidance](../SEARCH_METHOD_GUIDANCE.md) - Method selection

---

## 🎯 Evaluation Metrics Summary

### Global Metrics (Distribution-Level)
- **Distance JSD**: Jensen-Shannon divergence of trajectory length distribution
- **Duration JSD**: Jensen-Shannon divergence of trip duration distribution
- **Radius JSD**: Jensen-Shannon divergence of radius of gyration distribution

### Local Metrics (Trajectory-Level)
- **Hausdorff Distance**: Maximum deviation between trajectories
  - `Hausdorff_km`: Raw distance (km)
  - `Hausdorff_norm`: Normalized by trajectory length (km/point) 🆕
- **DTW Distance**: Dynamic Time Warping alignment cost
  - `DTW_km`: Raw distance (km)
  - `DTW_norm`: Normalized by trajectory length (km/point) 🆕
- **EDR**: Edit Distance on Real sequence (0-1)

### OD Matching Metrics 🆕 (Issue #15)
- **Origin Match Rate**: Proportion with correct origin grid
- **Destination Match Rate**: Proportion with correct destination grid
- **OD Pair Match Rate**: Proportion with correct OD pair
- **Both Correct Rate**: Proportion with both origin and destination correct

### Scenario Metrics
- **Per-Scenario Aggregates**: Metrics grouped by 9 scenarios
  - Temporal: weekday/weekend, peak/off_peak
  - Spatial: city_center/suburban, within/to/from center

### Abnormality Metrics (Wang et al. 2018)
- **Abp1**: Normal behavior rate
- **Abp2**: Temporal delay rate (congestion)
- **Abp3**: Route deviation rate (detours)
- **Abp4**: Both deviations rate (major anomalies)

---

## 🔄 Recent Updates (November 2025)

### Issue #8 - Beam Search Ablation
- **Documentation**: Comprehensive 26-hour study findings
- **Discovery**: Search-method dependent distillation benefits
- **Files**: [Beam Ablation Study](../BEAM_ABLATION_STUDY.md), [Search Method Guidance](../SEARCH_METHOD_GUIDANCE.md)

### Issue #14 - Normalized Metrics
- **Implementation**: Trajectory-length independent metrics
- **Impact**: Fair comparison across search methods
- **File**: [Normalized Metrics Impact Summary](../NORMALIZED_METRICS_IMPACT_SUMMARY.md)

### Issue #15 - Granular OD Metrics
- **Enhancement**: Separated OD matching into components
- **Benefits**: Better diagnostic information
- **Changes**: `evaluation.py` with 8 new metrics

### Issue #16 - Paired Statistical Tests
- **Implementation**: `tools/paired_statistical_tests.py`
- **Documentation**: [Paired Statistical Tests Guide](../PAIRED_STATISTICAL_TESTS_GUIDE.md)
- **Status**: Ready for integration into evaluation pipeline

---

## 🎓 Key Concepts

### Paired vs Unpaired Tests
- **Paired**: Same OD pairs across models (vanilla vs distilled)
- **Unpaired**: Different sets (real vs generated)
- **Power**: Paired tests have ~2x greater sensitivity

### Normalized vs Raw Metrics
- **Raw** (`_km`): Absolute distance, confounded by trajectory length
- **Normalized** (`_norm`): Per-point distance, fair comparison
- **Use Case**: Essential for comparing different generation methods

### Effect Size vs Significance
- **P-value**: Statistical significance (is there a difference?)
- **Effect size**: Practical significance (how big is the difference?)
- **Reporting**: Both should be reported together

### Scenario Analysis
- **Purpose**: Understand performance across contexts
- **Granularity**: 9 scenarios capture temporal and spatial patterns
- **Value**: Identify model strengths and weaknesses

---

## 📝 Contributing

When adding new evaluation documentation:
1. Add entry to this README in appropriate section
2. Include purpose, key findings, and file location
3. Mark with 🆕 if added within last month
4. Update "Recent Updates" section
5. Cross-reference related documents

---

## 🔗 Related Documentation

- **Main README**: [`docs/README.md`](../README.md)
- **Distillation Guide**: [`docs/LMTAD-Distillation.md`](../LMTAD-Distillation.md)
- **Project Root**: [`README.md`](../../README.md)

---

**Maintained by**: HOSER Research Team  
**Questions**: See [GitHub Issues](https://github.com/matercomus/HOSER/issues)
