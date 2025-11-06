# Effect Size and Confidence Interval Interpretation Guide

**Purpose**: Understanding practical significance beyond statistical significance

**Created**: 2025-11-06  
**Related**: Issue #18 (Missing Effect Sizes and Confidence Intervals)

---

## Overview

While p-values tell us if differences are statistically significant (unlikely to occur by chance), they don't tell us if differences are *practically meaningful*. This guide explains how to interpret effect sizes and confidence intervals to assess practical significance.

## Statistical vs Practical Significance

| Measure | What It Tells You | Example |
|---------|-------------------|---------|
| **P-value** | Probability difference is due to chance | p < 0.05: "Unlikely random" |
| **Effect Size** | Magnitude of the difference | h = 0.8: "Large effect" |
| **Confidence Interval** | Precision of the estimate | [10%, 15%]: "95% confident true value in this range" |

**Key Insight**: A result can be:
- ✅ Statistically significant BUT practically trivial (large N, small effect)
- ❌ Not statistically significant BUT practically important (small N, large effect)

**Best Practice**: Report all three measures for complete interpretation.

---

## Effect Sizes: Cohen's h for Proportions

### What is Cohen's h?

Cohen's h measures the difference between two proportions on a standardized scale.

**Formula**:
```
h = 2 × (arcsin(√p1) - arcsin(√p2))
```

Where:
- p1 = Proportion 1 (e.g., real abnormality rate)
- p2 = Proportion 2 (e.g., generated abnormality rate)

### Interpretation Guidelines

| Cohen's h | Magnitude | Interpretation |
|-----------|-----------|----------------|
| \|h\| < 0.2 | **Small** | Minimal practical difference |
| 0.2 ≤ \|h\| < 0.5 | **Medium** | Moderate practical difference |
| \|h\| ≥ 0.5 | **Large** | Substantial practical difference |

### Sign Interpretation

- **Positive h**: p1 > p2 (first proportion higher)
- **Negative h**: p1 < p2 (first proportion lower)
- **h ≈ 0**: Proportions nearly equal

### Example Interpretations

#### Example 1: Real vs Generated Abnormality Rate
```
Real abnormality rate: 22.0%
Generated abnormality rate: 18.5%
Cohen's h: 0.18
Effect size: small
```

**Interpretation**: "Generated data shows slightly lower abnormality rates than real data, but the difference is small in practical terms. The model captures abnormality patterns reasonably well."

#### Example 2: Large Effect
```
Real abnormality rate: 9.5%
Generated abnormality rate: 2.1%
Cohen's h: 0.85
Effect size: large
```

**Interpretation**: "Generated data has substantially lower abnormality rates than real data. This represents a large practical difference, indicating the model may be overly conservative in flagging abnormalities."

#### Example 3: Medium Effect
```
Real abnormality rate: 0.4%
Generated abnormality rate: 53.0%
Cohen's h: -0.96
Effect size: large (negative direction)
```

**Interpretation**: "Generated data shows dramatically higher abnormality rates than real data. The large negative effect indicates the model generates substantially more abnormal trajectories than observed in reality."

---

## Confidence Intervals (CIs)

### What are Confidence Intervals?

A 95% confidence interval means: "If we repeated this experiment 100 times, we'd expect the true value to fall within this range in 95 of those experiments."

### Interpretation Guidelines

#### 1. Precision
**Narrow CI**: Precise estimate, high confidence
```
Rate: 18.5%, 95% CI: [17.8%, 19.2%]
→ We're very confident the true rate is close to 18.5%
```

**Wide CI**: Imprecise estimate, low confidence
```
Rate: 18.5%, 95% CI: [10.2%, 28.9%]
→ True rate could be anywhere from 10% to 29%
```

#### 2. Overlap Assessment
**No Overlap**: Strong evidence of difference
```
Real: 22.0%, 95% CI: [21.0%, 23.0%]
Generated: 18.5%, 95% CI: [17.5%, 19.5%]
→ CIs don't overlap → Strong evidence of difference
```

**Overlap**: Weaker evidence, possible no difference
```
Real: 22.0%, 95% CI: [20.0%, 24.0%]
Generated: 21.5%, 95% CI: [19.5%, 23.5%]
→ CIs overlap → Difference may not be meaningful
```

#### 3. Sample Size Effects
**Large Sample → Narrow CI**:
```
n=5000: Rate 18.5%, CI: [17.7%, 19.3%] (width: 1.6%)
```

**Small Sample → Wide CI**:
```
n=100: Rate 18.5%, CI: [11.2%, 28.1%] (width: 16.9%)
```

### Using Wilson Score Interval

Our implementation uses the Wilson score interval (not normal approximation) because it:
- ✅ Works well for proportions near 0 or 1
- ✅ Never produces invalid intervals (e.g., negative proportions)
- ✅ More accurate for small sample sizes
- ✅ Recommended by statisticians over Wald interval

---

## Practical Significance Framework

### Step 1: Check Statistical Significance
```
p-value < 0.05 (adjusted): Statistically significant
p-value ≥ 0.05 (adjusted): Not statistically significant
```

### Step 2: Assess Effect Size
```
|h| < 0.2: Small practical difference
0.2 ≤ |h| < 0.5: Medium practical difference
|h| ≥ 0.5: Large practical difference
```

### Step 3: Examine Confidence Intervals
```
Narrow CI + No overlap: High confidence in difference
Wide CI or overlap: Lower confidence, interpret cautiously
```

### Combined Interpretation Matrix

| Statistical Significance | Effect Size | Practical Significance |
|--------------------------|-------------|------------------------|
| ✅ Significant | Large (h ≥ 0.5) | ✅ **Strong evidence of meaningful difference** |
| ✅ Significant | Medium (0.2-0.5) | ⚠️ **Moderate evidence, consider context** |
| ✅ Significant | Small (h < 0.2) | ⚠️ **Statistically real but may be trivial** |
| ❌ Not significant | Large (h ≥ 0.5) | ⚠️ **Potentially important, may need more data** |
| ❌ Not significant | Medium (0.2-0.5) | 🤷 **Inconclusive, more evidence needed** |
| ❌ Not significant | Small (h < 0.2) | ✅ **No meaningful difference** |

---

## Application to Wang Abnormality Detection

### Context
Wang abnormality detection identifies trajectories with unusual patterns (detours, speed anomalies, temporal delays). We compare real vs generated trajectories.

### Interpretation Goals
1. **Realism**: Does generated data have similar abnormality rates to real data?
2. **Model quality**: Are differences small (good) or large (problematic)?
3. **Pattern matching**: Do specific abnormality types match reality?

### Example Analysis

```json
{
  "model": "distilled",
  "od_source": "test",
  "real_rate": 22.05,
  "generated_rate": 18.50,
  "chi2": 42.15,
  "p_value": 8.4e-11,
  "p_value_adjusted": 4.2e-10,
  "significant": true,
  "cohens_h": 0.18,
  "effect_size": "small",
  "real_ci_95": [21.0, 23.1],
  "generated_ci_95": [17.5, 19.5]
}
```

**Interpretation**:
- ✅ **Statistical significance**: p < 0.001 (highly significant)
- ✅ **Effect size**: h = 0.18 (small)
- ✅ **Confidence intervals**: Narrow, non-overlapping
- ✅ **Conclusion**: "Distilled model produces slightly fewer abnormal trajectories than reality (18.5% vs 22.1%), but the practical difference is small (h=0.18). This represents good realism—the model captures abnormality patterns with minor deviation."

### Comparison: Good vs Poor Realism

**Good Realism** (Small Effect):
```
Real: 9.5%, Generated: 8.8%, h = 0.10 (small)
→ Model captures abnormality patterns well
```

**Poor Realism** (Large Effect):
```
Real: 0.4%, Generated: 53.0%, h = -0.96 (large)
→ Model generates far too many abnormalities
```

---

## Best Practices

### 1. Always Report All Three
```
✅ Good: "Generated rate (18.5%, 95% CI: [17.5%, 19.5%]) was significantly 
         lower than real rate (22.0%, 95% CI: [21.0%, 23.0%]), p < 0.001, 
         with a small effect size (h = 0.18)."

❌ Bad:  "Generated rate was significantly different, p < 0.001."
```

### 2. Prioritize Practical Significance
- Don't report "significant" without context
- Effect size tells you if it matters
- Large samples → everything is "significant"

### 3. Use Domain Knowledge
- What constitutes "meaningful" difference in your domain?
- For abnormality detection: ±5% might be acceptable, ±50% is catastrophic
- Context matters more than arbitrary thresholds

### 4. Visualize Uncertainty
- Plot rates with error bars (CIs)
- Makes uncertainty immediately visible
- Helps readers assess practical significance

---

## References

### Effect Sizes
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.
- Cohen's h specifically for proportions: Section 6.2

### Confidence Intervals
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209-212.
- Agresti, A., & Coull, B. A. (1998). Approximate is better than "exact" for interval estimation of binomial proportions. *The American Statistician*, 52(2), 119-126.

### Practical Significance
- Sullivan, G. M., & Feinn, R. (2012). Using effect size—or why the P value is not enough. *Journal of Graduate Medical Education*, 4(3), 279-282.
- American Psychological Association. (2020). *Publication Manual* (7th ed.). Section 3.5: Statistical and Mathematical Copy.

---

## Tools and Scripts

### Computing Effect Sizes and CIs

**Script**: `tools/analyze_wang_results.py`

**Functions**:
```python
# Effect size
cohens_h = compute_cohens_h(real_rate, generated_rate)
effect_size_interpretation = interpret_cohens_h(cohens_h)

# Confidence intervals
real_ci = compute_proportion_ci(real_abnormal_count, real_total_count)
gen_ci = compute_proportion_ci(gen_abnormal_count, gen_total_count)
```

**Output**: `wang_results_aggregated.json`
- Contains all statistical tests with effect sizes and CIs
- Use for comprehensive analysis and reporting

---

## Quick Reference Card

### Cohen's h Interpretation
- **h < 0.2**: Small effect (minor practical difference)
- **0.2 ≤ h < 0.5**: Medium effect (moderate practical difference)
- **h ≥ 0.5**: Large effect (substantial practical difference)

### Confidence Interval Checks
- **Narrow**: High precision, reliable estimate
- **Wide**: Low precision, interpret cautiously
- **No overlap**: Strong evidence of difference
- **Overlap**: Weaker evidence, may not differ

### Decision Rule
1. Check p-value (statistical significance)
2. Check effect size (practical magnitude)
3. Check CIs (precision and overlap)
4. Integrate all three for final interpretation

**Remember**: Statistical significance alone is never enough!

---

**For more details**, see:
- `tools/analyze_wang_results.py` - Implementation
- `docs/results/WANG_ABNORMALITY_DETECTION_RESULTS.md` - Applied results
- Issue #18 - Original implementation issue
