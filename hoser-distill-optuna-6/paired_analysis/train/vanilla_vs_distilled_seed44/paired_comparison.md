# Paired Statistical Comparison

## Models

- Model 1: vanilla
- Model 2: distilled_seed44

## Summary

- Number of matched trajectory pairs: 64
- Significance level (α): 0.05

## Metric Comparisons

### hausdorff_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: 0.635
- **vanilla mean**: 0.7384
- **distilled_seed44 mean**: 0.3018
- **Mean difference**: 0.4366
- **Std difference**: 0.6870

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### hausdorff_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: 0.685
- **vanilla mean**: 0.0399
- **distilled_seed44 mean**: 0.0177
- **Mean difference**: 0.0222
- **Std difference**: 0.0324

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0003). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: 0.302
- **vanilla mean**: 12.9599
- **distilled_seed44 mean**: 4.7197
- **Mean difference**: 8.2402
- **Std difference**: 27.3100

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: 0.517
- **vanilla mean**: 0.4580
- **distilled_seed44 mean**: 0.1745
- **Mean difference**: 0.2835
- **Std difference**: 0.5487

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### edr (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: 0.857
- **vanilla mean**: 0.6257
- **distilled_seed44 mean**: 0.2963
- **Mean difference**: 0.3293
- **Std difference**: 0.3843

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0348). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

