# Paired Statistical Comparison

## Models

- Model 1: distilled
- Model 2: distilled_seed44

## Summary

- Number of matched trajectory pairs: 778
- Significance level (α): 0.05

## Metric Comparisons

### hausdorff_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0142
- **Cohen's d**: 0.080
- **distilled mean**: 0.9239
- **distilled_seed44 mean**: 0.8541
- **Mean difference**: 0.0699
- **Std difference**: 0.8773

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### hausdorff_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0303
- **Cohen's d**: 0.058
- **distilled mean**: 0.0244
- **distilled_seed44 mean**: 0.0232
- **Mean difference**: 0.0012
- **Std difference**: 0.0205

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0450
- **Cohen's d**: 0.072
- **distilled mean**: 27.3920
- **distilled_seed44 mean**: 23.8362
- **Mean difference**: 3.5557
- **Std difference**: 49.1856

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0156
- **Cohen's d**: 0.082
- **distilled mean**: 0.5305
- **distilled_seed44 mean**: 0.4773
- **Mean difference**: 0.0532
- **Std difference**: 0.6500

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### edr (✗ Not Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0539
- **Cohen's d**: 0.065
- **distilled mean**: 0.4781
- **distilled_seed44 mean**: 0.4596
- **Mean difference**: 0.0184
- **Std difference**: 0.2822

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

