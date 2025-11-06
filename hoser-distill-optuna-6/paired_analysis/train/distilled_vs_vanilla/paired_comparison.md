# Paired Statistical Comparison

## Models

- Model 1: distilled
- Model 2: vanilla

## Summary

- Number of matched trajectory pairs: 61
- Significance level (α): 0.05

## Metric Comparisons

### hausdorff_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0027
- **Cohen's d**: -0.387
- **distilled mean**: 0.3956
- **vanilla mean**: 0.7267
- **Mean difference**: -0.3311
- **Std difference**: 0.8563

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0488). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### hausdorff_norm (✓ Significant)

- **Test**: Paired t-test
- **P-value**: 0.0006
- **Cohen's d**: -0.467
- **distilled mean**: 0.0207
- **vanilla mean**: 0.0395
- **Mean difference**: -0.0188
- **Std difference**: 0.0402

### dtw_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0066
- **Cohen's d**: -0.177
- **distilled mean**: 7.3560
- **vanilla mean**: 12.8962
- **Mean difference**: -5.5402
- **Std difference**: 31.2830

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0058
- **Cohen's d**: -0.307
- **distilled mean**: 0.2523
- **vanilla mean**: 0.4520
- **Mean difference**: -0.1997
- **Std difference**: 0.6508

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0031). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### edr (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: -0.633
- **distilled mean**: 0.3303
- **vanilla mean**: 0.6193
- **Mean difference**: -0.2890
- **Std difference**: 0.4569

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0268). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

