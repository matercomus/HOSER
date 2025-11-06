# Paired Statistical Comparison

## Models

- Model 1: distilled
- Model 2: vanilla

## Summary

- Number of matched trajectory pairs: 62
- Significance level (α): 0.05

## Metric Comparisons

### hausdorff_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0002
- **Cohen's d**: -0.194
- **distilled mean**: 0.4785
- **vanilla mean**: 0.6859
- **Mean difference**: -0.2075
- **Std difference**: 1.0713

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### hausdorff_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0001
- **Cohen's d**: -0.511
- **distilled mean**: 0.0208
- **vanilla mean**: 0.0354
- **Mean difference**: -0.0146
- **Std difference**: 0.0286

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0309). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0004
- **Cohen's d**: 0.018
- **distilled mean**: 12.2994
- **vanilla mean**: 11.2041
- **Mean difference**: 1.0954
- **Std difference**: 60.4184

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0009
- **Cohen's d**: -0.090
- **distilled mean**: 0.3107
- **vanilla mean**: 0.3951
- **Mean difference**: -0.0844
- **Std difference**: 0.9366

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### edr (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: -0.663
- **distilled mean**: 0.3276
- **vanilla mean**: 0.5555
- **Mean difference**: -0.2279
- **Std difference**: 0.3438

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0047). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

