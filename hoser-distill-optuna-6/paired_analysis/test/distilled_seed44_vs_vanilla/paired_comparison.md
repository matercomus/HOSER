# Paired Statistical Comparison

## Models

- Model 1: distilled_seed44
- Model 2: vanilla

## Summary

- Number of matched trajectory pairs: 65
- Significance level (α): 0.05

## Metric Comparisons

### hausdorff_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0001
- **Cohen's d**: -0.412
- **distilled_seed44 mean**: 0.3618
- **vanilla mean**: 0.6845
- **Mean difference**: -0.3227
- **Std difference**: 0.7837

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### hausdorff_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0000
- **Cohen's d**: -0.494
- **distilled_seed44 mean**: 0.0189
- **vanilla mean**: 0.0348
- **Mean difference**: -0.0160
- **Std difference**: 0.0323

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0017). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_km (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0006
- **Cohen's d**: -0.269
- **distilled_seed44 mean**: 5.1523
- **vanilla mean**: 11.7003
- **Mean difference**: -6.5480
- **Std difference**: 24.3231

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### dtw_norm (✓ Significant)

- **Test**: Wilcoxon signed-rank test
- **P-value**: 0.0006
- **Cohen's d**: -0.323
- **distilled_seed44 mean**: 0.2106
- **vanilla mean**: 0.4063
- **Mean difference**: -0.1957
- **Std difference**: 0.6053

**Warnings**:
- Differences may not be normally distributed (Shapiro-Wilk p=0.0000). Consider using Wilcoxon test instead of paired t-test.
- Using non-parametric Wilcoxon test due to non-normal differences

### edr (✓ Significant)

- **Test**: Paired t-test
- **P-value**: 0.0000
- **Cohen's d**: -0.714
- **distilled_seed44 mean**: 0.2889
- **vanilla mean**: 0.5383
- **Mean difference**: -0.2494
- **Std difference**: 0.3494

