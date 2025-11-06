# Cross-Seed Statistical Analysis

## Overview

This document presents cross-seed statistical analysis of HOSER evaluation results. All metrics are reported as **mean ± std** with confidence intervals.

**Confidence Level**: 95%

---

## Beijing Dataset

### Distilled

#### TEST OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0134 | 0.0013 | [0.0017, 0.0250] | 9.70% | 2 |
| Duration_JSD | 0.0188 | 0.0012 | [0.0077, 0.0299] | 6.56% | 2 |
| Radius_JSD | 0.0096 | 0.0001 | [0.0084, 0.0108] | 1.39% | 2 |
| Hausdorff_km | 0.7328 | 0.0032 | [0.7041, 0.7614] | 0.44% | 2 |
| DTW_km | 14.9811 | 0.6300 | [9.3212, 20.6410] | 4.20% | 2 |
| EDR | 0.4403 | 0.0018 | [0.4241, 0.4566] | 0.41% | 2 |
| OD Match Rate | 98.19% | - | - | - | 2 |

#### TRAIN OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0118 | 0.0003 | [0.0088, 0.0149] | 2.87% | 2 |
| Duration_JSD | 0.0183 | 0.0008 | [0.0115, 0.0251] | 4.13% | 2 |
| Radius_JSD | 0.0101 | 0.0004 | [0.0066, 0.0135] | 3.80% | 2 |
| Hausdorff_km | 0.7882 | 0.0200 | [0.6082, 0.9682] | 2.54% | 2 |
| DTW_km | 16.8523 | 0.0613 | [16.3018, 17.4028] | 0.36% | 2 |
| EDR | 0.4523 | 0.0091 | [0.3703, 0.5343] | 2.02% | 2 |
| OD Match Rate | 98.85% | - | - | - | 2 |

### Unknown

#### TEST OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0627 | 0.0780 | [-0.1311, 0.2565] | 124.41% | 3 |
| Duration_JSD | 0.0207 | 0.0031 | [0.0130, 0.0284] | 15.03% | 3 |
| Radius_JSD | 0.0710 | 0.1166 | [-0.2188, 0.3607] | 164.37% | 3 |
| Hausdorff_km | 0.8231 | 0.2296 | [0.2528, 1.3934] | 27.89% | 3 |
| DTW_km | 21.5510 | 11.1916 | [-6.2505, 49.3526] | 51.93% | 3 |
| EDR | 0.4947 | 0.0159 | [0.4551, 0.5343] | 3.22% | 3 |
| OD Match Rate | 63.03% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD, Distance_gen_mean, Duration_JSD, Duration_gen_mean, Radius_JSD, Radius_gen_mean, Hausdorff_km, DTW_km, matched_od_pairs

#### TRAIN OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0613 | 0.0720 | [-0.1177, 0.2403] | 117.50% | 3 |
| Duration_JSD | 0.0207 | 0.0029 | [0.0135, 0.0279] | 14.03% | 3 |
| Radius_JSD | 0.0680 | 0.1125 | [-0.2114, 0.3475] | 165.28% | 3 |
| Hausdorff_km | 0.8246 | 0.2759 | [0.1393, 1.5100] | 33.46% | 3 |
| DTW_km | 21.6829 | 12.1366 | [-8.4662, 51.8319] | 55.97% | 3 |
| EDR | 0.5052 | 0.0012 | [0.5021, 0.5083] | 0.25% | 3 |
| OD Match Rate | 65.26% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD, Distance_gen_mean, Duration_JSD, Duration_gen_mean, Radius_JSD, Radius_gen_mean, Hausdorff_km, DTW_km, matched_od_pairs

### Vanilla

#### TEST OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0134 | 0.0000 | [0.0134, 0.0134] | 0.00% | 1 |
| Duration_JSD | 0.0148 | 0.0000 | [0.0148, 0.0148] | 0.00% | 1 |
| Radius_JSD | 0.0130 | 0.0000 | [0.0130, 0.0130] | 0.00% | 1 |
| Hausdorff_km | 0.6728 | 0.0000 | [0.6728, 0.6728] | 0.00% | 1 |
| DTW_km | 9.7636 | 0.0000 | [9.7636, 9.7636] | 0.00% | 1 |
| EDR | 0.4835 | 0.0000 | [0.4835, 0.4835] | 0.00% | 1 |
| OD Match Rate | 59.90% | - | - | - | 1 |

#### TRAIN OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0128 | 0.0000 | [0.0128, 0.0128] | 0.00% | 1 |
| Duration_JSD | 0.0151 | 0.0000 | [0.0151, 0.0151] | 0.00% | 1 |
| Radius_JSD | 0.0147 | 0.0000 | [0.0147, 0.0147] | 0.00% | 1 |
| Hausdorff_km | 0.6020 | 0.0000 | [0.6020, 0.6020] | 0.00% | 1 |
| DTW_km | 8.5875 | 0.0000 | [8.5875, 8.5875] | 0.00% | 1 |
| EDR | 0.4837 | 0.0000 | [0.4837, 0.4837] | 0.00% | 1 |
| OD Match Rate | 61.56% | - | - | - | 1 |

## Porto Dataset

### Distill Phase1

#### TEST OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0055 | 0.0008 | [0.0036, 0.0073] | 13.80% | 3 |
| Duration_JSD | 0.0256 | 0.0025 | [0.0193, 0.0319] | 9.92% | 3 |
| Radius_JSD | 0.0105 | 0.0008 | [0.0086, 0.0125] | 7.36% | 3 |
| Hausdorff_km | 0.5494 | 0.0152 | [0.5115, 0.5872] | 2.77% | 3 |
| DTW_km | 15.2055 | 0.4231 | [14.1545, 16.2565] | 2.78% | 3 |
| EDR | 0.4661 | 0.0160 | [0.4262, 0.5059] | 3.44% | 3 |
| OD Match Rate | 88.81% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD

#### TRAIN OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0059 | 0.0013 | [0.0026, 0.0092] | 22.37% | 3 |
| Duration_JSD | 0.0254 | 0.0021 | [0.0201, 0.0306] | 8.37% | 3 |
| Radius_JSD | 0.0089 | 0.0010 | [0.0065, 0.0113] | 10.74% | 3 |
| Hausdorff_km | 0.5495 | 0.0168 | [0.5077, 0.5913] | 3.06% | 3 |
| DTW_km | 14.8563 | 0.4925 | [13.6329, 16.0796] | 3.31% | 3 |
| EDR | 0.4667 | 0.0151 | [0.4293, 0.5041] | 3.23% | 3 |
| OD Match Rate | 91.33% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD, Radius_JSD

### Distill Phase2

#### TEST OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0048 | 0.0012 | [0.0017, 0.0078] | 25.68% | 3 |
| Duration_JSD | 0.0278 | 0.0053 | [0.0146, 0.0410] | 19.14% | 3 |
| Radius_JSD | 0.0091 | 0.0018 | [0.0048, 0.0135] | 19.28% | 3 |
| Hausdorff_km | 0.5487 | 0.0109 | [0.5215, 0.5758] | 1.99% | 3 |
| DTW_km | 15.3560 | 0.3545 | [14.4753, 16.2367] | 2.31% | 3 |
| EDR | 0.4633 | 0.0082 | [0.4431, 0.4836] | 1.76% | 3 |
| OD Match Rate | 87.62% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD, Duration_JSD, Duration_gen_mean, Radius_JSD

#### TRAIN OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0050 | 0.0011 | [0.0022, 0.0077] | 22.34% | 3 |
| Duration_JSD | 0.0277 | 0.0055 | [0.0140, 0.0414] | 19.85% | 3 |
| Radius_JSD | 0.0084 | 0.0009 | [0.0061, 0.0106] | 10.86% | 3 |
| Hausdorff_km | 0.5565 | 0.0086 | [0.5352, 0.5778] | 1.54% | 3 |
| DTW_km | 15.3956 | 0.2915 | [14.6716, 16.1197] | 1.89% | 3 |
| EDR | 0.4683 | 0.0100 | [0.4434, 0.4933] | 2.15% | 3 |
| OD Match Rate | 90.90% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD, Duration_JSD, Duration_gen_mean, Radius_JSD

### Vanilla

#### TEST OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0060 | 0.0016 | [0.0020, 0.0099] | 26.52% | 3 |
| Duration_JSD | 0.0268 | 0.0010 | [0.0243, 0.0293] | 3.72% | 3 |
| Radius_JSD | 0.0105 | 0.0015 | [0.0068, 0.0142] | 14.11% | 3 |
| Hausdorff_km | 0.5534 | 0.0265 | [0.4877, 0.6191] | 4.78% | 3 |
| DTW_km | 15.5080 | 1.3025 | [12.2725, 18.7435] | 8.40% | 3 |
| EDR | 0.4720 | 0.0243 | [0.4116, 0.5324] | 5.15% | 3 |
| OD Match Rate | 86.88% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD, Duration_gen_mean, Radius_JSD

#### TRAIN OD

| Metric | Mean | Std Dev | 95% CI | CV% | N Seeds |
|--------|------|---------|--------|-----|--------|
| Distance_JSD | 0.0060 | 0.0016 | [0.0019, 0.0100] | 27.38% | 3 |
| Duration_JSD | 0.0270 | 0.0012 | [0.0241, 0.0299] | 4.31% | 3 |
| Radius_JSD | 0.0088 | 0.0016 | [0.0049, 0.0128] | 17.95% | 3 |
| Hausdorff_km | 0.5609 | 0.0256 | [0.4974, 0.6244] | 4.56% | 3 |
| DTW_km | 15.6592 | 1.0335 | [13.0919, 18.2264] | 6.60% | 3 |
| EDR | 0.4738 | 0.0181 | [0.4288, 0.5187] | 3.82% | 3 |
| OD Match Rate | 89.95% | - | - | - | 3 |

⚠️  **High Variance Metrics** (CV > 10%): Distance_JSD, Duration_gen_mean, Radius_JSD

---

## Statistical Notes

- **Mean ± Std**: Arithmetic mean with standard deviation
- **95% CI**: 95% confidence interval using t-distribution
- **CV%**: Coefficient of variation (std/mean × 100)
- **N Seeds**: Number of random seeds in analysis

**Interpretation**:
- Low CV (<5%): Stable, seed-independent results
- Medium CV (5-10%): Moderate seed sensitivity
- High CV (>10%): High seed sensitivity, interpret with caution
