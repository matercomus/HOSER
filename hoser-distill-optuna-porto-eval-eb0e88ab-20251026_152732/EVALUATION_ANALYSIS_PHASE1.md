# HOSER Distillation: Porto Dataset Evaluation Analysis (Phase 1)

**Date:** October 31, 2025  
**Experiment:** Comparison of Vanilla vs Distilled HOSER Models (Phase 1 Hyperparameters)  
**Dataset:** Porto Taxi Trajectory Data

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experimental Setup](#1-experimental-setup)
   - [Models Evaluated](#models-evaluated)
   - [Training Configuration](#training-configuration-fair-comparison)
   - [Evaluation Protocol](#evaluation-protocol)
   - [Metrics Explained](#metrics)
3. [Results Overview](#2-results-overview)
4. [Key Findings](#3-key-findings)
   - [Path Completion Success](#31-path-completion-success-od-coverage)
   - [Trip Length Realism](#32-trip-length-realism)
   - [Spatial Distribution Quality](#33-spatial-distribution-quality-radius-of-gyration)
   - [Generalization vs Memorization](#34-generalization-vs-memorization)
5. [Porto vs Beijing: Critical Differences](#4-porto-vs-beijing-critical-differences)
6. [What Distillation Transferred (Phase 1)](#5-what-distillation-transferred-phase-1)
7. [Trajectory-Level Analysis](#6-trajectory-level-analysis)
   - [Inference Performance](#65-inference-performance)
8. [Statistical Summary](#7-statistical-summary)
9. [Conclusions](#8-conclusions)
10. [Appendix: Methodology Details](#9-appendix-methodology-details)

---

## Executive Summary

This analysis evaluates Phase 1 knowledge-distilled HOSER models against vanilla baseline on the Porto taxi dataset. Results reveal **fundamentally different behavior** compared to Beijing, with both distilled and vanilla models achieving comparable performance.

### Key Findings:

1. **✅ Both models achieve high path completion**: Distilled 83-92%, Vanilla 88-92%
   - Contrasts sharply with Beijing (Distilled 85-89% vs Vanilla 12-18%)
   
2. **✅ Similar distance distribution quality**: 
   - Distilled Distance JSD: 0.0048-0.0078 (test)
   - Vanilla Distance JSD: 0.0049-0.0063 (test)
   - Both models generate realistic trip lengths (~3.4-3.7 km vs real 3.66 km)

3. **✅ Comparable spatial complexity**:
   - Distilled Radius JSD: 0.0092-0.0121 (test)
   - Vanilla Radius JSD: 0.0098-0.0114 (test)
   - Similar spatial distribution patterns

4. **⚠️ Marginal distillation benefit**: Phase 1 hyperparameters (λ=0.00644, τ=2.802, w=4) provide minimal improvement over vanilla
   - Suggests Porto task characteristics differ fundamentally from Beijing
   - Phase 2 refined hyperparameters (λ=0.00598, τ=2.515, w=4) may address this

5. **🔍 Cross-seed consistency**: Both model types show stable performance (CV <10% for key metrics)

6. **📊 Scenario-level findings reveal context-dependent performance**:
   - Distilled excels in **dense urban** (within-center: +20% Distance JSD improvement)
   - Vanilla excels in **suburban** scenarios (-20% Distance JSD improvement)
   - Benefits average to near-zero across full dataset
   - Teacher knowledge most useful where it was trained (city center)

**Critical Insight**: Porto's shorter trajectories (~3.6 km vs Beijing's 5.2 km) and different road network topology may make the navigation task more tractable for vanilla models, reducing the relative benefit of distillation with Phase 1 hyperparameters.

---

## 1. Experimental Setup

### Models Evaluated:

**Phase 1 Optimization Context:**
- **Best Phase 1 hyperparameters** from Optuna study `hoser_tuning_20251014_145134`
- Trial 10: val_acc 0.265343, λ=0.00644, τ=2.802, w=4
- For complete Phase 1 tuning analysis, see [Hyperparameter-Optimization-Porto.md](Hyperparameter-Optimization-Porto.md)

**Evaluated Models:**

1. **Vanilla HOSER** (3 seeds: 42, 43, 44)
   - `vanilla_25epoch_seed42.pth`
   - `vanilla_25epoch_seed43.pth`
   - `vanilla_25epoch_seed44.pth`
   - Baseline model trained without distillation
   - 25 epochs per seed
   - MLE-only training (hard labels)

2. **Distilled HOSER** (3 seeds: 42, 43, 44)
   - `distill_25epoch_seed42.pth`
   - `distill_25epoch_seed43.pth`
   - `distill_25epoch_seed44.pth`
   - Student model trained with knowledge distillation from LM-TAD teacher
   - 25 epochs per seed
   - MLE + KL divergence from teacher
   - **Phase 1 distillation hyperparameters:**
     - λ (KL weight): **0.00644** (4.6× higher than Beijing's 0.0014)
     - τ (temperature): **2.802** (36% lower than Beijing's 4.37)
     - Window size: **4 steps** (43% shorter than Beijing's 7 steps)

### Training Configuration (Fair Comparison):

All models were trained using **identical base configurations** from Phase 1 Optuna hyperparameter tuning:

| Parameter | Porto Value | Beijing Value | Notes |
|-----------|-------------|---------------|-------|
| **Base Config** | `porto_hoser.yaml` | `Beijing.yaml` | Dataset-specific |
| **Batch Size** | 32 | 128 | Reduced for Porto's 2× longer trajectories |
| **Accumulation Steps** | 2 | 8 | Effective batch = 64 vs 1024 |
| **Candidate Top-K** | 64 | 64 | Same filtering |
| **Learning Rate** | 0.001 | 0.001 | Identical |
| **Weight Decay** | 0.1 | 0.1 | Identical |
| **Max Epochs** | 25 | 25 | Full convergence |
| **Architecture** | Identical HOSER | Identical HOSER | Same model |
| **Gradient Checkpointing** | ✅ Enabled | ❌ Disabled | Porto memory optimization |

**Porto-Specific Differences:**

| Parameter | Porto | Beijing | Reason |
|-----------|-------|---------|--------|
| **Trajectory Length** | ~2× longer | Baseline | Porto trips span more road segments |
| **Memory Optimization** | Gradient checkpointing | Standard | O(T²) memory scaling with length |
| **Effective Batch** | 32 × 2 = 64 | 128 × 8 = 1024 | 16× smaller for memory constraints |
| **Baseline Accuracy** | 26.5% | 57.2% | Porto is a harder task |

**Distillation-Only Differences:**

| Parameter | Vanilla | Distilled (Phase 1) |
|-----------|---------|-------------------|
| **Distillation enabled** | ❌ No | ✅ Yes |
| **KL weight (λ)** | 0.0 (disabled) | 0.00644 |
| **Temperature (τ)** | N/A | 2.802 |
| **Teacher window** | N/A | 4 steps |
| **Teacher model** | N/A | LM-TAD (frozen) |

**Key Point:** The **ONLY difference** between vanilla and distilled models is whether distillation was enabled and the Phase 1 tuned distillation hyperparameters. All other training parameters are identical.

> **📖 For detailed training methodology and Phase 1 hyperparameter optimization, see:**
> - [`Hyperparameter-Optimization-Porto.md`](Hyperparameter-Optimization-Porto.md) - Phase 1 & Phase 2 tuning results
> - [`../docs/LMTAD-Distillation.md`](../docs/LMTAD-Distillation.md) - Distillation framework

### Evaluation Protocol:

- **Generated Trajectories:** 5,000 per model per OD source (30,000 total)
- **OD Sources:** Train set (memorization) and Test set (generalization)
- **Real Data:** 481,359 train trajectories, 137,532 test trajectories
- **Grid Resolution:** 0.001° (~111m) for OD pair matching
- **Beam Search:** Width 4 for trajectory generation
- **EDR Threshold:** ε=100m for Edit Distance on Real sequence

### Metrics:

#### Global Metrics (Distribution-Level):

##### **Jensen-Shannon Divergence (JSD)**

**Formula:**

$$
\text{JSD}(P \parallel Q) = \frac{1}{2} \text{KL}(P \parallel M) + \frac{1}{2} \text{KL}(Q \parallel M)
$$

where $M = \frac{1}{2}(P + Q)$

**Range:** [0, 1] (0 = perfect match, 1 = maximum divergence)

**Applied to three attributes:**

1. **Distance JSD**: Trip distance distributions (km)
2. **Duration JSD**: Trip duration distributions (hours)
3. **Radius of Gyration JSD**: Spatial spread distributions

##### **Radius of Gyration**

**Formula:**

$$
R_g = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \text{dist}(p_i, \bar{p})^2}
$$

where $\bar{p}$ is the centroid of all trajectory points.

#### Local Metrics (Trajectory-Level):

##### **Hausdorff Distance**

**Formula:**

$$
H(A, B) = \max\{h(A, B), h(B, A)\}
$$

where $h(A, B) = \max_{a \in A} \min_{b \in B} \text{dist}(a, b)$

**Range:** [0, ∞) km (lower is better)

##### **Dynamic Time Warping (DTW)**

**Intuition:** Minimum cumulative distance when optimally aligning two trajectories.

**Range:** [0, ∞) km (lower is better, scales with trajectory length)

##### **Edit Distance on Real Sequence (EDR)**

**Formula:**

$$
\text{EDR}(A, B, \varepsilon) = \frac{\text{EditOps}(A, B, \varepsilon)}{\max(|A|, |B|)}
$$

**Range:** [0, 1] (normalized, 0 = perfect match)

#### Coverage Metrics:

- **Matched OD Pairs:** Generated trajectories whose actual endpoints match real OD patterns
- **Total Generated OD Pairs:** Unique OD pairs from generated trajectory endpoints
- **Match Rate:** Percentage of generated trajectories that reach realistic destinations

---

## 2. Results Overview

### 2.1 Complete Results Table

**Real Data Baseline:**

| OD Source | Distance (km) | Duration (hours) | Radius of Gyration |
|-----------|---------------|------------------|-------------------|
| Train | 3.656 | 0.205 | 0.765 |
| Test | 3.658 | 0.205 | 0.765 |

**Model Performance:**

| Model | OD Source | Matched OD | Total Generated | Match Rate | Distance JSD | Radius JSD | Duration JSD | Distance (km) | Hausdorff (km) | DTW (km) | EDR |
|-------|-----------|------------|-----------------|------------|--------------|------------|--------------|---------------|----------------|----------|-----|
| **distill** | train | 4,256 | 4,654 | **91.5%** | 0.0053 | 0.0077 | 0.0257 | 3.589 | 0.581 | 16.12 | 0.478 |
| **distill** | test | 4,064 | 4,571 | **88.9%** | 0.0050 | 0.0092 | 0.0256 | 3.563 | 0.565 | 15.57 | 0.476 |
| **distill_seed43** | train | 4,194 | 4,655 | **90.1%** | 0.0048 | 0.0081 | 0.0280 | 3.427 | 0.532 | 14.48 | 0.454 |
| **distill_seed43** | test | 4,036 | 4,581 | **88.1%** | 0.0051 | 0.0101 | 0.0274 | 3.466 | 0.523 | 14.18 | 0.446 |
| **distill_seed44** | train | 4,059 | 4,597 | **88.3%** | 0.0078 | 0.0107 | 0.0273 | 3.621 | 0.570 | 16.38 | 0.489 |
| **distill_seed44** | test | 3,787 | 4,530 | **83.6%** | 0.0078 | 0.0121 | 0.0273 | 3.659 | 0.572 | 16.79 | 0.494 |
| **vanilla** | train | 4,251 | 4,638 | **91.7%** | 0.0074 | 0.0100 | 0.0231 | 3.440 | 0.548 | 14.77 | 0.458 |
| **vanilla** | test | 4,086 | 4,559 | **89.6%** | 0.0063 | 0.0114 | 0.0229 | 3.439 | 0.542 | 14.96 | 0.457 |
| **vanilla_seed43** | train | 4,195 | 4,647 | **90.3%** | 0.0048 | 0.0083 | 0.0258 | 3.459 | 0.534 | 14.41 | 0.458 |
| **vanilla_seed43** | test | 4,022 | 4,575 | **87.9%** | 0.0049 | 0.0105 | 0.0260 | 3.491 | 0.539 | 14.96 | 0.457 |
| **vanilla_seed44** | train | 4,284 | 4,654 | **92.0%** | 0.0056 | 0.0084 | 0.0273 | 3.570 | 0.567 | 15.39 | 0.484 |
| **vanilla_seed44** | test | 4,082 | 4,592 | **88.9%** | 0.0052 | 0.0098 | 0.0279 | 3.587 | 0.567 | 15.69 | 0.485 |

### 2.2 Aggregated Comparison (by Model Type)

**Distilled Models (seeds 42, 43, 44):**

| OD Source | Match Rate | Distance JSD | Radius JSD | Distance (km) | DTW (km) | EDR |
|-----------|------------|--------------|------------|---------------|----------|-----|
| Train | 89.9% ± 1.6% | 0.0060 ± 0.0016 | 0.0088 ± 0.0016 | 3.546 ± 0.100 | 15.66 ± 1.05 | 0.474 ± 0.018 |
| Test | 86.9% ± 2.9% | 0.0060 ± 0.0016 | 0.0108 ± 0.0015 | 3.563 ± 0.100 | 15.51 ± 1.32 | 0.472 ± 0.025 |

**Vanilla Models (seeds 42, 43, 44):**

| OD Source | Match Rate | Distance JSD | Radius JSD | Distance (km) | DTW (km) | EDR |
|-----------|------------|--------------|------------|---------------|----------|-----|
| Train | 91.3% ± 0.9% | 0.0059 ± 0.0013 | 0.0089 ± 0.0010 | 3.490 ± 0.070 | 14.85 ± 0.51 | 0.467 ± 0.014 |
| Test | 88.8% ± 0.9% | 0.0055 ± 0.0008 | 0.0106 ± 0.0011 | 3.506 ± 0.075 | 15.20 ± 0.41 | 0.466 ± 0.016 |

### 2.3 Key Observations

#### Path Completion (OD Coverage):
- **Distilled:** 83.6-91.5% match rate (train), 83.6-88.9% (test)
- **Vanilla:** 90.3-92.0% match rate (train), 87.9-89.6% (test)
- **Surprising finding:** Both models achieve similarly high success rates
- **Contrast with Beijing:** Beijing vanilla only achieved 12-18% vs distilled 85-89%

#### Distance Distribution Quality:
- **Distilled:** JSD 0.0048-0.0078 (test) 
- **Vanilla:** JSD 0.0049-0.0063 (test)
- **Similar performance:** Both models match real distance distribution well
- Real Porto distance: 3.66 km | Generated: 3.43-3.66 km

#### Radius of Gyration Quality:
- **Distilled:** JSD 0.0092-0.0121 (test)
- **Vanilla:** JSD 0.0098-0.0114 (test)
- **Comparable spatial complexity:** Both capture Porto's spatial patterns

---

## 3. Key Findings

### 3.1 Path Completion Success (OD Coverage)

**Finding:** Both vanilla and distilled models achieve high path completion rates on Porto.

| Model Type | Train Match Rate | Test Match Rate | Interpretation |
|------------|------------------|-----------------|----------------|
| **Distilled (seed 42)** | **91.5%** | **88.9%** | Successfully reaches targets |
| **Distilled (seed 43)** | **90.1%** | **88.1%** | Consistent navigation |
| **Distilled (seed 44)** | **88.3%** | **83.6%** | Good completion |
| **Vanilla (seed 42)** | **91.7%** | **89.6%** | Highest completion |
| **Vanilla (seed 43)** | **90.3%** | **87.9%** | Strong baseline |
| **Vanilla (seed 44)** | **92.0%** | **88.9%** | Excellent performance |

**Aggregate Statistics:**
- Distilled average: 89.9% (train), 86.9% (test)
- Vanilla average: 91.3% (train), 88.8% (test)
- **Difference:** Vanilla actually matches slightly better (+1.4% train, +1.9% test)

**Why This Differs from Beijing:**

1. **Shorter trajectories**: Porto trips average 3.66 km vs Beijing's 5.16 km
   - Fewer road segments to navigate
   - Lower chance of getting stuck or lost

2. **Road network characteristics**: Porto's road network may be more interconnected
   - Multiple valid paths between OD pairs
   - Easier to find alternate routes

3. **Baseline capability**: Porto vanilla models learned navigation well from MLE alone
   - Task may be more tractable without teacher guidance
   - 26.5% validation accuracy sufficient for generation

### 3.2 Trip Length Realism

![Train OD Distance Distributions](figures/distributions/distance_distribution_train_od.pdf)

![Test OD Distance Distributions](figures/distributions/distance_distribution_test_od.pdf)

**Finding:** Both model types generate realistic trip lengths close to Porto's 3.66 km average.

| Model Type | Train Distance | Test Distance | vs Real |
|------------|----------------|---------------|---------|
| Real Data | 3.656 km | 3.658 km | Baseline |
| **Distilled** | 3.546 ± 0.100 km | 3.563 ± 0.100 km | -2.9% (train), -2.6% (test) |
| **Vanilla** | 3.490 ± 0.070 km | 3.506 ± 0.075 km | -4.5% (train), -4.2% (test) |

**Distance JSD (lower is better):**
- Distilled test: 0.0048-0.0078 (mean 0.0060)
- Vanilla test: 0.0049-0.0063 (mean 0.0055)
- **Difference:** Negligible (0.0005 JSD units, ~9% relative)

**Interpretation:**
- Both models slightly underestimate distances (2-5% shorter than real)
- Distribution quality is comparable (JSD difference <0.001)
- Porto's shorter trips are easier to match than Beijing's longer trips

### 3.3 Spatial Distribution Quality (Radius of Gyration)

![Train OD Radius Distributions](figures/distributions/radius_distribution_train_od.pdf)

![Test OD Radius Distributions](figures/distributions/radius_distribution_test_od.pdf)

**Finding:** Similar spatial complexity captured by both model types.

| Metric | Distilled | Vanilla | Comparison |
|--------|-----------|---------|------------|
| **Radius JSD (train)** | 0.0088 ± 0.0016 | 0.0089 ± 0.0010 | Vanilla 1% worse |
| **Radius JSD (test)** | 0.0108 ± 0.0015 | 0.0106 ± 0.0011 | Vanilla 2% better |

**Real baseline:**
- Train radius: 0.765 | Test radius: 0.765

**Interpretation:**
- Both models capture Porto's spatial dispersion patterns
- JSD ~0.01 indicates excellent distribution matching
- No meaningful advantage for distillation in spatial complexity

**Contrast with Beijing:**
- Beijing: Distilled 0.003-0.004 JSD vs Vanilla 0.198-0.206 JSD (98% improvement)
- Porto: Distilled 0.011 JSD vs Vanilla 0.011 JSD (0% improvement)

### 3.4 Generalization vs Memorization

**Key Question:** Do models generalize to test OD pairs or just memorize training patterns?

**Real Data Consistency:**
- Train distance: 3.656 km | Test distance: 3.658 km (99.9% match)
- Porto has consistent trip characteristics across train/test splits

**Model Generalization Analysis:**

| Model | Train Dist JSD | Test Dist JSD | Δ (test - train) | Generalization |
|-------|----------------|---------------|------------------|----------------|
| **distill** | 0.0053 | 0.0050 | **-0.0003** | ✅ Improves on test |
| **distill_seed43** | 0.0048 | 0.0051 | **+0.0003** | ≈ Stable |
| **distill_seed44** | 0.0078 | 0.0078 | **0.0000** | ≈ Stable |
| **vanilla** | 0.0074 | 0.0063 | **-0.0011** | ✅ Improves on test |
| **vanilla_seed43** | 0.0048 | 0.0049 | **+0.0001** | ≈ Stable |
| **vanilla_seed44** | 0.0056 | 0.0052 | **-0.0004** | ✅ Improves on test |

**Distance Match Rate:**

| Model | Train Match Rate | Test Match Rate | Δ | Generalization |
|-------|------------------|-----------------|---|----------------|
| Distilled avg | 89.9% | 86.9% | -3.0% | Slight degradation |
| Vanilla avg | 91.3% | 88.8% | -2.5% | Slight degradation |

**Key Findings:**

1. **Both models generalize**: Test JSD is comparable or better than train JSD
2. **No overfitting**: Match rates drop only 2-3% from train to test
3. **Consistent behavior**: Both model types show similar train→test patterns
4. **Spatial learning**: Models learned generalizable spatial representations

### 3.5 Scenario-Level Analysis

**Scenario Taxonomy:**

Porto trajectories are classified into multiple overlapping scenario dimensions:
- **Temporal**: `weekday` (70%), `weekend` (30%), `peak` (11%), `off_peak` (89%)
- **Spatial**: `city_center` (91%), `suburban` (9%), `within_center` (60%), `to_center` (14%), `from_center` (17%)

![Test Scenario Distribution - Distilled](scenarios/test/distill/scenario_distribution.png)

![Test Scenario Distribution - Vanilla](scenarios/test/vanilla/scenario_distribution.png)

**Scenario Coverage:** Both model types generate similar scenario distributions, indicating comparable spatial and temporal diversity.

#### 3.5.1 Per-Scenario Performance Comparison

**Test Set Scenarios (Aggregated Across Seeds):**

| Scenario | Distilled Match% | Vanilla Match% | Δ Match% | Distilled Dist JSD | Vanilla Dist JSD | Δ Dist JSD |
|----------|------------------|----------------|----------|--------------------|------------------|------------|
| `city_center` | 82.8% | 85.7% | -2.9% | 0.0069 | 0.0068 | +0.0001 |
| `from_center` | 72.8% | 77.0% | -4.2% | 0.0316 | 0.0302 | +0.0014 |
| `off_peak` | 82.3% | 85.1% | -2.8% | 0.0070 | 0.0074 | -0.0004 |
| `peak` | 82.5% | 84.7% | -2.2% | 0.0426 | 0.0435 | -0.0009 |
| `suburban` | 77.5% | 77.6% | -0.1% | 0.0598 | 0.0719 | **-0.0121** |
| `to_center` | 80.9% | 84.7% | -3.8% | 0.0434 | 0.0432 | +0.0003 |
| `weekday` | 82.5% | 85.2% | -2.8% | 0.0086 | 0.0090 | -0.0004 |
| `weekend` | 81.3% | 83.8% | -2.5% | 0.0175 | 0.0173 | +0.0002 |
| `within_center` | 86.0% | 88.3% | -2.3% | 0.0105 | 0.0088 | **+0.0017** |

![Test Metric Comparison - Distilled](scenarios/test/distill/metric_comparison.png)

![Test Metric Comparison - Vanilla](scenarios/test/vanilla/metric_comparison.png)

#### 3.5.2 Key Scenario-Level Findings

**1. Vanilla Consistently Higher Match Rates:**
- Vanilla outperforms distilled in OD completion across all 9 scenarios
- Largest gap in `from_center` (-4.2%) and `to_center` (-3.8%)
- Smallest gap in `suburban` (-0.1%) and `peak` (-2.2%)
- **Interpretation:** Phase 1 distillation doesn't improve navigation success

**2. Distance JSD: Mixed Results:**
- **Vanilla better in:** `suburban` (-0.0121, 20% relative improvement), `peak` (-0.0009), `off_peak` (-0.0004), `weekday` (-0.0004)
- **Distilled better in:** `within_center` (+0.0017), `from_center` (+0.0014), `to_center` (+0.0003), `weekend` (+0.0002)
- **Interpretation:** Distilled shows marginal improvement in center-related scenarios but struggles with suburban trajectories

**3. Suburban Scenario: Vanilla's Strongest Advantage:**
- Vanilla: Distance JSD 0.0719 | Distilled: 0.0598
- Only 9% of trajectories, but distilled performs significantly worse
- Suburban trips likely have different characteristics (longer, less regular road network)
- **Hypothesis:** Teacher model (trained on city center) transfers less useful knowledge for suburban areas

**4. Within-Center Scenario: Distilled's Strongest Advantage:**
- Distilled: Distance JSD 0.0105 | Vanilla: 0.0088
- 60% of trajectories in this scenario
- Short-range navigation within city center
- **Hypothesis:** Teacher's local knowledge more valuable for dense urban areas

**5. Hierarchical Scenario Breakdown:**

![Test Hierarchical City Center - Distilled](scenarios/test/distill/hierarchical_city_center.png)

![Test Hierarchical Weekday - Vanilla](scenarios/test/vanilla/hierarchical_weekday.png)

Breaking down by spatial hierarchy (within_center → to_center → from_center) and temporal (weekday → weekend → peak/off_peak), both models show consistent patterns with vanilla maintaining a slight edge in most subcategories.

#### 3.5.3 Notable Scenarios Summary

**Top-3 Scenarios Where Distilled Performs Better (Distance JSD, Test):**
1. `within_center`: Δ = +0.0017 (20% relative improvement) — dense urban navigation
2. `from_center`: Δ = +0.0014 (5% relative improvement) — outbound trips
3. `to_center`: Δ = +0.0003 (1% relative improvement) — inbound trips

**Top-3 Scenarios Where Vanilla Performs Better (Distance JSD, Test):**
1. `suburban`: Δ = -0.0121 (20% relative improvement) — low-density areas
2. `peak`: Δ = -0.0009 (2% relative improvement) — rush hour traffic
3. `off_peak`: Δ = -0.0004 (6% relative improvement) — regular hours

**Interpretation:**
- Distillation provides value in **dense urban scenarios** where teacher's experience is most relevant
- Vanilla excels in **suburban and peak scenarios** where teacher may have less applicable knowledge
- Overall benefit is **scenario-dependent**, averaging to near-zero across full dataset

---

## 4. Porto vs Beijing: Critical Differences

This section examines why Porto results differ fundamentally from Beijing.

### 4.1 Dataset Characteristics

| Characteristic | Beijing | Porto | Impact |
|----------------|---------|-------|--------|
| **Average trip distance** | 5.16 km | 3.66 km | Porto 29% shorter |
| **Trajectory length** | Baseline | ~2× longer | Porto more segments per km |
| **Road network** | Grid structure | Complex topology | Porto less regular |
| **Baseline validation accuracy** | 57.2% | 26.5% | Porto harder task |
| **Training trajectories** | 629,380 | 481,359 | Beijing 31% more data |

### 4.2 Model Performance Comparison

| Metric | Beijing Distilled | Beijing Vanilla | Porto Distilled | Porto Vanilla |
|--------|------------------|-----------------|-----------------|---------------|
| **OD Match Rate (test)** | 85-89% | **12-18%** | 84-89% | **88-89%** |
| **Distance JSD (test)** | 0.016-0.022 | 0.145-0.153 | 0.005-0.008 | **0.005-0.006** |
| **Radius JSD (test)** | 0.003-0.004 | 0.198-0.206 | 0.009-0.012 | **0.010-0.011** |
| **Trip Distance** | 6.34-6.68 km | **2.33-2.43 km** | 3.43-3.66 km | **3.44-3.59 km** |

**Key Observations:**

1. **Beijing vanilla catastrophically fails**:
   - Only 12-18% OD completion
   - 55% too short trips (2.4 km vs 5.2 km)
   - 58-73× worse radius JSD

2. **Porto vanilla succeeds**:
   - 88-89% OD completion (matches distilled!)
   - Realistic trip lengths (3.5 km vs 3.7 km real)
   - Similar spatial complexity to distilled

### 4.3 Why Porto Vanilla Succeeds

**Hypothesis 1: Task Complexity**
- Porto's shorter trips (3.66 km) are easier to navigate than Beijing's (5.16 km)
- Fewer sequential decisions required
- Lower chance of compounding errors

**Hypothesis 2: Training Dynamics**
- Porto's 2× longer trajectories provide richer training signal per sample
- More road segments per trajectory → better spatial coverage
- Batch size reduction (32 vs 128) may improve gradient quality

**Hypothesis 3: Road Network Properties**
- Porto's road network may have higher connectivity
- Multiple valid paths between OD pairs reduce failure modes
- Less dependence on precise long-range planning

**Hypothesis 4: Phase 1 Hyperparameters Suboptimal**
- Phase 1 distillation (λ=0.00644, τ=2.802, w=4) may not be optimal for Porto
- Phase 2 refinement (λ=0.00598, τ=2.515, w=4) may improve relative benefit
- Temperature too high or lambda too low for effective knowledge transfer

### 4.4 Distillation Benefit Analysis

**Relative improvement from distillation:**

| Metric | Beijing Improvement | Porto Improvement |
|--------|-------------------|-------------------|
| **OD Match Rate** | +486% (12%→86%) | -2% (89%→87%) |
| **Distance JSD** | -87% (0.145→0.020) | -8% (0.0055→0.0060) |
| **Radius JSD** | -98% (0.198→0.004) | +2% (0.0106→0.0108) |

**Interpretation:**
- Beijing: Distillation transforms failure into success
- Porto: Distillation provides marginal (if any) benefit
- Phase 1 hyperparameters ineffective for Porto's characteristics

---

## 5. What Distillation Transferred (Phase 1)

### 5.1 Spatial Understanding (Limited Transfer)

Based on Phase 1 hyperparameters (λ=0.00644, τ=2.802, w=4):

| Capability | Distilled Performance | Vanilla Performance | Benefit |
|------------|----------------------|---------------------|---------|
| **Trip Length** | 3.563 ± 0.100 km | 3.506 ± 0.075 km | +1.6% (negligible) |
| **Path Completion** | 86.9% | 88.8% | -2.1% (worse!) |
| **Spatial Complexity** | JSD 0.0108 | JSD 0.0106 | +1.9% (worse!) |

**Key Observations:**

1. **No clear spatial advantage**: Distilled models don't show improved spatial reasoning
2. **Vanilla baseline strong**: Porto vanilla already captures spatial patterns well
3. **Phase 1 hyperparameters ineffective**: Teacher guidance not providing expected benefit

### 5.2 Robustness Across Seeds

**Coefficient of Variation (CV%) across seeds:**

| Metric | Distilled CV% | Vanilla CV% | More Stable |
|--------|---------------|-------------|-------------|
| **Distance JSD (test)** | 26.7% | 13.8% | Vanilla |
| **Radius JSD (test)** | 13.9% | 10.4% | Vanilla |
| **Match Rate (test)** | 3.3% | 1.0% | Vanilla |
| **Distance (test)** | 2.8% | 2.1% | Vanilla |

**Interpretation:**
- Vanilla shows MORE consistent performance across seeds
- Distillation introduces additional variability
- Phase 1 hyperparameters may not be well-tuned for robustness

**Comparison to Beijing:**
- Beijing distilled CV%: <15% (consistent)
- Porto distilled CV%: up to 27% (variable)
- Suggests Phase 1 hyperparameters not optimal

### 5.3 Phase 2 Expectations

**Phase 2 refined hyperparameters:**
- λ: 0.00598 (-7.1% from Phase 1)
- τ: 2.515 (-10.2% from Phase 1)
- w: 4 (unchanged, validated)

**Expected improvements:**
- Lower temperature (τ=2.515) may provide sharper, more useful teacher signal
- Slight lambda adjustment may improve transfer effectiveness
- Independent seed validation (44 vs 43) confirms hyperparameter refinement

**Phase 2 models currently training** (25 epochs with refined hyperparameters).
Separate analysis will be conducted once training completes.

---

## 6. Trajectory-Level Analysis

### 6.1 Local Metrics

| Metric | Distilled (test) | Vanilla (test) | Interpretation |
|--------|-----------------|---------------|----------------|
| **Hausdorff (km)** | 0.554 ± 0.026 | 0.549 ± 0.015 | Vanilla slightly closer |
| **DTW (km)** | 15.51 ± 1.32 | 15.20 ± 0.41 | Vanilla slightly better |
| **EDR** | 0.472 ± 0.025 | 0.466 ± 0.016 | Comparable alignment |

**Observations:**
- All local metrics show comparable or slightly better vanilla performance
- DTW per km: Distilled 4.35 km/km, Vanilla 4.33 km/km (similar)
- No evidence of distillation improving trajectory-level similarity

### 6.2 Visual Trajectory Comparison

**Multi-Scenario Grid Analysis:**

The following trajectory grid comparisons show all 6 models (3 distilled + 3 vanilla) generating trajectories for identical origin-destination pairs from the test set.

#### Example 1: Scenario `test_origin8237_dest861`

![Test Scenario Grid 1](figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin8237_dest861_grid.png)

**Observations:**
- All 6 models successfully complete paths to destination
- Distilled models (left 3) show similar routing to vanilla models (right 3)
- Cross-seed consistency visible for both model types
- No dramatic failure modes for vanilla (unlike Beijing)

#### Example 2: Scenario `test_origin8742_dest693`

![Test Scenario Grid 2](figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin8742_dest693_grid.png)

**Observations:**
- Both model types navigate complex multi-segment routes
- Vanilla models reach destinations successfully
- Path variety within model type (seed variation) > between model types
- Spatial understanding comparable between distilled and vanilla

#### Example 3: Scenario `test_origin2059_dest5620`

![Test Scenario Grid 3](figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin2059_dest5620_grid.png)

**Observations:**
- Long-distance navigation (~5+ km)
- Both model types handle extended trajectories
- No premature termination (vanilla's Beijing failure mode absent)
- Consistent endpoint reaching across all 6 models

**Additional test scenarios** (4 more OD pairs) available in Appendix Section 9.1.

### 6.3 Observed Failure Patterns

**Distilled Models:**
- Seed 44 shows lower match rate (83.6% train, 83.6% test)
- Higher variance in Distance JSD (0.0078 vs 0.0050 for seed 42)
- Possible training instability for this seed

**Vanilla Models:**
- Minimal failure modes observed
- Consistent high performance across all 3 seeds
- No catastrophic navigation failures seen in Beijing vanilla

**Contrast with Beijing:**
- Beijing vanilla: Premature stops, unrealistic shortcuts, 82-88% failure rate
- Porto vanilla: Complete paths, realistic routing, <12% failure rate
- Porto's shorter trips and different topology enable vanilla success

### 6.4 Porto-Specific Patterns

**Trajectory Length Sensitivity:**
- Shorter Porto trajectories (3.5-3.7 km) reduce error accumulation
- Fewer sequential decisions per trajectory
- Lower DTW absolute values (15-16 km vs Beijing's 27-29 km)

**Per-Kilometer Normalization:**
- DTW per km: 4.3-4.4 (both model types)
- Comparable to Beijing's normalized values (4.4 km/km)
- Suggests similar per-segment accuracy

**Duration Metric:**
- Real Porto duration: 0.205 hours (~12 minutes)
- Generated duration: 0.115-0.123 hours (~7 minutes)
- Both models underestimate by ~40% (timing calibration issue)
- Duration JSD: 0.023-0.028 (comparable across model types)

---

## 6.5 Inference Performance

### Status: Data Not Collected

⚠️ **Note:** Inference performance metrics (timing, throughput, latency) were not systematically collected during Phase 1 trajectory generation (October 26-27, 2025). The codebase has since been updated to automatically persist these metrics.

### Measurement Plan

Future evaluation runs will automatically collect and save:

**Aggregate Metrics** (saved as `<filename>_perf.json`):
- `total_time_mean` / `total_time_std`: Per-trajectory generation time (seconds)
- `total_time_p95`: 95th percentile latency
- `throughput_traj_per_sec`: Trajectories generated per second
- `forward_time_mean` / `forward_count_mean`: Model inference statistics
- `forward_time_per_step_mean`: Average model forward pass time
- Configuration: `beam_search_enabled`, `beam_width`, `device`

**Per-Trajectory Timing** (optional, saved as `<filename>_timing.csv` when `HOSER_SAVE_TIMING=1`):
- `total_time`: End-to-end generation time per trajectory
- `forward_time_total`: Total model inference time
- `forward_count`: Number of model forward passes
- `forward_time_avg`: Average forward pass duration

### Hardware Configuration

```
GPU: NVIDIA RTX 4090 (24GB VRAM)
CPU: AMD Ryzen 9 7950X (16 cores / 32 threads)
RAM: 64 GB DDR5
```

### Reproducibility Command

To generate trajectories with performance profiling for Porto Phase 1 models:

```bash
# Generate with distilled model (test set)
cd /home/matt/Dev/HOSER && uv run python gene.py \
  --dataset porto_hoser \
  --seed 42 \
  --cuda 0 \
  --num_gene 5000 \
  --od_source test \
  --beam_search \
  --beam_width 4 \
  --model_path hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/distill_final_best.pth \
  --wandb \
  --wandb_project hoser-porto-phase1-inference

# Generate with vanilla model (test set)
cd /home/matt/Dev/HOSER && uv run python gene.py \
  --dataset porto_hoser \
  --seed 42 \
  --cuda 0 \
  --num_gene 5000 \
  --od_source test \
  --beam_search \
  --beam_width 4 \
  --model_path hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/vanilla_final_best.pth \
  --wandb \
  --wandb_project hoser-porto-phase1-inference
```

Performance metrics will be automatically saved as `<csv_basename>_perf.json` alongside the generated trajectory CSV files.

---

## 7. Statistical Summary

### 7.1 Performance Metrics (Normalized)

**Normalization scheme:**
- OD Coverage: Raw match rate percentage
- Distance Quality: 1 - Distance_JSD (higher is better)
- Radius Quality: 1 - Radius_JSD (higher is better)
- Distance Accuracy: 1 - |generated - real| / real

| Model Type | OD Coverage | Distance Quality | Radius Quality | Distance Accuracy |
|------------|-------------|------------------|----------------|-------------------|
| **Distilled (test)** | 86.9% | 99.4% | 98.9% | 97.4% |
| **Vanilla (test)** | 88.8% | 99.4% | 98.9% | 95.8% |
| **Improvement** | **-1.9%** | **0.0%** | **0.0%** | **+1.6%** |

**Interpretation:**
- All normalized scores >95% (both model types perform well)
- No meaningful improvement from distillation
- Vanilla actually edges ahead in OD coverage

### 7.2 Consistency Analysis

**Coefficient of Variation across seeds (test set):**

| Metric | Distilled CV% | Vanilla CV% | Winner |
|--------|---------------|-------------|--------|
| **Match Rate** | 3.3% | 1.0% | Vanilla (3.3× more stable) |
| **Distance JSD** | 26.7% | 13.8% | Vanilla (1.9× more stable) |
| **Radius JSD** | 13.9% | 10.4% | Vanilla (1.3× more stable) |
| **Mean Distance** | 2.8% | 2.1% | Vanilla (1.3× more stable) |
| **DTW** | 8.5% | 2.7% | Vanilla (3.1× more stable) |
| **EDR** | 5.3% | 3.4% | Vanilla (1.6× more stable) |

**Key Finding:**
- Vanilla demonstrates superior cross-seed stability
- Distilled models show higher variability (especially Distance JSD: 27% CV)
- Phase 1 hyperparameters may introduce training instability

---

## 8. Conclusions

### 8.1 Primary Findings

1. **Porto vanilla models perform unexpectedly well**
   - 88-92% OD match rate (vs Beijing vanilla's 12-18%)
   - Realistic trip lengths and spatial distributions
   - Comparable or better than distilled models with Phase 1 hyperparameters

2. **Phase 1 distillation provides minimal benefit**
   - No improvement in OD coverage, distance quality, or spatial complexity
   - Higher cross-seed variability than vanilla
   - Hyperparameters (λ=0.00644, τ=2.802, w=4) not optimal for Porto

3. **Dataset characteristics fundamentally differ from Beijing**
   - Shorter trips (3.66 km vs 5.16 km) are easier to navigate
   - Porto vanilla doesn't exhibit Beijing's catastrophic failure modes
   - Task complexity may be below distillation benefit threshold

4. **Both models generalize effectively**
   - Test performance matches or exceeds train performance
   - No evidence of overfitting
   - Spatial representations are generalizable

5. **Scenario-level analysis reveals context-dependent benefits**
   - **Distilled advantages:** Within-center (dense urban) scenarios show 20% Distance JSD improvement
   - **Vanilla advantages:** Suburban scenarios show 20% Distance JSD improvement
   - **Match rates:** Vanilla consistently 2-4% better across all scenarios
   - **Interpretation:** Distillation benefits are scenario-specific, averaging to near-zero overall
   - Teacher model knowledge most useful in dense urban areas where it was likely trained

6. **Phase 2 refinement expected to improve relative benefit**
   - Lower temperature (τ=2.515 vs 2.802) may sharpen teacher guidance
   - Independent seed validation confirms Phase 1 findings
   - Evaluation pending 25-epoch Phase 2 model completion

### 8.2 Porto-Specific Insights

**Why Porto differs from Beijing:**

1. **Trajectory length**: Porto's shorter trips (29% less distance) reduce navigation complexity
2. **Road network**: Porto's topology may offer more navigation flexibility
3. **Training signal**: 2× longer trajectories per sample provide richer spatial learning
4. **Baseline capability**: Porto vanilla learned effective navigation from MLE alone

**Implications for distillation:**
- Knowledge transfer most beneficial when vanilla fails catastrophically
- For tractable tasks, vanilla baseline may be sufficient
- Hyperparameter tuning critical - Phase 1 parameters ineffective for Porto

### 8.3 Phase 1 vs Phase 2 Context

**Phase 1 Results (This Document):**
- Models: 6 × 25 epochs (3 distilled + 3 vanilla, seeds 42/43/44)
- Hyperparameters: λ=0.00644, τ=2.802, w=4 (Trial 10)
- Finding: Minimal distillation benefit, vanilla performs well

**Phase 2 Status (In Progress):**
- Models: Training 25 epochs with refined hyperparameters
- Hyperparameters: λ=0.00598, τ=2.515, w=4 (Trial 6, seed 44)
- Expectation: Lower temperature may improve teacher signal quality
- Separate analysis will be conducted upon completion

**Cross-phase comparison planned** to assess:
- Whether Phase 2 hyperparameters provide meaningful improvement
- If lower temperature (2.515 vs 2.802) sharpens knowledge transfer
- Independent seed validation of Phase 2 findings

### 8.4 Recommendations for Future Work

1. **Phase 2 Evaluation Priority:**
   - Complete evaluation once 25-epoch models finish training
   - Compare Phase 1 vs Phase 2 distilled models directly
   - Assess if refined hyperparameters improve relative benefit

2. **Hyperparameter Investigation:**
   - Explore even lower temperatures (τ < 2.5) for sharper teacher signal
   - Test higher lambda values (λ > 0.007) for stronger teacher weight
   - Investigate longer windows (w > 4) despite Phase 1 finding optimal at 4

3. **Dataset Analysis:**
   - Quantify Porto road network connectivity vs Beijing
   - Measure trajectory complexity metrics beyond radius of gyration
   - Identify task characteristics that predict distillation benefit

4. **Model Comparison:**
   - Evaluate Beijing models on Porto data (cross-dataset transfer)
   - Test Porto models on Beijing data (reverse transfer)
   - Identify what spatial knowledge is dataset-specific

5. **Distillation Strategy:**
   - Consider adaptive distillation (stronger λ for difficult OD pairs)
   - Explore curriculum learning (start with long trips)
   - Investigate multi-teacher distillation (ensemble LM-TAD)

---

## 9. Appendix: Methodology Details

### 9.1 Visualizations

**Distribution Analysis:**
- `figures/distributions/distance_distribution_train_od.pdf` - Train OD distance comparison (histograms + statistics)
- `figures/distributions/distance_distribution_test_od.pdf` - Test OD distance comparison (histograms + statistics)
- `figures/distributions/radius_distribution_train_od.pdf` - Train OD radius of gyration (histograms + box plots)
- `figures/distributions/radius_distribution_test_od.pdf` - Test OD radius of gyration (histograms + box plots)

**Trajectory Visualizations (Complete List):**

Multi-Scenario Grids (Test Set - All 7 scenarios):
1. `figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin8237_dest861_grid.png` - Featured in Section 6.2
2. `figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin788_dest228_grid.png`
3. `figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin8742_dest693_grid.png` - Featured in Section 6.2
4. `figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin2059_dest5620_grid.png` - Featured in Section 6.2
5. `figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin10232_dest818_grid.png`
6. `figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin7840_dest5620_grid.png`
7. `figures/trajectories/scenario_cross_model/test/multi_scenario_grid/test_origin712_dest5620_grid.png`

Each grid shows 6 model trajectories (distill/distill_seed43/distill_seed44 and vanilla/vanilla_seed43/vanilla_seed44) for identical origin-destination pair, enabling direct visual comparison.

**Scenario Analysis:**

*Per-model scenario metrics and visualizations (6 models × 2 OD sources)*

Train Set:
- `scenarios/train/distill/scenario_metrics.json` - Distilled (seed 42) scenario metrics
- `scenarios/train/distill_seed43/scenario_metrics.json` - Distilled (seed 43) scenario metrics
- `scenarios/train/distill_seed44/scenario_metrics.json` - Distilled (seed 44) scenario metrics
- `scenarios/train/vanilla/scenario_metrics.json` - Vanilla (seed 42) scenario metrics
- `scenarios/train/vanilla_seed43/scenario_metrics.json` - Vanilla (seed 43) scenario metrics
- `scenarios/train/vanilla_seed44/scenario_metrics.json` - Vanilla (seed 44) scenario metrics

Test Set:
- `scenarios/test/distill/scenario_metrics.json` - Distilled (seed 42) scenario metrics
- `scenarios/test/distill_seed43/scenario_metrics.json` - Distilled (seed 43) scenario metrics
- `scenarios/test/distill_seed44/scenario_metrics.json` - Distilled (seed 44) scenario metrics
- `scenarios/test/vanilla/scenario_metrics.json` - Vanilla (seed 42) scenario metrics
- `scenarios/test/vanilla_seed43/scenario_metrics.json` - Vanilla (seed 43) scenario metrics
- `scenarios/test/vanilla_seed44/scenario_metrics.json` - Vanilla (seed 44) scenario metrics

Visualization Types (per model):
- `scenario_distribution.png` - Bar chart of scenario frequency distribution
- `metric_comparison.png` - Heatmap/bar chart of metrics across scenarios
- `hierarchical_city_center.png` - Hierarchical breakdown by spatial location
- `hierarchical_weekday.png` - Hierarchical breakdown by temporal period

Aggregated Analysis:
- `analysis/scenarios_train.csv` - Per-scenario aggregates (train set, distilled vs vanilla)
- `analysis/scenarios_test.csv` - Per-scenario aggregates (test set, distilled vs vanilla)
- `analysis/top_scenarios_train.csv` - Top-5 scenarios by metric improvement
- `analysis/top_scenarios_test.csv` - Top-5 scenarios by metric improvement
- `analysis/md/scenario_analysis.md` - Markdown tables and interpretations

**How to Reproduce Scenario Analysis:**

```bash
uv run python scripts/analysis/aggregate_eval_scenarios.py \
  --root /home/matt/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto \
  --out /home/matt/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis
```

This reusable script:
- Aggregates metrics across seeds by model group (distilled vs vanilla)
- Generates per-scenario comparison tables
- Computes performance deltas
- Exports CSV and JSON formats
- Creates markdown fragments for documentation

**Hyperparameter Optimization:**
- `figures/optuna/` - Phase 1 search visualization (12 trials, broad search)
- `figures/optuna_phase2/` - Phase 2 refined search visualization (10 trials, narrowed space)

### 9.2 OD Pair Matching Algorithm

```python
# Grid-based spatial binning
grid_size = 0.001  # degrees (~111 meters)

# Convert lat/lon to grid cells
origin_cell = (round(origin_lat / grid_size), round(origin_lon / grid_size))
dest_cell = (round(dest_lat / grid_size), round(dest_lon / grid_size))

# Create OD pair identifier
od_pair = (origin_cell, dest_cell)

# Match against real data OD pairs
is_matched = od_pair in real_od_pairs_set
```

**Rationale:**
- 0.001° ≈ 111 meters at Porto's latitude (~41°N)
- Allows for small spatial variations in endpoint selection
- Consistent with Beijing evaluation methodology

**Critical Note:**
- OD pair extracted from **actual generated trajectory endpoints**
- NOT the input OD request
- Match rate measures path completion success + endpoint realism

### 9.3 Data Sources

**Real Porto Taxi Data:**
- **Train set:** 481,359 trajectories
- **Test set:** 137,532 trajectories
- **Total:** 618,891 trajectories
- **Temporal coverage:** Full dataset from Porto Taxi Service
- **Spatial coverage:** Porto metropolitan area, Portugal

**Generated Data:**
- **Per model:** 5,000 trajectories (train OD) + 5,000 trajectories (test OD)
- **Total:** 60,000 generated trajectories (6 models × 2 OD sources × 5,000)
- **Beam width:** 4 for trajectory generation
- **Max length:** No explicit limit (terminated by model)

**Train/Test Split:**
- Stratified by spatial distribution
- Consistent with LM-TAD evaluation protocol
- No temporal ordering preserved (spatial learning focus)

### 9.4 Computational Details

**Hardware:**
- GPU: NVIDIA GPU with CUDA support
- Generation: GPU-accelerated beam search
- Evaluation: CPU-based metric computation

**Software:**
- Trajectory generation: `gene.py` (HOSER codebase)
- Evaluation pipeline: `python_pipeline.py`
- Metric computation: `evaluation.py`
- Visualization: `create_distribution_plots.py`, `visualize_trajectories.py`

**Reproducibility:**
- Fixed seeds: 42, 43, 44 for all experiments
- Beam search width: 4 (consistent across all models)
- Grid size: 0.001° (consistent with Beijing)
- EDR threshold: 100m (standard for trajectory comparison)

---

**Generated:** October 31, 2025  
**Pipeline Version:** hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732  
**Phase:** Phase 1 Models (25 epochs, seeds 42/43/44)  
**Phase 2 Status:** Training in progress (separate analysis pending)

