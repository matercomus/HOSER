# OD-Pair Baseline Statistics - Methodology and Results

**Purpose**: Establish normal trajectory behavior baselines for statistical abnormality detection  
**Methodology**: Wang et al. 2018 - ISPRS Int. J. Geo-Inf. 7(1), 25  
**Datasets**: Beijing, BJUT_Beijing, Porto

---

## Methodology

### Baseline Computation Process

**Step 1**: Load all real trajectories (train + test combined)

**Step 2**: Compute metrics per trajectory:
- Route length (meters): Sum of road segment count × 100m estimate
- Travel time (seconds): End timestamp - start timestamp  
- Average speed (km/h): Distance / time

**Step 3**: Group by OD pair (origin, destination)

**Step 4**: Compute statistics per OD pair:
- Mean, standard deviation
- Min, median (p50), 95th percentile (p95)
- Sample count

**Step 5**: Compute global statistics (for OD pairs with insufficient samples)

**Step 6**: Save comprehensive baseline JSON file

---

## Beijing Baseline Results

**File**: `baselines/baselines_beijing.json`  
**Computed**: 2025-11-04  
**Source**: 809,203 trajectories (629k train + 180k test)

### Coverage Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total trajectories | 809,203 | Large dataset |
| Total OD pairs | 712,435 | High diversity (88% unique) |
| OD pairs with ≥5 samples | 4,268 (0.6%) | Very sparse coverage |

**Interpretation**: Beijing has extremely diverse OD patterns - most OD pairs occur only 1-2 times. Only 0.6% of OD pairs have sufficient samples (≥5) for robust statistical baselines.

**Implication**: Statistical detection will primarily use global statistics for most trajectories.

### Global Statistics

| Metric | Mean | Std Dev |
|--------|------|---------|
| Route length | 2,831m (2.8km) | - |
| Travel time | 772s (12.9 min) | - |
| Average speed | 15.4 km/h | - |

**Interpretation**: 
- Short trips on average (2.8km)
- Slow speeds (15.4 km/h) suggest urban traffic
- Compatible with Beijing taxi dataset characteristics

---

## BJUT_Beijing Baseline Results

**File**: `baselines/baselines_bjut_beijing.json`  
**Computed**: 2025-11-04  
**Source**: 33,876 trajectories (28k train + 6k test)

### Coverage Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total trajectories | 33,876 | Medium dataset |
| Total OD pairs | 30,523 | Extremely high diversity (90% unique) |
| OD pairs with ≥5 samples | 139 (0.5%) | Extremely sparse coverage |

**Interpretation**: BJUT has even higher OD diversity than Beijing - 90% of trajectories have unique OD pairs. Statistical baselines will be challenging.

### Global Statistics

| Metric | Mean | Std Dev |
|--------|------|---------|
| Route length | 2,571m (2.6km) | - |
| Travel time | 143s (2.4 min) | - |
| Average speed | 85.9 km/h | - |

**Interpretation**:
- Similar route length to Beijing (2.6km vs 2.8km)
- **Much faster** travel time (2.4 min vs 12.9 min)
- **Very high speed** (86 km/h vs 15 km/h)

**Hypothesis**: BJUT dataset may include highway/expressway trajectories, not just urban roads.

---

## Porto Baseline Results

**File**: `baselines/baselines_porto_hoser.json`  
**Computed**: 2025-11-04  
**Source**: 618,891 trajectories (509k train + 138k test)

### Coverage Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total trajectories | 618,891 | Large dataset |
| Total OD pairs | 346,455 | Moderate diversity (56% unique) |
| OD pairs with ≥5 samples | 17,735 (5.1%) | **Much better coverage than Beijing/BJUT** |

**Interpretation**: Porto has significantly better OD-pair coverage than Beijing - 5.1% of OD pairs have ≥5 samples (vs 0.6% for Beijing). This means statistical detection will be more OD-specific for Porto.

**Key Insight**: Porto taxi routes are more repetitive (more common routes), while Beijing routes are more diverse/unique.

### Global Statistics

| Metric | Mean | Std Dev |
|--------|------|---------|
| Route length | 4,004m (4.0km) | ±2,005m |
| Travel time | 479s (8.0 min) | ±243s |
| Average speed | 33.1 km/h | ±23.7 km/h |

**Interpretation**: 
- Longer trips than Beijing (4.0km vs 2.8km)
- Faster average speed (33 km/h vs 15 km/h)
- Shorter duration despite longer distance (8 min vs 13 min)
- **Suggests efficient road network with less congestion**

---

## Comparison: Beijing vs BJUT vs Porto

| Metric | Beijing | BJUT | Porto | Notes |
|--------|---------|------|-------|-------|
| **Trajectories** | 809k | 34k | 619k | Porto: 2nd largest |
| **OD pairs** | 712k | 31k | 346k | Beijing most diverse |
| **OD ≥5 samples** | 4,268 (0.6%) | 139 (0.5%) | 17,735 (5.1%) | **Porto 8.5x better** |
| **Mean length** | 2.8km | 2.6km | 4.0km | Porto trips longer |
| **Mean duration** | 12.9 min | 2.4 min | 8.0 min | Porto balanced |
| **Mean speed** | 15.4 km/h | 85.9 km/h | 33.1 km/h | Wide variation |

### Speed Ratio Analysis

| Comparison | Speed Ratio | Interpretation |
|------------|-------------|----------------|
| Porto vs Beijing | 2.15x faster | Less congestion, better roads |
| BJUT vs Beijing | 5.58x faster | Likely highway/expressway data |
| BJUT vs Porto | 2.60x faster | BJUT unusually fast |

**Key Findings**:

1. **Porto has best OD-pair coverage** (5.1% vs <1% for Beijing/BJUT)
   - More repeated routes → better statistical baselines
   - 17,735 high-quality OD pairs vs 4,268 (Beijing) and 139 (BJUT)

2. **Three distinct speed profiles**:
   - Beijing: 15 km/h (congested urban)
   - Porto: 33 km/h (efficient urban)
   - BJUT: 86 km/h (highway/expressway)

3. **Route length patterns**:
   - Beijing/BJUT: ~2.7km (short urban trips)
   - Porto: 4.0km (longer trips, possibly more airport/suburban routes)

4. **Efficiency comparison** (speed × time):
   - Porto: Longer distance, moderate time → efficient
   - Beijing: Short distance, long time → congested
   - BJUT: Short distance, very short time → high-speed roads

---

## Implications for Statistical Detection

### Challenge: OD-Pair Coverage Varies by Dataset

**Beijing/BJUT**: <1% of OD pairs have ≥5 samples
- Will rely heavily on global statistics
- Less OD-specific detection
- More false negatives expected

**Porto**: 5.1% of OD pairs have ≥5 samples  
- Better OD-specific detection capability
- More robust statistical baselines
- Higher detection precision expected

**Solutions**:
1. **Use global statistics as fallback** for OD pairs without baselines
2. **Dataset-adaptive thresholds** based on coverage
3. **Document coverage limitation** in results
4. **Compare detection rates** across datasets to quantify impact

### Wang et al. Threshold Applicability

**Fixed thresholds from paper** (Wuhan dataset):
- Lρ = 5,000m (route deviation)
- Tρ = 300s (temporal delay)

**Beijing context**:
- Mean route: 2,800m → Lρ adds +5,000m (179% increase)
- Mean time: 772s → Tρ adds +300s (39% increase)
- **Assessment**: Lρ too lenient, Tρ reasonable

**BJUT context**:
- Mean route: 2,600m → Lρ adds +5,000m (192% increase)
- Mean time: 143s → Tρ adds +300s (210% increase)
- **Assessment**: Both thresholds extremely lenient!

**Porto context**:
- Mean route: 4,000m → Lρ adds +5,000m (125% increase)
- Mean time: 479s → Tρ adds +300s (63% increase)
- **Assessment**: Lρ somewhat lenient, Tρ reasonable

**Conclusion**: 
- Fixed thresholds work best for Porto (closest to Wuhan characteristics)
- Statistical multipliers (2.5σ) more appropriate for Beijing/BJUT
- Hybrid strategy (minimum of fixed/statistical) recommended

---

## Recommendations

### For Implementation

1. **Use hybrid threshold strategy** (minimum of fixed/statistical)
   - Handles both high-speed (BJUT) and congested (Beijing) scenarios
   - Balances absolute and relative deviation detection

2. **Dataset-specific fallback rates**:
   - Beijing/BJUT: Expect 95%+ global fallback rate
   - Porto: Expect ~50% OD-specific, 50% global fallback

3. **Document coverage in results**:
   - Report % trajectories using OD-specific vs global baselines
   - Note potential OD-specificity bias in findings

4. **Quality tiers**:
   - High-quality: OD pairs with ≥10 samples (Porto: ~8k pairs)
   - Medium-quality: OD pairs with 5-9 samples (Porto: ~9k pairs)
   - Low-quality: Global fallback (all datasets: majority)

### For Research

1. **Compare detection rates across datasets**:
   - Hypothesis: Porto will show more nuanced abnormality patterns
   - Beijing/BJUT may show more extreme outliers only

2. **Investigate speed characteristics**:
   - BJUT: 86 km/h (verify if highway/expressway data)
   - Porto: 33 km/h (efficient urban vs Beijing 15 km/h congested)

3. **Baseline coverage impact study**:
   - Compare detection precision: Porto (5.1% coverage) vs Beijing (0.6%)
   - Quantify false negative rate from sparse coverage

4. **Cross-dataset validation**:
   - Use Porto baselines to detect abnormalities in Porto-generated trajectories
   - Compare with Beijing cross-dataset analysis (Beijing → BJUT)

---

## Files Generated

```
baselines/
├── baselines_beijing.json (337 MB)      # 809k trajectories, 712k OD pairs
├── baselines_bjut_beijing.json (14 MB)  # 34k trajectories, 31k OD pairs
└── baselines_porto_hoser.json (157 MB)  # 619k trajectories, 346k OD pairs
  
Computation logs:
├── baselines_beijing_20251104.log
├── baselines_bjut_beijing_20251104.log
└── baselines_porto_20251104.log
```

**Status**: ✅ All three datasets have baselines computed

**Next Steps**:
1. Run Wang statistical abnormality detection on all datasets
2. Compare abnormality rates: Real data vs Generated trajectories
3. Analyze pattern distributions (Abp1-4) across datasets
4. Quantify impact of OD-pair coverage on detection precision

