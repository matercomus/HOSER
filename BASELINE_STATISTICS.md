# OD-Pair Baseline Statistics - Methodology and Results

**Purpose**: Establish normal trajectory behavior baselines for statistical abnormality detection  
**Methodology**: Wang et al. 2018 - ISPRS Int. J. Geo-Inf. 7(1), 25  
**Datasets**: Beijing, BJUT_Beijing

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

## Comparison: Beijing vs BJUT

| Metric | Beijing | BJUT | Ratio |
|--------|---------|------|-------|
| Trajectories | 809k | 34k | 24:1 |
| OD pairs | 712k | 31k | 23:1 |
| OD pairs with ≥5 samples | 4,268 (0.6%) | 139 (0.5%) | 31:1 |
| Mean route length | 2.8km | 2.6km | 1.08:1 |
| Mean duration | 12.9 min | 2.4 min | 5.4:1 |
| Mean speed | 15.4 km/h | 85.9 km/h | 1:5.6 |

**Key Findings**:

1. **Similar route lengths** (2.6-2.8km)
2. **Dramatically different speeds** (15 vs 86 km/h)
   - Beijing: Urban taxi, traffic congestion
   - BJUT: Highway/expressway, free-flowing traffic
3. **Both datasets extremely sparse** (<1% OD coverage)
4. **Statistical detection will rely heavily on global baselines**

---

## Implications for Statistical Detection

### Challenge: Low OD-Pair Coverage

**Problem**: <1% of OD pairs have ≥5 samples

**Solutions**:
1. **Use global statistics as fallback** for OD pairs without baselines
2. **Lower minimum sample threshold** to 3 instead of 5
3. **Spatial clustering**: Group nearby OD pairs (e.g., same grid cell)
4. **Accept limitation**: Document that detection is less OD-specific

### Wang et al. Threshold Applicability

**Fixed thresholds from paper**:
- Lρ = 5,000m (route deviation)
- Tρ = 300s (temporal delay)

**Beijing context**:
- Mean route: 2,800m → Lρ adds 179% tolerance (very lenient)
- Mean time: 772s → Tρ adds 39% tolerance (reasonable)

**BJUT context**:
- Mean route: 2,600m → Lρ adds 192% tolerance (very lenient)
- Mean time: 143s → Tρ adds 210% tolerance (extremely lenient!)

**Conclusion**: Fixed thresholds from Wang et al. (designed for Wuhan taxis) may be too lenient for Beijing/BJUT. Statistical multipliers (2.5σ) likely more appropriate.

---

## Recommendations

### For Implementation

1. **Primarily use statistical thresholds** (σ-based)
2. **Use fixed thresholds as absolute bounds** (catch extreme cases)
3. **Implement spatial OD clustering** to increase coverage
4. **Document sparse coverage limitation**

### For Research

1. **Compare fixed vs statistical thresholds**
2. **Quantify impact of sparse OD coverage**
3. **Investigate BJUT speed characteristics** (86 km/h unusual for urban)
4. **Consider dataset-specific threshold calibration**

---

## Files Generated

```
baselines/
  baselines_beijing.json (11.4MB)
  baselines_bjut_beijing.json (~300KB)
  
Computation logs:
  baselines_beijing_computation.log
  baselines_bjut_computation.log
```

**Next**: Implement WangStatisticalDetector using these baselines (Phase 2)

