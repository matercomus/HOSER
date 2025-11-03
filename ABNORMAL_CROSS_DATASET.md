# Abnormal Trajectory Cross-Dataset Analysis Notes

## Why Road Network Mapping is Necessary

### The 99% False Positive Discovery

**Observed behavior** (without mapping):
```
Analyzing Beijing-trained models on BJUT Beijing dataset:

Beijing Dataset (same network as training):
  ✅ Real: 0% abnormal
  ✅ Generated: 0% abnormal
  → Realistic distribution maintained

BJUT Dataset (different road network):
  ✅ Real: 0% abnormal  
  ❌ Generated: 99% abnormal ← FALSE POSITIVE!
```

**Root cause identified**:
- Generated trajectories contain Beijing road IDs
- BJUT detection expects BJUT road IDs
- Road ID mismatch interpreted as abnormal patterns
- Result: Nearly every trajectory flagged as abnormal

### Why This Matters for Research

This false positive would lead to incorrect conclusions:
- ❌ "Models fail completely on unseen datasets" (wrong!)
- ❌ "Models hallucinate abnormal patterns" (wrong!)
- ✅ "Need road network translation for cross-dataset analysis" (correct!)

## Expected Results After Mapping

### Realistic Baseline Expectations

**Within-network analysis** (no translation needed):
```
Beijing Real:      0-5% abnormal
Beijing Generated: 0-8% abnormal
Difference:        0-3 percentage points ✅
```

**Cross-network analysis** (with translation):
```
BJUT Real:                0-5% abnormal
Beijing→BJUT Generated:   0-12% abnormal
Difference:               0-7 percentage points ✅ (acceptable for transfer)
```

### Abnormality Category Distribution

**Expected patterns**:
- Speeding: 0-5% (most common)
- Detour: 0-3% (route optimization issues)
- Suspicious stops: 0-2% (rare in normal driving)
- Circuitous: 0-4% (routing artifacts)
- Unusual duration: 0-3% (traffic/stops)

**If categories differ significantly**:
- May indicate model artifacts
- Or dataset-specific driving patterns
- Requires deeper investigation

## Interpretation Framework

### Scenario A: Low Abnormality in Both (<5%)

**Interpretation**: Clean datasets and realistic models
- Models maintain normal driving patterns
- Good generalization to unseen network
- **Research value**: Validates model quality

### Scenario B: Low Real, Moderate Generated (5-15%)

**Interpretation**: Acceptable transfer learning gap
- Models slightly less realistic on unseen network
- Still within acceptable bounds
- **Research value**: Quantifies transfer learning capability

### Scenario C: Low Real, High Generated (>20%)

**Interpretation**: Poor generalization or detection miscalibration
- **Check**: Translation quality (>95%?)
- **Check**: Detection thresholds appropriate for both datasets?
- **Check**: Category breakdown (all high or specific?)
- **Research value**: Identifies improvement areas

### Scenario D: High in Both (>15%)

**Interpretation**: Detection threshold too sensitive
- Adjust thresholds in `config/abnormal_detection.yaml`
- Or datasets contain genuinely challenging scenarios
- **Research value**: Identifies dataset characteristics

## Translation Quality Impact on Results

### Excellent Translation (>98%)

**Can confidently compare**:
- Abnormality rates
- Category distributions
- Model rankings

**Interpretation strength**: High

### Good Translation (95-98%)

**Can compare with caveats**:
- Overall trends reliable
- Absolute values have ±2% uncertainty
- Category comparisons less precise

**Interpretation strength**: Medium

### Fair Translation (85-95%)

**Limited comparisons**:
- Only gross trends reliable
- Specific percentages unreliable
- Use for screening only

**Interpretation strength**: Low

### Poor Translation (<85%)

**Results not interpretable**:
- Too much noise from unmapped roads
- Cannot draw meaningful conclusions
- Need to improve mapping or skip cross-dataset analysis

**Interpretation strength**: None

## Research Questions Answered

### 1. Do models generalize across road networks?

**Method**: Compare abnormal rates on source vs target network

**Good**: Difference <10 percentage points
**Poor**: Difference >20 percentage points

### 2. What abnormal patterns do models reproduce?

**Method**: Category breakdown analysis

**Example findings**:
- Models good at avoiding speeding
- Models struggle with circuitous routes
- Models over-generate detours

### 3. How does translation quality affect results?

**Method**: Correlate translation rate with abnormal rate variance

**Expected**: Higher translation rate → lower variance

## Known Issues and Solutions

### Issue 1: No Abnormalities Found

**Symptoms**: 0% abnormal in real data even with sensitive thresholds

**Causes**:
- Detection thresholds still too strict
- Dataset is extremely clean
- Detection algorithm not working

**Solutions**:
1. Gradually loosen thresholds until finding 1-5%
2. Try different categories independently
3. Validate algorithm on known abnormal examples

### Issue 2: JSON Serialization Error

**Symptoms**: "Object of type bool is not JSON serializable"

**Cause**: Detection results contain non-serializable objects

**Solution**: Convert all numpy/pandas types to native Python before JSON dump

### Issue 3: Asymmetric Mapping

**Symptoms**: A→B mapping differs from B→A mapping

**Cause**: Nearest neighbor is not symmetric

**Solution**: Use forward mapping (source→target) consistently

## Recommended Workflow

1. **Run baseline detection** (within-network):
   ```bash
   --only abnormal  # Uses Beijing network for Beijing data
   ```

2. **Create and validate mapping**:
   ```bash
   --only road_network_translate  # Creates mapping + translates
   ```

3. **Re-run detection with translated files**:
   - Modify abnormal phase to use translated files from `gene_translated/`
   - Compare results against real BJUT baseline

4. **Extract OD pairs** from real BJUT abnormal trajectories:
   ```bash
   --only abnormal_od_extract
   ```

5. **Test models** on challenging OD pairs:
   ```bash
   --only abnormal_od_generate,abnormal_od_evaluate
   ```

## Documentation Standards

All experiments should record:
- Mapping quality metrics
- Translation success rates
- Detection thresholds used
- Abnormality rates by category
- Model-specific patterns
- Any anomalies or unexpected results

This enables reproducibility and facilitates future research.
