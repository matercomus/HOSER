# Road Network Mapping Methodology

## Overview

This document describes the methodology for mapping road network IDs between different datasets (e.g., Beijing → BJUT Beijing) to enable cross-dataset trajectory analysis.

## Problem Statement

When analyzing generated trajectories from one dataset on a different road network, direct road ID comparison fails:

- **Beijing road network**: 40,060 roads with IDs [0, 40059]
- **BJUT Beijing road network**: 87,481 roads with IDs [0, 87480]
- **Generated trajectories**: Use Beijing road IDs (models trained on Beijing)
- **Cross-dataset analysis**: Requires BJUT road IDs

**Without mapping**: 99% false positive abnormality rate due to ID mismatch.

## Mapping Methodology

### 1. Coordinate Extraction

**Input**: `.geo` files in CSV format with columns:
- `geo_id`: Road network ID
- `coordinates`: LineString JSON `[[lon1, lat1], [lon2, lat2], ...]`

**Process**:
```python
def extract_road_centerpoint(coordinates_str):
    coords = json.loads(coordinates_str)  # Parse LineString
    lons = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    center_lon = mean(lons)
    center_lat = mean(lats)
    return (center_lat, center_lon)
```

**Rationale**: 
- Centerpoint represents road location
- Simple average provides good approximation
- Works for roads of any length/complexity

### 2. Distance Calculation

**Method**: Haversine formula for great-circle distance

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    
    a = sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2)
    c = 2·atan2(√a, √(1-a))
    
    return R · c  # Distance in meters
```

**Accuracy**: ±0.5% for distances <1km (sufficient for road matching)

### 3. Nearest Neighbor Matching

**Algorithm**:
```
For each source road:
    1. Calculate haversine distance to all target roads
    2. Find target road with minimum distance
    3. If distance <= threshold (50m default):
        mapping[source_id] = target_id
    Else:
        Mark as unmapped
```

**Threshold Rationale**:
- 50m: Typical urban road spacing
- Allows for minor coordinate errors
- Prevents incorrect matches across parallel roads

### 4. Quality Validation

**Automatic Quality Assessment**:

| Metric | Good | Fair | Poor |
|--------|------|------|------|
| Mapping Rate | >85% | 70-85% | <70% |
| Avg Distance | <15m | 15-30m | >30m |
| Action | Proceed | Warn | Fail |

**Distribution Analysis**:
- 0-10m: High confidence matches
- 10-20m: Good matches
- 20-30m: Acceptable matches
- 30-40m: Low confidence
- 40-50m: Very low confidence

## Statistics Collected

### Mapping Quality (`*_stats.json`)

```json
{
  "metadata": {
    "mapping_method": "nearest_neighbor_haversine",
    "max_distance_threshold_m": 50.0,
    "timestamp": "2025-11-03T18:00:00"
  },
  "mapping_quality": {
    "total_mapped": 35420,
    "total_unmapped": 4640,
    "mapping_rate_pct": 88.4,
    "avg_distance_m": 12.5,
    "median_distance_m": 8.3,
    "p95_distance_m": 35.7,
    "max_distance_m": 48.2
  },
  "distance_distribution": {
    "0-10m": 18500,   # 52% - High confidence
    "10-20m": 12000,  # 34% - Good
    "20-30m": 3200,   # 9%  - Acceptable
    "30-40m": 1500,   # 4%  - Low confidence
    "40-50m": 220     # 1%  - Very low
  },
  "many_to_one_count": 5200,  # Multiple source roads → same target
  "unmapped_roads_sample": [...]  # First 50 for debugging
}
```

### Translation Quality (`*_translation_stats.json`)

```json
{
  "translation_results": {
    "points_translated": 185420,
    "points_unmapped": 3206,
    "translation_rate_pct": 98.3,
    "trajectories_fully_translated": 4850,  # 97% success
    "trajectories_with_gaps": 120,          # 2.4% partial
    "trajectories_failed": 30               # 0.6% failed
  },
  "unmapped_roads_encountered": {
    "unique_count": 145,
    "total_occurrences": 3206,
    "top_10_unmapped": [...]  # Most frequent unmapped roads
  }
}
```

## Known Limitations

### 1. One-to-Many and Many-to-One Mappings

**Many-to-one** (common):
- Multiple source roads map to same target road
- Occurs when source network has finer granularity
- Example: 2 Beijing roads merge into 1 BJUT road
- **Impact**: Some granularity lost in translation

**One-to-many** (rare with nearest neighbor):
- Not possible with nearest neighbor algorithm
- Each source maps to exactly one target

### 2. Unmapped Roads

**Causes**:
- Roads near dataset boundaries
- Roads unique to one dataset
- Coordinate errors in .geo files
- Distance exceeds threshold

**Handling**:
- Keep original road ID (logged as warning)
- Trajectory marked as "with gaps"
- If >30% points unmapped → trajectory fails

### 3. Coordinate Accuracy

- Centerpoint approximation may differ for complex geometries
- Long curved roads: centerpoint may be off actual route
- Short roads (<10m): High accuracy
- **Mitigation**: Use median point for long roads (future enhancement)

## Example Mappings

### Good Match (Distance: 5.2m)

```
Source Road 1234 (Beijing):
  Coordinates: [[116.3898, 39.8987], [116.3898, 39.8986]]
  Centerpoint: (39.89865, 116.3898)

Target Road 56789 (BJUT):
  Coordinates: [[116.3898, 39.8987], [116.3898, 39.8985]]
  Centerpoint: (39.8986, 116.3898)

Distance: 5.2m → EXCELLENT MATCH
```

### Fair Match (Distance: 28.4m)

```
Source Road 5678:
  Centerpoint: (39.9234, 116.4123)

Nearest Target Road 12345:
  Centerpoint: (39.9236, 116.4121)

Distance: 28.4m → ACCEPTABLE (parallel road or slight offset)
```

### No Match (Distance: 75.3m)

```
Source Road 9999:
  Centerpoint: (39.8500, 116.3000)
  Reason: Road near dataset boundary, not in BJUT network

Distance to nearest: 75.3m → UNMAPPED (exceeds 50m threshold)
```

## Validation Procedures

### Pre-Translation Validation

1. **Check mapping rate**: Should be >85% for good quality
2. **Check avg distance**: Should be <15m
3. **Review unmapped roads**: Check if they're boundary roads
4. **Check distance distribution**: Should have >50% in 0-10m bin

### Post-Translation Validation

1. **Check translation rate**: Should be >95% of points
2. **Check trajectory success**: Should have >90% fully translated
3. **Review top unmapped roads**: Identify systematic issues
4. **Spot-check examples**: Manually verify some translated trajectories

### Red Flags

- ⚠️  Many-to-one count > 20% of mappings (granularity mismatch)
- ⚠️  High p95 distance (>40m) suggests poor alignment
- ⚠️  Many unmapped in city center (data quality issue)
- ⚠️  Translation rate <90% (poor mapping quality)

## Usage

### Create Mapping

```bash
uv run python tools/map_road_networks.py \
  --source data/Beijing/roadmap.geo \
  --target data/BJUT_Beijing/roadmap.geo \
  --output road_mapping_beijing_to_bjut.json \
  --max-distance 50
```

**Outputs**:
- `road_mapping_beijing_to_bjut.json` - Mapping dict
- `road_mapping_beijing_to_bjut_stats.json` - Quality stats

### Translate Trajectories

```bash
uv run python tools/translate_trajectories.py \
  --input gene/Beijing/seed42/distilled_train.csv \
  --mapping road_mapping_beijing_to_bjut.json \
  --output gene_translated/BJUT_Beijing/distilled_train.csv
```

**Outputs**:
- Translated CSV file
- `{filename}_translation_stats.json`

### Integrated Pipeline

```bash
# Automatically creates mapping and translates
uv run python python_pipeline.py --eval-dir . \
  --only road_network_translate,abnormal
```

## Cross-Dataset Analysis Interpretation

### Expected Results After Mapping

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

### Translation Quality Impact on Results

**Excellent Translation (>98%)**:
- Can confidently compare abnormality rates, category distributions, and model rankings
- **Interpretation strength**: High

**Good Translation (95-98%)**:
- Can compare with caveats: overall trends reliable, absolute values have ±2% uncertainty
- **Interpretation strength**: Medium

**Fair Translation (85-95%)**:
- Limited comparisons: only gross trends reliable, specific percentages unreliable
- **Interpretation strength**: Low

**Poor Translation (<85%)**:
- Results not interpretable: too much noise from unmapped roads
- **Interpretation strength**: None

### Interpretation Framework

**Scenario A: Low Abnormality in Both (<5%)**
- **Interpretation**: Clean datasets and realistic models
- **Research value**: Validates model quality

**Scenario B: Low Real, Moderate Generated (5-15%)**
- **Interpretation**: Acceptable transfer learning gap
- **Research value**: Quantifies transfer learning capability

**Scenario C: Low Real, High Generated (>20%)**
- **Interpretation**: Poor generalization or detection miscalibration
- **Check**: Translation quality (>95%?), detection thresholds appropriate?
- **Research value**: Identifies improvement areas

**Scenario D: High in Both (>15%)**
- **Interpretation**: Detection threshold too sensitive or datasets contain challenging scenarios
- **Research value**: Identifies dataset characteristics

## Known Issues and Solutions

### Issue 1: No Abnormalities Found

**Symptoms**: 0% abnormal in real data even with sensitive thresholds

**Causes**: Detection thresholds too strict, extremely clean dataset, or algorithm not working

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

## Future Enhancements

1. **Bidirectional validation**: Map both ways and check consistency
2. **Segment matching**: Match road segments instead of centerpoints
3. **Topology validation**: Verify connectivity preserved
4. **Visual inspection tool**: Plot matches on map for manual review
5. **Adaptive thresholds**: Use different thresholds for urban vs rural roads

### 8. OD Pair Translation in Abnormal Workflow

**New Feature**: Automatic OD pair translation integrated into the abnormal OD workflow to enable cross-dataset evaluation.

**Problem**: When running abnormal OD analysis with cross-dataset evaluation (e.g., OD pairs from BJUT_Beijing, models trained on Porto), the extracted OD pairs use source dataset road IDs but models expect target dataset road IDs.

**Solution**: 
- **Translation is always required** - not conditional detection
- **Fail-fast assertions** - no fallbacks, assert all required files exist
- **Quality filtering** - filter out OD pairs with poor mapping quality (distance > threshold)
- **Programmatic interfaces** - all functions use Python modules, never CLI/subprocess

#### Translation Workflow

**File**: `tools/run_abnormal_od_workflow.py`

**Integration Points**:
1. **Configuration** - `tools/config_loader.py` extends `EvaluationConfig` with translation fields:
   ```python
   source_dataset: str      # Where OD pairs come from
   target_dataset: str      # Where models are trained  
   translation_max_distance: float = 20.0  # Quality threshold
   translation_mapping_file: Path  # Road mapping JSON
   ```

2. **Translation Step** - Always executed after `extract_abnormal_od_pairs()`:
   ```python
   # Workflow: detect → extract → translate → generate → evaluate
   self.detect_abnormalities()
   self.extract_abnormal_od_pairs()
   self.translate_od_pairs()  # NEW: Always required
   self.generate_trajectories()
   self.evaluate_trajectories()
   ```

3. **Fail-Fast Behavior**:
   - Assert mapping file exists: `Translation mapping file not found: {path}. Translation is required.`
   - Assert source != target: `Source dataset and target dataset must be different for translation.`
   - Assert all OD pairs mapped: `Origin road {id} not found in mapping. All roads must be mapped.`
   - Assert quality threshold: `All OD pairs filtered out. No pairs passed quality threshold {threshold}m.`

#### Quality Filtering

**Default threshold**: 20.0 meters (configurable via `translation_max_distance`)

**Process**:
- For each OD pair (origin, destination):
  - Check origin mapping distance ≤ threshold
  - Check destination mapping distance ≤ threshold  
  - Include pair only if BOTH pass
- Statistics tracked:
  - Total pairs before/after filtering
  - Filter rate percentage
  - Breakdown by filter reason (origin, destination, both)

#### Extended JSON Format

**Translated OD pairs file** includes metadata:
```json
{
  "dataset": "BJUT_Beijing",           # Source dataset
  "translated_dataset": "porto_hoser",  # Target dataset
  "translation_applied": true,          # Flag for detection
  "translation_stats": {
    "total_pairs_before": 150,
    "total_pairs_after": 120,
    "filtered_pairs": 30,
    "filter_rate_pct": 20.0,
    "max_distance_threshold_m": 20.0
  },
  "od_pairs_by_category": {
    "speeding": [[1001, 2002], ...],  # Translated road IDs
    "detour": [[3003, 4004], ...]
  }
}
```

#### Configuration Example

**In `config/evaluation.yaml`**:
```yaml
# Required for cross-dataset evaluation
dataset: porto_hoser                    # Target (models trained here)
source_dataset: BJUT_Beijing           # Source (OD pairs from here)
translation_max_distance: 20.0         # Filter threshold (meters)

# Optional: explicit mapping file path
translation_mapping_file: road_mapping_bjut_beijing_to_porto_hoser.json

# Auto-detected if not specified:
# road_mapping_{source}_to_{target}.json
```

#### Error Messages

**All errors include clear action items**:
- `Mapping file not found: {path}. Translation is required.`
  - **Action**: Create mapping file or verify path
- `Source dataset and target dataset must be different for translation.`
  - **Action**: Check configuration, they should be different
- `All OD pairs filtered out. No pairs passed quality threshold 20.0m.`
  - **Action**: Increase `translation_max_distance` threshold
- `Origin road 12345 not found in mapping. All roads must be mapped.`
  - **Action**: Regenerate mapping to include this road

#### Quality Metrics

**Mapping Quality Validation** (from mapping creation):
- ✅ Good: Mapping rate >85%, Average distance <15m
- ⚠️ Fair: Mapping rate >70%, Average distance <30m  
- ❌ Poor: Mapping rate <70% or Average distance >30m

**Filtering Impact**:
- Conservative (≤10m): High confidence, fewer pairs
- Moderate (≤20m): Balanced quality/coverage (default)
- Liberal (≤50m): More pairs, lower confidence

#### API Usage

**Programmatic Interface**:
```python
from tools.run_abnormal_od_workflow import run_abnormal_od_workflow

analysis_dir = run_abnormal_od_workflow(
    eval_dir=Path("hoser-distill-optuna-eval"),
    dataset="BJUT_Beijing",           # Source dataset
    real_data_dir=Path("data/BJUT_Beijing"),
    num_trajectories=50,
    max_pairs_per_category=20,
    seed=42,
    skip_detection=True              # If detection already done
)
```

**Automatic Behavior**:
- Loads configuration with translation settings
- Creates road mapping if needed
- Translates and filters OD pairs automatically
- Uses translated pairs for generation and evaluation

**Note**: Translation is **always required** for cross-dataset scenarios. The workflow will fail fast if any required files or configurations are missing.

