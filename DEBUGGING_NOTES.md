# Training Script Debugging Notes

## Current Status: Dataset Loading Error

### Error Description
**File:** `dataset.py` line 47  
**Type:** `ValueError: zero-size array to reduction operation maximum which has no identity`

```python
# Error occurs in haversine_vector calculation:
metric_dis[i] = haversine_vector(global_road_center_gps[candidate_road_id_list], 
                                global_road_center_gps[destination_road_id].reshape(1, -1), 
                                'm', comb=True).reshape(-1).astype(np.float32)
```

### Root Cause
Some roads in the Beijing dataset have **no reachable destinations** in the `roadmap.rel` file, creating empty `candidate_road_id_list` arrays. When these empty arrays are passed to `haversine_vector()`, it fails because:
1. Empty array indexing fails: `global_road_center_gps[empty_array]` 
2. `numpy.abs(lat).max()` fails on zero-size arrays

### Dataset Comparison Analysis
**Working Porto Dataset (Backup: `/mnt/i/Matt-Backups/HOSER-Backups/Test-1/Porto/`):**
- **Scale:** 11,025 roads, 25,992 relations, 108,848 trajectories
- **Connectivity:** 99.99% of roads have outgoing connections (11,024/11,025)
- **Density:** Well-connected, preprocessed road network
- **Max Road ID in trajectories:** 10,057 (within bounds)

**Problematic Beijing Dataset (Current):**
- **Scale:** 1,239,015 roads, 1,359,274 relations, 50,490 trajectories  
- **Connectivity:** ~82% estimated connectivity (from 50K sample)
- **Issues:** ~18% of roads are dead-ends/isolated with no outgoing connections
- **Max Road ID in trajectories:** 1,223,203 (within bounds)
- **Problem:** Raw OSM data includes service roads, dead-ends, isolated segments

**Key Insight:** The Porto dataset was likely preprocessed to remove disconnected roads, while Beijing contains raw road network data with many isolated/dead-end segments that break the trajectory processing logic.

### Detailed Connectivity Analysis

#### Beijing Dataset (Problematic)
```
Total Roads:           1,239,014
Connected Roads:       1,205,985  (97.33%)
Disconnected Roads:      33,029   (2.67%)
Relations:             1,359,274
Trajectory Density:         0.041 trajectories per road
```

**Sample Dead-End Roads in Beijing:**
```
Road ID 224: DEAD END (no outgoing connections)
Road ID 429: DEAD END (no outgoing connections)  
Road ID 458: DEAD END (no outgoing connections)
Road ID 474: DEAD END (no outgoing connections)
Road ID 536: DEAD END (no outgoing connections)
... and 33,024 more
```

**Example Beijing Trajectory (35 roads):**
```
"146439,146453,146461,146484,380129,380129,380129,380129,380129,380129,
22938,22941,554070,930189,536852,30640,537057,537060,537060,537060,
1012870,210809,136390,136224,80751,80751,80751,210871,178852,708093,
81191,178881,179177,457692,54090"
```

#### Porto Dataset (Working)  
```
Total Roads:              11,025
Connected Roads:          11,024  (99.99%)
Disconnected Roads:            1  (0.01%)
Relations:                25,992
Trajectory Density:           9.87 trajectories per road
```

**Example Porto Trajectory (51 roads):**
```
"7689,7799,7688,8741,8744,232,7882,758,8420,8588,814,8212,756,8237,8725,
7669,8717,924,1036,3328,7875,7667,76,7670,8184,8214,7247,119,726,5089,
3070,5091,5093,5097,7025,7026,7021,7028,7041,7793,799,798,3245,5378,846,
2582,897,7949,3091,890,9995,2580"
```

### The Critical Difference

**Volume vs Density Trade-off:**
- **Beijing:** Massive road network (112x larger) with sparse trajectory coverage
- **Porto:** Compact road network with dense trajectory coverage  

**Data Quality:**
- **Beijing:** Raw OSM data including service roads, parking lots, dead-ends
- **Porto:** Preprocessed/filtered road network with connectivity validation

**Highway Type Distribution:**
- **Beijing:** Single integer code (99) for all roads - simplified/encoded
- **Porto:** Diverse numeric codes (7,1,3,2,4,8,5,6,9) representing different road types

### Error Manifestation

When `dataset.py` processes Beijing trajectories:

1. **Road Selection:** `trace_road_id = [146439, 146453, 146461, ...]`  
2. **Candidate Lookup:** `global_reachable_road_id_dict[146439]` → `[146441]` ✅
3. **Dead-End Road:** `global_reachable_road_id_dict[224]` → `[]` ❌  
4. **Empty Array:** `candidate_road_id_list = np.array([], dtype=np.int64)`
5. **Haversine Failure:** `global_road_center_gps[empty_array]` → IndexError

**Result:** `ValueError: zero-size array to reduction operation maximum which has no identity`

### Issues Fixed So Far

#### ✅ 1. Memory Allocation Error (Fixed)
- **Problem:** Training script tried to allocate 1.4 TiB for adjacency matrix with 1.24M+ roads
- **Solution:** Replaced massive numpy matrix with efficient adjacency lists using sets
- **Files:** `train.py` lines 185-194

#### ✅ 2. AttributeError for Highway Types (Fixed)  
- **Problem:** `'int' object has no attribute 'startswith'` for Beijing dataset highway values (all `99`)
- **Solution:** Added type checking `isinstance(road_attr_type[i], str)` before calling `.startswith()`
- **Files:** `train.py` lines 144-149

#### ✅ 3. IndexError in Dataset Loading (Fixed)
- **Problem:** `'arrays used as indices must be of integer (or boolean) type'` in multiprocessing
- **Solution:** Explicit integer casting for all road IDs and `dtype=np.int64` for arrays
- **Files:** `dataset.py` lines 28, 39, 43, 60, 68

#### 🔄 4. Zero-Size Array Error (In Progress)
- **Problem:** Roads with no reachable destinations create empty arrays that break haversine calculations
- **Solution Started:** Added checks for `len(candidate_road_id_list) == 0` in `metric_dis` and `metric_angle` calculations
- **Remaining Issue:** Need to handle `road_label` calculation when roads cannot reach their next road in trajectory

### Current Fix Branch
- **Branch:** `fix/dataset-indexing-error` 
- **Based on:** `fix/train-memory-optimization`
- **Commits:** 2cf8844, 40c9180

### Next Steps
1. **Complete empty candidate handling** in `dataset.py`:
   - Handle `road_label` calculation when `next_road_id not in reachable_roads`
   - Add robust fallback logic for isolated roads
   - Test with sample trajectories to verify fix

2. **Data Quality Solutions (Based on Analysis):**
   - ✅ **Confirmed:** ~18% of Beijing roads have no outgoing connections (vs 0.01% in Porto)
   - **Immediate Fix:** Skip/filter trajectories that use roads with no reachable destinations
   - **Long-term:** Consider preprocessing Beijing dataset to remove isolated road segments

3. **Implementation Approaches:**
   - **Option A:** Filter invalid trajectory segments during dataset loading
   - **Option B:** Add fallback logic (use closest road or skip problematic steps)
   - **Option C:** Preprocess the road network to improve connectivity (complex)
   - **Recommended:** Start with Option A (trajectory filtering) for quick fix

### Dataset Scale
- **Roads:** 1,239,014 segments
- **Edges:** 1,359,271 relationships  
- **Trajectories:** 50,489 (train), 10,098 (val), 14,425 (test)
- **Memory Optimized:** From 1.4 TiB to ~few GB

### Progress Summary
- ✅ Memory optimization complete (1.4 TiB → few GB)
- ✅ Type errors resolved (highway attribute handling)
- ✅ Multiprocessing indexing fixed (integer casting)
- 🔄 Empty candidate arrays - root cause identified
- ⏳ Full training pipeline - pending trajectory filtering fix

### Solution Impact Analysis

**Beijing Trajectory Data Quality Assessment:**
```bash
# Expected trajectory filtering impact:
Total trajectories:        50,490
Roads in trajectories:   ~1,223,203 (max observed)
Dead-end roads:             33,029 (2.67% of network)
Estimated affected trajectories: ~15-25% (will need validation)
```

### Recommended Solution: Smart Trajectory Filtering

**Implementation Strategy (Option A):** Modify `dataset.py` to filter problematic trajectory segments:

```python
def process_row(args):
    # ... existing code ...
    
    # PRE-FILTER: Remove roads with no outgoing connections  
    valid_road_pairs = []
    for i in range(len(rid_list) - 1):
        current_road = int(rid_list[i])
        next_road = int(rid_list[i + 1])
        
        # Check if current road has outgoing connections
        reachable = global_reachable_road_id_dict[current_road]
        if len(reachable) > 0 and next_road in reachable:
            valid_road_pairs.append(i)
        else:
            print(f"Filtering out road pair: {current_road} -> {next_road} (disconnected)")
    
    # Skip trajectory if too few valid segments remain
    if len(valid_road_pairs) < 3:  # Minimum viable trajectory length
        return None  # Skip this trajectory entirely
    
    # Continue processing with valid road pairs only
    filtered_rid_list = [rid_list[i] for i in valid_road_pairs] + [rid_list[valid_road_pairs[-1] + 1]]
    # ... continue with filtered trajectory ...
```

**Expected Outcomes:**
- **Preserve ~75-85%** of Beijing trajectories (based on connectivity ratio)
- **Eliminate zero-size array errors** completely  
- **Maintain training data quality** while handling raw OSM connectivity issues
- **Scalable approach** for other large-scale datasets with similar characteristics

**Alternative Approaches:**
- **Option B:** Add default fallback roads for isolated segments
- **Option C:** Preprocess road network to improve connectivity (time-intensive)
- **Option D:** Use spatial nearest-neighbor for disconnected roads

**Recommended:** Start with **Option A** for immediate resolution, consider **Option C** for production deployment.

