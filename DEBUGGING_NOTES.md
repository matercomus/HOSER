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
   - Consider filtering out invalid trajectories or adding fallback logic
   - Test with sample trajectories to verify fix

2. **Data Quality Investigation:**
   - Analyze how many roads have no reachable destinations
   - Check if this is expected in the Beijing dataset structure
   - Determine if trajectories should be filtered or graph connectivity improved

3. **Alternative Approaches:**
   - Pre-filter trajectories to remove invalid road sequences
   - Add default/fallback roads for isolated nodes
   - Investigate if this is a data preprocessing issue

### Dataset Scale
- **Roads:** 1,239,014 segments
- **Edges:** 1,359,271 relationships  
- **Trajectories:** 50,489 (train), 10,098 (val), 14,425 (test)
- **Memory Optimized:** From 1.4 TiB to ~few GB

### Progress Summary
- ✅ Memory optimization complete
- ✅ Type errors resolved  
- ✅ Multiprocessing indexing fixed
- 🔄 Empty candidate arrays - in progress
- ⏳ Full training pipeline - pending

