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

### Recommended Solution
**Quick Fix (Option A):** Modify `dataset.py` to filter out trajectory segments where roads have no reachable destinations:

```python
# In process_row function, before road_label calculation:
valid_indices = []
for i in range(len(trace_road_id)):
    road_id = int(rid_list[i])
    next_road_id = int(rid_list[i + 1])
    if len(global_reachable_road_id_dict[road_id]) > 0 and next_road_id in global_reachable_road_id_dict[road_id]:
        valid_indices.append(i)

# Only process valid trajectory segments
if len(valid_indices) > 0:
    # Continue with current logic using valid_indices
else:
    # Skip this trajectory entirely
```

This preserves the training pipeline while handling the connectivity issues in the Beijing dataset.

