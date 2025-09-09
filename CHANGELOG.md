# Changelog

## [2025-09-09] - Training Memory Efficiency and PyTorch API Update

### Fixed
- **Critical Out-of-Memory (OOM) Error**: Resolved a major memory bottleneck in `dataset.py` where the entire dataset was loaded into RAM, causing crashes during training with many workers. The script now runs with a minimal memory footprint.
- **`_pickle.UnpicklingError`**: Fixed a loading error caused by a security update in PyTorch >= 2.6 that defaults `torch.load` to `weights_only=True`. Explicitly set `weights_only=False` for trusted, self-generated cache files.
- **`FutureWarning` Deprecations**: Updated all instances of the deprecated `torch.cuda.amp` API to the modern `torch.amp` API in `train.py`, eliminating console warnings.

### Changed
- **`dataset.py`**: Re-architected the `Dataset` class to use an on-disk caching strategy.
  - Instead of processing in-memory, the script now preprocesses each trajectory once and saves it as a `.pt` file.
  - The dataset now lazy-loads data samples on demand, dramatically reducing RAM usage.
- **`train.py`**: Adapted the training script to work with the new caching dataset.
  - Removed manual in-memory statistics calculation, instead loading pre-calculated stats from the cache.

### Performance
- **Memory Usage**: Reduced RAM consumption during training from >30GB (causing OOM) to a negligible amount, allowing the script to run smoothly.
- **Preprocessing**: A one-time preprocessing step is now required, which caches the entire dataset to disk. Subsequent training runs start much faster without this overhead.

## [2025-09-05] - Road ID Sequential Indexing and Dead-End Handling

### Fixed
- **`data/preprocess/partition_road_network.py`**: Modified to use actual road IDs from data files
  - Replaced sequential index assumption with actual `geo_id` values from roadmap.geo
  - Added bidirectional mapping between actual road IDs and sequential indices for KaHIP compatibility
  - Created `road_id_mapping.csv` output file to preserve ID relationships
  - Fixed KeyError when road IDs were non-sequential (e.g., 191210, 1598, etc.)

- **`data/preprocess/get_zone_trans_mat.py`**: Updated to work with actual road IDs
  - Added mapping file loading to convert actual road IDs to sequential indices
  - Fixed IndexError when accessing partition data with non-sequential road IDs
  - Added error handling for unmapped road IDs in trajectory data

- **`gene.py`**: Enhanced trajectory generation to handle dead-end roads
  - Added dead-end destination filtering to prevent selecting roads with no outgoing connections
  - Implemented dead-end path handling in search algorithm to skip problematic roads during pathfinding
  - Added timestamp sampling with replacement to allow generating more trajectories than training examples
  - Fixed AssertionError when encountering dead-end roads during trajectory generation

### Changed
- **Dataset preprocessing**: All scripts now work with actual road IDs from data files instead of assuming sequential 0-based indexing
- **Trajectory generation**: Successfully generates 100 trajectories from Beijing dataset with 73 dead-end roads
- **Error handling**: Robust handling of network connectivity issues in sparse road networks

### Performance
- **Beijing dataset analysis**: 2,791 roads total, 73 dead-end roads, 1 isolated road
- **Trajectory generation**: Successfully generated 100 trajectories in ~74 seconds
- **Memory efficiency**: Maintained efficient processing despite non-sequential road ID mapping
