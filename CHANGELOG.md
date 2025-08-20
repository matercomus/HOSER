# Changelog

## [2024-08-20] - Dataset Preprocessing Optimization

### Changed
- **`data/preprocess/partition_road_network.py`**: Major memory optimization for large road networks
  - Replaced memory-intensive adjacency matrix (1.4 TiB for 1.24M roads) with efficient adjacency lists using sets and dictionaries
  - Added configurable `KAHIP_PATH` constant for external KaHIP installation path
  - Added progress logging and error handling for better debugging
  - Updated default dataset from `'Beijing-BJUT'` to `'Beijing'` to match expected naming convention
  - Successfully tested with Beijing dataset: 1,239,014 roads, 1,359,271 edges

### Fixed  
- Memory allocation error when processing large datasets (Beijing-BJUT with 1.24M+ road segments)
- Script now works efficiently with datasets of any size without requiring massive amounts of RAM

### Performance
- Reduced memory usage from 1.4 TiB to ~few GB for large road networks
- Partitioning completed in ~87 seconds for 1.24M road network using KaHIP
- Generated balanced 300-zone partition with cut value of 3694

## [2024-08-20] - Training Script Memory Optimization

### Changed
- **`train.py`**: Applied same memory optimization for large road networks
  - Replaced memory-intensive road adjacency matrix (1.4 TiB for 1.24M roads) with efficient adjacency lists using sets
  - Fixed AttributeError when processing integer highway type values in Beijing dataset
  - Added type checking for highway attribute processing to handle both string and integer values

### Fixed
- Memory allocation error in training script when loading large datasets
- AttributeError: 'int' object has no attribute 'startswith' for Beijing dataset highway values
- Training script now works with datasets containing 1.24M+ road segments

## [2024-08-20] - Dataset Loading IndexError Fix

### Changed
- **`dataset.py`**: Fixed multiprocessing data type issues for large datasets
  - Explicitly cast all road IDs to integers to prevent type conversion issues
  - Added dtype=np.int64 to candidate road ID arrays
  - Enhanced type safety in trajectory processing functions

### Fixed
- IndexError: 'arrays used as indices must be of integer (or boolean) type' in dataset loading
- Multiprocessing data type conversion issues causing array indexing failures
- Dataset loading now works reliably with large Beijing dataset (1.24M+ roads)
