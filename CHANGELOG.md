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

## [2025-01-27] - Comprehensive Dataset Analysis and Error Documentation

### Added
- **`DEBUGGING_NOTES.md`**: Comprehensive analysis of Beijing vs Porto dataset differences
  - Detailed connectivity analysis: Beijing 97.33% vs Porto 99.99% road connectivity
  - Concrete examples of 33,029 dead-end roads in Beijing causing zero-size array errors
  - Real trajectory samples showing scale differences (Beijing: 35 roads, Porto: 51 roads)
  - Error manifestation flow from disconnected roads to haversine calculation failures
  - Smart trajectory filtering solution with expected 75-85% data preservation

### Analysis
- **Dataset Characteristics:**
  - Beijing: 1,239,014 roads, 0.041 trajectories/road (sparse, raw OSM data)
  - Porto: 11,025 roads, 9.87 trajectories/road (dense, preprocessed data)
  - Scale factor: Beijing 112x larger network, 240x lower trajectory density
- **Root Cause:** Raw OSM data includes service roads, parking lots, isolated segments vs preprocessed connectivity
- **Error Pattern:** Dead-end roads → empty candidate arrays → `ValueError: zero-size array to reduction operation maximum`

### Documentation
- Enhanced debugging notes with concrete road IDs, trajectory examples, and connectivity statistics
- Implementation strategy for trajectory filtering with minimum viable segment validation
- Alternative solution approaches (spatial nearest-neighbor, network preprocessing)
- Clear roadmap for resolving zero-size array errors while preserving data quality
- **Dead-end road analysis**: Detailed characterization of Beijing dead-end roads vs connected roads
- **Road classification table**: Complete taxonomy of road types and connectivity patterns
- **Concrete examples**: Real road segments (224, 146439) showing dead-end vs connected patterns

## [2025-01-27] - Zone Transition Matrix Generation Fix

### Fixed
- **`data/preprocess/get_zone_trans_mat.py`**: TypeError when processing single-road trajectories
  - Fixed `TypeError: 'int' object is not iterable` caused by trajectories reduced to single road IDs
  - Added handling for single-road trajectories resulting from dead-end filtering preprocessing
  - Convert single integers to lists for consistent processing: `eval('943754')` → `[943754]`
  - Skip trajectories with fewer than 2 roads (no zone transitions possible)

### Added
- **Trajectory filtering statistics**: Log count of skipped single-road trajectories
- **Robust preprocessing pipeline**: Handle edge cases from dead-end road filtering
- **Beijing-specific processing**: Limited script to Beijing dataset for current development

### Performance
- **Successfully processed**: 8,345 trajectories with only 6 single-road trajectories skipped (0.07%)
- **Zone transition matrix generated**: 704K output file for Beijing dataset
- **Preprocessing pipeline complete**: Ready for training script execution
