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
