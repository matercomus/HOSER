# HOSER Preprocessing Pipeline Analysis

## Overview

This document provides a comprehensive analysis of the HOSER (Holistic Semantic Representation for Navigational Trajectory Generation) preprocessing pipeline. The analysis focuses on understanding data encoding, preprocessing steps, and how to integrate additional information like Points of Interest (POI).

## Dataset Structure

### Input Files

The Beijing dataset contains the following key files:

```bash
data/Beijing/
├── roadmap.geo        # Road geometry information (1,180,954 roads)
├── roadmap.rel        # Road connectivity relationships (1,295,594 connections)
├── train.csv          # Training trajectories (8,352 trajectories)
├── val.csv            # Validation trajectories
├── test.csv           # Test trajectories
├── road_network_partition  # Zone assignments (300 zones)
└── zone_trans_mat.npy # Zone transition matrix (300x300)
```

### Data Format Analysis

#### 1. Road Geometry File (`roadmap.geo`)

**Structure:**
```csv
geo_id,type,coordinates,highway,lanes,tunnel,bridge,roundabout,oneway,length,maxspeed,u,v
```

**Example:**
```csv
0,LineString,"[[116.3894271850586,39.9062614440918],[116.38945007324219,39.9060173034668]]",99,0,0,0,0,0,27.217,0.0,9063921182,9063921183
```

**Key Fields:**
- `geo_id`: Unique road identifier (0-1,180,953)
- `coordinates`: GeoJSON-style coordinate array `[[lon,lat], [lon,lat], ...]`
- `highway`: Road type encoded as integer (99 = unclassified)
- `length`: Road segment length in meters
- `u,v`: Start and end node IDs for the road segment

#### 2. Road Relations File (`roadmap.rel`)

**Structure:**
```csv
rel_id,type,origin_id,destination_id
```

**Example:**
```csv
0,geo,0,2
1,geo,0,3
```

**Purpose:** Defines which roads are directly connected to each other, forming the road network graph.

#### 3. Trajectory Files (`train.csv`, `val.csv`, `test.csv`)

**Structure:**
```csv
mm_id,user_id,traj_id,rid_list,time_list
```

**Example:**
```csv
0,571901467743189,571901467743189,"125227,125239,125245,495127,...","2019-11-25T00:00:52Z,2019-11-25T00:01:52Z,..."
```

**Key Fields:**
- `rid_list`: Comma-separated sequence of road IDs representing the trajectory path
- `time_list`: Corresponding timestamps in ISO format

## Preprocessing Pipeline

### Step 1: Road Network Partitioning (`partition_road_network.py`)

**Purpose:** Divide the large road network into smaller zones for hierarchical processing.

**Algorithm Used:** KaHIP (Karlsruhe High Quality Partitioning)

**Process:**

1. **Build Adjacency Lists**
   ```python
   # Build adjacency lists from road relations
   adj_lists = {i: set() for i in range(num_roads)}
   for _, row in rel.iterrows():
       origin_id = row['origin_id']
       destination_id = row['destination_id']
       adj_lists[origin_id].add(destination_id)
       adj_lists[destination_id].add(origin_id)
   ```

2. **Create KaHIP Input Format**
   ```bash
   # Format: num_nodes num_edges
   # Then for each node: list of adjacent nodes (1-indexed)
   1180954 647797
   2 3 4
   1 5 6
   ...
   ```

3. **Run KaHIP Partitioning**
   ```bash
   kaffpa ./graph_input.tmp --k 300 --seed 0 --preconfiguration=strong --output_filename=road_network_partition
   ```

**Output:** `road_network_partition` file containing zone assignment for each road:
```
87  # Road 0 is in zone 87
87  # Road 1 is in zone 87
87  # Road 2 is in zone 87
...
110 # Road n is in zone 110
```

**Key Parameters:**
- `--k 300`: Create 300 zones
- `--seed 0`: Reproducible partitioning
- `--preconfiguration=strong`: High-quality partitioning

### Step 2: Zone Transition Matrix Generation (`get_zone_trans_mat.py`)

**Purpose:** Create a transition probability matrix between zones based on actual trajectory data.

**Process:**

1. **Load Zone Assignments**
   ```python
   road2zone = []
   with open(f'../{dataset}/road_network_partition', 'r') as file:
       for line in file:
           road2zone.append(int(line.strip()))
   ```

2. **Initialize Transition Matrix**
   ```python
   zone_cnt = max(road2zone) + 1  # 300 zones (0-299)
   zone_trans_mat = np.zeros((zone_cnt, zone_cnt), dtype=np.int64)
   ```

3. **Process Each Trajectory**
   ```python
   for _, row in tqdm(traj.iterrows(), total=len(traj)):
       rid_list = eval(row['rid_list'])
       
       # Handle single-road trajectories
       if isinstance(rid_list, int):
           rid_list = [rid_list]
       
       # Skip trajectories with less than 2 roads
       if len(rid_list) < 2:
           continue
           
       # Convert road sequence to zone sequence
       zone_list = [road2zone[rid] for rid in rid_list]
       
       # Count zone transitions
       for prev_zone, next_zone in zip(zone_list[:-1], zone_list[1:]):
           if prev_zone != next_zone:
               zone_trans_mat[prev_zone, next_zone] += 1
   ```

**Output:** `zone_trans_mat.npy` - A 300×300 matrix where `matrix[i][j]` contains the number of transitions from zone i to zone j.

## Data Encoding in the Training Pipeline

### Road Network Features

The training script (`train.py`) creates several types of features:

#### 1. Road Attributes
```python
# Length normalization
road_attr_len = geo['length'].to_numpy().astype(np.float32)
road_attr_len = np.log1p(road_attr_len)  # Log transformation
road_attr_len = (road_attr_len - np.mean(road_attr_len)) / np.std(road_attr_len)  # Z-score normalization

# Highway type encoding
road_attr_type = geo['highway'].values.tolist()
le = LabelEncoder()
road_attr_type = le.fit_transform(road_attr_type)

# Spatial coordinates (centroid of road segment)
road_attr_lon = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.x for _, row in geo.iterrows()])
road_attr_lat = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.y for _, row in geo.iterrows()])
# Both coordinates are Z-score normalized
```

#### 2. Graph Topology Features
```python
# Edge connections between adjacent roads
road_edge_index = np.stack([adj_row, adj_col], axis=0)

# Intersection attributes (angle and reachability)
intersection_attr = np.stack([
    np.array(adj_angle).astype(np.float32),        # Angle between roads
    np.array(adj_reachability).astype(np.float32), # Whether roads are connected
], axis=1)
```

#### 3. Zone-Level Features
```python
# Zone connectivity graph
zone_edge_index = np.stack(zone_trans_mat.nonzero())

# Normalized transition weights
D_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.sum(zone_trans_mat, axis=1), 1.0))
zone_trans_mat_norm = zone_trans_mat * D_inv_sqrt[:, np.newaxis] * D_inv_sqrt[np.newaxis, :]
zone_edge_weight = zone_trans_mat_norm[zone_edge_index[0], zone_edge_index[1]]
```

## Adding Point of Interest (POI) Information

### Current Data Flow
```
Raw Trajectories → Road Network → Zones → Zone Transitions → Training
```

### Proposed POI Integration

#### Option 1: Road-Level POI Features

**Step 1: Create POI Database**
```csv
poi_id,name,category,longitude,latitude,rating
1,Beijing University,education,116.3019,39.9925,4.5
2,Starbucks Coffee,food,116.3955,39.9042,4.2
3,Beijing Hospital,healthcare,116.4142,39.9042,4.0
```

**Step 2: Map POIs to Roads**
```python
def find_nearest_roads_to_poi(poi_lat, poi_lon, geo_df, max_distance=100):
    """Find roads within max_distance meters of POI"""
    nearest_roads = []
    for idx, row in geo_df.iterrows():
        road_geom = LineString(eval(row['coordinates']))
        poi_point = Point(poi_lon, poi_lat)
        distance = road_geom.distance(poi_point) * 111000  # Convert to meters
        if distance <= max_distance:
            nearest_roads.append((idx, distance, row['length']))
    return nearest_roads

# Create POI features for each road
road_poi_features = np.zeros((num_roads, num_poi_categories))
for poi in poi_data:
    nearby_roads = find_nearest_roads_to_poi(poi['lat'], poi['lon'], geo)
    for road_id, distance, road_length in nearby_roads:
        # Weight by inverse distance and road length
        weight = (1.0 / (1.0 + distance)) * (poi['rating'] / 5.0)
        category_idx = poi_category_encoder.transform([poi['category']])[0]
        road_poi_features[road_id, category_idx] += weight
```

**Step 3: Integrate into Model**
```python
# Add to road network encoder configuration
config.road_network_encoder_feature.road_attr.poi_features = road_poi_features

# Modify model to accept POI features
class RoadNetworkEncoder(nn.Module):
    def __init__(self, config, features):
        self.poi_embedding = nn.Linear(num_poi_categories, embedding_dim)
        
    def forward(self, road_ids):
        poi_features = self.features.road_attr.poi_features[road_ids]
        poi_embeddings = self.poi_embedding(poi_features)
        # Combine with existing road features
        road_embeddings = road_embeddings + poi_embeddings
```

#### Option 2: Zone-Level POI Aggregation

**Step 1: Aggregate POIs by Zone**
```python
def aggregate_pois_by_zone(poi_data, road2zone, geo_df):
    zone_poi_features = {}
    
    for zone_id in range(max(road2zone) + 1):
        zone_roads = [i for i, z in enumerate(road2zone) if z == zone_id]
        zone_pois = {'education': 0, 'food': 0, 'healthcare': 0, 'entertainment': 0}
        
        for road_id in zone_roads:
            road_geom = LineString(eval(geo_df.loc[road_id, 'coordinates']))
            for poi in poi_data:
                poi_point = Point(poi['lon'], poi['lat'])
                if road_geom.distance(poi_point) * 111000 <= 200:  # 200m radius
                    zone_pois[poi['category']] += poi['rating']
        
        zone_poi_features[zone_id] = list(zone_pois.values())
    
    return zone_poi_features
```

#### Option 3: Dynamic POI Context

**Real-time POI Influence**
```python
def get_trajectory_poi_context(trajectory_roads, poi_data, time_of_day):
    """Get POI context for trajectory based on time and location"""
    poi_context = []
    
    for road_id in trajectory_roads:
        road_geom = LineString(eval(geo.loc[road_id, 'coordinates']))
        nearby_pois = []
        
        for poi in poi_data:
            poi_point = Point(poi['lon'], poi['lat'])
            distance = road_geom.distance(poi_point) * 111000
            
            if distance <= 500:  # 500m radius
                # Time-based weighting
                time_weight = get_time_weight(poi['category'], time_of_day)
                poi_influence = (poi['rating'] / 5.0) * time_weight * (1.0 / (1.0 + distance/100))
                nearby_pois.append(poi_influence)
        
        poi_context.append(sum(nearby_pois))
    
    return poi_context

def get_time_weight(poi_category, time_of_day):
    """Weight POI influence based on time of day"""
    weights = {
        'education': [0.2, 0.9, 0.9, 0.2],  # Morning, Noon, Afternoon, Evening
        'food': [0.3, 0.9, 0.7, 0.9],
        'entertainment': [0.1, 0.3, 0.6, 0.9],
        'healthcare': [0.6, 0.8, 0.8, 0.3]
    }
    hour = int(time_of_day.split(':')[0])
    period = hour // 6  # 0-5: 0, 6-11: 1, 12-17: 2, 18-23: 3
    return weights.get(poi_category, [0.5, 0.5, 0.5, 0.5])[period]
```

## Implementation Recommendations

### 1. Start with Road-Level POI Features (Option 1)
- Easiest to implement
- Directly integrates with existing road embeddings
- Minimal changes to model architecture

### 2. Preprocessing Script Modifications

**New script: `add_poi_features.py`**
```python
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point

def main(dataset, poi_file):
    # Load data
    geo = pd.read_csv(f'../{dataset}/roadmap.geo')
    poi_data = pd.read_csv(poi_file)
    
    # Create POI features
    road_poi_features = create_road_poi_features(geo, poi_data)
    
    # Save features
    np.save(f'../{dataset}/road_poi_features.npy', road_poi_features)
    
    print(f'Created POI features: {road_poi_features.shape}')
```

### 3. Model Integration

**Modify config file to include POI features:**
```yaml
road_network_encoder_config:
  use_poi_features: true
  poi_feature_dim: 10  # Number of POI categories
  poi_embedding_dim: 32
```

## Real Data Examples

### Road Network Example
From the Beijing dataset:
- **Total roads**: 1,180,954 road segments
- **Total connections**: 1,295,594 road-to-road relationships
- **Total trajectories**: 8,352 training trajectories

**Sample Road Entry:**
```csv
geo_id: 0
type: LineString
coordinates: [[116.3894271850586,39.9062614440918],[116.38945007324219,39.9060173034668]]
highway: 99 (unclassified road type)
length: 27.217 meters
```

### Zone Partitioning Results
The KaHIP algorithm creates **300 zones** with balanced distribution:
- Zone 87: Contains roads 0-9 (clustered spatially)
- Top zones by road count:
  - Zone 109: 4,054 roads
  - Zone 97: 4,054 roads  
  - Zone 169: 4,054 roads
  - Zone 107: 4,054 roads
  - Zone 76: 4,054 roads

### Trajectory Example
**Sample Trajectory:**
```python
trajectory_id: 571901467743189
road_sequence: [125227, 125239, 125245, 495127, 342550, ...]
timestamps: ['2019-11-25T00:00:52Z', '2019-11-25T00:01:52Z', ...]
total_segments: 36 road segments
duration: ~28 minutes
```

## Data Exploration Commands

```bash
# Examine road network statistics
cd /home/matt/Dev/HOSER/data/Beijing
wc -l roadmap.geo roadmap.rel train.csv

# Check zone distribution  
python3 -c "
import collections
with open('road_network_partition') as f:
    zones = [int(line.strip()) for line in f]
zone_counts = collections.Counter(zones)
print('Zone distribution:')
for zone, count in zone_counts.most_common(10):
    print(f'Zone {zone}: {count} roads')
"

# Analyze trajectory lengths
python3 -c "
import csv
with open('train.csv') as f:
    reader = csv.DictReader(f)
    lengths = []
    for row in reader:
        rid_list = eval(row['rid_list'])
        lengths.append(len(rid_list))
lengths.sort()
print(f'Min trajectory length: {min(lengths)}')
print(f'Max trajectory length: {max(lengths)}')
print(f'Median trajectory length: {lengths[len(lengths)//2]}')
"

# Check coordinate ranges (for POI mapping)
python3 -c "
import csv, json
lons, lats = [], []
with open('roadmap.geo') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 1000: break  # Sample first 1000 roads
        coords = eval(row['coordinates'])
        for lon, lat in coords:
            lons.append(lon); lats.append(lat)
print(f'Longitude range: {min(lons):.4f} to {max(lons):.4f}')
print(f'Latitude range: {min(lats):.4f} to {max(lats):.4f}')
"
```

## Validation and Testing

### POI Integration Validation
1. **Spatial Validation**: Ensure POIs are correctly mapped to nearby roads
2. **Category Balance**: Check distribution of POI categories across zones
3. **Model Performance**: Compare model accuracy with/without POI features
4. **Ablation Study**: Test different POI influence radii and weighting schemes

### Monitoring Commands
```bash
# Check POI feature distribution
python3 -c "import numpy as np; features = np.load('road_poi_features.npy'); print(f'Shape: {features.shape}', f'Non-zero: {np.count_nonzero(features)}')"

# Validate zone-POI mapping
python3 -c "
poi_by_zone = {}
with open('road_network_partition') as f:
    for i, line in enumerate(f):
        zone = int(line.strip())
        poi_by_zone[zone] = poi_by_zone.get(zone, 0) + 1
print('Top 10 zones by road count:', sorted(poi_by_zone.items(), key=lambda x: x[1], reverse=True)[:10])
"
```

## Creating a Complete POI Integration Script

### Full Implementation Example

```python
# add_poi_features.py
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
import json
from sklearn.preprocessing import LabelEncoder

def load_poi_data(poi_file):
    """Load and categorize POI data"""
    # Example POI data structure
    pois = [
        {"id": 1, "name": "Beijing University", "category": "education", 
         "lon": 116.3019, "lat": 39.9925, "rating": 4.5},
        {"id": 2, "name": "Forbidden City", "category": "tourism", 
         "lon": 116.3973, "lat": 39.9163, "rating": 4.8},
        {"id": 3, "name": "Starbucks", "category": "food", 
         "lon": 116.3955, "lat": 39.9042, "rating": 4.2},
        # Add more POIs...
    ]
    return pois

def map_pois_to_roads(geo_df, poi_data, max_distance=200):
    """Map POIs to nearby roads within max_distance meters"""
    road_poi_features = np.zeros((len(geo_df), len(poi_categories)))
    
    for idx, road_row in geo_df.iterrows():
        if idx % 10000 == 0:
            print(f"Processing road {idx}/{len(geo_df)}")
            
        # Parse road coordinates
        coords = eval(road_row['coordinates'])
        road_geom = LineString(coords)
        
        # Find nearby POIs
        for poi in poi_data:
            poi_point = Point(poi['lon'], poi['lat'])
            # Convert degrees to meters (approximate)
            distance = road_geom.distance(poi_point) * 111000
            
            if distance <= max_distance:
                category_idx = poi_category_encoder.transform([poi['category']])[0]
                # Weight by inverse distance and rating
                weight = (poi['rating'] / 5.0) * (1.0 / (1.0 + distance/100))
                road_poi_features[idx, category_idx] += weight
    
    return road_poi_features

def create_poi_zone_features(road_poi_features, road2zone):
    """Aggregate POI features by zone"""
    max_zone = max(road2zone) + 1
    zone_poi_features = np.zeros((max_zone, road_poi_features.shape[1]))
    
    for road_id, zone_id in enumerate(road2zone):
        zone_poi_features[zone_id] += road_poi_features[road_id]
    
    # Normalize by roads per zone
    for zone_id in range(max_zone):
        roads_in_zone = sum(1 for z in road2zone if z == zone_id)
        if roads_in_zone > 0:
            zone_poi_features[zone_id] /= roads_in_zone
    
    return zone_poi_features

def main():
    dataset = 'Beijing'
    
    # Load existing data
    geo = pd.read_csv(f'../{dataset}/roadmap.geo')
    print(f"Loaded {len(geo)} roads")
    
    # Load zone assignments
    road2zone = []
    with open(f'../{dataset}/road_network_partition', 'r') as f:
        for line in f:
            road2zone.append(int(line.strip()))
    
    # Load or create POI data
    poi_data = load_poi_data('poi_data.json')
    
    # Create POI categories encoder
    global poi_categories, poi_category_encoder
    poi_categories = list(set(poi['category'] for poi in poi_data))
    poi_category_encoder = LabelEncoder()
    poi_category_encoder.fit(poi_categories)
    
    print(f"POI categories: {poi_categories}")
    
    # Create road-level POI features
    road_poi_features = map_pois_to_roads(geo, poi_data)
    print(f"Created road POI features: {road_poi_features.shape}")
    
    # Create zone-level POI features
    zone_poi_features = create_poi_zone_features(road_poi_features, road2zone)
    print(f"Created zone POI features: {zone_poi_features.shape}")
    
    # Save features
    np.save(f'../{dataset}/road_poi_features.npy', road_poi_features)
    np.save(f'../{dataset}/zone_poi_features.npy', zone_poi_features)
    
    # Save metadata
    metadata = {
        'poi_categories': poi_categories,
        'num_categories': len(poi_categories),
        'max_distance': 200,
        'num_roads': len(geo),
        'num_zones': len(set(road2zone))
    }
    
    with open(f'../{dataset}/poi_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("POI integration complete!")

if __name__ == '__main__':
    main()
```

## Key Insights and Best Practices

### 1. Hierarchical Structure
The HOSER preprocessing creates a two-level hierarchy (roads → zones) that naturally accommodates POI integration:
- **Road-level POIs**: Direct influence on individual road segments
- **Zone-level POIs**: Aggregate influence on geographic regions

### 2. Balanced Partitioning
KaHIP creates well-balanced zones (each with ~4,054 roads), ensuring:
- Even computational load during training
- Consistent zone-level feature representation
- Effective hierarchical learning

### 3. Scalability Considerations
With 1.18M roads and 8.3K trajectories:
- Process POIs in batches to manage memory
- Use spatial indexing (R-tree) for large POI datasets
- Consider different distance thresholds for different POI types

### 4. Integration Points
POI features can be integrated at multiple levels:
- **Model input**: Add POI embeddings to road features
- **Attention mechanism**: Use POI context for trajectory attention
- **Loss function**: Weight trajectory predictions by POI relevance

### 5. Validation Strategy
- **Spatial accuracy**: Verify POI-road mappings using visualization
- **Category balance**: Ensure all POI types are represented
- **Model impact**: A/B test trajectory generation with/without POI features

This comprehensive analysis provides both theoretical understanding and practical implementation guidance for extending the HOSER preprocessing pipeline with POI information.
