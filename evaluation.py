import argparse
import os
import json
import ast
import math
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
from shapely.geometry import LineString
from tqdm import tqdm
from scipy.stats import entropy
from fastdtw import fastdtw
from haversine import haversine, haversine_vector
from geopy import distance  # type: ignore
import wandb
import yaml


def js_divergence(p, q):
    """
    Calculate Jensen-Shannon divergence between two distributions.
    Uses the original author's implementation for consistency.
    """
    p = p / (np.sum(p) + 1e-14)
    q = q / (np.sum(q) + 1e-14)
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def extract_wandb_metadata_from_path(path: str) -> dict:
    """
    Extract WandB project and run ID from a path that may contain wandb run metadata.
    
    Args:
        path: Path that may be in or reference a wandb run directory
    
    Returns:
        Dict with 'project' and 'run_id' keys, or None values if not found
    """
    result = {'project': None, 'run_id': None, 'run_name': None}
    
    if not path or 'wandb' not in path:
        return result
    
    try:
        # Find the wandb run directory in the path
        parts = path.split(os.sep)
        wandb_idx = None
        run_idx = None
        
        for i, part in enumerate(parts):
            if part == 'wandb':
                wandb_idx = i
            elif part.startswith('run-'):
                run_idx = i
                # Extract run ID from directory name (e.g., "run-20250929_191519-0vw2ywd9")
                run_dir_name = part
                if '-' in run_dir_name:
                    # Format: run-<timestamp>-<run_id>
                    result['run_id'] = run_dir_name.split('-')[-1]
                break
        
        if wandb_idx is None or run_idx is None:
            return result
        
        # Reconstruct wandb run directory path
        wandb_run_dir = os.path.join(*parts[:run_idx+1])
        config_path = os.path.join(wandb_run_dir, 'files', 'config.yaml')
        
        if not os.path.exists(config_path):
            return result
        
        # Parse config.yaml to extract project name and run name
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Navigate the nested structure: wandb.value.project
        if 'wandb' in config and 'value' in config['wandb']:
            wandb_config = config['wandb']['value']
            result['project'] = wandb_config.get('project')
            result['run_name'] = wandb_config.get('run_name')
        
        return result
        
    except Exception:
        return result


def detect_wandb_metadata(run_dir: str, generated_file: str = None) -> dict:
    """
    Auto-detect WandB metadata from run directory and generated file path.
    
    Returns dict with:
    - project: WandB project name
    - training_run_id: Run ID of training run (if detected)
    - training_run_name: Run name of training run (if detected)
    - generation_run_id: Run ID of generation run (if detected)
    - generation_run_name: Run name of generation run (if detected)
    """
    default_project = 'hoser-eval'
    metadata = {
        'project': default_project,
        'training_run_id': None,
        'training_run_name': None,
        'generation_run_id': None,
        'generation_run_name': None,
    }
    
    # Extract from generated file path (generation run)
    if generated_file:
        gen_meta = extract_wandb_metadata_from_path(generated_file)
        if gen_meta['run_id']:
            metadata['generation_run_id'] = gen_meta['run_id']
            metadata['generation_run_name'] = gen_meta['run_name']
            if gen_meta['project']:
                metadata['project'] = gen_meta['project']
                print(f"📊 Detected generation run: {gen_meta['run_name']} (ID: {gen_meta['run_id']})")
    
    # Extract from run_dir (training run or generation run)
    train_meta = extract_wandb_metadata_from_path(run_dir)
    if train_meta['run_id']:
        # If we didn't get generation metadata, this might be the generation run
        if metadata['generation_run_id'] is None:
            metadata['generation_run_id'] = train_meta['run_id']
            metadata['generation_run_name'] = train_meta['run_name']
        else:
            # Otherwise it's likely the training run
            metadata['training_run_id'] = train_meta['run_id']
            metadata['training_run_name'] = train_meta['run_name']
        
        if train_meta['project'] and metadata['project'] == default_project:
            metadata['project'] = train_meta['project']
            print(f"📊 Detected project from run directory: {train_meta['project']}")
    
    # Check if run_dir contains wandb subdirectories (training outputs)
    if os.path.isdir(run_dir):
        wandb_dir = os.path.join(run_dir, 'wandb')
        if os.path.exists(wandb_dir):
            for item in os.listdir(wandb_dir):
                if item.startswith('run-'):
                    run_path = os.path.join(wandb_dir, item)
                    meta = extract_wandb_metadata_from_path(run_path)
                    if meta['run_id'] and metadata['training_run_id'] is None:
                        metadata['training_run_id'] = meta['run_id']
                        metadata['training_run_name'] = meta['run_name']
                        if meta['project']:
                            metadata['project'] = meta['project']
                            print(f"📊 Detected training run: {meta['run_name']} (ID: {meta['run_id']})")
                        break
    
    if metadata['project'] == default_project:
        print(f"ℹ️  No WandB metadata found, using default project: {default_project}")
    
    return metadata


# --- Data Loading ---

# Global caches for expensive data loading
_ROAD_NETWORK_CACHE = {}
_TRAJECTORY_CACHE = {}

def load_road_network(geo_path):
    """
    Loads road network data using Polars with aggressive validation.
    FAILS FAST if data is malformed or incomplete.
    Uses global cache to avoid reloading/reprocessing.
    """
    # Check cache first
    cache_key = str(geo_path)
    if cache_key in _ROAD_NETWORK_CACHE:
        print("📂 Using cached road network...")
        return _ROAD_NETWORK_CACHE[cache_key]
    
    print("📂 Loading road network...")
    
    if not os.path.exists(geo_path):
        raise FileNotFoundError(f"🚨 FATAL: Road network file not found: {geo_path}")
    
    # Load with Polars (specify schema for problematic columns)
    try:
        geo_df = pl.read_csv(geo_path, schema_overrides={"lanes": pl.Utf8, "oneway": pl.Utf8})
    except Exception as e:
        raise RuntimeError(f"🚨 FATAL: Failed to parse road network CSV: {e}")
    
    # Validate required columns
    required_cols = ['geo_id', 'coordinates']
    missing = [col for col in required_cols if col not in geo_df.columns]
    if missing:
        raise ValueError(f"🚨 FATAL: Road network missing required columns: {missing}\nAvailable: {list(geo_df.columns)}")
    
    # Rename geo_id to road_id for consistency
    if 'geo_id' in geo_df.columns and 'road_id' not in geo_df.columns:
        geo_df = geo_df.rename({'geo_id': 'road_id'})
    
    # Validate road_id is unique and sequential
    road_ids = geo_df['road_id'].to_list()
    if len(road_ids) != len(set(road_ids)):
        raise ValueError("🚨 FATAL: Duplicate road_ids found in road network!")
    
    max_id = max(road_ids)
    min_id = min(road_ids)
    if min_id != 0 or max_id != len(road_ids) - 1:
        print(f"⚠️  WARNING: Road IDs not sequential. Min: {min_id}, Max: {max_id}, Count: {len(road_ids)}")
        print(f"    This will cause indexing issues. Expected: 0 to {len(road_ids)-1}")

    # Pre-calculate road center GPS coordinates
    print(f"📐 Calculating road centroids for {len(geo_df):,} roads...")
    road_center_gps = []
    failed_roads = []
    
    for idx, row in enumerate(geo_df.iter_rows(named=True)):
        try:
            coordinates = json.loads(row['coordinates'])
            road_line = LineString(coordinates=coordinates)
            center_coord = road_line.centroid
            # Store as (lat, lon) for consistency with haversine
            road_center_gps.append((center_coord.y, center_coord.x))
        except (json.JSONDecodeError, TypeError, Exception) as e:
            failed_roads.append((row['road_id'], str(e)))
            road_center_gps.append((None, None))

    if failed_roads:
        print(f"⚠️  WARNING: {len(failed_roads)} roads failed centroid calculation")
        if len(failed_roads) > 10:
            print(f"    First 10 failures: {failed_roads[:10]}")
        else:
            print(f"    Failures: {failed_roads}")
        
        if len(failed_roads) > len(geo_df) * 0.1:  # >10% failure
            raise RuntimeError(f"🚨 FATAL: Too many roads ({len(failed_roads)}) failed centroid calculation!")
    
    # Add center_gps column
    geo_df = geo_df.with_columns(pl.Series('center_gps', road_center_gps))
    
    # Drop rows with null centroids
    before_count = len(geo_df)
    geo_df = geo_df.filter(pl.col('center_gps').is_not_null())
    dropped = before_count - len(geo_df)
    
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} roads with invalid coordinates")
    
    # Convert to pandas for compatibility with existing code (only for indexed access)
    geo_pd = geo_df.to_pandas()
    
    print(f"✅ Road network loaded: {len(geo_pd):,} valid roads")
    
    # Cache the processed road network
    _ROAD_NETWORK_CACHE[cache_key] = geo_pd
    
    return geo_pd

def load_trajectories(traj_path, is_real_data, max_road_id=None):
    """
    Loads trajectories using Polars with aggressive validation.
    FAILS FAST if data is malformed or road IDs are out of bounds.
    Caches real trajectory data to avoid reloading.
    """
    # Check cache for real trajectories only (generated trajectories change)
    cache_key = str(traj_path)
    if is_real_data and cache_key in _TRAJECTORY_CACHE:
        print(f"📂 Using cached real trajectories from {os.path.basename(traj_path)}...")
        return _TRAJECTORY_CACHE[cache_key]
    
    print(f"📂 Loading trajectories from {traj_path}...")
    
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"🚨 FATAL: Trajectory file not found: {traj_path}")
    
    # Load with Polars
    try:
        traj_df = pl.read_csv(traj_path)
    except Exception as e:
        raise RuntimeError(f"🚨 FATAL: Failed to parse trajectory CSV: {e}")
    
    # Validate required columns
    if is_real_data:
        required_cols = ['rid_list', 'time_list']
    else:
        required_cols = ['gene_trace_road_id', 'gene_trace_datetime']
    
    missing = [col for col in required_cols if col not in traj_df.columns]
    if missing:
        raise ValueError(f"🚨 FATAL: Trajectory file missing required columns: {missing}\nAvailable: {traj_df.columns}")
    
    trajectories = []
    invalid_count = 0
    out_of_bounds_count = 0
    
    print(f"🔄 Parsing {len(traj_df):,} trajectories...")
    
    if is_real_data:
        # Real data from test.csv
        for idx, row in enumerate(traj_df.iter_rows(named=True)):
            try:
                # rid_list is comma-separated string
                rids = [int(r) for r in str(row['rid_list']).split(',')]
                timestamps = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') 
                             for t in str(row['time_list']).split(',')]
                
                # Validate road IDs if max_road_id provided
                if max_road_id is not None:
                    invalid_rids = [r for r in rids if r < 0 or r > max_road_id]
                    if invalid_rids:
                        out_of_bounds_count += 1
                        if out_of_bounds_count <= 5:  # Show first 5
                            print(f"⚠️  Traj {idx}: road IDs out of bounds: {invalid_rids[:10]}")
                        continue
                
                if len(rids) != len(timestamps):
                    invalid_count += 1
                    continue
                
                trajectories.append(list(zip(rids, timestamps)))
            except Exception as e:
                invalid_count += 1
                if invalid_count <= 5:
                    print(f"⚠️  Traj {idx} parse error: {e}")
    else:
        # Generated data
        for idx, row in enumerate(traj_df.iter_rows(named=True)):
            try:
                # gene_trace_road_id is JSON array
                rid_str = str(row['gene_trace_road_id'])
                rids = json.loads(rid_str)

                # gene_trace_datetime is string representation of list
                datetime_list_str = ast.literal_eval(str(row['gene_trace_datetime']))
                timestamps = [datetime.strptime(t.strip('"'), '%Y-%m-%dT%H:%M:%SZ') 
                             for t in datetime_list_str]
                
                # Validate road IDs if max_road_id provided
                if max_road_id is not None:
                    invalid_rids = [r for r in rids if r < 0 or r > max_road_id]
                    if invalid_rids:
                        out_of_bounds_count += 1
                        if out_of_bounds_count <= 5:
                            print(f"⚠️  Traj {idx}: road IDs out of bounds: {invalid_rids[:10]}")
                        continue
                
                if len(rids) != len(timestamps):
                    invalid_count += 1
                    continue
                
                trajectories.append(list(zip(rids, timestamps)))
            except Exception as e:
                invalid_count += 1
                if invalid_count <= 5:
                    print(f"⚠️  Traj {idx} parse error: {e}")
    
    if invalid_count > 0:
        print(f"⚠️  WARNING: {invalid_count} trajectories failed parsing")
        if invalid_count > len(traj_df) * 0.1:  # >10% failure
            raise RuntimeError(f"🚨 FATAL: Too many trajectories ({invalid_count}) failed parsing!")
    
    if out_of_bounds_count > 0:
        print(f"⚠️  WARNING: {out_of_bounds_count} trajectories have out-of-bounds road IDs")
        if out_of_bounds_count > len(traj_df) * 0.1:  # >10% failure
            raise RuntimeError(f"🚨 FATAL: Too many trajectories ({out_of_bounds_count}) have invalid road IDs!")
    
    if len(trajectories) == 0:
        raise RuntimeError(f"🚨 FATAL: No valid trajectories loaded from {traj_path}")
    
    print(f"✅ Loaded {len(trajectories):,} valid trajectories (dropped {invalid_count + out_of_bounds_count})")
    
    # Cache real trajectories to avoid reloading
    if is_real_data:
        _TRAJECTORY_CACHE[cache_key] = trajectories
        
    return trajectories

# --- Metric Calculation ---

class GlobalMetrics:
    """
    Calculate global (distribution-level) metrics using the original author's method.
    
    Key differences from previous version:
    - Distance: Uses Haversine between road centroids (not road length field)
    - Duration: Per-segment durations (not total trip duration)
    - Radius: Simple distance averaging (not RMS formula)
    - JS Divergence: Uses original author's implementation
    """
    def __init__(self, real_trajs, generated_trajs, geo_df):
        self.real_trajs = real_trajs
        self.generated_trajs = generated_trajs
        self.geo_df = geo_df.set_index('road_id')

        # Create road_id → GPS mapping dict for safe access
        self.road_gps = {}
        for road_id, row in self.geo_df.iterrows():
            try:
                center_gps = row['center_gps']
                # center_gps is a tuple like (lat, lon)
                if center_gps is not None and len(center_gps) == 2:
                    lat, lon = center_gps
                    if lat is not None and lon is not None:
                        # Store as (lon, lat) for compatibility with original code
                        self.road_gps[road_id] = (lon, lat)
            except (KeyError, TypeError, Exception) as e:
                pass  # Silently skip invalid entries
        
        print(f"✅ GlobalMetrics: Loaded GPS for {len(self.road_gps):,} roads")

    def _calculate_distance_distribution(self, trajectories):
        """Calculate trip distances using Haversine between road centroids."""
        distances = []
        skipped_trajs = 0
        
        for traj in tqdm(trajectories, desc="Calculating distances"):
            if len(traj) < 2:
                continue
            
            road_ids = [p[0] for p in traj]
            travel_distance = 0
            valid_segments = 0
            
            for i in range(1, len(road_ids)):
                road_prev = road_ids[i-1]
                road_curr = road_ids[i]
                
                # Check if both roads have valid GPS coordinates
                if road_prev not in self.road_gps or road_curr not in self.road_gps:
                    continue
                
                gps_prev = self.road_gps[road_prev]
                gps_curr = self.road_gps[road_curr]
                
                if gps_prev is None or gps_curr is None:
                    continue
                
                try:
                    # Use Haversine distance between consecutive road centroids
                    segment_dist = distance.great_circle(
                        (gps_prev[1], gps_prev[0]),  # (lat, lon)
                        (gps_curr[1], gps_curr[0])
                    ).kilometers
                    travel_distance += segment_dist
                    valid_segments += 1
                except Exception as e:
                    print(f"⚠️  Distance calculation error for roads {road_prev}→{road_curr}: {e}")
                    continue
            
            # Only include trajectories with at least one valid segment
            if valid_segments > 0:
                distances.append(travel_distance)
            else:
                skipped_trajs += 1
        
        if skipped_trajs > 0:
            print(f"⚠️  Skipped {skipped_trajs} trajectories with no valid road segments")
        
        return distances

    def _calculate_duration_distribution(self, trajectories):
        """Calculate per-segment durations (not total trip duration)."""
        durations = []
        for traj in tqdm(trajectories, desc="Calculating durations"):
            if len(traj) < 2:
                continue
            
            # Extract per-segment durations in minutes
            for i in range(1, len(traj)):
                time_duration = (traj[i][1] - traj[i-1][1]).total_seconds() / 60.0
                durations.append(time_duration)
        
        return durations

    def _calculate_radius_distribution(self, trajectories):
        """Calculate radius of gyration using simple distance averaging."""
        radii = []
        for traj in tqdm(trajectories, desc="Calculating radii"):
            if len(traj) < 1:
                continue
            
            road_ids = [p[0] for p in traj]
            
            # Filter out road IDs that don't have GPS data
            valid_rids = [rid for rid in road_ids if rid in self.road_gps]
            if not valid_rids:
                continue
            
            # Calculate mean center
            lons = [self.road_gps[rid][0] for rid in valid_rids]
            lats = [self.road_gps[rid][1] for rid in valid_rids]
            lon_mean = np.mean(lons)
            lat_mean = np.mean(lats)
            
            # Calculate average distance from center (original author's method)
            rad_list = []
            for rid in valid_rids:
                lon = self.road_gps[rid][0]
                lat = self.road_gps[rid][1]
                dis = distance.great_circle((lat_mean, lon_mean), (lat, lon)).kilometers
                rad_list.append(dis)
            
            rad = np.mean(rad_list)
            radii.append(rad)
        
        return radii

    def evaluate(self):
        """
        Evaluate global metrics using JS divergence.
        Returns dict with Distance/Duration/Radius JSD values.
        """
        print("📊 Calculating global metrics...")
        
        # Calculate distributions for real and generated trajectories
        real_dist = self._calculate_distance_distribution(self.real_trajs)
        gen_dist = self._calculate_distance_distribution(self.generated_trajs)
        
        real_dur = self._calculate_duration_distribution(self.real_trajs)
        gen_dur = self._calculate_duration_distribution(self.generated_trajs)
        
        real_rad = self._calculate_radius_distribution(self.real_trajs)
        gen_rad = self._calculate_radius_distribution(self.generated_trajs)
        
        results = {}
        
        # Calculate JS divergence for each metric
        for name, real_vals, gen_vals in [
            ("Distance", real_dist, gen_dist), 
                                          ("Duration", real_dur, gen_dur), 
            ("Radius", real_rad, gen_rad)
        ]:
            if not real_vals or not gen_vals:
                print(f"⚠️ Warning: Empty {name} distribution, skipping")
                continue
            
            # Create histogram bins
            real_max = np.max(real_vals)
            bins = np.linspace(0, real_max, 100).tolist()
            bins.append(float('inf'))
            bins = np.array(bins)
            
            # Calculate histograms
            real_hist, _ = np.histogram(real_vals, bins)
            gen_hist, _ = np.histogram(gen_vals, bins)
            
            # Calculate JS divergence using original author's implementation
            jsd = js_divergence(real_hist, gen_hist)
            results[f"{name}_JSD"] = jsd
            
            # Also store mean values for reference
            results[f"{name}_real_mean"] = np.mean(real_vals)
            results[f"{name}_gen_mean"] = np.mean(gen_vals)
            
        print("✅ Global metrics calculated.")
        return results


class LocalMetrics:
    """
    Calculate local (trajectory-level) metrics using the original author's method.
    
    Key changes:
    - Uses 0.001° grid for OD pair grouping (matches Beijing.yaml grid_size)
    - Computes grid bounds dynamically from actual road data
    - Adds EDR (Edit Distance on Real sequence) metric
    - Keeps vectorized Hausdorff from our version (faster)
    """
    def __init__(self, real_trajs, generated_trajs, geo_df, grid_size=0.001, edr_eps=100.0):
        self.real_trajs = real_trajs
        self.generated_trajs = generated_trajs
        self.geo_df = geo_df.set_index('road_id')
        self.grid_size = grid_size  # in degrees (matching Beijing.yaml)
        self.edr_eps = edr_eps  # EDR threshold in meters
        
        # Create road_id → GPS mapping dict for safe access
        self.road_gps = {}
        for road_id, row in self.geo_df.iterrows():
            try:
                center_gps = row['center_gps']
                # center_gps is a tuple like (lat, lon)
                if center_gps is not None and len(center_gps) == 2:
                    lat, lon = center_gps
                    if lat is not None and lon is not None:
                        # Store as (lon, lat) for compatibility with original code
                        self.road_gps[road_id] = (lon, lat)
            except (KeyError, TypeError, Exception) as e:
                pass  # Silently skip invalid entries
        
        # Setup dynamic grid bounds from actual data
        self._setup_grid()

    def _setup_grid(self):
        """
        Setup grid system with bounds computed from actual road data.
        Uses degree-based grid matching Beijing.yaml configuration.
        """
        # Get all road center coordinates from valid GPS entries
        valid_gps = [gps for gps in self.road_gps.values() if gps is not None and gps[0] is not None and gps[1] is not None]
        
        if not valid_gps:
            raise RuntimeError("🚨 FATAL: No valid GPS coordinates found for grid setup!")
        
        lons = [gps[0] for gps in valid_gps]
        lats = [gps[1] for gps in valid_gps]
        
        self.min_lon = min(lons)
        self.max_lon = max(lons)
        self.min_lat = min(lats)
        self.max_lat = max(lats)
        
        # Calculate grid dimensions in degrees (matching Beijing.yaml grid_size: 0.001)
        self.img_width = math.ceil((self.max_lon - self.min_lon) / self.grid_size) + 1
        self.img_height = math.ceil((self.max_lat - self.min_lat) / self.grid_size) + 1
        
        # Convert grid size to approximate meters for display
        # At Beijing latitude (~39.9°N), 1 degree ≈ 111km
        grid_size_m = self.grid_size * 111000  # Approximate meters
        
        print(f"📐 Grid setup: {self.img_width} × {self.img_height} cells ({self.grid_size}° ≈ {grid_size_m:.0f}m resolution)")
        print(f"   Bounds: lon [{self.min_lon:.4f}, {self.max_lon:.4f}], lat [{self.min_lat:.4f}, {self.max_lat:.4f}]")

    def gps2grid(self, lon, lat):
        """
        Convert GPS coordinates to grid cell indices.
        Uses degree-based grid matching Beijing.yaml configuration.
        """
        # X: longitude-based grid cell
        x = math.floor((lon - self.min_lon) / self.grid_size)
        
        # Y: latitude-based grid cell  
        y = math.floor((lat - self.min_lat) / self.grid_size)
        
        # Clamp to grid bounds (no assertions, just clamp)
        x = max(0, min(x, self.img_width - 1))
        y = max(0, min(y, self.img_height - 1))
        
        return x, y

    def _get_od_key(self, origin_rid, dest_rid):
        """
        Get OD pair key using grid system.
        This matches the original author's approach.
        """
        o_lon, o_lat = self.road_gps[origin_rid]
        d_lon, d_lat = self.road_gps[dest_rid]
        
        o_rid_x, o_rid_y = self.gps2grid(o_lon, o_lat)
        d_rid_x, d_rid_y = self.gps2grid(d_lon, d_lat)
        
        # Key: (origin_grid_id, dest_grid_id)
        key = (
            o_rid_x * self.img_height + o_rid_y,
            d_rid_x * self.img_height + d_rid_y
        )
        return key

    def _group_by_od(self, trajectories):
        """Group trajectories by OD pair using grid system."""
        od_groups = {}
        for i, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            
            try:
                origin_rid = traj[0][0]
                dest_rid = traj[-1][0]
                key = self._get_od_key(origin_rid, dest_rid)
                
                if key not in od_groups:
                    od_groups[key] = []
                od_groups[key].append(i)
            except (IndexError, KeyError):
                # Road ID not in dataset, skip
                continue
        
        return od_groups

    def _get_coord_traj(self, trajectory):
        """Extract GPS coordinates from trajectory."""
        road_ids = [p[0] for p in trajectory]
        # Return as (lat, lon) for haversine compatibility
        return np.array([(self.road_gps[rid][1], self.road_gps[rid][0]) 
                         for rid in road_ids])

    def _calculate_hausdorff_haversine(self, u, v):
        """
        Calculates the Hausdorff distance between two trajectories u and v
        using the Haversine distance. Vectorized implementation (kept from our version).
        u and v are numpy arrays of shape (n, 2) and (m, 2) of (lat, lon) points.
        """
        # haversine_vector with comb=True creates a pairwise distance matrix
        dist_matrix = haversine_vector(list(u), list(v), comb=True)

        # h(u, v) = max(min(dist_matrix, axis=1))
        h_u_v = np.max(np.min(dist_matrix, axis=1))

        # h(v, u) = max(min(dist_matrix, axis=0))
        h_v_u = np.max(np.min(dist_matrix, axis=0))

        return max(h_u_v, h_v_u)

    def _calculate_dtw_polars(self, traj1_coords, traj2_coords):
        """
        Calculate DTW distance using fastdtw.
        
        Args:
            traj1_coords: List of (lat, lon) tuples for trajectory 1
            traj2_coords: List of (lat, lon) tuples for trajectory 2
            
        Returns:
            DTW distance in kilometers
        """
        # Use fastdtw directly to avoid polars-ts compatibility issues
        dist, _ = fastdtw(traj1_coords, traj2_coords, dist=haversine)
        return dist

    def _calculate_edr(self, t0, t1, eps=100):
        """
        Calculate Edit Distance on Real sequence (EDR).
        From original author's implementation.
        
        Args:
            t0, t1: Trajectories as arrays of (lat, lon) points
            eps: Distance threshold in meters (default 100m)
        
        Returns:
            EDR normalized by max trajectory length
        """
        rad = math.pi / 180.0
        R = 6378137.0  # Earth radius in meters
        
        def great_circle_distance(lon1, lat1, lon2, lat2):
            dlat = rad * (lat2 - lat1)
            dlon = rad * (lon2 - lon1)
            a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
                 math.cos(rad * lat1) * math.cos(rad * lat2) *
                 math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            d = R * c
            return d
        
        n0 = len(t0)
        n1 = len(t1)
        C = np.full((n0 + 1, n1 + 1), np.inf)
        C[:, 0] = np.arange(n0 + 1)
        C[0, :] = np.arange(n1 + 1)

        for i in range(1, n0 + 1):
            for j in range(1, n1 + 1):
                # t0 and t1 are (lat, lon) from _get_coord_traj, need to pass (lon, lat) to distance function
                if great_circle_distance(
                    t0[i - 1][1], t0[i - 1][0],  # lon, lat from t0
                    t1[j - 1][1], t1[j - 1][0]   # lon, lat from t1
                ) < eps:
                    subcost = 0
                else:
                    subcost = 1
                C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
        
        edr = float(C[n0][n1]) / max([n0, n1])
        return edr

    def evaluate(self):
        """
        Evaluate local metrics: Hausdorff, DTW, EDR.
        Uses grid-based OD pair matching (original author's method).
        """
        print("📊 Calculating local metrics...")
        real_od_groups = self._group_by_od(self.real_trajs)
        gen_od_groups = self._group_by_od(self.generated_trajs)

        hausdorff_scores = []
        dtw_scores = []
        edr_scores = []
        
        num_matched_od = 0
        num_total_gen_od = len(gen_od_groups)
        
        for od_pair, gen_indices in tqdm(gen_od_groups.items(), desc="Comparing OD pairs"):
            if od_pair not in real_od_groups:
                continue

            num_matched_od += 1
            real_indices = real_od_groups[od_pair]
            
            # Compare each generated trajectory with real trajectories from same OD pair
            # Use min(len(real), len(gen)) comparisons per OD pair (original author's method)
            for i in range(min(len(real_indices), len(gen_indices))):
                real_idx = real_indices[i]
                gen_idx = gen_indices[i]
                
                real_traj_coords = self._get_coord_traj(self.real_trajs[real_idx])
                gen_traj_coords = self._get_coord_traj(self.generated_trajs[gen_idx])
                
                # Hausdorff (km) - vectorized version
                h_dist = self._calculate_hausdorff_haversine(gen_traj_coords, real_traj_coords)
                hausdorff_scores.append(h_dist)

                # DTW (km) - using polars-ts for better performance
                dtw_dist = self._calculate_dtw_polars(gen_traj_coords, real_traj_coords)
                dtw_scores.append(dtw_dist)
                
                # EDR (unitless, 0-1)
                edr = self._calculate_edr(real_traj_coords, gen_traj_coords, eps=self.edr_eps)
                edr_scores.append(edr)

        results = {
            "Hausdorff_km": np.mean(hausdorff_scores) if hausdorff_scores else 0,
            "DTW_km": np.mean(dtw_scores) if dtw_scores else 0,
            "EDR": np.mean(edr_scores) if edr_scores else 0,
            "matched_od_pairs": num_matched_od,
            "total_generated_od_pairs": num_total_gen_od,
        }
        
        print(f"✅ Local metrics calculated ({num_matched_od}/{num_total_gen_od} OD pairs matched).")
        return results

# --- Main Execution ---

def find_generated_file(directory):
    """Finds the newest generated trajectory file in the directory by timestamp in filename."""
    import re
    from datetime import datetime
    
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$')
    
    # Collect all CSV files that match the timestamp pattern
    timestamped_files = []
    other_csv_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename not in ['train.csv', 'val.csv', 'test.csv', 'road_id_mapping.csv']:
            match = timestamp_pattern.match(filename)
            if match:
                try:
                    # Parse timestamp from filename
                    timestamp_str = match.group(1)
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                    timestamped_files.append((timestamp, filename))
                except ValueError:
                    # If timestamp parsing fails, treat as other CSV
                    other_csv_files.append(filename)
            else:
                other_csv_files.append(filename)
    
    # Return the newest timestamped file if any exist
    if timestamped_files:
        timestamped_files.sort(key=lambda x: x[0], reverse=True)  # Sort by timestamp, newest first
        newest_file = timestamped_files[0][1]
        return os.path.join(directory, newest_file)
    
    # Fallback to any other CSV file
    if other_csv_files:
        return os.path.join(directory, other_csv_files[0])
    
    return None

def evaluate_trajectories_programmatic(
    generated_file: str,
    dataset: str = "Beijing",
    od_source: str = "test",
    grid_size: float = 0.001,
    edr_eps: float = 100.0,
    enable_wandb: bool = False,
    wandb_project: str = None,
    wandb_run_name: str = None,
    wandb_tags: list = None
) -> dict:
    """
    Programmatic interface for trajectory evaluation.
    
    Args:
        generated_file: Path to generated trajectory CSV file
        dataset: Dataset name (e.g., 'Beijing')
        od_source: 'train' or 'test' - which dataset to use for real trajectories
        grid_size: Grid size in degrees for OD pair matching
        edr_eps: EDR threshold in meters
        enable_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        wandb_tags: WandB tags
    
    Returns:
        Dictionary containing evaluation results
    """
    from datetime import datetime
    import os
    import json
    
    # Set up data paths (handle symlink)
    data_dir = Path(f'../data/{dataset}')
    if data_dir.is_symlink():
        data_dir = data_dir.resolve()
    
    # Use appropriate dataset based on OD source
    real_path = data_dir / f'{od_source}.csv'
    geo_path = data_dir / 'roadmap.geo'
    
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real data not found: {real_path}")
    if not os.path.exists(geo_path):
        raise FileNotFoundError(f"Road network not found: {geo_path}")
    
    # Load data
    geo_df = load_road_network(geo_path)
    max_road_id = geo_df['road_id'].max()
    
    real_trajectories = load_trajectories(real_path, is_real_data=True, max_road_id=max_road_id)
    generated_trajectories = load_trajectories(generated_file, is_real_data=False, max_road_id=max_road_id)
    
    print(f"Loaded {len(real_trajectories)} real and {len(generated_trajectories)} generated trajectories")
    
    # Run evaluations
    global_metrics = GlobalMetrics(real_trajectories, generated_trajectories, geo_df).evaluate()
    local_metrics = LocalMetrics(real_trajectories, generated_trajectories, geo_df, 
                                grid_size=grid_size, edr_eps=edr_eps).evaluate()

    # Combine and display results
    all_results = {**global_metrics, **local_metrics}
    
    # Add metadata about the evaluation
    all_results["metadata"] = {
        "generated_file": str(generated_file),
        "real_data_file": str(real_path),
        "road_network_file": str(geo_path),
        "od_source": od_source,
        "evaluation_timestamp": datetime.now().isoformat(),
        "real_trajectories_count": len(real_trajectories),
        "generated_trajectories_count": len(generated_trajectories),
        "grid_size": grid_size,
        "edr_eps": edr_eps,
    }

    print("\n--- Evaluation Results ---")
    for metric, value in all_results.items():
        if metric != "metadata" and isinstance(value, float):
            print(f"{metric:<20} {value:.4f}")
        elif metric != "metadata":
            print(f"{metric:<20} {value}")
    print("--------------------------\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'./eval/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'results.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Results saved to {results_file}")

    # Optional: log to Weights & Biases
    if enable_wandb:
        run_name = wandb_run_name or f"eval-{os.path.basename(generated_file)}-{timestamp}"
        
        # Build config
        wandb_config = {
            'generated_file': generated_file,
            'real_data_file': real_path,
            'road_network_file': geo_path,
            'evaluation_timestamp': timestamp,
            'real_trajectories_count': len(real_trajectories),
            'generated_trajectories_count': len(generated_trajectories),
            'grid_size': grid_size,
            'edr_eps': edr_eps,
        }
        
        wandb.init(project=wandb_project or 'hoser-eval', name=run_name, tags=wandb_tags or ['eval'], config=wandb_config)
        
        # Log scalar metrics
        log_payload = {k: v for k, v in all_results.items() if k != 'metadata' and isinstance(v, float)}
        wandb.log(log_payload)
        
        # Save the results file as an artifact
        wandb.save(results_file)
        
        wandb.finish()
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory generation models.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory containing hoser_format folder.')
    parser.add_argument('--generated_file', type=str, help='Path to specific generated CSV file. If not provided, will search for generated files in run_dir/hoser_format/')
    parser.add_argument('--wandb', action='store_true', help='Log results to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name (auto-detected from run metadata if not specified)')
    parser.add_argument('--wandb_run_name', type=str, default='', help='WandB run name (optional)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=['eval'], help='WandB tags')
    parser.add_argument('--grid_size', type=float, default=0.001, help='Grid size in degrees for OD pair matching (default: 0.001 from Beijing.yaml)')
    parser.add_argument('--edr_eps', type=float, default=100.0, help='EDR threshold in meters (default: 100.0)')
    args = parser.parse_args()
    
    # Detect WandB metadata early (we'll use it even if --wandb is not set, for reference)
    wandb_metadata = detect_wandb_metadata(args.run_dir, args.generated_file)
    
    # Auto-detect WandB project if not manually specified
    if args.wandb_project is None:
        args.wandb_project = wandb_metadata['project']
    elif wandb_metadata['project'] != 'hoser-eval':
        # User specified project, but we detected one too - use user's
        pass
    else:
        # User specified project explicitly, suppress auto-detection message
        print(f"📊 Using specified WandB project: {args.wandb_project}")

    # Check for hoser_format subdirectory (new format) or use run_dir directly (legacy)
    hoser_format_path = os.path.join(args.run_dir, 'hoser_format')
    if os.path.isdir(hoser_format_path):
        data_dir = hoser_format_path
        print(f"📂 Using hoser_format directory: {hoser_format_path}")
    elif os.path.exists(os.path.join(args.run_dir, 'test.csv')):
        data_dir = args.run_dir
        print(f"📂 Using data directory directly: {args.run_dir}")
    else:
        print(f"❌ Error: Neither 'hoser_format' subdirectory nor dataset files found in {args.run_dir}")
        return

    # Define paths based on the data directory
    real_path = os.path.join(data_dir, 'test.csv')
    geo_path = os.path.join(data_dir, 'roadmap.geo')
    
    # Use provided generated file or search for one
    if args.generated_file:
        generated_path = args.generated_file
        if not os.path.exists(generated_path):
            print(f"❌ Error: Generated file not found: {generated_path}")
            return
        print(f"📂 Using provided generated file: {generated_path}")
    else:
        generated_path = find_generated_file(data_dir)
        if not generated_path:
            print(f"❌ Error: No generated trajectory CSV file found in {data_dir}")
            return
        print(f"📂 Found generated file: {os.path.basename(generated_path)}")

    # Create output directory inside the run_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(args.run_dir, f'eval_{timestamp}')
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    geo_df = load_road_network(geo_path)
    
    # Get max road ID for validation
    max_road_id = geo_df['road_id'].max()
    print(f"🔍 Max road ID in network: {max_road_id}")
    
    real_trajectories = load_trajectories(real_path, is_real_data=True, max_road_id=max_road_id)
    generated_trajectories = load_trajectories(generated_path, is_real_data=False, max_road_id=max_road_id)
    
    # Run evaluations
    global_metrics = GlobalMetrics(real_trajectories, generated_trajectories, geo_df).evaluate()
    local_metrics = LocalMetrics(real_trajectories, generated_trajectories, geo_df, 
                                grid_size=args.grid_size, edr_eps=args.edr_eps).evaluate()

    # Combine and display results
    all_results = {**global_metrics, **local_metrics}
    
    # Add metadata about the evaluation including WandB run traceability
    all_results["metadata"] = {
        "run_directory": args.run_dir,
        "generated_file": generated_path,
        "real_data_file": real_path,
        "road_network_file": geo_path,
        "evaluation_timestamp": datetime.now().isoformat(),
        "real_trajectories_count": len(real_trajectories),
        "generated_trajectories_count": len(generated_trajectories),
        "training_run_id": wandb_metadata['training_run_id'],
        "training_run_name": wandb_metadata['training_run_name'],
        "generation_run_id": wandb_metadata['generation_run_id'],
        "generation_run_name": wandb_metadata['generation_run_name'],
    }

    print("\n--- Evaluation Results ---")
    for metric, value in all_results.items():
        if metric != "metadata" and isinstance(value, float):
            print(f"{metric:<20} {value:.4f}")
        elif metric != "metadata":
            print(f"{metric:<20} {value}")
    print("--------------------------\n")
    
    # Save results
    results_file = os.path.join(output_path, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Results saved to {results_file}")
    print(f"Generated file: {os.path.basename(generated_path)}")
    print(f"Run directory: {args.run_dir}")

    # Optional: log to Weights & Biases
    if args.wandb:
        run_name = args.wandb_run_name or f"eval-{os.path.basename(args.run_dir)}-{timestamp}"
        
        # Build config with run traceability
        wandb_config = {
            'run_directory': args.run_dir,
            'generated_file': generated_path,
            'real_data_file': real_path,
            'road_network_file': geo_path,
            'evaluation_timestamp': timestamp,
            'real_trajectories_count': len(real_trajectories),
            'generated_trajectories_count': len(generated_trajectories),
            # Add traceability to source runs
            'training_run_id': wandb_metadata['training_run_id'],
            'training_run_name': wandb_metadata['training_run_name'],
            'generation_run_id': wandb_metadata['generation_run_id'],
            'generation_run_name': wandb_metadata['generation_run_name'],
        }
        
        wandb.init(project=args.wandb_project, name=run_name, tags=args.wandb_tags, config=wandb_config)
        
        # Log scalar metrics
        log_payload = {k: v for k, v in all_results.items() if k != 'metadata' and isinstance(v, float)}
        wandb.log(log_payload)
        
        # Save the results file as an artifact
        wandb.save(results_file)
        
        # Log summary with run IDs for easy reference
        if wandb_metadata['training_run_id']:
            print(f"🔗 Training run: {wandb_metadata['training_run_name']} (ID: {wandb_metadata['training_run_id']})")
        if wandb_metadata['generation_run_id']:
            print(f"🔗 Generation run: {wandb_metadata['generation_run_name']} (ID: {wandb_metadata['generation_run_id']})")
        
        wandb.finish()

if __name__ == '__main__':
    main()
