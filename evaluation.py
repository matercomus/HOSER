import argparse
import os
import json
import ast
from datetime import datetime
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw
from haversine import haversine, haversine_vector
import wandb

# --- Data Loading ---

def load_road_network(geo_path):
    """Loads road network data from geo files, handling optional header."""
    print("📂 Loading road network...")
    
    try:
        # First, try to read with a header
        geo_df = pd.read_csv(geo_path)
        if 'geo_id' in geo_df.columns:
            geo_df = geo_df.rename(columns={'geo_id': 'road_id'})
        elif 'road_id' not in geo_df.columns: # If geo_id is not there, maybe road_id is
             raise ValueError("Header malformed")
    except (ValueError, pd.errors.ParserError):
        # If that fails, read without a header and assign names
        col_names = ['road_id', 'type', 'coordinates', 'highway', 'oneway', 'length', 'name', 'lanes', 'bridge', 'access', 'maxspeed', 'ref', 'tunnel', 'junction', 'width']
        geo_df = pd.read_csv(geo_path, header=None, names=col_names)

    # Pre-calculate road center GPS coordinates
    road_center_gps = []
    for _, row in geo_df.iterrows():
        # The coordinates are stored as a string, so they need to be parsed.
        # Using json.loads is safer than eval().
        try:
            coordinates = json.loads(row['coordinates'])
            road_line = LineString(coordinates=coordinates)
            center_coord = road_line.centroid
            road_center_gps.append((center_coord.y, center_coord.x))
        except (json.JSONDecodeError, TypeError):
            print(f"⚠️ Warning: Could not parse coordinates for road_id {row['road_id']}. Skipping.")
            road_center_gps.append((None, None))

    geo_df['center_gps'] = road_center_gps
    geo_df = geo_df.dropna(subset=['center_gps'])
    
    print("✅ Road network loaded.")
    return geo_df

def load_trajectories(traj_path, is_real_data):
    """Loads trajectories from a CSV file."""
    print(f"📂 Loading trajectories from {traj_path}...")
    traj_df = pd.read_csv(traj_path)
    
    trajectories = []
    if is_real_data:
        # Handles real data from files like test.csv
        for _, row in traj_df.iterrows():
            # rid_list is a comma-separated string like "1,2,3"
            rids = [int(r) for r in str(row['rid_list']).split(',')]
            timestamps = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in row['time_list'].split(',')]
            trajectories.append(list(zip(rids, timestamps)))
    else:  # Generated data - new format
        for _, row in traj_df.iterrows():
            # gene_trace_road_id is a string like "[2977, 2979, 26496, ...]"
            rid_str = str(row['gene_trace_road_id'])
            rids = json.loads(rid_str)

            # gene_trace_datetime is a string representation of a list of strings
            datetime_list_str = ast.literal_eval(row['gene_trace_datetime'])
            timestamps = [datetime.strptime(t.strip('"'), '%Y-%m-%dT%H:%M:%SZ') for t in datetime_list_str]
            trajectories.append(list(zip(rids, timestamps)))
            
    print(f"✅ Loaded {len(trajectories)} trajectories.")
    return trajectories

# --- Metric Calculation ---

class GlobalMetrics:
    def __init__(self, real_trajs, generated_trajs, geo_df):
        self.real_trajs = real_trajs
        self.generated_trajs = generated_trajs
        self.geo_df = geo_df.set_index('road_id')

    def _calculate_features(self, trajectories):
        distances = []
        durations = []
        radii = []

        for traj in tqdm(trajectories, desc="Calculating global features"):
            if len(traj) < 2:
                continue
            
            # Duration
            duration = (traj[-1][1] - traj[0][1]).total_seconds() / 60.0  # in minutes
            durations.append(duration)

            # Distance
            road_ids = [p[0] for p in traj]
            distance = self.geo_df.loc[road_ids]['length'].sum() / 1000.0 # in km
            distances.append(distance)
            
            # Radius of Gyration
            points = np.array([self.geo_df.loc[rid]['center_gps'] for rid in road_ids])
            center_of_mass = np.mean(points, axis=0)
            radius = np.sqrt(np.mean(np.sum((points - center_of_mass)**2, axis=1)))
            radii.append(radius)
            
        return distances, durations, radii

    def evaluate(self):
        print("📊 Calculating global metrics...")
        real_dist, real_dur, real_rad = self._calculate_features(self.real_trajs)
        gen_dist, gen_dur, gen_rad = self._calculate_features(self.generated_trajs)
        
        results = {}
        
        for name, real_vals, gen_vals in [("Distance", real_dist, gen_dist), 
                                          ("Duration", real_dur, gen_dur), 
                                          ("Radius", real_rad, gen_rad)]:
            
            min_val = min(min(real_vals), min(gen_vals)) if real_vals and gen_vals else 0
            max_val = max(max(real_vals), max(gen_vals)) if real_vals and gen_vals else 1
            bins = np.linspace(min_val, max_val, 101)
            
            real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
            gen_hist, _ = np.histogram(gen_vals, bins=bins, density=True)
            
            # Add small constant to avoid division by zero in jensenshannon
            real_hist += 1e-10
            gen_hist += 1e-10

            jsd = jensenshannon(real_hist, gen_hist, base=2.0)
            results[f"{name} (JSD)"] = jsd
            
        print("✅ Global metrics calculated.")
        return results


class LocalMetrics:
    def __init__(self, real_trajs, generated_trajs, geo_df, grid_size=200):
        self.real_trajs = real_trajs
        self.generated_trajs = generated_trajs
        self.geo_df = geo_df.set_index('road_id')
        self.grid_size = grid_size # in meters
        self._setup_grid()

    def _setup_grid(self):
        # Using a simple lat/lon grid. For more accuracy, projection would be needed.
        centers = np.array(self.geo_df['center_gps'].tolist())
        self.min_lat, self.min_lon = centers.min(axis=0)
        self.max_lat, self.max_lon = centers.max(axis=0)
        
        # Approximate conversion from meters to degrees
        lat_degree_per_m = 1 / 111111
        lon_degree_per_m = 1 / (111111 * np.cos(np.deg2rad(self.min_lat)))
        
        self.lat_step = self.grid_size * lat_degree_per_m
        self.lon_step = self.grid_size * lon_degree_per_m

        self.lat_bins = int(np.ceil((self.max_lat - self.min_lat) / self.lat_step))
        self.lon_bins = int(np.ceil((self.max_lon - self.min_lon) / self.lon_step))

    def _get_grid_id(self, road_id):
        lat, lon = self.geo_df.loc[road_id]['center_gps']
        lat_idx = int((lat - self.min_lat) / self.lat_step)
        lon_idx = int((lon - self.min_lon) / self.lon_step)
        return lat_idx * self.lon_bins + lon_idx

    def _group_by_od(self, trajectories):
        od_groups = {}
        for i, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            origin_grid = self._get_grid_id(traj[0][0])
            dest_grid = self._get_grid_id(traj[-1][0])
            if (origin_grid, dest_grid) not in od_groups:
                od_groups[(origin_grid, dest_grid)] = []
            od_groups[(origin_grid, dest_grid)].append(i)
        return od_groups

    def _get_coord_traj(self, trajectory):
        road_ids = [p[0] for p in trajectory]
        return np.array([self.geo_df.loc[rid]['center_gps'] for rid in road_ids])

    def _calculate_hausdorff_haversine(self, u, v):
        """
        Calculates the Hausdorff distance between two trajectories u and v
        using the Haversine distance. Vectorized implementation.
        u and v are numpy arrays of shape (n, 2) and (m, 2) of (lat, lon) points.
        """
        # haversine_vector with comb=True creates a pairwise distance matrix
        dist_matrix = haversine_vector(list(u), list(v), comb=True)

        # h(u, v) = max(min(dist_matrix, axis=1))
        h_u_v = np.max(np.min(dist_matrix, axis=1))

        # h(v, u) = max(min(dist_matrix, axis=0))
        h_v_u = np.max(np.min(dist_matrix, axis=0))

        return max(h_u_v, h_v_u)

    def evaluate(self):
        print("📊 Calculating local metrics...")
        real_od_groups = self._group_by_od(self.real_trajs)
        gen_od_groups = self._group_by_od(self.generated_trajs)

        hausdorff_scores = []
        dtw_scores = []
        
        for od_pair, gen_indices in tqdm(gen_od_groups.items(), desc="Comparing OD pairs"):
            if od_pair not in real_od_groups:
                continue

            real_indices = real_od_groups[od_pair]
            
            for gen_idx in gen_indices:
                gen_traj_coords = self._get_coord_traj(self.generated_trajs[gen_idx])
                
                h_dists = []
                d_dists = []

                for real_idx in real_indices:
                    real_traj_coords = self._get_coord_traj(self.real_trajs[real_idx])
                    
                    # Hausdorff (km)
                    h_dist = self._calculate_hausdorff_haversine(gen_traj_coords, real_traj_coords)
                    h_dists.append(h_dist)

                    # DTW (km)
                    dist, _ = fastdtw(gen_traj_coords, real_traj_coords, dist=haversine)
                    d_dists.append(dist)

                if h_dists:
                    hausdorff_scores.append(np.mean(h_dists))
                if d_dists:
                    dtw_scores.append(np.mean(d_dists))

        results = {
            "Hausdorff (km)": np.mean(hausdorff_scores) if hausdorff_scores else 0,
            "DTW (km)": np.mean(dtw_scores) if dtw_scores else 0
        }
        
        print("✅ Local metrics calculated.")
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory generation models.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory containing hoser_format folder.')
    parser.add_argument('--generated_file', type=str, help='Path to specific generated CSV file. If not provided, will search for generated files in run_dir/hoser_format/')
    parser.add_argument('--wandb', action='store_true', help='Log results to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='hoser-eval', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default='', help='WandB run name (optional)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=['eval'], help='WandB tags')
    args = parser.parse_args()

    hoser_format_path = os.path.join(args.run_dir, 'hoser_format')
    if not os.path.isdir(hoser_format_path):
        print(f"❌ Error: 'hoser_format' directory not found in {args.run_dir}")
        return

    # Define paths based on the hoser_format directory
    real_path = os.path.join(hoser_format_path, 'test.csv')
    geo_path = os.path.join(hoser_format_path, 'roadmap.geo')
    
    # Use provided generated file or search for one
    if args.generated_file:
        generated_path = args.generated_file
        if not os.path.exists(generated_path):
            print(f"❌ Error: Generated file not found: {generated_path}")
            return
        print(f"📂 Using provided generated file: {generated_path}")
    else:
        generated_path = find_generated_file(hoser_format_path)
        if not generated_path:
            print(f"❌ Error: No generated trajectory CSV file found in {hoser_format_path}")
            return
        print(f"📂 Found generated file: {os.path.basename(generated_path)}")

    # Create output directory inside the run_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(args.run_dir, f'eval_{timestamp}')
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    geo_df = load_road_network(geo_path)
    real_trajectories = load_trajectories(real_path, is_real_data=True)
    generated_trajectories = load_trajectories(generated_path, is_real_data=False)
    
    # Run evaluations
    global_metrics = GlobalMetrics(real_trajectories, generated_trajectories, geo_df).evaluate()
    local_metrics = LocalMetrics(real_trajectories, generated_trajectories, geo_df).evaluate()

    # Combine and display results
    all_results = {**global_metrics, **local_metrics}
    
    # Add metadata about the evaluation
    all_results["metadata"] = {
        "run_directory": args.run_dir,
        "generated_file": generated_path,
        "real_data_file": real_path,
        "road_network_file": geo_path,
        "evaluation_timestamp": datetime.now().isoformat(),
        "real_trajectories_count": len(real_trajectories),
        "generated_trajectories_count": len(generated_trajectories)
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
        wandb.init(project=args.wandb_project, name=run_name, tags=args.wandb_tags, config={
            'run_directory': args.run_dir,
            'generated_file': generated_path,
            'real_data_file': real_path,
            'road_network_file': geo_path,
            'evaluation_timestamp': timestamp,
            'real_trajectories_count': len(real_trajectories),
            'generated_trajectories_count': len(generated_trajectories),
        })
        # Log scalar metrics
        log_payload = {k: v for k, v in all_results.items() if k != 'metadata' and isinstance(v, float)}
        wandb.log(log_payload)
        # Save the results file as an artifact
        wandb.save(results_file)
        wandb.finish()

if __name__ == '__main__':
    main()
