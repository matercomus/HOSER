# %%
import argparse
import os
import json
import ast
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# %%
def find_generated_file(directory):
    """Finds the generated trajectory file in the directory."""
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename not in ['train.csv', 'val.csv', 'test.csv', 'road_id_mapping.csv']:
            return os.path.join(directory, filename)
    return None

def load_road_network(geo_path):
    """Loads road network data from geo files, handling optional header."""
    print("📂 Loading road network...")
    try:
        geo_df = pd.read_csv(geo_path)
        if 'geo_id' in geo_df.columns:
            geo_df = geo_df.rename(columns={'geo_id': 'road_id'})
        elif 'road_id' not in geo_df.columns:
             raise ValueError("Header malformed, 'geo_id' or 'road_id' not found")
    except (ValueError, pd.errors.ParserError):
        col_names = ['road_id', 'type', 'coordinates', 'highway', 'oneway', 'length', 'name', 'lanes', 'bridge', 'access', 'maxspeed', 'ref', 'tunnel', 'junction', 'width']
        geo_df = pd.read_csv(geo_path, header=None, names=col_names)
    
    geometry_list = []
    for _, row in geo_df.iterrows():
        try:
            coords = json.loads(row['coordinates'])
            geometry_list.append(LineString(coords))
        except (json.JSONDecodeError, TypeError):
            geometry_list.append(None)
    
    geo_df['geometry'] = geometry_list
    geo_df = geo_df.dropna(subset=['geometry'])
    print("✅ Road network loaded.")
    return geo_df

def calculate_visit_counts(traj_path, num_roads, is_real_data):
    """Calculates road segment visit counts from a trajectory file."""
    print(f"📊 Calculating visit counts for {'Real' if is_real_data else 'Generated'} data...")
    traj_df = pd.read_csv(traj_path)
    visit_counts = np.zeros(num_roads, dtype=np.int64)

    for _, row in tqdm(traj_df.iterrows(), total=len(traj_df), desc="Processing trajectories"):
        rids = []
        try:
            if is_real_data:
                rids = [int(r) for r in str(row['rid_list']).split(',')]
            else:
                rid_str = row['gene_trace_road_id'].replace("np.int64(", "").replace(")", "")
                rids = json.loads(rid_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Warning: Skipping row due to parsing error: {e}")
            continue

        for rid in rids:
            if rid < num_roads:
                visit_counts[rid] += 1
    
    print("✅ Visit counts calculated.")
    return visit_counts

def plot_heatmap(geo_df, visit_counts, title, output_path):
    """Generates and saves a single heatmap plot."""
    print(f"🎨 Generating heatmap for {title}...")
    
    data = {'geometry': geo_df['geometry'], 'visit_count': visit_counts}
    # Align index for proper joining
    roads_gdf = gpd.GeoDataFrame(data, index=geo_df.index, crs="EPSG:4326")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    roads_gdf.plot(column='visit_count', cmap='Reds', linewidth=0.5, ax=ax, vmin=0, vmax=np.percentile(visit_counts[visit_counts>0], 99))

    ax.set_title(title, fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Heatmap saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for trajectory data.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory containing hoser_format folder.')
    args = parser.parse_args()

    hoser_format_path = os.path.join(args.run_dir, 'hoser_format')
    if not os.path.isdir(hoser_format_path):
        print(f"❌ Error: 'hoser_format' directory not found in {args.run_dir}")
        return

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.run_dir, f'heatmap_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    geo_path = os.path.join(hoser_format_path, 'roadmap.geo')
    real_traj_path = os.path.join(hoser_format_path, 'test.csv')
    generated_traj_path = find_generated_file(hoser_format_path)

    if not generated_traj_path:
        print(f"❌ Error: No generated trajectory CSV file found in {hoser_format_path}")
        return

    # Load data and calculate visit counts
    geo_df = load_road_network(geo_path)
    num_roads = geo_df['road_id'].max() + 1

    real_visit_counts = calculate_visit_counts(real_traj_path, num_roads, is_real_data=True)
    generated_visit_counts = calculate_visit_counts(generated_traj_path, num_roads, is_real_data=False)
    
    # Generate and save plots
    plot_heatmap(geo_df, real_visit_counts, "Real", os.path.join(output_dir, "Real.png"))
    plot_heatmap(geo_df, generated_visit_counts, "Generated", os.path.join(output_dir, "Generated.png"))

if __name__ == '__main__':
    main()


