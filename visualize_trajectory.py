import pandas as pd
import json
import matplotlib.pyplot as plt

# Load the roadmap
geo = pd.read_csv('data/Beijing/roadmap.geo')
generated = pd.read_csv('gene/Beijing/seed0/2025-09-05_15-20-40.csv')

# Example: Plot the first trajectory
first_traj = eval(generated.iloc[0]['gene_trace_road_id'])
# Remove np.int64() wrapper if present
road_ids = [int(str(rid).split('(')[-1].split(')')[0]) if 'int64' in str(rid) else rid for rid in first_traj]

plt.figure(figsize=(10, 8))

# Plot all roads in grey
for _, road in geo.iterrows():
    coords = json.loads(road['coordinates'])
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    plt.plot(lons, lats, 'grey', alpha=0.3, linewidth=0.5)

# Plot the trajectory in red
for i, road_id in enumerate(road_ids):
    road = geo[geo['geo_id'] == road_id].iloc[0]
    coords = json.loads(road['coordinates'])
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    plt.plot(lons, lats, 'red', linewidth=3, label=f'Segment {i+1}')
    
    # Mark start and end
    if i == 0:
        plt.scatter(lons[0], lats[0], color='green', s=100, zorder=5, label='Start')
    if i == len(road_ids) - 1:
        plt.scatter(lons[-1], lats[-1], color='blue', s=100, zorder=5, label='End')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Generated Trajectory: Roads {road_ids}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
