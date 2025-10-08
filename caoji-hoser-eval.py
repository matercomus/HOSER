# %%
import os
import math
from datetime import datetime
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from scipy.stats import entropy
from geopy import distance
import hausdorff
from fastdtw import fastdtw

import seaborn as sns
import matplotlib.pyplot as plt

from map_manager import MapManager

sns.set_theme()

# %%
def js_divergence(p, q):
    p = p / (np.sum(p) + 1e-14)
    q = q / (np.sum(q) + 1e-14)
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

# %%
real = pd.read_csv('../../cleaned-data/new_test.csv')
geo = pd.read_csv('../../cleaned-data/roadmap.geo')

gene_data = dict()

for dir_name in os.listdir('./gene'):
    if dir_name != 'HOSER':
        continue
    gene_data[dir_name] = []
    for file_name in os.listdir(os.path.join('./gene', dir_name)):
        file_path = os.path.join('./gene', dir_name, file_name)
        gene_data[dir_name].append(file_path)

map_manager = MapManager('BJ_Taxi')

# %% [markdown]
# Calculate the center point of the road

# %%
road_gps = []
for _, row in geo.iterrows():
    coordinates = eval(row['coordinates'])
    road_line = LineString(coordinates=coordinates)
    center_coord = road_line.centroid
    center_lon, center_lat = center_coord.x, center_coord.y
    road_gps.append((center_lon, center_lat))

# %% [markdown]
# Calculate distance

# %%
real_distance = []
for _, row in real.iterrows():
    rid_list = eval(row['rid_list'])
    travel_distance = 0
    for i in range(1, len(rid_list)):
        travel_distance += distance.great_circle((road_gps[rid_list[i-1]][1], road_gps[rid_list[i-1]][0]), (road_gps[rid_list[i]][1], road_gps[rid_list[i]][0])).kilometers
    real_distance.append(travel_distance)

real_max_distance = np.max(real_distance)
distance_bins = np.linspace(0, real_max_distance, 100).tolist()
distance_bins.append(float('inf'))
distance_bins = np.array(distance_bins)
real_distance_distribution, _ = np.histogram(real_distance, distance_bins)

print(f'real average: {np.mean(real_distance):.4f}')
print('=' * 40)

for name, path_list in gene_data.items():
    value_list = []
    js_list = []

    for path in path_list:
        data = pd.read_csv(path)
        distance_list = []
        for _, row in data.iterrows():
            rid_list = eval(row['gene_trace_road_id'])
            travel_distance = 0
            for i in range(1, len(rid_list)):
                travel_distance += distance.great_circle((road_gps[rid_list[i-1]][1], road_gps[rid_list[i-1]][0]), (road_gps[rid_list[i]][1], road_gps[rid_list[i]][0])).kilometers
            distance_list.append(travel_distance)

        value_list.append(np.mean(distance_list))

        distance_distribution, _ = np.histogram(distance_list, distance_bins)
        js_list.append(js_divergence(real_distance_distribution, distance_distribution))

    value_list = np.array(value_list)
    js_list = np.array(js_list)

    print(f'{name} average: {np.mean(value_list):.4f}±{np.std(value_list):.4f}')
    print(f'{name} JS divergence: {np.mean(js_list):.4f}±{np.std(js_list):.4f}')
    print('=' * 40)

# %% [markdown]
# Calculate radius

# %%
real_radius = []
for _, row in real.iterrows():
    rid_list = eval(row['rid_list'])
    lon_mean = np.mean([road_gps[rid][0] for rid in rid_list])
    lat_mean = np.mean([road_gps[rid][1] for rid in rid_list])
    rad = []
    for rid in rid_list:
        lon = road_gps[rid][0]
        lat = road_gps[rid][1]
        dis = distance.great_circle((lat_mean, lon_mean), (lat, lon)).kilometers
        rad.append(dis)
    rad = np.mean(rad)
    real_radius.append(rad)

real_max_radius = np.max(real_radius)
radius_bins = np.linspace(0, real_max_radius, 100).tolist()
radius_bins.append(float('inf'))
radius_bins = np.array(radius_bins)
real_radius_distribution, _ = np.histogram(real_radius, radius_bins)

print(f'real average: {np.mean(real_radius):.4f}')
print('=' * 40)

for name, path_list in gene_data.items():
    value_list = []
    js_list = []

    for path in path_list:
        data = pd.read_csv(path)
        radius_list = []
        for _, row in data.iterrows():
            rid_list = eval(row['gene_trace_road_id'])
            if isinstance(rid_list, int):
                rid_list = [rid_list]
            lon_mean = np.mean([road_gps[rid][0] for rid in rid_list])
            lat_mean = np.mean([road_gps[rid][1] for rid in rid_list])
            rad = []
            for rid in rid_list:
                lon = road_gps[rid][0]
                lat = road_gps[rid][1]
                dis = distance.great_circle((lat_mean, lon_mean), (lat, lon)).kilometers
                rad.append(dis)
            rad = np.mean(rad)
            radius_list.append(rad)

        value_list.append(np.mean(radius_list))

        radius_distribution, _ = np.histogram(radius_list, radius_bins)
        js_list.append(js_divergence(real_radius_distribution, radius_distribution))

    value_list = np.array(value_list)
    js_list = np.array(js_list)

    print(f'{name} average: {np.mean(value_list):.4f}±{np.std(value_list):.4f}')
    print(f'{name} JS divergence: {np.mean(js_list):.4f}±{np.std(js_list):.4f}')
    print('=' * 40)

# %% [markdown]
# Calculate duration

# %%
real_time_duration = []
for _, row in real.iterrows():
    time_list = row['time_list'].split(',')
    time_list = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in time_list]
    for i in range(1, len(time_list)):
        time_duration = (time_list[i]-time_list[i-1]).total_seconds() / 60
        real_time_duration.append(time_duration)

real_max_time_duration = np.max(real_time_duration)
time_duration_bins = np.linspace(0, real_max_time_duration, 100).tolist()
time_duration_bins.append(float('inf'))
time_duration_bins = np.array(time_duration_bins)
real_time_duration_distribution, _ = np.histogram(real_time_duration, time_duration_bins)

print(f'real average: {np.mean(real_time_duration):.4f}')
print('=' * 40)

for name, path_list in gene_data.items():
    value_list = []
    js_list = []

    for path in path_list:
        data = pd.read_csv(path)
        time_duration = []
        for _, row in data.iterrows():
            time_list = eval(row['gene_trace_datetime'])
            for t1, t2 in zip(time_list[:-1], time_list[1:]):
                t1 = datetime.strptime(t1, '%Y-%m-%dT%H:%M:%SZ')
                t2 = datetime.strptime(t2, '%Y-%m-%dT%H:%M:%SZ')
                time_duration.append((t2 - t1).total_seconds() / 60)

        value_list.append(np.mean(time_duration))

        time_duration_distribution, _ = np.histogram(time_duration, time_duration_bins)
        js_list.append(js_divergence(real_time_duration_distribution, time_duration_distribution))

    value_list = np.array(value_list)
    js_list = np.array(js_list)

    print(f'{name} average: {np.mean(value_list):.4f}±{np.std(value_list):.4f}')
    print(f'{name} JS divergence: {np.mean(js_list):.4f}±{np.std(js_list):.4f}')
    print('=' * 40)

# %% [markdown]
# Calculate micro-level similarity

# %%
def haversine(array_x, array_y):
    R = 6378.0
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = (pow(math.sin(dlat/2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon/2.0), 2.0))
    return R * 2 * math.asin(math.sqrt(a))

rad = math.pi / 180.0
R = 6378137.0

def great_circle_distance(lon1, lat1, lon2, lat2):
    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def edr(t0, t1, eps):
    n0 = len(t0)
    n1 = len(t1)
    C = np.full((n0 + 1, n1 + 1), np.inf)
    C[:, 0] = np.arange(n0 + 1)
    C[0, :] = np.arange(n1 + 1)

    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr

# %%
real_id = dict()
for i in range(len(real)):
    rid_list = eval(real.loc[i, 'rid_list'])
    o_rid, d_rid = rid_list[0], rid_list[-1]
    o_rid_x, o_rid_y = map_manager.gps2grid(*road_gps[o_rid])
    d_rid_x, d_rid_y = map_manager.gps2grid(*road_gps[d_rid])
    key = (o_rid_x * map_manager.img_height + o_rid_y, d_rid_x * map_manager.img_height + d_rid_y)
    if key not in real_id:
        real_id[key] = [i]
    else:
        real_id[key].append(i)

# %%
for name, path_list in gene_data.items():
    hausdorff_value_list = []
    dtw_value_list = []
    edr_value_list = []

    for path in path_list:
        data = pd.read_csv(path)

        od2traj_id_gen = dict()
        for i in range(len(data)):
            gen_rid_list = eval(data.loc[i, 'gene_trace_road_id'])
            o_rid, d_rid = gen_rid_list[0], gen_rid_list[-1]
            o_rid_x, o_rid_y = map_manager.gps2grid(*road_gps[o_rid])
            d_rid_x, d_rid_y = map_manager.gps2grid(*road_gps[d_rid])
            key = (o_rid_x * map_manager.img_height + o_rid_y, d_rid_x * map_manager.img_height + d_rid_y)
            if key not in od2traj_id_gen:
                od2traj_id_gen[key] = [i]
            else:
                od2traj_id_gen[key].append(i)

        hausdorff_list = []
        dtw_list = []
        edr_list = []

        for k in od2traj_id_gen.keys():
            if k in real_id:
                for i in range(min(len(real_id[k]), len(od2traj_id_gen[k]))):
                    real_rid_list = eval(real.loc[real_id[k][i], 'rid_list'])
                    real_gps_list = [road_gps[rid][::-1] for rid in real_rid_list]
                    real_gps_list = np.array(real_gps_list)

                    gene_rid_list = eval(data.loc[od2traj_id_gen[k][i], 'gene_trace_road_id'])
                    gene_gps_list = [road_gps[rid][::-1] for rid in gene_rid_list]
                    gene_gps_list = np.array(gene_gps_list)

                    hausdorff_list.append(hausdorff.hausdorff_distance(real_gps_list, gene_gps_list, distance='haversine'))
                    dtw_list.append(fastdtw(real_gps_list, gene_gps_list, dist=haversine)[0])
                    edr_list.append(edr(real_gps_list, gene_gps_list, 100))

        hausdorff_value_list.append(np.mean(hausdorff_list))
        dtw_value_list.append(np.mean(dtw_list))
        edr_value_list.append(np.mean(edr_list))

    print(f'{name}, Hausdorff: {np.mean(hausdorff_value_list):.4f}±{np.std(hausdorff_value_list):.4f}, '
          f'DTW: {np.mean(dtw_value_list):.4f}±{np.std(dtw_value_list):.4f}, EDR: {np.mean(edr_value_list):.4f}±{np.std(edr_value_list):.4f}')
    print('=' * 40)


