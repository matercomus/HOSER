import math
import multiprocessing
import os
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from haversine import haversine_vector
from shapely.geometry import LineString
import torch

from utils import get_angle


def init_shared_variables(reachable_road_id_dict, geo, road_center_gps):
    global global_reachable_road_id_dict, global_geo, global_road_center_gps
    global_reachable_road_id_dict = reachable_road_id_dict
    global_geo = geo
    global_road_center_gps = road_center_gps

def load_single_file(file_path):
    """Load a single .pt file for parallel caching."""
    return torch.load(file_path, weights_only=False)

def process_and_save_row(args):
    index, row, cache_dir = args

    rid_list = eval(row['rid_list'])
    time_list = row['time_list'].split(',')
    time_list = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in time_list]

    # Ensure all road IDs are integers
    trace_road_id = np.array([int(rid) for rid in rid_list[:-1]], dtype=np.int64)
    temporal_info = np.array([(t.hour * 60.0 + t.minute + t.second / 60.0) / 1440.0 for t in time_list[:-1]]).astype(np.float32)

    if time_list[0].weekday() >= 5:
        temporal_info *= -1.0

    trace_distance_mat = haversine_vector(global_road_center_gps[trace_road_id], global_road_center_gps[trace_road_id], 'm', comb=True).astype(np.float32)
    trace_distance_mat = np.clip(trace_distance_mat, 0.0, 1000.0) / 1000.0
    trace_time_interval_mat = np.abs(temporal_info[:, None] * 1440.0 - temporal_info * 1440.0)
    trace_time_interval_mat = np.clip(trace_time_interval_mat, 0.0, 5.0) / 5.0
    trace_len = len(trace_road_id)
    destination_road_id = int(rid_list[-1])

    candidate_road_id = np.empty(len(trace_road_id), dtype=object)
    for i, road_id in enumerate(trace_road_id):
        candidate_road_id[i] = np.array(global_reachable_road_id_dict[road_id], dtype=np.int64)

    metric_dis = np.empty(len(trace_road_id), dtype=object)
    for i, candidate_road_id_list in enumerate(candidate_road_id):
        if len(candidate_road_id_list) == 0:
            # Handle roads with no reachable destinations
            metric_dis[i] = np.array([], dtype=np.float32)
        else:
            metric_dis[i] = haversine_vector(global_road_center_gps[candidate_road_id_list], global_road_center_gps[destination_road_id].reshape(1, -1), 'm', comb=True).reshape(-1).astype(np.float32)
            metric_dis[i] = np.log1p((metric_dis[i] - np.min(metric_dis[i])) / 100)

    metric_angle = np.empty(len(trace_road_id), dtype=object)
    for i, (road_id, candidate_road_id_list) in enumerate(zip(trace_road_id, candidate_road_id)):
        if len(candidate_road_id_list) == 0:
            # Handle roads with no reachable destinations
            metric_angle[i] = np.array([], dtype=np.float32)
        else:
            angle1 = np.vectorize(lambda candidate: get_angle(*(eval(global_geo.loc[road_id, 'coordinates'])[-1]), *(eval(global_geo.loc[candidate, 'coordinates'])[-1])))(candidate_road_id_list)
            angle2 = get_angle(*(eval(global_geo.loc[road_id, 'coordinates'])[-1]), *(eval(global_geo.loc[destination_road_id, 'coordinates'])[-1]))
            angle = np.abs(angle1 - angle2).astype(np.float32)
            angle = np.where(angle > math.pi, 2 * math.pi - angle, angle) / math.pi
            metric_angle[i] = angle

    candidate_len = np.array([len(candidate_road_id_list) for candidate_road_id_list in candidate_road_id])

    road_label = np.array([global_reachable_road_id_dict[int(rid_list[i])].index(int(rid_list[i + 1])) for i in range(len(trace_road_id))])
    timestamp_label = np.array([(time_list[i+1] - time_list[i]).total_seconds() for i in range(len(trace_road_id))]).astype(np.float32)

    data_to_save = {
        'trace_road_id': trace_road_id,
        'temporal_info': temporal_info,
        'trace_distance_mat': trace_distance_mat,
        'trace_time_interval_mat': trace_time_interval_mat,
        'trace_len': trace_len,
        'destination_road_id': destination_road_id,
        'candidate_road_id': candidate_road_id,
        'metric_dis': metric_dis,
        'metric_angle': metric_angle,
        'candidate_len': candidate_len,
        'road_label': road_label,
        'timestamp_label': timestamp_label,
    }
    
    output_path = os.path.join(cache_dir, f'data_{index}.pt')
    torch.save(data_to_save, output_path)

    return timestamp_label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, geo_file, rel_file, traj_file):
        cache_dir = os.path.splitext(traj_file)[0] + '_cache'
        stats_file = os.path.join(cache_dir, 'stats.json')

        if not os.path.exists(cache_dir) or not os.listdir(cache_dir) or not os.path.exists(stats_file):
            print(f"Cache not found or incomplete for {traj_file}. Preprocessing...")
            os.makedirs(cache_dir, exist_ok=True)
            
            geo = pd.read_csv(geo_file)
            rel = pd.read_csv(rel_file)
            traj = pd.read_csv(traj_file)

            road_center_gps = []
            for _, row in geo.iterrows():
                coordinates = eval(row['coordinates'])
                road_line = LineString(coordinates=coordinates)
                center_coord = road_line.centroid
                road_center_gps.append((center_coord.y, center_coord.x))
            road_center_gps = np.array(road_center_gps)

            reachable_road_id_dict = dict()
            num_roads = len(geo)
            for i in range(num_roads):
                reachable_road_id_dict[i] = []
            for _, row in rel.iterrows():
                origin_id = int(row['origin_id'])
                destination_id = int(row['destination_id'])
                reachable_road_id_dict[origin_id].append(destination_id)

            all_timestamp_labels = []
            with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_shared_variables, initargs=(reachable_road_id_dict, geo, road_center_gps)) as pool:
                tasks = [(index, row, cache_dir) for index, row in traj.iterrows()]
                for labels in tqdm(pool.imap_unordered(process_and_save_row, tasks), total=len(traj), desc=f'Preprocessing {os.path.basename(traj_file)}'):
                    all_timestamp_labels.extend(labels)

            timestamp_label_array = np.array(all_timestamp_labels, dtype=np.float32)
            mean = np.log1p(timestamp_label_array).mean()
            std = np.log1p(timestamp_label_array).std()
            with open(stats_file, 'w') as f:
                json.dump({'mean': mean.item(), 'std': std.item()}, f)

            self.timestamp_label_log1p_mean = mean
            self.timestamp_label_log1p_std = std
        else:
            print(f"Loading preprocessed data from cache {cache_dir}")
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            self.timestamp_label_log1p_mean = stats['mean']
            self.timestamp_label_log1p_std = stats['std']

        self.file_paths = sorted(
            [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pt')],
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )
        
        # Smart caching: estimate RAM requirement and available memory
        import psutil
        
        # Sample a few files to estimate average size
        sample_size = min(100, len(self.file_paths))
        sample_bytes = sum(os.path.getsize(f) for f in self.file_paths[:sample_size])
        avg_file_size = sample_bytes / sample_size
        estimated_ram_gb = (avg_file_size * len(self.file_paths)) / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        self.cached_data = None
        # Cache if dataset fits in 60% of available RAM
        if estimated_ram_gb < (available_ram_gb * 0.6):
            print(f"📦 Caching {len(self.file_paths):,} samples to RAM (~{estimated_ram_gb:.1f}GB, {available_ram_gb:.1f}GB available)")
            
            # Parallel loading with all CPU cores
            num_workers = multiprocessing.cpu_count()
            print(f"🚀 Using {num_workers} cores for parallel loading...")
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                self.cached_data = list(tqdm(
                    pool.imap(load_single_file, self.file_paths),
                    total=len(self.file_paths),
                    desc="Loading cache"
                ))
            
            print(f"✅ Cached {len(self.cached_data):,} samples in memory")
        else:
            print(f"📁 Dataset needs ~{estimated_ram_gb:.1f}GB but only {available_ram_gb:.1f}GB available — streaming from disk")
            print(f"   To enable caching, free up more RAM or reduce dataset size")

    def __len__(self):
        return len(self.file_paths) if self.cached_data is None else len(self.cached_data)
    
    def __getitem__(self, i):
        if self.cached_data is not None:
            data = self.cached_data[i]
        else:
            # Stream from disk with explicit file handle management
            with open(self.file_paths[i], 'rb') as f:
                data = torch.load(f, weights_only=False)

        # Handle missing grid tokens (for backward compatibility)
        trace_grid_token = data.get('trace_grid_token', None)
        candidate_grid_token = data.get('candidate_grid_token', None)

        return (
            data['trace_road_id'],
            data['temporal_info'],
            data['trace_distance_mat'],
            data['trace_time_interval_mat'],
            data['trace_len'],
            data['destination_road_id'],
            data['candidate_road_id'],
            data['metric_dis'],
            data['metric_angle'],
            data['candidate_len'],
            data['road_label'],
            data['timestamp_label'],
            trace_grid_token,
            candidate_grid_token,
        )

    def get_stats(self):
        return {
            'mean': self.timestamp_label_log1p_mean,
            'std': self.timestamp_label_log1p_std
        }