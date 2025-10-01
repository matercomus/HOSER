import math
import multiprocessing
import os
import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from haversine import haversine_vector
from shapely.geometry import LineString
from tqdm import tqdm

from utils import get_angle


def init_shared_variables(reachable_road_id_dict, geo, road_center_gps):
    global global_reachable_road_id_dict, global_geo, global_road_center_gps
    global_reachable_road_id_dict = reachable_road_id_dict
    global_geo = geo
    global_road_center_gps = road_center_gps


def process_and_save_row(args):
    index, row, cache_dir = args

    rid_list = eval(row['rid_list'])
    time_list = row['time_list'].split(',')
    time_list = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in time_list]

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
            metric_dis[i] = np.array([], dtype=np.float32)
        else:
            metric_dis[i] = haversine_vector(global_road_center_gps[candidate_road_id_list], global_road_center_gps[destination_road_id].reshape(1, -1), 'm', comb=True).reshape(-1).astype(np.float32)
            metric_dis[i] = np.log1p((metric_dis[i] - np.min(metric_dis[i])) / 100)

    metric_angle = np.empty(len(trace_road_id), dtype=object)
    for i, (road_id, candidate_road_id_list) in enumerate(zip(trace_road_id, candidate_road_id)):
        if len(candidate_road_id_list) == 0:
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

            reachable_road_id_dict = {i: [] for i in range(len(geo))}
            for _, row in rel.iterrows():
                reachable_road_id_dict[int(row['origin_id'])].append(int(row['destination_id']))

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

        file_paths = sorted(
            [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pt')],
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )

        if not file_paths:
            raise RuntimeError(f"No cached samples found in {cache_dir}")

        raw_samples: List[dict] = []
        self.max_trace_len = 0
        self.max_candidate_len = 0
        has_trace_grid_token = False
        has_candidate_grid_token = False

        for path in tqdm(file_paths, desc='Loading cached samples'):
            sample = torch.load(path, weights_only=False)
            raw_samples.append(sample)
            trace_len = int(sample['trace_len'])
            self.max_trace_len = max(self.max_trace_len, trace_len)
            cand_len_arr = sample['candidate_len']
            if cand_len_arr.size > 0:
                self.max_candidate_len = max(self.max_candidate_len, int(cand_len_arr.max()))
            if sample.get('trace_grid_token') is not None:
                has_trace_grid_token = True
            if sample.get('candidate_grid_token') is not None:
                has_candidate_grid_token = True

        N = len(raw_samples)
        T = self.max_trace_len
        C = max(self.max_candidate_len, 1)

        self.trace_road_id = torch.zeros((N, T), dtype=torch.long)
        self.temporal_info = torch.zeros((N, T), dtype=torch.float32)
        self.trace_distance_mat = torch.zeros((N, T, T), dtype=torch.float32)
        self.trace_time_interval_mat = torch.zeros((N, T, T), dtype=torch.float32)
        self.trace_len = torch.zeros(N, dtype=torch.long)
        self.destination_road_id = torch.zeros(N, dtype=torch.long)
        self.candidate_road_id = torch.zeros((N, T, C), dtype=torch.long)
        self.metric_dis = torch.zeros((N, T, C), dtype=torch.float32)
        self.metric_angle = torch.zeros((N, T, C), dtype=torch.float32)
        self.candidate_len = torch.zeros((N, T), dtype=torch.long)
        self.road_label = torch.full((N, T), -100, dtype=torch.long)
        self.timestamp_label = torch.zeros((N, T), dtype=torch.float32)

        self.trace_grid_token = torch.zeros((N, T), dtype=torch.long) if has_trace_grid_token else None
        self.candidate_grid_token = torch.zeros((N, T, C), dtype=torch.long) if has_candidate_grid_token else None

        for idx, sample in enumerate(raw_samples):
            trace_len = int(sample['trace_len'])
            self.trace_len[idx] = trace_len
            self.destination_road_id[idx] = int(sample['destination_road_id'])

            self.trace_road_id[idx, :trace_len] = torch.as_tensor(sample['trace_road_id'], dtype=torch.long)
            self.temporal_info[idx, :trace_len] = torch.as_tensor(sample['temporal_info'], dtype=torch.float32)
            self.trace_distance_mat[idx, :trace_len, :trace_len] = torch.as_tensor(sample['trace_distance_mat'], dtype=torch.float32)
            self.trace_time_interval_mat[idx, :trace_len, :trace_len] = torch.as_tensor(sample['trace_time_interval_mat'], dtype=torch.float32)
            self.road_label[idx, :trace_len] = torch.as_tensor(sample['road_label'], dtype=torch.long)
            self.timestamp_label[idx, :trace_len] = torch.as_tensor(sample['timestamp_label'], dtype=torch.float32)

            cand_len_arr = sample['candidate_len']
            if cand_len_arr.size > 0:
                self.candidate_len[idx, :trace_len] = torch.as_tensor(cand_len_arr, dtype=torch.long)

            cand_ids = sample['candidate_road_id']
            dis_vals = sample['metric_dis']
            angle_vals = sample['metric_angle']
            for step in range(trace_len):
                cand = cand_ids[step]
                cand_len = len(cand)
                if cand_len == 0:
                    continue
                self.candidate_road_id[idx, step, :cand_len] = torch.as_tensor(cand, dtype=torch.long)
                self.metric_dis[idx, step, :cand_len] = torch.as_tensor(dis_vals[step], dtype=torch.float32)
                self.metric_angle[idx, step, :cand_len] = torch.as_tensor(angle_vals[step], dtype=torch.float32)

            if self.trace_grid_token is not None:
                trace_grid = sample.get('trace_grid_token')
                if trace_grid is not None:
                    self.trace_grid_token[idx, :trace_len] = torch.as_tensor(trace_grid, dtype=torch.long)

            if self.candidate_grid_token is not None:
                candidate_grid = sample.get('candidate_grid_token')
                if candidate_grid is not None:
                    for step in range(trace_len):
                        cand_grid = candidate_grid[step]
                        cand_len = len(cand_grid)
                        if cand_len == 0:
                            continue
                        self.candidate_grid_token[idx, step, :cand_len] = torch.as_tensor(cand_grid, dtype=torch.long)

        del raw_samples

    def __len__(self):
        return self.trace_road_id.size(0)

    def __getitem__(self, i):
        trace_grid = None if self.trace_grid_token is None else self.trace_grid_token[i]
        candidate_grid = None if self.candidate_grid_token is None else self.candidate_grid_token[i]

        return (
            self.trace_road_id[i],
            self.temporal_info[i],
            self.trace_distance_mat[i],
            self.trace_time_interval_mat[i],
            self.trace_len[i],
            self.destination_road_id[i],
            self.candidate_road_id[i],
            self.metric_dis[i],
            self.metric_angle[i],
            self.candidate_len[i],
            self.road_label[i],
            self.timestamp_label[i],
            trace_grid,
            candidate_grid,
        )

    def get_stats(self):
        return {
            'mean': self.timestamp_label_log1p_mean,
            'std': self.timestamp_label_log1p_std
        }
