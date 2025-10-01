import argparse
import json
import multiprocessing
import os
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from haversine import haversine_vector
from shapely.geometry import LineString
from tqdm import tqdm

from dataset import _chunk_list, _init_pack_worker, _pack_chunk, _scan_sample_for_metadata, META_FILENAME, AGGREGATE_FILENAME
from utils import get_angle


def init_shared_variables(reachable_road_id_dict, geo, road_center_gps):
    global global_reachable_road_id_dict, global_geo, global_road_center_gps
    global_reachable_road_id_dict = reachable_road_id_dict
    global_geo = geo
    global_road_center_gps = road_center_gps


def process_and_save_row_temp(args):
    index, row, cache_dir = args
    # This is a temporary copy of the logic in dataset.py to avoid circular imports

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
    max_candidate_len = int(candidate_len.max()) if candidate_len.size > 0 else 0

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

    return {
        'timestamp_label': timestamp_label,
        'trace_len': trace_len,
        'max_candidate_len': max_candidate_len,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Builds aggregate dataset tensors for HOSER training."
    )
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HOSER-format dataset')
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)

    for split in ['train', 'val', 'test']:
        traj = DATA_DIR / f'{split}.csv'
        if not traj.exists():
            print(f'Skip {traj}, file not found')
            continue
        cache_dir = traj.with_suffix('_cache')
        aggregate_file = cache_dir / AGGREGATE_FILENAME
        if aggregate_file.exists():
            print(f'Aggregate already exists for {split}: {aggregate_file}')
            continue

        print(f'Building aggregate tensors for {split}...')
        geo_file = DATA_DIR / 'roadmap.geo'
        rel_file = DATA_DIR / 'roadmap.rel'
        geo = pd.read_csv(geo_file)
        rel = pd.read_csv(rel_file)
        traj_df = pd.read_csv(traj)

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

        os.makedirs(cache_dir, exist_ok=True)

        all_timestamp_labels = []
        max_trace_len = 0
        max_candidate_len = 0
        
        # First pass to generate individual .pt files and collect metadata
        with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_shared_variables, initargs=(reachable_road_id_dict, geo, road_center_gps)) as pool:
            tasks = [(index, row, cache_dir) for index, row in traj_df.iterrows()]
            for result in tqdm(pool.imap_unordered(process_and_save_row_temp, tasks), total=len(traj_df), desc=f'Preprocessing {split}'):
                all_timestamp_labels.extend(result['timestamp_label'])
                max_trace_len = max(max_trace_len, result['trace_len'])
                max_candidate_len = max(max_candidate_len, result['max_candidate_len'])

        stats = {
            'mean': float(np.log1p(all_timestamp_labels).mean()),
            'std': float(np.log1p(all_timestamp_labels).std()),
        }
        with (cache_dir / 'stats.json').open('w') as f:
            json.dump(stats, f)

        metadata = {
            'max_trace_len': int(max_trace_len),
            'max_candidate_len': int(max_candidate_len),
            'has_trace_grid_token': False,  # Assume false for now, update if needed
            'has_candidate_grid_token': False,
        }
        with (cache_dir / META_FILENAME).open('w') as f:
            json.dump(metadata, f)

        # Second pass to pack into aggregate tensors
        file_paths = sorted(
            [str(p) for p in cache_dir.glob('data_*.pt')],
            key=lambda x: int(Path(x).stem.split('_')[1])
        )

        chunk_size = 512
        tensors = []
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=_init_pack_worker,
            initargs=(max_trace_len, max_candidate_len, False, False),
        ) as pool:
            for packed in tqdm(
                pool.imap_unordered(_pack_chunk, list(_chunk_list(file_paths, chunk_size))),
                total=int(np.ceil(len(file_paths) / chunk_size)),
                desc=f'Packing {split} tensors',
            ):
                tensors.append(packed)

        data = {
            'trace_road_id': torch.cat([t['trace_road_id'] for t in tensors], dim=0),
            'temporal_info': torch.cat([t['temporal_info'] for t in tensors], dim=0),
            'trace_distance_mat': torch.cat([t['trace_distance_mat'] for t in tensors], dim=0),
            'trace_time_interval_mat': torch.cat([t['trace_time_interval_mat'] for t in tensors], dim=0),
            'trace_len': torch.cat([t['trace_len'] for t in tensors], dim=0),
            'destination_road_id': torch.cat([t['destination_road_id'] for t in tensors], dim=0),
            'candidate_road_id': torch.cat([t['candidate_road_id'] for t in tensors], dim=0),
            'metric_dis': torch.cat([t['metric_dis'] for t in tensors], dim=0),
            'metric_angle': torch.cat([t['metric_angle'] for t in tensors], dim=0),
            'candidate_len': torch.cat([t['candidate_len'] for t in tensors], dim=0),
            'road_label': torch.cat([t['road_label'] for t in tensors], dim=0),
            'timestamp_label': torch.cat([t['timestamp_label'] for t in tensors], dim=0),
        }
        torch.save(data, aggregate_file)
        print(f'Aggregate saved: {aggregate_file}')

    print('Done building all aggregates')

if __name__ == '__main__':
    main()
