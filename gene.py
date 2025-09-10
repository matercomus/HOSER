import os
import argparse
from queue import PriorityQueue
import math
from datetime import datetime, timedelta
import random
from collections import Counter
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import LineString
from haversine import haversine, haversine_vector
import torch
import torch.nn.functional as F
import pickle
import sys

from utils import set_seed, create_nested_namespace, get_angle
from models.hoser import HOSER

sys.setrecursionlimit(2000) # Set recursion limit for deep data structures


class SearchNode:
    def __init__(self, trace_road_id, trace_datetime, log_prob):
        self.trace_road_id = trace_road_id
        self.trace_datetime = trace_datetime
        self.log_prob = log_prob

    def __ge__(self, other):
        return self.log_prob >= other.log_prob
    
    def __le__(self, other):
        return self.log_prob <= other.log_prob

    def __gt__(self, other):
        return self.log_prob > other.log_prob
    
    def __lt__(self, other):
        return self.log_prob < other.log_prob


class Searcher:
    def __init__(self, model, reachable_road_id_dict, geo, road_center_gps, road_end_coords, timestamp_label_array_log1p_mean, timestamp_label_array_log1p_std, device):
        # Model should already be on device and in eval mode
        self.model = model
        self.reachable_road_id_dict = reachable_road_id_dict
        self.geo = geo
        self.road_center_gps = road_center_gps
        self.road_end_coords = road_end_coords
        self.timestamp_label_array_log1p_mean = timestamp_label_array_log1p_mean
        self.timestamp_label_array_log1p_std = timestamp_label_array_log1p_std
        self.device = device

        # Setup road network features once
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                self.model.setup_road_network_features()

    def search(self, origin_road_id, origin_datetime, destination_road_id, max_search_step=5000):
        vis_set = set()
        pq = PriorityQueue()
        road_id2log_prob = dict()

        best_trace = None
        min_dis = float('inf')

        origin_node = SearchNode(trace_road_id=[origin_road_id], trace_datetime=[origin_datetime], log_prob=0)

        road_id2log_prob[origin_road_id] = 0
        pq.put((-origin_node.log_prob, origin_node))

        search_step = 0
        while (not pq.empty()) and (search_step < max_search_step):
            neg_log_prob, cur_node = pq.get()
            cur_road_id = cur_node.trace_road_id[-1]

            if cur_road_id in vis_set:
                continue
            vis_set.add(cur_road_id)

            if cur_road_id == destination_road_id:
                best_trace = cur_node.trace_road_id, cur_node.trace_datetime
                break

            dis = haversine(self.road_center_gps[cur_road_id], self.road_center_gps[destination_road_id], unit='m')
            if dis < min_dis:
                min_dis = dis
                best_trace = cur_node.trace_road_id, cur_node.trace_datetime

            reachable_road_id_list = self.reachable_road_id_dict[cur_road_id]
            if len(reachable_road_id_list) == 0:
                # Dead-end road reached, skip this path
                continue

            # Predicts the next spatio-temporal point based on the current state
            trace_road_id = np.array(cur_node.trace_road_id)
            temporal_info = np.array([(t.hour * 60.0 + t.minute + t.second / 60.0) / 1440.0 for t in cur_node.trace_datetime]).astype(np.float32)

            if cur_node.trace_datetime[0].weekday() >= 5:
                temporal_info *= -1.0

            trace_distance_mat = haversine_vector(self.road_center_gps[trace_road_id], self.road_center_gps[trace_road_id], 'm', comb=True).astype(np.float32)
            trace_distance_mat = np.clip(trace_distance_mat, 0.0, 1000.0) / 1000.0
            trace_time_interval_mat = np.abs(temporal_info[:, None] * 1440.0 - temporal_info * 1440.0)
            trace_time_interval_mat = np.clip(trace_time_interval_mat, 0.0, 5.0) / 5.0
            trace_len = len(trace_road_id)

            candidate_road_id = np.array(reachable_road_id_list)

            metric_dis = haversine_vector(self.road_center_gps[candidate_road_id], self.road_center_gps[destination_road_id].reshape(1, -1), 'm', comb=True).reshape(-1).astype(np.float32)
            metric_dis = np.log1p((metric_dis - np.min(metric_dis)) / 100)

            cur_road_end_coord = self.road_end_coords[cur_road_id]
            candidate_end_coords = self.road_end_coords[candidate_road_id]
            dest_end_coord = self.road_end_coords[destination_road_id]

            vec1 = candidate_end_coords - cur_road_end_coord
            angle1 = np.arctan2(vec1[:, 1], vec1[:, 0])

            vec2 = dest_end_coord - cur_road_end_coord
            angle2 = np.arctan2(vec2[1], vec2[0])

            angle = np.abs(angle1 - angle2).astype(np.float32)
            angle = np.where(angle > math.pi, 2 * math.pi - angle, angle) / math.pi
            metric_angle = angle

            # Create tensors directly on device to avoid CPU->GPU transfer overhead
            batch_trace_road_id = torch.tensor([trace_road_id], dtype=torch.long, device=self.device)
            batch_temporal_info = torch.tensor([temporal_info], dtype=torch.float32, device=self.device)
            batch_trace_distance_mat = torch.tensor([trace_distance_mat], dtype=torch.float32, device=self.device)
            batch_trace_time_interval_mat = torch.tensor([trace_time_interval_mat], dtype=torch.float32, device=self.device)
            batch_trace_len = torch.tensor([trace_len], dtype=torch.long, device=self.device)
            batch_destination_road_id = torch.tensor([destination_road_id], dtype=torch.long, device=self.device)
            batch_candidate_road_id = torch.tensor([candidate_road_id], dtype=torch.long, device=self.device)
            batch_metric_dis = torch.tensor([metric_dis], dtype=torch.float32, device=self.device)
            batch_metric_angle = torch.tensor([metric_angle], dtype=torch.float32, device=self.device)

            # Inference without autocast for single samples (autocast adds overhead)
            with torch.no_grad():
                logits, time_pred = self.model.infer(batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle)

            logits = logits[0]
            output = F.softmax(logits, dim=-1)
            log_output = torch.log(output)
            log_output += cur_node.log_prob

            time_pred = time_pred[0]
            time_pred = time_pred * self.timestamp_label_array_log1p_std + self.timestamp_label_array_log1p_mean
            time_pred = torch.expm1(time_pred)
            time_pred = torch.clamp(time_pred, min=0.0)

            for index, candidate_road_id_val in enumerate(reachable_road_id_list):
                candidate_log_prob = log_output[index].item()
                next_datatime = cur_node.trace_datetime[-1] + timedelta(seconds=round(time_pred[index].item()))

                if candidate_road_id_val not in road_id2log_prob or candidate_log_prob > road_id2log_prob[candidate_road_id_val]:
                    new_node = SearchNode(
                        trace_road_id=cur_node.trace_road_id+[candidate_road_id_val],
                        trace_datetime=cur_node.trace_datetime+[next_datatime],
                        log_prob=candidate_log_prob
                    )
                    pq.put((-candidate_log_prob, new_node))
                    road_id2log_prob[candidate_road_id_val] = candidate_log_prob

            search_step += 1

        assert best_trace is not None
        return best_trace[0], best_trace[1]


# Removed multiprocessing functions - no longer needed

def load_and_preprocess_data(dataset):
    cache_path = f'./data/{dataset}/gene_preprocessed_cache.pkl'
    if os.path.exists(cache_path):
        print(f"✅ Loading preprocessed data from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("🚀 Preprocessing data for trajectory generation (first time only)...")

    geo_file = f'./data/{dataset}/roadmap.geo'
    rel_file = f'./data/{dataset}/roadmap.rel'
    train_traj_file = f'./data/{dataset}/train.csv'
    road_network_partition_file = f'./data/{dataset}/road_network_partition'
    zone_trans_mat_file = f'./data/{dataset}/zone_trans_mat.npy'

    print("📂 Loading road network data...")
    geo = pd.read_csv(geo_file)
    rel = pd.read_csv(rel_file)
    num_roads = len(geo)
    print(f"✅ Loaded {num_roads} road segments.")

    print("🛰️  Calculating road coordinates and centers...")
    geo['coordinates_parsed'] = [eval(coord_str) for coord_str in tqdm(geo['coordinates'], desc="Parsing Coords")]
    road_center_gps = np.array([LineString(coords).centroid.coords[0][::-1] for coords in tqdm(geo['coordinates_parsed'], desc="Centroids")])
    road_end_coords = np.array([coords[-1] for coords in geo['coordinates_parsed']])
    print("✅ GPS calculation complete.")

    print("📝 Preprocessing road attributes...")
    road_attr_len = geo['length'].to_numpy(dtype=np.float32)
    road_attr_len = np.log1p(road_attr_len)
    road_attr_len = (road_attr_len - np.mean(road_attr_len)) / np.std(road_attr_len)

    road_attr_type = geo['highway'].values.tolist()
    if dataset in ['Beijing', 'San_Francisco']:
        for i in range(len(road_attr_type)):
            if isinstance(road_attr_type[i], str) and road_attr_type[i].startswith('[') and road_attr_type[i].endswith(']'):
                info = eval(road_attr_type[i])
                road_attr_type[i] = info[0] if info[0] != 'unclassified' else info[1]
    le = LabelEncoder()
    road_attr_type = le.fit_transform(road_attr_type)
    
    road_attr_lon = np.array([c[0] for c in road_center_gps]).astype(np.float32)
    road_attr_lon = (road_attr_lon - np.mean(road_attr_lon)) / np.std(road_attr_lon)
    road_attr_lat = np.array([c[1] for c in road_center_gps]).astype(np.float32)
    road_attr_lat = (road_attr_lat - np.mean(road_attr_lat)) / np.std(road_attr_lat)
    print("✅ Attribute preprocessing complete.")

    reachable_road_id_dict = {i: [] for i in range(num_roads)}
    for _, row in rel.iterrows():
        reachable_road_id_dict[row['origin_id']].append(row['destination_id'])

    print("🔗 Building road network graph...")
    coord2road_id = {}
    for road_id, row in tqdm(geo.iterrows(), total=num_roads, desc="Building graph"):
        coord = row['coordinates_parsed']
        start_coord, end_coord = tuple(coord[0]), tuple(coord[-1])
        coord2road_id.setdefault(start_coord, []).append(road_id)
        coord2road_id.setdefault(end_coord, []).append(road_id)

    adj_row, adj_col, adj_angle, adj_reachability = [], [], [], []
    road_adj_lists = {i: set() for i in range(num_roads)}
    for v in coord2road_id.values():
        for r1 in v:
            for r2 in v:
                if r1 != r2:
                    road_adj_lists[r1].add(r2)

    for road_id in range(num_roads):
        for adj_road_id in road_adj_lists[road_id]:
            adj_row.append(road_id)
            adj_col.append(adj_road_id)

            r1_coords, r2_coords = geo.at[road_id, 'coordinates_parsed'], geo.at[adj_road_id, 'coordinates_parsed']
            r1_angle = get_angle(r1_coords[0][1], r1_coords[0][0], r1_coords[-1][1], r1_coords[-1][0])
            r2_angle = get_angle(r2_coords[0][1], r2_coords[0][0], r2_coords[-1][1], r2_coords[-1][0])
            
            angle = abs(r1_angle - r2_angle)
            angle = (math.pi * 2 - angle) if angle > math.pi else angle
            adj_angle.append(angle / math.pi)
            adj_reachability.append(1.0 if adj_road_id in reachable_road_id_dict[road_id] else 0.0)
    print("✅ Graph construction complete.")
    
    road_edge_index = np.stack([np.array(adj_row, dtype=np.int64), np.array(adj_col, dtype=np.int64)])
    intersection_attr = np.stack([np.array(adj_angle, dtype=np.float32), np.array(adj_reachability, dtype=np.float32)], axis=1)

    zone_trans_mat = np.load(zone_trans_mat_file)
    zone_edge_index = np.stack(zone_trans_mat.nonzero())
    zone_trans_mat = zone_trans_mat.astype(np.float32)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.sum(zone_trans_mat, axis=1), 1.0))
    zone_trans_mat_norm = zone_trans_mat * D_inv_sqrt[:, np.newaxis] * D_inv_sqrt[np.newaxis, :]
    zone_edge_weight = zone_trans_mat_norm[zone_edge_index[0], zone_edge_index[1]]

    road2zone = np.loadtxt(road_network_partition_file, dtype=np.int32)

    print("🗺️  Loading training trajectories for OD matrix...")
    train_traj = pd.read_csv(train_traj_file)
    print(f"✅ Loaded {len(train_traj)} trajectories.")

    print("... Calculating OD matrix...")
    od_counts = Counter()
    for rid_list_str in tqdm(train_traj['rid_list'], desc="Calculating OD"):
        rid_list = eval(rid_list_str)
        if len(rid_list) >= 2:
            od_counts[(rid_list[0], rid_list[-1])] += 1.0
    print("✅ OD matrix calculation complete.")

    print("🕒 Calculating timestamp statistics...")
    timestamp_labels = [
        (datetime.strptime(t2, '%Y-%m-%dT%H:%M:%SZ') - datetime.strptime(t1, '%Y-%m-%dT%H:%M:%SZ')).total_seconds()
        for time_list_str in train_traj['time_list']
        for t1, t2 in zip(time_list_str.split(',')[:-1], time_list_str.split(',')[1:])
    ]
    timestamp_label_array = np.array(timestamp_labels, dtype=np.float32)
    timestamp_label_array_log1p_mean = np.mean(np.log1p(timestamp_label_array))
    timestamp_label_array_log1p_std = np.std(np.log1p(timestamp_label_array))
    print(f'📊 Timestamp Stats: mean {timestamp_label_array_log1p_mean:.3f}, std {timestamp_label_array_log1p_std:.3f}')
    
    data = {
        "geo": geo, "rel": rel, "num_roads": num_roads, "road_center_gps": road_center_gps,
        "road_end_coords": road_end_coords, "road_attr_len": road_attr_len, "road_attr_type": road_attr_type,
        "road_attr_lon": road_attr_lon, "road_attr_lat": road_attr_lat, "road_edge_index": road_edge_index,
        "intersection_attr": intersection_attr, "zone_edge_index": zone_edge_index,
        "zone_edge_weight": zone_edge_weight, "road2zone": road2zone, "train_traj": train_traj,
        "reachable_road_id_dict": reachable_road_id_dict, "od_counts": od_counts,
        "timestamp_label_array_log1p_mean": timestamp_label_array_log1p_mean,
        "timestamp_label_array_log1p_std": timestamp_label_array_log1p_std
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved preprocessed data to cache: {cache_path}")

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_gene', type=int, default=5000)
    parser.add_argument('--processes', type=int, default=1, help='(Deprecated - single process is optimal for GPU inference)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = f'cuda:{args.cuda}'
    
    # Set PyTorch threading for optimal single-process performance
    torch.set_num_threads(torch.get_num_threads())  # Use all available threads
    torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner

    data = load_and_preprocess_data(args.dataset)
    num_roads = data['num_roads']

    with open(f'./config/{args.dataset}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = create_nested_namespace(config)
    
    config.road_network_encoder_config.road_id_num_embeddings = num_roads
    config.road_network_encoder_config.type_num_embeddings = len(np.unique(data['road_attr_type']))
    config.road_network_encoder_feature.road_attr.len = data['road_attr_len']
    config.road_network_encoder_feature.road_attr.type = data['road_attr_type']
    config.road_network_encoder_feature.road_attr.lon = data['road_attr_lon']
    config.road_network_encoder_feature.road_attr.lat = data['road_attr_lat']
    config.road_network_encoder_feature.road_edge_index = data['road_edge_index']
    config.road_network_encoder_feature.intersection_attr = data['intersection_attr']
    config.road_network_encoder_feature.zone_edge_index = data['zone_edge_index']
    config.road_network_encoder_feature.zone_edge_weight = data['zone_edge_weight']

    print(" lọc... Filtering valid OD pairs...")
    valid_destinations = {i for i in range(num_roads) if len(data['reachable_road_id_dict'][i]) > 0}
    valid_od_pairs = [(o, d) for (o, d), count in data['od_counts'].items() if d in valid_destinations]
    valid_od_counts = np.array([data['od_counts'][od] for od in valid_od_pairs], dtype=np.float32)

    if not valid_od_pairs:
        raise ValueError("No valid origin-destination pairs found (all destinations are dead-ends)")

    od_probabilities = valid_od_counts / valid_od_counts.sum()
    print(f"✅ Found {len(valid_od_pairs)} valid OD pairs.")

    print(f"🎲 Sampling {args.num_gene} OD pairs for generation...")
    sampled_indices = np.random.choice(len(valid_od_pairs), size=args.num_gene, p=od_probabilities)
    od_coords = [valid_od_pairs[i] for i in sampled_indices]

    time_list = list(data['train_traj']['time_list'])
    origin_datetime_list = [datetime.strptime(row.split(',')[0], '%Y-%m-%dT%H:%M:%SZ') for row in random.choices(time_list, k=args.num_gene)]

    print("🧠 Loading trained model...")
    model = HOSER(
        config.road_network_encoder_config,
        config.road_network_encoder_feature,
        config.trajectory_encoder_config,
        config.navigator_config,
        data['road2zone'],
    )

    save_dir = f'./save/{args.dataset}/seed{args.seed}'
    model_state_dict = torch.load(os.path.join(save_dir, 'best.pth'), map_location='cpu')
    model.load_state_dict(model_state_dict)
    print("✅ Model loaded successfully.")

    gene_trace_road_id = [None] * args.num_gene
    gene_trace_datetime = [None] * args.num_gene

    print(f"🧬 Starting trajectory generation for {args.num_gene} trajectories...")
    
    # Move model to GPU once and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Create searcher instance
    searcher = Searcher(model, data['reachable_road_id_dict'], data['geo'], data['road_center_gps'], 
                       data['road_end_coords'], data['timestamp_label_array_log1p_mean'], 
                       data['timestamp_label_array_log1p_std'], device)
    
    # Process trajectories
    for i, ((origin_road_id, destination_road_id), origin_datetime) in enumerate(
            tqdm(zip(od_coords, origin_datetime_list), total=len(od_coords), desc='Generating trajectories')):
        trace_road_id, trace_datetime = searcher.search(origin_road_id, origin_datetime, destination_road_id)
        gene_trace_road_id[i] = trace_road_id
        gene_trace_datetime[i] = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in trace_datetime]

    res_df = pd.DataFrame({
        'gene_trace_road_id': gene_trace_road_id,
        'gene_trace_datetime': gene_trace_datetime,
    })
    
    gene_dir = f'./gene/{args.dataset}/seed{args.seed}'
    os.makedirs(gene_dir, exist_ok=True)
    now = datetime.now()
    output_path = os.path.join(gene_dir, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv')
    res_df.to_csv(output_path, index=False)
    print(f"🎉 Trajectory generation complete. Saved to {output_path}")
