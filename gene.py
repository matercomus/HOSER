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


def process_trajectory_mp(args_tuple):
    """Multiprocessing worker function for trajectory generation"""
    idx, origin_road_id, destination_road_id, origin_datetime, model_state, data, device, beam_search, beam_width, model_config = args_tuple
    
    # Import inside worker to avoid CUDA issues
    from models.hoser import HOSER
    
    # Use CPU for multiprocessing to avoid CUDA conflicts
    device = 'cpu'
    
    # Create model and searcher for this process
    model = HOSER(
        model_config['road_network_encoder_config'],
        model_config['road_network_encoder_feature'],
        model_config['trajectory_encoder_config'],
        model_config['navigator_config'],
        model_config['road2zone'],
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    searcher = Searcher(model, data['reachable_road_id_dict'], data['geo'], 
                      data['road_center_gps'], data['road_end_coords'], 
                      data['timestamp_label_array_log1p_mean'], 
                      data['timestamp_label_array_log1p_std'], device)
    
    if beam_search:
        trace_road_id, trace_datetime = searcher.beam_search(origin_road_id, origin_datetime, destination_road_id, beam_width=beam_width)
    else:
        trace_road_id, trace_datetime = searcher.search(origin_road_id, origin_datetime, destination_road_id)
    
    return idx, trace_road_id, trace_datetime


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
        
        # Pre-compute constants for timestamp operations
        self.timestamp_mean_tensor = torch.tensor(timestamp_label_array_log1p_mean, device=device)
        self.timestamp_std_tensor = torch.tensor(timestamp_label_array_log1p_std, device=device)

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

            # Create tensors efficiently - expand dims instead of wrapping in list
            batch_trace_road_id = torch.from_numpy(trace_road_id).unsqueeze(0).to(self.device)
            batch_temporal_info = torch.from_numpy(temporal_info).unsqueeze(0).to(self.device)
            batch_trace_distance_mat = torch.from_numpy(trace_distance_mat).unsqueeze(0).to(self.device)
            batch_trace_time_interval_mat = torch.from_numpy(trace_time_interval_mat).unsqueeze(0).to(self.device)
            batch_trace_len = torch.tensor(trace_len, dtype=torch.long, device=self.device).unsqueeze(0)
            batch_destination_road_id = torch.tensor(destination_road_id, dtype=torch.long, device=self.device).unsqueeze(0)
            batch_candidate_road_id = torch.from_numpy(candidate_road_id).to(self.device).unsqueeze(0)
            batch_metric_dis = torch.from_numpy(metric_dis).unsqueeze(0).to(self.device)
            batch_metric_angle = torch.from_numpy(metric_angle).unsqueeze(0).to(self.device)

            # Inference without autocast for single samples (autocast adds overhead)
            with torch.no_grad():
                logits, time_pred = self.model.infer(batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle)

            logits = logits[0]
            # Use log_softmax for numerical stability and efficiency
            log_output = F.log_softmax(logits, dim=-1)
            log_output += cur_node.log_prob

            time_pred = time_pred[0]
            time_pred = time_pred * self.timestamp_std_tensor + self.timestamp_mean_tensor
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
    
    def beam_search(self, origin_road_id, origin_datetime, destination_road_id, beam_width=8, max_search_step=5000):
        """Beam search that processes multiple candidates in parallel for better GPU utilization"""
        vis_set = set()
        road_id2best = {}  # Maps road_id to best (log_prob, node) tuple
        
        best_trace = None
        min_dis = float('inf')
        
        origin_node = SearchNode(trace_road_id=[origin_road_id], trace_datetime=[origin_datetime], log_prob=0)
        road_id2best[origin_road_id] = (0, origin_node)
        
        beam = [origin_node]  # Current beam
        
        search_step = 0
        while beam and search_step < max_search_step:
            # Check for destination in current beam
            for node in beam:
                cur_road_id = node.trace_road_id[-1]
                if cur_road_id == destination_road_id:
                    return node.trace_road_id, node.trace_datetime
                
                # Update best trace based on distance
                dis = haversine(self.road_center_gps[cur_road_id], self.road_center_gps[destination_road_id], unit='m')
                if dis < min_dis:
                    min_dis = dis
                    best_trace = node.trace_road_id, node.trace_datetime
            
            # Collect all candidates from current beam
            all_candidates = []
            candidate_info = []  # (beam_idx, reachable_roads) pairs
            
            for beam_idx, node in enumerate(beam):
                cur_road_id = node.trace_road_id[-1]
                if cur_road_id in vis_set:
                    continue
                vis_set.add(cur_road_id)
                
                reachable_road_id_list = self.reachable_road_id_dict[cur_road_id]
                if reachable_road_id_list:
                    all_candidates.append(node)
                    candidate_info.append((beam_idx, reachable_road_id_list))
            
            if not all_candidates:
                break
            
            # Prepare batch data for all candidates
            batch_data = self._prepare_beam_batch_data(all_candidates, candidate_info, destination_road_id)
            
            # Single batched inference for all candidates
            with torch.no_grad():
                logits, time_pred = self.model.infer(
                    batch_data['trace_road_id'],
                    batch_data['temporal_info'],
                    batch_data['trace_distance_mat'],
                    batch_data['trace_time_interval_mat'],
                    batch_data['trace_len'],
                    batch_data['destination_road_id'],
                    batch_data['candidate_road_id'],
                    batch_data['metric_dis'],
                    batch_data['metric_angle']
                )
            
            # Process results and collect new candidates
            new_candidates = []
            batch_idx = 0
            
            for node, (_, reachable_road_id_list) in zip(all_candidates, candidate_info):
                log_output = F.log_softmax(logits[batch_idx], dim=-1) + node.log_prob
                cur_time_pred = time_pred[batch_idx] * self.timestamp_std_tensor + self.timestamp_mean_tensor
                cur_time_pred = torch.expm1(cur_time_pred)
                cur_time_pred = torch.clamp(cur_time_pred, min=0.0)
                
                for j, candidate_road_id_val in enumerate(reachable_road_id_list):
                    candidate_log_prob = log_output[j].item()
                    next_datatime = node.trace_datetime[-1] + timedelta(seconds=round(cur_time_pred[j].item()))
                    
                    # Only keep if it's the best path to this road
                    if candidate_road_id_val not in road_id2best or candidate_log_prob > road_id2best[candidate_road_id_val][0]:
                        new_node = SearchNode(
                            trace_road_id=node.trace_road_id+[candidate_road_id_val],
                            trace_datetime=node.trace_datetime+[next_datatime],
                            log_prob=candidate_log_prob
                        )
                        road_id2best[candidate_road_id_val] = (candidate_log_prob, new_node)
                        new_candidates.append((candidate_log_prob, new_node))
                
                batch_idx += 1
            
            # Select top beam_width candidates for next iteration
            new_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = [node for _, node in new_candidates[:beam_width]]
            
            search_step += len(all_candidates)
        
        return best_trace[0], best_trace[1] if best_trace else (None, None)
    
    def vectorized_search(self, od_coords, origin_datetime_list, batch_size=256, max_search_step=5000):
        """Process multiple trajectories in parallel using vectorized GPU operations"""
        num_trajectories = len(od_coords)
        all_results = []
        
        # Convert road network data to GPU tensors for fast lookups
        road_gps_tensor = torch.tensor(self.road_center_gps, dtype=torch.float32, device=self.device)
        road_end_tensor = torch.tensor(self.road_end_coords, dtype=torch.float32, device=self.device)
        
        # Process in batches
        for batch_start in tqdm(range(0, num_trajectories, batch_size), desc='Processing trajectory batches'):
            batch_end = min(batch_start + batch_size, num_trajectories)
            batch_od = od_coords[batch_start:batch_end]
            batch_times = origin_datetime_list[batch_start:batch_end]
            
            # Initialize batch search state
            batch_results = [None] * len(batch_od)
            active_searches = list(range(len(batch_od)))
            
            # Current nodes for each active search
            current_nodes = {i: SearchNode(trace_road_id=[batch_od[i][0]], 
                                         trace_datetime=[batch_times[i]], 
                                         log_prob=0) 
                           for i in active_searches}
            
            # Best traces for fallback
            best_traces = {i: (current_nodes[i].trace_road_id, current_nodes[i].trace_datetime) 
                          for i in active_searches}
            min_distances = {i: float('inf') for i in active_searches}
            
            visited = {i: set() for i in active_searches}
            
            for step in range(max_search_step):
                if not active_searches:
                    break
                
                # Collect all active nodes and their candidates
                batch_nodes = []
                batch_destinations = []
                search_indices = []
                
                for idx in active_searches[:]:
                    node = current_nodes[idx]
                    destination = batch_od[idx][1]
                    
                    # Check if reached destination
                    if node.trace_road_id[-1] == destination:
                        batch_results[idx] = (node.trace_road_id, node.trace_datetime)
                        active_searches.remove(idx)
                        continue
                    
                    # Update best trace
                    cur_pos = road_gps_tensor[node.trace_road_id[-1]]
                    dest_pos = road_gps_tensor[destination]
                    dist = torch.norm(cur_pos - dest_pos).item() * 111000  # Approximate meters
                    
                    if dist < min_distances[idx]:
                        min_distances[idx] = dist
                        best_traces[idx] = (node.trace_road_id, node.trace_datetime)
                    
                    # Skip if visited
                    if node.trace_road_id[-1] in visited[idx]:
                        continue
                    visited[idx].add(node.trace_road_id[-1])
                    
                    # Get reachable roads
                    reachable = self.reachable_road_id_dict.get(node.trace_road_id[-1], [])
                    if reachable:
                        batch_nodes.append(node)
                        batch_destinations.append(destination)
                        search_indices.append((idx, reachable))
                
                if not batch_nodes:
                    break
                
                # Prepare vectorized batch data
                batch_data = self._prepare_vectorized_batch(batch_nodes, batch_destinations, 
                                                           search_indices, road_gps_tensor, road_end_tensor)
                
                # Single GPU inference for all candidates
                with torch.no_grad():
                    logits, time_pred = self.model.infer(
                        batch_data['trace_road_id'],
                        batch_data['temporal_info'],
                        batch_data['trace_distance_mat'],
                        batch_data['trace_time_interval_mat'],
                        batch_data['trace_len'],
                        batch_data['destination_road_id'],
                        batch_data['candidate_road_id'],
                        batch_data['metric_dis'],
                        batch_data['metric_angle']
                    )
                
                # Process results and select best candidates
                result_idx = 0
                for node, dest, (search_idx, reachable) in zip(batch_nodes, batch_destinations, search_indices):
                    # Get predictions for this search
                    node_logits = logits[result_idx]
                    node_time_pred = time_pred[result_idx]
                    
                    # Find best candidate
                    log_probs = F.log_softmax(node_logits[:len(reachable)], dim=-1) + node.log_prob
                    best_idx = torch.argmax(log_probs).item()
                    
                    # Create new node
                    best_road = reachable[best_idx]
                    time_delta = (node_time_pred[best_idx] * self.timestamp_std_tensor + self.timestamp_mean_tensor).exp() - 1
                    time_delta = max(0, time_delta.item())
                    
                    new_node = SearchNode(
                        trace_road_id=node.trace_road_id + [best_road],
                        trace_datetime=node.trace_datetime + [node.trace_datetime[-1] + timedelta(seconds=round(time_delta))],
                        log_prob=log_probs[best_idx].item()
                    )
                    
                    current_nodes[search_idx] = new_node
                    result_idx += 1
            
            # Collect results
            for i in range(len(batch_od)):
                if batch_results[i] is not None:
                    all_results.append(batch_results[i])
                else:
                    all_results.append(best_traces[i])
        
        return all_results
    
    def _prepare_vectorized_batch(self, nodes, destinations, search_indices, road_gps_tensor, road_end_tensor):
        """Prepare batch data for vectorized search with minimal CPU operations"""
        # Use maximum sizes for padding
        max_trace_len = max(len(n.trace_road_id) for n in nodes)
        max_candidates = max(len(reachable) for _, reachable in search_indices)
        
        # Pre-allocate tensors
        batch_size = len(nodes)
        trace_road_ids = torch.zeros((batch_size, max_trace_len), dtype=torch.long, device=self.device)
        temporal_infos = torch.zeros((batch_size, max_trace_len), dtype=torch.float32, device=self.device)
        trace_lens = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        destination_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        candidate_ids = torch.zeros((batch_size, max_candidates), dtype=torch.long, device=self.device)
        
        # Fill tensors
        for i, (node, dest, (_, reachable)) in enumerate(zip(nodes, destinations, search_indices)):
            trace_len = len(node.trace_road_id)
            trace_road_ids[i, :trace_len] = torch.tensor(node.trace_road_id, device=self.device)
            
            # Temporal info
            times = torch.tensor([(t.hour * 60.0 + t.minute + t.second / 60.0) / 1440.0 
                                 for t in node.trace_datetime], dtype=torch.float32, device=self.device)
            if node.trace_datetime[0].weekday() >= 5:
                times *= -1.0
            temporal_infos[i, :trace_len] = times
            
            trace_lens[i] = trace_len
            destination_ids[i] = dest
            candidate_ids[i, :len(reachable)] = torch.tensor(reachable, device=self.device)
        
        # Compute distance matrices using GPU operations
        trace_positions = road_gps_tensor[trace_road_ids]  # (batch, max_trace, 2)
        trace_dist_mat = torch.cdist(trace_positions, trace_positions) * 111000  # Approximate meters
        trace_dist_mat = torch.clamp(trace_dist_mat, 0, 1000) / 1000
        
        # Time interval matrix
        time_expanded = temporal_infos.unsqueeze(2) * 1440  # (batch, max_trace, 1)
        time_interval_mat = torch.abs(time_expanded - time_expanded.transpose(1, 2))
        time_interval_mat = torch.clamp(time_interval_mat, 0, 5) / 5
        
        # Candidate metrics
        candidate_positions = road_gps_tensor[candidate_ids]  # (batch, max_candidates, 2)
        dest_positions = road_gps_tensor[destination_ids].unsqueeze(1)  # (batch, 1, 2)
        metric_distances = torch.norm(candidate_positions - dest_positions, dim=2) * 111000
        metric_distances = torch.log1p((metric_distances - metric_distances.min(dim=1, keepdim=True)[0]) / 100)
        
        # Angles (simplified for GPU)
        last_road_ids = trace_road_ids[torch.arange(batch_size), trace_lens - 1]
        last_positions = road_end_tensor[last_road_ids]
        candidate_end_pos = road_end_tensor[candidate_ids]
        dest_end_pos = road_end_tensor[destination_ids].unsqueeze(1)
        
        vec1 = candidate_end_pos - last_positions.unsqueeze(1)
        angle1 = torch.atan2(vec1[:, :, 1], vec1[:, :, 0])
        vec2 = dest_end_pos - last_positions.unsqueeze(1)
        angle2 = torch.atan2(vec2[:, :, 1], vec2[:, :, 0])
        
        angles = torch.abs(angle1 - angle2.squeeze(1))
        angles = torch.where(angles > math.pi, 2 * math.pi - angles, angles) / math.pi
        
        # Set invalid candidates to high values
        valid_mask = candidate_ids > 0
        metric_distances = torch.where(valid_mask, metric_distances, torch.tensor(1e6, device=self.device))
        angles = torch.where(valid_mask, angles, torch.tensor(1.0, device=self.device))
        
        return {
            'trace_road_id': trace_road_ids,
            'temporal_info': temporal_infos,
            'trace_distance_mat': trace_dist_mat,
            'trace_time_interval_mat': time_interval_mat,
            'trace_len': trace_lens,
            'destination_road_id': destination_ids,
            'candidate_road_id': candidate_ids,
            'metric_dis': metric_distances,
            'metric_angle': angles
        }
    
    def _prepare_beam_batch_data(self, nodes, candidate_info, destination_road_id):
        """Prepare batch data for beam search"""
        # Find max lengths for padding
        max_trace_len = max(len(node.trace_road_id) for node in nodes)
        max_candidate_len = max(len(reachable) for _, reachable in candidate_info)
        
        # Initialize batch lists
        batch_trace_road_id = []
        batch_temporal_info = []
        batch_trace_distance_mat = []
        batch_trace_time_interval_mat = []
        batch_trace_len = []
        batch_destination_road_id = []
        batch_candidate_road_id = []
        batch_metric_dis = []
        batch_metric_angle = []
        
        for node, (_, reachable_road_id_list) in zip(nodes, candidate_info):
            # Prepare trace data
            trace_road_id = np.array(node.trace_road_id)
            temporal_info = np.array([(t.hour * 60.0 + t.minute + t.second / 60.0) / 1440.0 for t in node.trace_datetime]).astype(np.float32)
            
            if node.trace_datetime[0].weekday() >= 5:
                temporal_info *= -1.0
            
            trace_distance_mat = haversine_vector(self.road_center_gps[trace_road_id], self.road_center_gps[trace_road_id], 'm', comb=True).astype(np.float32)
            trace_distance_mat = np.clip(trace_distance_mat, 0.0, 1000.0) / 1000.0
            trace_time_interval_mat = np.abs(temporal_info[:, None] * 1440.0 - temporal_info * 1440.0)
            trace_time_interval_mat = np.clip(trace_time_interval_mat, 0.0, 5.0) / 5.0
            
            # Prepare candidate data
            candidate_road_id = np.array(reachable_road_id_list)
            metric_dis = haversine_vector(self.road_center_gps[candidate_road_id], self.road_center_gps[destination_road_id].reshape(1, -1), 'm', comb=True).reshape(-1).astype(np.float32)
            metric_dis = np.log1p((metric_dis - np.min(metric_dis)) / 100)
            
            cur_road_id = node.trace_road_id[-1]
            cur_road_end_coord = self.road_end_coords[cur_road_id]
            candidate_end_coords = self.road_end_coords[candidate_road_id]
            dest_end_coord = self.road_end_coords[destination_road_id]
            
            vec1 = candidate_end_coords - cur_road_end_coord
            angle1 = np.arctan2(vec1[:, 1], vec1[:, 0])
            vec2 = dest_end_coord - cur_road_end_coord
            angle2 = np.arctan2(vec2[1], vec2[0])
            
            angle = np.abs(angle1 - angle2).astype(np.float32)
            angle = np.where(angle > math.pi, 2 * math.pi - angle, angle) / math.pi
            
            # Pad to max lengths
            padded_trace_road_id = np.pad(trace_road_id, (0, max_trace_len - len(trace_road_id)), 'constant')
            padded_temporal_info = np.pad(temporal_info, (0, max_trace_len - len(temporal_info)), 'constant')
            padded_trace_distance_mat = np.pad(trace_distance_mat, ((0, max_trace_len - trace_distance_mat.shape[0]), (0, max_trace_len - trace_distance_mat.shape[1])), 'constant')
            padded_trace_time_interval_mat = np.pad(trace_time_interval_mat, ((0, max_trace_len - trace_time_interval_mat.shape[0]), (0, max_trace_len - trace_time_interval_mat.shape[1])), 'constant')
            padded_candidate_road_id = np.pad(candidate_road_id, (0, max_candidate_len - len(candidate_road_id)), 'constant')
            padded_metric_dis = np.pad(metric_dis, (0, max_candidate_len - len(metric_dis)), 'constant', constant_values=1e6)
            padded_metric_angle = np.pad(angle, (0, max_candidate_len - len(angle)), 'constant', constant_values=1.0)
            
            batch_trace_road_id.append(padded_trace_road_id)
            batch_temporal_info.append(padded_temporal_info)
            batch_trace_distance_mat.append(padded_trace_distance_mat)
            batch_trace_time_interval_mat.append(padded_trace_time_interval_mat)
            batch_trace_len.append(len(trace_road_id))
            batch_destination_road_id.append(destination_road_id)
            batch_candidate_road_id.append(padded_candidate_road_id)
            batch_metric_dis.append(padded_metric_dis)
            batch_metric_angle.append(padded_metric_angle)
        
        # Convert to tensors
        return {
            'trace_road_id': torch.tensor(np.array(batch_trace_road_id), dtype=torch.long, device=self.device),
            'temporal_info': torch.tensor(np.array(batch_temporal_info), dtype=torch.float32, device=self.device),
            'trace_distance_mat': torch.tensor(np.array(batch_trace_distance_mat), dtype=torch.float32, device=self.device),
            'trace_time_interval_mat': torch.tensor(np.array(batch_trace_time_interval_mat), dtype=torch.float32, device=self.device),
            'trace_len': torch.tensor(batch_trace_len, dtype=torch.long, device=self.device),
            'destination_road_id': torch.tensor(batch_destination_road_id, dtype=torch.long, device=self.device),
            'candidate_road_id': torch.tensor(np.array(batch_candidate_road_id), dtype=torch.long, device=self.device),
            'metric_dis': torch.tensor(np.array(batch_metric_dis), dtype=torch.float32, device=self.device),
            'metric_angle': torch.tensor(np.array(batch_metric_angle), dtype=torch.float32, device=self.device)
        }
    
    def batch_search(self, batch_origins, batch_origin_times, batch_destinations, max_search_step=5000):
        """Process multiple trajectory searches in parallel to better utilize GPU"""
        batch_size = len(batch_origins)
        results = [None] * batch_size
        
        # Initialize search states for each trajectory
        search_states = []
        for i in range(batch_size):
            state = {
                'vis_set': set(),
                'pq': PriorityQueue(),
                'road_id2log_prob': dict(),
                'best_trace': None,
                'min_dis': float('inf'),
                'done': False,
                'search_step': 0
            }
            origin_node = SearchNode(
                trace_road_id=[batch_origins[i]], 
                trace_datetime=[batch_origin_times[i]], 
                log_prob=0
            )
            state['road_id2log_prob'][batch_origins[i]] = 0
            state['pq'].put((-origin_node.log_prob, origin_node))
            search_states.append(state)
        
        # Continue until all searches are done
        while any(not state['done'] for state in search_states):
            # Collect active searches that need model inference
            active_indices = []
            active_nodes = []
            
            for i, state in enumerate(search_states):
                if state['done'] or state['pq'].empty() or state['search_step'] >= max_search_step:
                    if not state['done']:
                        state['done'] = True
                        results[i] = (state['best_trace'][0], state['best_trace'][1]) if state['best_trace'] else None
                    continue
                
                # Get next node from priority queue
                neg_log_prob, cur_node = state['pq'].get()
                cur_road_id = cur_node.trace_road_id[-1]
                
                if cur_road_id in state['vis_set']:
                    continue
                state['vis_set'].add(cur_road_id)
                
                # Check if reached destination
                if cur_road_id == batch_destinations[i]:
                    state['best_trace'] = cur_node.trace_road_id, cur_node.trace_datetime
                    state['done'] = True
                    results[i] = (state['best_trace'][0], state['best_trace'][1])
                    continue
                
                # Update best trace based on distance
                dis = haversine(self.road_center_gps[cur_road_id], self.road_center_gps[batch_destinations[i]], unit='m')
                if dis < state['min_dis']:
                    state['min_dis'] = dis
                    state['best_trace'] = cur_node.trace_road_id, cur_node.trace_datetime
                
                # Skip if dead-end
                reachable_road_id_list = self.reachable_road_id_dict[cur_road_id]
                if len(reachable_road_id_list) == 0:
                    continue
                
                active_indices.append(i)
                active_nodes.append((cur_node, reachable_road_id_list))
                state['search_step'] += 1
            
            if not active_indices:
                continue
            
            # Batch process all active nodes
            batch_data = self._prepare_batch_data(active_nodes, [batch_destinations[i] for i in active_indices])
            
            # Single batched inference
            with torch.no_grad():
                logits, time_pred = self.model.infer(
                    batch_data['trace_road_id'],
                    batch_data['temporal_info'],
                    batch_data['trace_distance_mat'],
                    batch_data['trace_time_interval_mat'],
                    batch_data['trace_len'],
                    batch_data['destination_road_id'],
                    batch_data['candidate_road_id'],
                    batch_data['metric_dis'],
                    batch_data['metric_angle']
                )
            
            # Process results and update priority queues
            for idx, (i, (cur_node, reachable_road_id_list)) in enumerate(zip(active_indices, active_nodes)):
                log_output = F.log_softmax(logits[idx], dim=-1) + cur_node.log_prob
                cur_time_pred = time_pred[idx] * self.timestamp_std_tensor + self.timestamp_mean_tensor
                cur_time_pred = torch.expm1(cur_time_pred)
                cur_time_pred = torch.clamp(cur_time_pred, min=0.0)
                
                for j, candidate_road_id_val in enumerate(reachable_road_id_list):
                    candidate_log_prob = log_output[j].item()
                    next_datatime = cur_node.trace_datetime[-1] + timedelta(seconds=round(cur_time_pred[j].item()))
                    
                    state = search_states[i]
                    if candidate_road_id_val not in state['road_id2log_prob'] or candidate_log_prob > state['road_id2log_prob'][candidate_road_id_val]:
                        new_node = SearchNode(
                            trace_road_id=cur_node.trace_road_id+[candidate_road_id_val],
                            trace_datetime=cur_node.trace_datetime+[next_datatime],
                            log_prob=candidate_log_prob
                        )
                        state['pq'].put((-candidate_log_prob, new_node))
                        state['road_id2log_prob'][candidate_road_id_val] = candidate_log_prob
        
        return results
    
    def _prepare_batch_data(self, active_nodes, destinations):
        """Prepare batch data for multiple nodes"""
        
        # Find max lengths for padding
        max_trace_len = max(len(node.trace_road_id) for node, _ in active_nodes)
        max_candidate_len = max(len(reachable) for _, reachable in active_nodes)
        
        # Initialize batch tensors
        batch_trace_road_id = []
        batch_temporal_info = []
        batch_trace_distance_mat = []
        batch_trace_time_interval_mat = []
        batch_trace_len = []
        batch_destination_road_id = []
        batch_candidate_road_id = []
        batch_metric_dis = []
        batch_metric_angle = []
        
        for (cur_node, reachable_road_id_list), dest_id in zip(active_nodes, destinations):
            # Prepare trace data
            trace_road_id = np.array(cur_node.trace_road_id)
            temporal_info = np.array([(t.hour * 60.0 + t.minute + t.second / 60.0) / 1440.0 for t in cur_node.trace_datetime]).astype(np.float32)
            
            if cur_node.trace_datetime[0].weekday() >= 5:
                temporal_info *= -1.0
            
            trace_distance_mat = haversine_vector(self.road_center_gps[trace_road_id], self.road_center_gps[trace_road_id], 'm', comb=True).astype(np.float32)
            trace_distance_mat = np.clip(trace_distance_mat, 0.0, 1000.0) / 1000.0
            trace_time_interval_mat = np.abs(temporal_info[:, None] * 1440.0 - temporal_info * 1440.0)
            trace_time_interval_mat = np.clip(trace_time_interval_mat, 0.0, 5.0) / 5.0
            
            # Prepare candidate data
            candidate_road_id = np.array(reachable_road_id_list)
            metric_dis = haversine_vector(self.road_center_gps[candidate_road_id], self.road_center_gps[dest_id].reshape(1, -1), 'm', comb=True).reshape(-1).astype(np.float32)
            metric_dis = np.log1p((metric_dis - np.min(metric_dis)) / 100)
            
            cur_road_id = cur_node.trace_road_id[-1]
            cur_road_end_coord = self.road_end_coords[cur_road_id]
            candidate_end_coords = self.road_end_coords[candidate_road_id]
            dest_end_coord = self.road_end_coords[dest_id]
            
            vec1 = candidate_end_coords - cur_road_end_coord
            angle1 = np.arctan2(vec1[:, 1], vec1[:, 0])
            vec2 = dest_end_coord - cur_road_end_coord
            angle2 = np.arctan2(vec2[1], vec2[0])
            
            angle = np.abs(angle1 - angle2).astype(np.float32)
            angle = np.where(angle > math.pi, 2 * math.pi - angle, angle) / math.pi
            
            # Pad to max lengths
            padded_trace_road_id = np.pad(trace_road_id, (0, max_trace_len - len(trace_road_id)), 'constant')
            padded_temporal_info = np.pad(temporal_info, (0, max_trace_len - len(temporal_info)), 'constant')
            padded_trace_distance_mat = np.pad(trace_distance_mat, ((0, max_trace_len - trace_distance_mat.shape[0]), (0, max_trace_len - trace_distance_mat.shape[1])), 'constant')
            padded_trace_time_interval_mat = np.pad(trace_time_interval_mat, ((0, max_trace_len - trace_time_interval_mat.shape[0]), (0, max_trace_len - trace_time_interval_mat.shape[1])), 'constant')
            padded_candidate_road_id = np.pad(candidate_road_id, (0, max_candidate_len - len(candidate_road_id)), 'constant')
            padded_metric_dis = np.pad(metric_dis, (0, max_candidate_len - len(metric_dis)), 'constant', constant_values=1e6)
            padded_metric_angle = np.pad(angle, (0, max_candidate_len - len(angle)), 'constant', constant_values=1.0)
            
            batch_trace_road_id.append(padded_trace_road_id)
            batch_temporal_info.append(padded_temporal_info)
            batch_trace_distance_mat.append(padded_trace_distance_mat)
            batch_trace_time_interval_mat.append(padded_trace_time_interval_mat)
            batch_trace_len.append(len(trace_road_id))
            batch_destination_road_id.append(dest_id)
            batch_candidate_road_id.append(padded_candidate_road_id)
            batch_metric_dis.append(padded_metric_dis)
            batch_metric_angle.append(padded_metric_angle)
        
        # Convert to tensors
        return {
            'trace_road_id': torch.tensor(np.array(batch_trace_road_id), dtype=torch.long, device=self.device),
            'temporal_info': torch.tensor(np.array(batch_temporal_info), dtype=torch.float32, device=self.device),
            'trace_distance_mat': torch.tensor(np.array(batch_trace_distance_mat), dtype=torch.float32, device=self.device),
            'trace_time_interval_mat': torch.tensor(np.array(batch_trace_time_interval_mat), dtype=torch.float32, device=self.device),
            'trace_len': torch.tensor(batch_trace_len, dtype=torch.long, device=self.device),
            'destination_road_id': torch.tensor(batch_destination_road_id, dtype=torch.long, device=self.device),
            'candidate_road_id': torch.tensor(np.array(batch_candidate_road_id), dtype=torch.long, device=self.device),
            'metric_dis': torch.tensor(np.array(batch_metric_dis), dtype=torch.float32, device=self.device),
            'metric_angle': torch.tensor(np.array(batch_metric_angle), dtype=torch.float32, device=self.device)
        }


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
    parser.add_argument('--processes', type=int, default=1, help='Number of CPU processes for parallel trajectory generation')
    parser.add_argument('--beam_search', action='store_true', help='Use beam search instead of A* search')
    parser.add_argument('--beam_width', type=int, default=8, help='Beam width for beam search')
    parser.add_argument('--vectorized', action='store_true', help='Use vectorized GPU-parallel search')
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

    print(" ... Filtering valid OD pairs...")
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
    
    # Try to compile the model for faster inference (PyTorch 2.0+)
    try:
        import torch._dynamo
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ Model compiled with torch.compile for faster inference")
    except (ImportError, AttributeError):
        pass  # torch.compile not available, use regular model
    
    # Create searcher instance
    searcher = Searcher(model, data['reachable_road_id_dict'], data['geo'], data['road_center_gps'], 
                       data['road_end_coords'], data['timestamp_label_array_log1p_mean'], 
                       data['timestamp_label_array_log1p_std'], device)
    
    # Process trajectories 
    if args.processes > 1:
        print(f"🚀 Using {args.processes} CPU processes for parallel search")
        print("💡 Note: Using CPU for multiprocessing to avoid CUDA conflicts")
        # Use multiprocessing for CPU-bound work
        from multiprocessing import Pool
        
        # Prepare arguments for multiprocessing
        model_state = model.state_dict()
        model_config = {
            'road_network_encoder_config': config.road_network_encoder_config,
            'road_network_encoder_feature': config.road_network_encoder_feature,
            'trajectory_encoder_config': config.trajectory_encoder_config,
            'navigator_config': config.navigator_config,
            'road2zone': data['road2zone'],
        }
        mp_args = [(i, od[0], od[1], t, model_state, data, device, args.beam_search, args.beam_width, model_config) 
                   for i, (od, t) in enumerate(zip(od_coords, origin_datetime_list))]
        
        with Pool(processes=args.processes) as pool:
            results = list(tqdm(pool.imap(process_trajectory_mp, mp_args), 
                              total=len(mp_args), desc='Generating trajectories'))
        
        # Sort results by index and store
        results.sort(key=lambda x: x[0])
        for idx, trace_road_id, trace_datetime in results:
            gene_trace_road_id[idx] = trace_road_id
            gene_trace_datetime[idx] = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in trace_datetime]
            
    elif args.vectorized:
        print("🚀 Using vectorized parallel search")
        # Process all trajectories in parallel
        results = searcher.vectorized_search(od_coords, origin_datetime_list)
        for i, (trace_road_id, trace_datetime) in enumerate(results):
            gene_trace_road_id[i] = trace_road_id
            gene_trace_datetime[i] = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in trace_datetime]
    else:
        if args.beam_search:
            print(f"🔍 Using beam search with width {args.beam_width}")
            def search_fn(o, t, d):
                return searcher.beam_search(o, t, d, beam_width=args.beam_width)
        else:
            print("🔍 Using standard A* search")
            search_fn = searcher.search
            
        for i, ((origin_road_id, destination_road_id), origin_datetime) in enumerate(
                tqdm(zip(od_coords, origin_datetime_list), total=len(od_coords), desc='Generating trajectories')):
            trace_road_id, trace_datetime = search_fn(origin_road_id, origin_datetime, destination_road_id)
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
