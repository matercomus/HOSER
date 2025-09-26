import os
import argparse
import json
import math
from datetime import datetime
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import LineString
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from utils import set_seed, create_nested_namespace, get_angle
from models.hoser import HOSER
from dataset import Dataset

# Distillation utilities (kept separate from original training)
from critics.distill_hook import DistillationManager, DistillConfig


class MyCollateFn:
    def __init__(self, timestamp_label_log1p_mean, timestamp_label_log1p_std):
        self.timestamp_label_log1p_mean = timestamp_label_log1p_mean
        self.timestamp_label_log1p_std = timestamp_label_log1p_std

    def __call__(self, items):
        batch_trace_road_id = []
        batch_temporal_info = []
        batch_trace_distance_mat = []
        batch_trace_time_interval_mat = []
        batch_trace_len = []
        batch_destination_road_id = []
        batch_candidate_road_id = []
        batch_metric_dis = []
        batch_metric_angle = []
        batch_candidate_len = []
        batch_road_label = []
        batch_timestamp_label = []

        for trace_road_id, temporal_info, trace_distance_mat, trace_time_interval_mat, trace_len, destination_road_id, candidate_road_id, metric_dis, metric_angle, candidate_len, road_label, timestamp_label in items:
            batch_trace_road_id.append(trace_road_id)
            batch_temporal_info.append(temporal_info)
            batch_trace_distance_mat.append(trace_distance_mat)
            batch_trace_time_interval_mat.append(trace_time_interval_mat)
            batch_trace_len.append(trace_len)
            batch_destination_road_id.append(destination_road_id)
            batch_candidate_road_id.append(candidate_road_id)
            batch_metric_dis.append(metric_dis)
            batch_metric_angle.append(metric_angle)
            batch_candidate_len.append(candidate_len)
            batch_road_label.append(road_label)
            batch_timestamp_label.append(timestamp_label)

        max_trace_len = max(batch_trace_len)
        max_candidate_len = max([max(x) for x in batch_candidate_len])

        for i in range(len(batch_trace_road_id)):
            trace_pad_len = max_trace_len - batch_trace_len[i]

            batch_trace_road_id[i] = np.pad(batch_trace_road_id[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_temporal_info[i] = np.pad(batch_temporal_info[i], (0, trace_pad_len), 'constant', constant_values=0.0)
            batch_trace_distance_mat[i] = np.pad(batch_trace_distance_mat[i], ((0, trace_pad_len), (0, trace_pad_len)), 'constant', constant_values=0.0)
            batch_trace_time_interval_mat[i] = np.pad(batch_trace_time_interval_mat[i], ((0, trace_pad_len), (0, trace_pad_len)), 'constant', constant_values=0.0)
            
            for j in range(len(batch_candidate_road_id[i])):
                candidate_pad_len = max_candidate_len - batch_candidate_len[i][j]

                batch_candidate_road_id[i][j] = np.pad(batch_candidate_road_id[i][j], (0, candidate_pad_len), 'constant', constant_values=0)
                batch_metric_dis[i][j] = np.pad(batch_metric_dis[i][j], (0, candidate_pad_len), 'constant', constant_values=0.0)
                batch_metric_angle[i][j] = np.pad(batch_metric_angle[i][j], (0, candidate_pad_len), 'constant', constant_values=0.0)

            batch_candidate_road_id[i] = np.concatenate((np.stack(batch_candidate_road_id[i]), np.zeros((trace_pad_len, max_candidate_len), dtype=np.int64)), axis=0)
            batch_metric_dis[i] = np.concatenate((np.stack(batch_metric_dis[i]), np.zeros((trace_pad_len, max_candidate_len), dtype=np.float32)), axis=0)
            batch_metric_angle[i] = np.concatenate((np.stack(batch_metric_angle[i]), np.zeros((trace_pad_len, max_candidate_len), dtype=np.float32)), axis=0)

            batch_candidate_len[i] = np.pad(batch_candidate_len[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_road_label[i] = np.pad(batch_road_label[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_timestamp_label[i] = np.pad(batch_timestamp_label[i], (0, trace_pad_len), 'constant', constant_values=0.0)

        batch_timestamp_label = (np.log1p(batch_timestamp_label) - self.timestamp_label_log1p_mean) / self.timestamp_label_log1p_std

        batch_trace_road_id = torch.from_numpy(np.array(batch_trace_road_id))
        batch_temporal_info = torch.from_numpy(np.array(batch_temporal_info))
        batch_trace_distance_mat = torch.from_numpy(np.array(batch_trace_distance_mat))
        batch_trace_time_interval_mat = torch.from_numpy(np.array(batch_trace_time_interval_mat))
        batch_trace_len = torch.from_numpy(np.array(batch_trace_len))
        batch_destination_road_id = torch.from_numpy(np.array(batch_destination_road_id))
        batch_candidate_road_id = torch.from_numpy(np.array(batch_candidate_road_id))
        batch_metric_dis = torch.from_numpy(np.array(batch_metric_dis))
        batch_metric_angle = torch.from_numpy(np.array(batch_metric_angle))
        batch_candidate_len = torch.from_numpy(np.array(batch_candidate_len))
        batch_road_label = torch.from_numpy(np.array(batch_road_label))
        batch_timestamp_label = torch.from_numpy(np.array(batch_timestamp_label))

        return batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle, batch_candidate_len, batch_road_label, batch_timestamp_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--distill', action='store_true', help='Enable LM-TAD distillation')
    parser.add_argument('--distill_repo', type=str, default='/home/matt/Dev/LMTAD')
    parser.add_argument('--distill_ckpt', type=str, default='')
    parser.add_argument('--distill_lambda', type=float, default=0.01)
    parser.add_argument('--distill_temperature', type=float, default=2.0)
    parser.add_argument('--distill_window', type=int, default=64)
    parser.add_argument('--distill_grid_size', type=float, default=0.001)
    parser.add_argument('--distill_downsample', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='', help='Path to HOSER-format data directory (overrides --dataset default path)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = f'cuda:{args.cuda}'

    # Prepare model config and related features (copied and generalized)
    base_data_dir = args.data_dir if args.data_dir else f'./data/{args.dataset}'
    geo_file = os.path.join(base_data_dir, 'roadmap.geo')
    rel_file = os.path.join(base_data_dir, 'roadmap.rel')
    train_traj_file = os.path.join(base_data_dir, 'train.csv')
    val_traj_file = os.path.join(base_data_dir, 'val.csv')
    test_traj_file = os.path.join(base_data_dir, 'test.csv')
    road_network_partition_file = os.path.join(base_data_dir, 'road_network_partition')
    zone_trans_mat_file = os.path.join(base_data_dir, 'zone_trans_mat.npy')

    # Sanity check for required files
    required_files = [geo_file, rel_file, train_traj_file, val_traj_file, test_traj_file, road_network_partition_file, zone_trans_mat_file]
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}. Set --data_dir to your HOSER-format directory.")

    save_dir = f'./save/{args.dataset}/seed{args.seed}_distill'
    tensorboard_log_dir = f'./tensorboard_log/{args.dataset}/seed{args.seed}_distill'
    loguru_log_dir = f'./log/{args.dataset}/seed{args.seed}_distill'

    with open(f'./config/{args.dataset}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = create_nested_namespace(config)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_log_dir)
    os.makedirs(loguru_log_dir, exist_ok=True)
    logger.add(os.path.join(loguru_log_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'), level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {message}")

    geo = pd.read_csv(geo_file)
    rel = pd.read_csv(rel_file)
    num_roads = len(geo)

    road_attr_len = geo['length'].to_numpy().astype(np.float32)
    road_attr_len = np.log1p(road_attr_len)
    road_attr_len = (road_attr_len - np.mean(road_attr_len)) / np.std(road_attr_len)

    road_attr_type = geo['highway'].values.tolist()
    if args.dataset in ['Beijing', 'San_Francisco']:
        for i in range(len(road_attr_type)):
            if isinstance(road_attr_type[i], str) and road_attr_type[i].startswith('[') and road_attr_type[i].endswith(']'):
                info = eval(road_attr_type[i])
                road_attr_type[i] = info[0] if info[0] != 'unclassified' else info[1]
    le = LabelEncoder()
    road_attr_type = le.fit_transform(road_attr_type)

    road_attr_lon = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.x for _, row in geo.iterrows()]).astype(np.float32)
    road_attr_lat = np.array([LineString(coordinates=eval(row['coordinates'])).centroid.y for _, row in geo.iterrows()]).astype(np.float32)

    road_attr_lon_std = (road_attr_lon - np.mean(road_attr_lon)) / np.std(road_attr_lon)
    road_attr_lat_std = (road_attr_lat - np.mean(road_attr_lat)) / np.std(road_attr_lat)

    # Boundaries for grid mapper
    min_lng = float(np.min([min([p[0] for p in eval(row['coordinates'])]) for _, row in geo.iterrows()]))
    max_lng = float(np.max([max([p[0] for p in eval(row['coordinates'])]) for _, row in geo.iterrows()]))
    min_lat = float(np.min([min([p[1] for p in eval(row['coordinates'])]) for _, row in geo.iterrows()]))
    max_lat = float(np.max([max([p[1] for p in eval(row['coordinates'])]) for _, row in geo.iterrows()]))

    adj_row = []
    adj_col = []
    adj_angle = []
    adj_reachability = []

    reachable_road_id_dict = {i: [] for i in range(num_roads)}
    for _, row in rel.iterrows():
        reachable_road_id_dict[row['origin_id']].append(row['destination_id'])

    coord2road_id = {}
    for road_id, row in geo.iterrows():
        coord = json.loads(row['coordinates'], parse_float=str)
        start_coord = tuple(coord[0])
        end_coord = tuple(coord[-1])
        coord2road_id.setdefault(start_coord, []).append(road_id)
        coord2road_id.setdefault(end_coord, []).append(road_id)

    road_adj_lists = {i: set() for i in range(num_roads)}
    for v in coord2road_id.values():
        for r1 in v:
            for r2 in v:
                if r1 != r2:
                    road_adj_lists[r1].add(r2)

    for road_id in range(num_roads):
        for adj_road_id in list(road_adj_lists[road_id]):
            adj_row.append(road_id)
            adj_col.append(adj_road_id)

            road_id_coord = eval(geo.loc[road_id, 'coordinates'])
            adj_road_id_coord = eval(geo.loc[adj_road_id, 'coordinates'])
            road_id_angle = get_angle(road_id_coord[0][1], road_id_coord[0][0], road_id_coord[-1][1], road_id_coord[-1][0])
            adj_road_id_angle = get_angle(adj_road_id_coord[0][1], adj_road_id_coord[0][0], adj_road_id_coord[-1][1], adj_road_id_coord[-1][0])
            angle = abs(road_id_angle - adj_road_id_angle)
            if angle > math.pi:
                angle = math.pi * 2 - angle
            angle /= math.pi
            adj_angle.append(angle)

            adj_reachability.append(1.0 if adj_road_id in reachable_road_id_dict[road_id] else 0.0)

    road_edge_index = np.stack([
        np.array(adj_row).astype(np.int64),
        np.array(adj_col).astype(np.int64),
    ], axis=0)
    intersection_attr = np.stack([
        np.array(adj_angle).astype(np.float32),
        np.array(adj_reachability).astype(np.float32),
    ], axis=1)

    zone_trans_mat = np.load(zone_trans_mat_file)
    zone_edge_index = np.stack(zone_trans_mat.nonzero())

    zone_trans_mat = zone_trans_mat.astype(np.float32)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(np.sum(zone_trans_mat, axis=1), 1.0))
    zone_trans_mat_norm = zone_trans_mat * D_inv_sqrt[:, np.newaxis] * D_inv_sqrt[np.newaxis, :]
    zone_edge_weight = zone_trans_mat_norm[zone_edge_index[0], zone_edge_index[1]]

    config.road_network_encoder_config.road_id_num_embeddings = num_roads
    config.road_network_encoder_config.type_num_embeddings = len(np.unique(road_attr_type))
    config.road_network_encoder_feature.road_attr.len = road_attr_len
    config.road_network_encoder_feature.road_attr.type = road_attr_type
    config.road_network_encoder_feature.road_attr.lon = road_attr_lon_std
    config.road_network_encoder_feature.road_attr.lat = road_attr_lat_std
    config.road_network_encoder_feature.road_edge_index = road_edge_index
    config.road_network_encoder_feature.intersection_attr = intersection_attr
    config.road_network_encoder_feature.zone_edge_index = zone_edge_index
    config.road_network_encoder_feature.zone_edge_weight = zone_edge_weight

    road2zone = []
    with open(road_network_partition_file, 'r') as file:
        for line in file:
            road2zone.append(int(line.strip()))
    road2zone = np.array(road2zone)

    # Datasets
    train_dataset = Dataset(geo_file, rel_file, train_traj_file)
    val_dataset = Dataset(geo_file, rel_file, val_traj_file)

    train_stats = train_dataset.get_stats()
    timestamp_mean = train_stats['mean']
    timestamp_std = train_stats['std']

    logger.info(f'[distill] timestamp_label_array_log1p_mean {timestamp_mean:.3f}')
    logger.info(f'[distill] timestamp_label_array_log1p_std {timestamp_std:.3f}')

    train_dataloader = DataLoader(
        train_dataset,
        config.optimizer_config.batch_size,
        shuffle=True,
        collate_fn=MyCollateFn(timestamp_mean, timestamp_std),
        num_workers=16,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        config.optimizer_config.batch_size,
        shuffle=False,
        collate_fn=MyCollateFn(timestamp_mean, timestamp_std),
        num_workers=16,
        pin_memory=False,
    )

    # Model
    model = HOSER(
        config.road_network_encoder_config,
        config.road_network_encoder_feature,
        config.trajectory_encoder_config,
        config.navigator_config,
        road2zone,
    ).to(device)

    logger.info('[distill] Initialized HOSER model for distillation training')

    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer_config.learning_rate, weight_decay=config.optimizer_config.weight_decay)

    # Distillation manager setup (optional)
    distill_mgr: Optional[DistillationManager] = None
    if args.distill:
        dcfg = DistillConfig(
            enabled=True,
            repo_path=args.distill_repo,
            ckpt_path=args.distill_ckpt,
            dtype='float16',
            window=int(args.distill_window),
            lambda_kl=float(args.distill_lambda),
            temperature=float(args.distill_temperature),
            sample_steps_per_trace=1,
            grid_size=float(args.distill_grid_size),
            downsample_factor=int(args.distill_downsample),
            verify_grid_dims=True,
        )
        distill_mgr = DistillationManager(
            dcfg,
            device=device,
            boundary_min_lat=min_lat,
            boundary_max_lat=max_lat,
            boundary_min_lng=min_lng,
            boundary_max_lng=max_lng,
            road_centroids_lat=road_attr_lat,
            road_centroids_lng=road_attr_lon,
            logger=logger,
        )
        logger.info('[distill] Distillation manager initialized')

    total_iters = config.optimizer_config.max_epoch * len(train_dataloader)
    warmup_iters = config.optimizer_config.max_epoch * len(train_dataloader) * config.optimizer_config.warmup_ratio
    iter_num = 0
    def get_lr(it):
        if it < warmup_iters:
            return config.optimizer_config.learning_rate * it / warmup_iters
        assert it <= total_iters
        decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return coeff * config.optimizer_config.learning_rate

    for epoch_id in range(config.optimizer_config.max_epoch):
        model.train()
        for batch_id, (batch_trace_road_id, batch_temporal_info, batch_trace_distance_mat, batch_trace_time_interval_mat, batch_trace_len, batch_destination_road_id, batch_candidate_road_id, batch_metric_dis, batch_metric_angle, batch_candidate_len, batch_road_label, batch_timestamp_label) in enumerate(tqdm(train_dataloader, desc=f'[training+distill] epoch{epoch_id+1}')):
            with torch.amp.autocast(device_type='cuda'):
                model.setup_road_network_features()

            batch_trace_road_id = batch_trace_road_id.to(device)
            batch_temporal_info = batch_temporal_info.to(device)
            batch_trace_distance_mat = batch_trace_distance_mat.to(device)
            batch_trace_time_interval_mat = batch_trace_time_interval_mat.to(device)
            batch_trace_len = batch_trace_len.to(device)
            batch_destination_road_id = batch_destination_road_id.to(device)
            batch_candidate_road_id = batch_candidate_road_id.to(device)
            batch_metric_dis = batch_metric_dis.to(device)
            batch_metric_angle = batch_metric_angle.to(device)
            batch_candidate_len = batch_candidate_len.to(device)
            batch_road_label = batch_road_label.to(device)
            batch_timestamp_label = batch_timestamp_label.to(device)

            iter_num += 1
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                logits, time_pred = model(
                    batch_trace_road_id,
                    batch_temporal_info,
                    batch_trace_distance_mat,
                    batch_trace_time_interval_mat,
                    batch_trace_len,
                    batch_destination_road_id,
                    batch_candidate_road_id,
                    batch_metric_dis,
                    batch_metric_angle,
                )

                logits_mask = torch.arange(logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < batch_trace_len.unsqueeze(1)
                selected_logits = logits[logits_mask]
                selected_candidate_len = batch_candidate_len[logits_mask]
                selected_road_label = batch_road_label[logits_mask]

                candidate_mask = torch.arange(selected_logits.size(1), dtype=torch.int64, device=device).unsqueeze(0) < selected_candidate_len.unsqueeze(1)
                masked_selected_logits = selected_logits.masked_fill(~candidate_mask, float('-inf'))

                loss_next_step = F.cross_entropy(masked_selected_logits, selected_road_label)

                selected_time_pred = time_pred[logits_mask][torch.arange(time_pred[logits_mask].size(0)), selected_road_label]
                selected_time_pred = selected_time_pred * timestamp_std + timestamp_mean
                selected_timestamp_label = batch_timestamp_label[logits_mask]
                selected_timestamp_label = selected_timestamp_label * timestamp_std + timestamp_mean

                loss_time_pred = torch.mean(torch.abs(selected_time_pred - selected_timestamp_label) / torch.clamp(selected_timestamp_label, min=1.0))

                # Optional KL from teacher
                if distill_mgr is not None:
                    kl_loss = distill_mgr.compute_kl_for_batch(
                        logits=logits,
                        batch_trace_road_id=batch_trace_road_id,
                        batch_trace_len=batch_trace_len,
                        batch_candidate_road_id=batch_candidate_road_id,
                        batch_candidate_len=batch_candidate_len,
                    )
                    loss = loss_next_step + loss_time_pred + distill_mgr.cfg.lambda_kl * kl_loss
                else:
                    kl_loss = torch.tensor(0.0, device=device)
                    loss = loss_next_step + loss_time_pred

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.optimizer_config.max_norm)
            scaler.step(optimizer)
            scaler.update()

            # Logging
            writer.add_scalar('loss_next_step', loss_next_step.item(), len(train_dataloader) * epoch_id + batch_id)
            writer.add_scalar('loss_time_pred', loss_time_pred.item(), len(train_dataloader) * epoch_id + batch_id)
            if distill_mgr is not None:
                writer.add_scalar('loss_kl', kl_loss.item(), len(train_dataloader) * epoch_id + batch_id)
            writer.add_scalar('loss', loss.item(), len(train_dataloader) * epoch_id + batch_id)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], len(train_dataloader) * epoch_id + batch_id)

        logger.info(f'[training+distill] epoch{epoch_id+1}, loss_next_step {loss_next_step.item():.3f}, loss_time_pred {loss_time_pred.item():.3f}' + (f', loss_kl {kl_loss.item():.3f}' if distill_mgr is not None else ''))

    torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
    logger.info(f'[distill] Saved best model to {save_dir}/best.pth')


