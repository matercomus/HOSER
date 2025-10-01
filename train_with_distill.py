import os
import argparse
import json
import math
from datetime import datetime
from typing import Optional
import yaml
import time # For profiling
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import LineString
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import wandb
import optuna

# Import math (already imported above, but making sure it's available)

from utils import set_seed, create_nested_namespace, get_angle
from typing import Optional  # noqa: F811
from models.hoser import HOSER
from dataset import Dataset

# Distillation utilities (kept separate from original training)
from critics.distill_hook import DistillationManager, DistillConfig


class MyCollateFn:
    def __init__(self, timestamp_label_log1p_mean, timestamp_label_log1p_std, max_len: Optional[int] = None):
        self.timestamp_label_log1p_mean = timestamp_label_log1p_mean
        self.timestamp_label_log1p_std = timestamp_label_log1p_std
        self.max_len = int(max_len) if max_len else None

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
        batch_trace_grid_token = []
        batch_candidate_grid_token = []

        for (
            trace_road_id,
            temporal_info,
            trace_distance_mat,
            trace_time_interval_mat,
            trace_len,
            destination_road_id,
            candidate_road_id,
            metric_dis,
            metric_angle,
            candidate_len,
            road_label,
            timestamp_label,
            trace_grid_token,
            candidate_grid_token,
        ) in items:
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
            batch_trace_grid_token.append(trace_grid_token)
            batch_candidate_grid_token.append(candidate_grid_token)

        max_trace_len = max(batch_trace_len)
        max_candidate_len = max([max(x) if len(x) > 0 else 0 for x in batch_candidate_len])

        for i in range(len(batch_trace_road_id)):
            trace_pad_len = max_trace_len - batch_trace_len[i]

            if self.max_len is not None and batch_trace_len[i] > self.max_len:
                crop = batch_trace_len[i] - self.max_len
                batch_trace_road_id[i] = batch_trace_road_id[i][crop:]
                batch_temporal_info[i] = batch_temporal_info[i][crop:]
                batch_trace_distance_mat[i] = batch_trace_distance_mat[i][crop:, crop:]
                batch_trace_time_interval_mat[i] = batch_trace_time_interval_mat[i][crop:, crop:]
                batch_candidate_road_id[i] = batch_candidate_road_id[i][crop:]
                batch_metric_dis[i] = batch_metric_dis[i][crop:]
                batch_metric_angle[i] = batch_metric_angle[i][crop:]
                batch_candidate_len[i] = batch_candidate_len[i][crop:]
                batch_road_label[i] = batch_road_label[i][crop:]
                batch_timestamp_label[i] = batch_timestamp_label[i][crop:]
                if batch_trace_grid_token[i] is not None:
                    batch_trace_grid_token[i] = batch_trace_grid_token[i][crop:]
                if batch_candidate_grid_token[i] is not None:
                    batch_candidate_grid_token[i] = batch_candidate_grid_token[i][crop:]
                batch_trace_len[i] = self.max_len
                trace_pad_len = max_trace_len - batch_trace_len[i]

            batch_trace_road_id[i] = np.pad(batch_trace_road_id[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_temporal_info[i] = np.pad(batch_temporal_info[i], (0, trace_pad_len), 'constant', constant_values=0.0)
            batch_trace_distance_mat[i] = np.pad(batch_trace_distance_mat[i], ((0, trace_pad_len), (0, trace_pad_len)), 'constant', constant_values=0.0)
            batch_trace_time_interval_mat[i] = np.pad(batch_trace_time_interval_mat[i], ((0, trace_pad_len), (0, trace_pad_len)), 'constant', constant_values=0.0)

            metric_dis_pad_list = []
            metric_angle_pad_list = []
            candidate_pad_arrays = []

            for j in range(len(batch_candidate_road_id[i])):
                candidate_pad_len = max_candidate_len - batch_candidate_len[i][j]
                candidate_pad_arrays.append(
                    np.pad(batch_candidate_road_id[i][j], (0, candidate_pad_len), 'constant', constant_values=0)
                )
                metric_dis_pad_list.append(
                    np.pad(batch_metric_dis[i][j], (0, candidate_pad_len), 'constant', constant_values=0.0)
                )
                metric_angle_pad_list.append(
                    np.pad(batch_metric_angle[i][j], (0, candidate_pad_len), 'constant', constant_values=0.0)
                )

            pad_matrix_shape = (trace_pad_len, max_candidate_len)
            batch_candidate_road_id[i] = np.concatenate(
                (np.stack(candidate_pad_arrays), np.zeros(pad_matrix_shape, dtype=np.int64)), axis=0
            )
            batch_metric_dis[i] = np.concatenate(
                (np.stack(metric_dis_pad_list), np.zeros(pad_matrix_shape, dtype=np.float32)), axis=0
            )
            batch_metric_angle[i] = np.concatenate(
                (np.stack(metric_angle_pad_list), np.zeros(pad_matrix_shape, dtype=np.float32)), axis=0
            )

            batch_candidate_len[i] = np.pad(batch_candidate_len[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_road_label[i] = np.pad(batch_road_label[i], (0, trace_pad_len), 'constant', constant_values=0)
            batch_timestamp_label[i] = np.pad(batch_timestamp_label[i], (0, trace_pad_len), 'constant', constant_values=0.0)
            
            # Handle grid tokens padding
            if batch_trace_grid_token[i] is not None:
                batch_trace_grid_token[i] = np.pad(batch_trace_grid_token[i], (0, trace_pad_len), 'constant', constant_values=0)
            else:
                batch_trace_grid_token[i] = np.zeros(max_trace_len, dtype=np.int64)
                
            if batch_candidate_grid_token[i] is not None:
                # Pad candidate grid tokens similar to other candidate data
                candidate_grid_pad_list = []
                for j in range(len(batch_candidate_grid_token[i])):
                    candidate_pad_len = max_candidate_len - batch_candidate_len[i][j]
                    candidate_grid_pad_list.append(
                        np.pad(batch_candidate_grid_token[i][j], (0, candidate_pad_len), 'constant', constant_values=0)
                    )
                pad_matrix_shape = (trace_pad_len, max_candidate_len)
                batch_candidate_grid_token[i] = np.concatenate(
                    (np.stack(candidate_grid_pad_list), np.zeros(pad_matrix_shape, dtype=np.int64)), axis=0
                )
            else:
                batch_candidate_grid_token[i] = np.zeros((max_trace_len, max_candidate_len), dtype=np.int64)

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
        batch_trace_grid_token = torch.from_numpy(np.array(batch_trace_grid_token))
        batch_candidate_grid_token = torch.from_numpy(np.array(batch_candidate_grid_token))

        return (
            batch_trace_road_id,
            batch_temporal_info,
            batch_trace_distance_mat,
            batch_trace_time_interval_mat,
            batch_trace_len,
            batch_destination_road_id,
            batch_candidate_road_id,
            batch_metric_dis,
            batch_metric_angle,
            batch_candidate_len,
            batch_road_label,
            batch_timestamp_label,
            batch_trace_grid_token,
            batch_candidate_grid_token,
        )


def main(
    dataset: str = None,
    config_path: str = '',
    seed: int = 0,
    cuda: int = 0,
    data_dir: str = '',
    return_metrics: bool = False,
    optuna_trial = None,  # Pass Optuna trial for intermediate reporting
):
    """
    Main training function that can be called programmatically or from CLI.
    
    Args:
        dataset: Dataset name (e.g., 'Beijing')
        config_path: Path to YAML config file
        seed: Random seed
        cuda: CUDA device index
        data_dir: Path to HOSER-format data directory
        return_metrics: Whether to return validation metrics
        optuna_trial: Optuna trial object for intermediate value reporting
    
    Returns:
        dict: Training metrics if return_metrics=True, else None
    """
    # Handle CLI invocation
    if dataset is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--config', type=str, default='', help='Path to YAML config (overrides dataset default path)')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--cuda', type=int, default=0)
        parser.add_argument('--data_dir', type=str, default='', help='Path to HOSER-format data directory (overrides YAML)')
        parser.add_argument('--return_metrics', action='store_true', help='Return validation metrics (for Optuna)')
        args = parser.parse_args()
        
        dataset = args.dataset
        config_path = args.config
        seed = args.seed
        cuda = args.cuda
        data_dir = args.data_dir
        return_metrics = args.return_metrics

    set_seed(seed)
    device = f'cuda:{cuda}'

    # Prepare model config and related features (copied and generalized)
    # Prefer explicit --config; else fall back to dataset default; else Beijing
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path:
        _config_path = config_path
    else:
        _default_ds = dataset if dataset else 'Beijing'
        _config_path = os.path.join(_script_dir, 'config', f'{_default_ds}.yaml')
    if not os.path.exists(_config_path):
        raise FileNotFoundError(f"Config not found at {_config_path}. Ensure the dataset YAML exists.")
    with open(_config_path, 'r') as file:
        raw_config = yaml.safe_load(file)
    config = create_nested_namespace(raw_config)

    # Determine dataset name (for save/log dir names) from CLI or config filename
    dataset_name = dataset if dataset else os.path.splitext(os.path.basename(_config_path))[0]

    cfg_data_dir = getattr(config, 'data_dir', None) if hasattr(config, 'data_dir') else None
    base_data_dir = data_dir if data_dir else (cfg_data_dir if cfg_data_dir else f'./data/{dataset_name}')
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

    save_dir = f'./save/{dataset_name}/seed{seed}_distill'
    tensorboard_log_dir = f'./tensorboard_log/{dataset_name}/seed{seed}_distill'
    loguru_log_dir = f'./log/{dataset_name}/seed{seed}_distill'

    # config already loaded above

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
    if dataset_name in ['Beijing', 'San_Francisco']:
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

    # Dataloader knobs from config
    dl_num_workers = getattr(getattr(config, 'dataloader', {}), 'num_workers', 16)
    dl_pin = bool(getattr(getattr(config, 'dataloader', {}), 'pin_memory', False))
    dl_prefetch = getattr(getattr(config, 'dataloader', {}), 'prefetch_factor', 4)
    dl_persist = bool(getattr(getattr(config, 'dataloader', {}), 'persistent_workers', False))

    train_dataloader = DataLoader(
        train_dataset,
        config.optimizer_config.batch_size,
        shuffle=True,
        collate_fn=MyCollateFn(timestamp_mean, timestamp_std, getattr(config.trajectory_encoder_config, 'max_len', None)),
        num_workers=dl_num_workers,
        pin_memory=dl_pin,
        persistent_workers=dl_persist,
        prefetch_factor=dl_prefetch if dl_num_workers > 0 else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        config.optimizer_config.batch_size,
        shuffle=False,
        collate_fn=MyCollateFn(timestamp_mean, timestamp_std, getattr(config.trajectory_encoder_config, 'max_len', None)),
        num_workers=dl_num_workers,
        pin_memory=dl_pin,
        persistent_workers=dl_persist,
        prefetch_factor=dl_prefetch if dl_num_workers > 0 else None,
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

    # Initialize Weights & Biases if enabled
    wb_enable = bool(getattr(getattr(config, 'wandb', {}), 'enable', False))
    if wb_enable:
        wb_project = getattr(getattr(config, 'wandb', {}), 'project', 'hoser-distill')
        wb_run_name = getattr(getattr(config, 'wandb', {}), 'run_name', '') or f"{dataset_name}_b{config.optimizer_config.batch_size}_acc{getattr(config.optimizer_config,'accum_steps',1)}"
        wb_tags = list(getattr(getattr(config, 'wandb', {}), 'tags', []))
        # Log entire YAML config to wandb
        wandb.init(project=wb_project, name=wb_run_name, tags=wb_tags, config=raw_config)

    # Training performance knobs
    allow_tf32 = bool(getattr(getattr(config, 'training', {}), 'allow_tf32', False))
    cudnn_bench = bool(getattr(getattr(config, 'training', {}), 'cudnn_benchmark', False))
    compile_flag = bool(getattr(getattr(config, 'training', {}), 'torch_compile', False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_bench
    # torch.compile and cudagraphs interaction: allow disabling cudagraphs
    disable_cudagraphs = bool(getattr(getattr(config, 'training', {}), 'disable_cudagraphs', False))
    if compile_flag:
        try:
            # Get compile mode from config, default to "default" for faster compilation
            # Options: "default" (fast compile), "reduce-overhead" (balanced), "max-autotune" (best runtime)
            compile_mode = getattr(getattr(config, 'training', {}), 'torch_compile_mode', 'default')
            model = torch.compile(
                model,
                mode=compile_mode,
                disable=disable_cudagraphs
            )
            logger.info(f'[perf] torch.compile enabled ({compile_mode} mode)')
        except Exception as e:
            logger.info(f'[perf] torch.compile failed: {e}')

    # Cache road/zone embeddings once per run (do not recompute every batch)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            model.setup_road_network_features()

    scaler = torch.amp.GradScaler()

    # Apply compiled optimizer with LR scheduler pattern from PyTorch tutorial
    # This enables better compilation and caching of optimizer + scheduler
    base_lr = config.optimizer_config.learning_rate

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=config.optimizer_config.weight_decay
    )

    # Create LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.optimizer_config.max_epoch * len(train_dataloader),
        eta_min=base_lr * 0.1  # Minimum LR is 10% of base
    )

    # Distillation manager setup (optional)
    distill_mgr: Optional[DistillationManager] = None
    # Prefer config over CLI, with CLI as fallback if given
    distill_window_cfg = None
    if hasattr(config, 'distill') and hasattr(config.distill, 'window'):
        distill_window_cfg = int(config.distill.window)

    # Enable distill if config says so
    enable_distill = False
    if hasattr(config, 'distill') and hasattr(config.distill, 'enable'):
        enable_distill = bool(config.distill.enable)
    if enable_distill:
        dcfg = DistillConfig(
            enabled=True,
            repo_path=getattr(config.distill, 'repo', '/home/matt/Dev/LMTAD'),
            ckpt_path=getattr(config.distill, 'ckpt', ''),
            dtype='float16',
            window=int(distill_window_cfg or 64),
            lambda_kl=float(getattr(config.distill, 'lambda', 0.01)),
            temperature=float(getattr(config.distill, 'temperature', 2.0)),
            sample_steps_per_trace=1,
            grid_size=float(getattr(config.distill, 'grid_size', 0.001)),
            downsample_factor=int(getattr(config.distill, 'downsample', 1)),
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
    iter_num = 0

    accum_steps = int(getattr(config.optimizer_config, 'accum_steps', 1))
    
    # Track validation metrics for Optuna
    validation_metrics = []
    best_val_acc = 0.0
    profiler_ran = False # Ensure profiler runs only once
    
    for epoch_id in range(config.optimizer_config.max_epoch):
        model.train()
        for batch_id, (
            batch_trace_road_id,
            batch_temporal_info,
            batch_trace_distance_mat,
            batch_trace_time_interval_mat,
            batch_trace_len,
            batch_destination_road_id,
            batch_candidate_road_id,
            batch_metric_dis,
            batch_metric_angle,
            batch_candidate_len,
            batch_road_label,
            batch_timestamp_label,
            batch_trace_grid_token,
            batch_candidate_grid_token,
        ) in enumerate(tqdm(train_dataloader, desc=f'[training+distill] epoch{epoch_id+1}')):
            
            # --- Profiling Start ---
            torch.cuda.synchronize()
            t_start = time.time()

            batch_trace_road_id = batch_trace_road_id.to(device, non_blocking=True)
            batch_temporal_info = batch_temporal_info.to(device, non_blocking=True)
            batch_trace_distance_mat = batch_trace_distance_mat.to(device, non_blocking=True)
            batch_trace_time_interval_mat = batch_trace_time_interval_mat.to(device, non_blocking=True)
            batch_trace_len = batch_trace_len.to(device, non_blocking=True)
            batch_destination_road_id = batch_destination_road_id.to(device, non_blocking=True)
            # Optional candidate top-K cap to reduce memory/time
            top_k = int(getattr(getattr(config, 'data', {}), 'candidate_top_k', 0) or 0)
            if top_k > 0:
                # Move candidate selection to GPU for better performance
                with torch.no_grad():
                    # Move data to GPU first
                    batch_candidate_road_id = batch_candidate_road_id.to(device, non_blocking=True)
                    batch_metric_dis_gpu = batch_metric_dis.to(device, non_blocking=True)
                    batch_metric_angle = batch_metric_angle.to(device, non_blocking=True)
                    batch_candidate_len_gpu = batch_candidate_len.to(device, non_blocking=True)
                    
                    # Get the actual max candidates in this batch
                    max_cand = batch_candidate_road_id.size(-1)
                    k = min(top_k, max_cand)
                    
                    # Sort by distance and get indices
                    sorted_indices = torch.argsort(batch_metric_dis_gpu, dim=-1, descending=False)
                    
                    # Create a mask for valid candidates (within candidate_len)
                    B, T, C = batch_candidate_road_id.shape
                    indices = torch.arange(C, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, C]
                    valid_mask = indices < batch_candidate_len_gpu.unsqueeze(-1)  # [B, T, C]
                    
                    # Only keep top-k among valid candidates
                    # First, move invalid candidates to the end by setting their indices to a large value
                    sorted_indices_masked = torch.where(
                        valid_mask.gather(-1, sorted_indices),
                        sorted_indices,
                        torch.tensor(C, device=device)  # Push invalid to end
                    )
                    
                    # Now take only the first k
                    idx = sorted_indices_masked[..., :k]
                    
                    # For positions where idx >= C (invalid), replace with 0 to avoid index errors
                    idx = torch.clamp(idx, max=C-1)
                    
                    # Gather using the clamped indices
                    batch_candidate_road_id = torch.gather(batch_candidate_road_id, -1, idx)
                    batch_metric_dis = torch.gather(batch_metric_dis_gpu, -1, idx)
                    batch_metric_angle = torch.gather(batch_metric_angle, -1, idx)
                    batch_candidate_len = torch.clamp(batch_candidate_len.to(device, non_blocking=True), max=k)
                    
                    # Update road labels to match the filtered candidates
                    # Vectorized approach for efficiency
                    batch_road_label_device = batch_road_label.to(device, non_blocking=True)
                    
                    # Create a reverse mapping: for each position, find where the original label ended up
                    # Use the original sorted_indices (before masking) for label mapping
                    # We need to find where batch_road_label appears in the top-k sorted indices
                    
                    # Expand dimensions for broadcasting
                    labels_expanded = batch_road_label_device.unsqueeze(-1)  # [B, T, 1]
                    idx_expanded = sorted_indices[..., :k]  # [B, T, k] - use original sorted indices
                    
                    # Find matches: where does each label appear in the sorted indices?
                    matches = (idx_expanded == labels_expanded).float()  # [B, T, k]
                    
                    # Get the position of the match (if any)
                    has_match = matches.sum(dim=-1) > 0  # [B, T]
                    match_positions = torch.argmax(matches, dim=-1)  # [B, T]
                    
                    # Set labels: use match position if found, otherwise -100 (ignored by cross_entropy)
                    new_road_label = torch.where(has_match, match_positions, torch.tensor(-100, device=device))
                    batch_road_label = new_road_label

            batch_candidate_road_id = batch_candidate_road_id.to(device, non_blocking=True)
            batch_metric_dis = batch_metric_dis.to(device, non_blocking=True)
            batch_metric_angle = batch_metric_angle.to(device, non_blocking=True)
            batch_candidate_len = batch_candidate_len.to(device, non_blocking=True)
            batch_road_label = batch_road_label.to(device, non_blocking=True)
            batch_timestamp_label = batch_timestamp_label.to(device, non_blocking=True)
            batch_trace_grid_token = batch_trace_grid_token.to(device, non_blocking=True)
            batch_candidate_grid_token = batch_candidate_grid_token.to(device, non_blocking=True)

            torch.cuda.synchronize()
            t_data = time.time()

            iter_num += 1
            # Update LR using scheduler (correct order: optimizer before scheduler)
            # Note: scheduler.step() should be called AFTER optimizer.step()

            if (batch_id % accum_steps) == 0:
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

                # Debug check for out-of-bounds labels
                if torch.any(selected_road_label >= selected_logits.size(1)):
                    print(f"ERROR: Road labels out of bounds!")
                    print(f"  selected_logits.shape: {selected_logits.shape}")
                    print(f"  selected_road_label max: {selected_road_label.max().item()}")
                    print(f"  selected_road_label: {selected_road_label}")
                    print(f"  selected_candidate_len: {selected_candidate_len}")
                    raise ValueError("Road label exceeds logits dimension")

                loss_next_step = F.cross_entropy(masked_selected_logits, selected_road_label)

                selected_time_pred = time_pred[logits_mask][torch.arange(time_pred[logits_mask].size(0)), selected_road_label]
                selected_time_pred = selected_time_pred * timestamp_std + timestamp_mean
                selected_timestamp_label = batch_timestamp_label[logits_mask]
                selected_timestamp_label = selected_timestamp_label * timestamp_std + timestamp_mean

                loss_time_pred = torch.mean(torch.abs(selected_time_pred - selected_timestamp_label) / torch.clamp(selected_timestamp_label, min=1.0))

            torch.cuda.synchronize()
            t_forward = time.time()

            # Optional KL from teacher
            if distill_mgr is not None:
                kl_loss = distill_mgr.compute_kl_for_batch(
                    logits=logits,
                    batch_trace_road_id=batch_trace_road_id,
                    batch_trace_len=batch_trace_len,
                    batch_candidate_road_id=batch_candidate_road_id,
                    batch_candidate_len=batch_candidate_len,
                )

                # Check for NaN in KL loss
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    print("Warning: NaN/inf KL loss detected, setting to 0")
                    kl_loss = torch.tensor(0.0, device=device)

                loss = loss_next_step + loss_time_pred + distill_mgr.cfg.lambda_kl * kl_loss
            else:
                kl_loss = torch.tensor(0.0, device=device)
                loss = loss_next_step + loss_time_pred

            # Check for NaN/inf in total loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Critical: NaN/inf total loss detected! loss_next_step={loss_next_step.item():.4f}, loss_time_pred={loss_time_pred.item():.4f}, kl_loss={kl_loss.item():.4f}")
                # Set loss to a large value to prevent training but don't crash
                loss = torch.tensor(100.0, device=device, requires_grad=True)

            torch.cuda.synchronize()
            t_distill = time.time()

            scaler.scale(loss / accum_steps).backward()

            torch.cuda.synchronize()
            t_backward = time.time()

            if ((batch_id + 1) % accum_steps) == 0:
                # Unscale and clip just before stepping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.optimizer_config.max_norm)
                scaler.step(optimizer)
                scaler.update()

                # Step the LR scheduler AFTER optimizer step (PyTorch best practice)
                scheduler.step()

            torch.cuda.synchronize()
            t_optim = time.time()

            # --- Profiling Report ---
            if not profiler_ran and batch_id == 100: # After 100 warmup steps
                logger.info("--- Performance Profile (ms/batch) ---")
                total_time = (t_optim - t_start) * 1000
                profile_data = {
                    "Data Transfer": (t_data - t_start) * 1000,
                    "HOSER Forward": (t_forward - t_data) * 1000,
                    "Distill KL (Teacher)": (t_distill - t_forward) * 1000,
                    "Backward Pass": (t_backward - t_distill) * 1000,
                    "Optimizer Step": (t_optim - t_backward) * 1000,
                }
                for name, duration in profile_data.items():
                    percentage = (duration / total_time) * 100 if total_time > 0 else 0
                    logger.info(f"{name:<25}: {duration:>8.2f} ms ({percentage:.1f}%)")
                logger.info(f"{'Total':<25}: {total_time:>8.2f} ms")
                logger.info("----------------------------------------")
                profiler_ran = True


            # Logging
            step_idx = len(train_dataloader) * epoch_id + batch_id
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('loss_next_step', loss_next_step.item(), step_idx)
            writer.add_scalar('loss_time_pred', loss_time_pred.item(), step_idx)
            if distill_mgr is not None:
                writer.add_scalar('loss_kl', kl_loss.item(), step_idx)
            writer.add_scalar('loss', loss.item(), step_idx)
            writer.add_scalar('learning_rate', current_lr, step_idx)
            if wb_enable:
                wandb.log({
                    'loss_next_step': loss_next_step.item(),
                    'loss_time_pred': loss_time_pred.item(),
                    'loss': loss.item(),
                    **({'loss_kl': kl_loss.item()} if distill_mgr is not None else {}),
                    'lr': current_lr,
                    'epoch': epoch_id + 1,
                    'iter': step_idx,
                })

        logger.info(f'[training+distill] epoch{epoch_id+1}, loss_next_step {loss_next_step.item():.3f}, loss_time_pred {loss_time_pred.item():.3f}' + (f', loss_kl {kl_loss.item():.3f}' if distill_mgr is not None else ''))

        # -----------------------------
        # Validation (per-epoch)
        # -----------------------------
        model.eval()
        val_next_step_correct_cnt, val_next_step_total_cnt = 0, 0
        val_time_pred_mape_sum, val_time_pred_total_cnt = 0, 0
        with torch.no_grad():
            for (
                batch_trace_road_id,
                batch_temporal_info,
                batch_trace_distance_mat,
                batch_trace_time_interval_mat,
                batch_trace_len,
                batch_destination_road_id,
                batch_candidate_road_id,
                batch_metric_dis,
                batch_metric_angle,
                batch_candidate_len,
                batch_road_label,
                batch_timestamp_label,
                batch_trace_grid_token,
                batch_candidate_grid_token,
            ) in val_dataloader:
                batch_trace_road_id = batch_trace_road_id.to(device, non_blocking=True)
                batch_temporal_info = batch_temporal_info.to(device, non_blocking=True)
                batch_trace_distance_mat = batch_trace_distance_mat.to(device, non_blocking=True)
                batch_trace_time_interval_mat = batch_trace_time_interval_mat.to(device, non_blocking=True)
                batch_trace_len = batch_trace_len.to(device, non_blocking=True)
                batch_destination_road_id = batch_destination_road_id.to(device, non_blocking=True)

                # Apply same candidate top-k cap as training (GPU-optimized)
                top_k = int(getattr(getattr(config, 'data', {}), 'candidate_top_k', 0) or 0)
                if top_k > 0:
                    with torch.no_grad():
                        # Move data to GPU first
                        batch_candidate_road_id = batch_candidate_road_id.to(device, non_blocking=True)
                        batch_metric_dis_gpu = batch_metric_dis.to(device, non_blocking=True)
                        batch_metric_angle = batch_metric_angle.to(device, non_blocking=True)
                        batch_candidate_len_gpu = batch_candidate_len.to(device, non_blocking=True)
                        
                        # Get the actual max candidates in this batch
                        max_cand = batch_candidate_road_id.size(-1)
                        k = min(top_k, max_cand)
                        
                        # Sort by distance and get indices
                        sorted_indices = torch.argsort(batch_metric_dis_gpu, dim=-1, descending=False)
                        
                        # Create a mask for valid candidates (within candidate_len)
                        B, T, C = batch_candidate_road_id.shape
                        indices = torch.arange(C, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, C]
                        valid_mask = indices < batch_candidate_len_gpu.unsqueeze(-1)  # [B, T, C]
                        
                        # Only keep top-k among valid candidates
                        # First, move invalid candidates to the end by setting their indices to a large value
                        sorted_indices_masked = torch.where(
                            valid_mask.gather(-1, sorted_indices),
                            sorted_indices,
                            torch.tensor(C, device=device)  # Push invalid to end
                        )
                        
                        # Now take only the first k
                        idx = sorted_indices_masked[..., :k]
                        
                        # For positions where idx >= C (invalid), replace with 0 to avoid index errors
                        idx = torch.clamp(idx, max=C-1)
                        
                        # Gather using the clamped indices
                        batch_candidate_road_id = torch.gather(batch_candidate_road_id, -1, idx)
                        batch_metric_dis = torch.gather(batch_metric_dis_gpu, -1, idx)
                        batch_metric_angle = torch.gather(batch_metric_angle, -1, idx)
                        batch_candidate_len = torch.clamp(batch_candidate_len.to(device, non_blocking=True), max=k)
                        
                        # Update road labels to match the filtered candidates
                        # Vectorized approach for efficiency
                        batch_road_label_device = batch_road_label.to(device, non_blocking=True)
                        
                        # Create a reverse mapping: for each position, find where the original label ended up
                        # Use the original sorted_indices (before masking) for label mapping
                        # We need to find where batch_road_label appears in the top-k sorted indices
                        
                        # Expand dimensions for broadcasting
                        labels_expanded = batch_road_label_device.unsqueeze(-1)  # [B, T, 1]
                        idx_expanded = sorted_indices[..., :k]  # [B, T, k] - use original sorted indices
                        
                        # Find matches: where does each label appear in the sorted indices?
                        matches = (idx_expanded == labels_expanded).float()  # [B, T, k]
                        
                        # Get the position of the match (if any)
                        has_match = matches.sum(dim=-1) > 0  # [B, T]
                        match_positions = torch.argmax(matches, dim=-1)  # [B, T]
                        
                        # Set labels: use match position if found, otherwise -100 (ignored by cross_entropy)
                        new_road_label = torch.where(has_match, match_positions, torch.tensor(-100, device=device))
                        batch_road_label = new_road_label

                batch_candidate_road_id = batch_candidate_road_id.to(device, non_blocking=True)
                batch_metric_dis = batch_metric_dis.to(device, non_blocking=True)
                batch_metric_angle = batch_metric_angle.to(device, non_blocking=True)
                batch_candidate_len = batch_candidate_len.to(device, non_blocking=True)
                batch_road_label = batch_road_label.to(device, non_blocking=True)
                batch_timestamp_label = batch_timestamp_label.to(device, non_blocking=True)
                batch_trace_grid_token = batch_trace_grid_token.to(device, non_blocking=True)
                batch_candidate_grid_token = batch_candidate_grid_token.to(device, non_blocking=True)

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

                val_next_step_correct_cnt += torch.sum(torch.argmax(masked_selected_logits, dim=1) == selected_road_label).item()
                val_next_step_total_cnt += torch.sum(batch_trace_len).item()

                selected_time_pred = time_pred[logits_mask][torch.arange(time_pred[logits_mask].size(0)), selected_road_label]
                selected_time_pred = selected_time_pred * timestamp_std + timestamp_mean
                selected_timestamp_label = batch_timestamp_label[logits_mask]
                selected_timestamp_label = selected_timestamp_label * timestamp_std + timestamp_mean
                val_time_pred_mape_sum += torch.sum(torch.abs(selected_time_pred - selected_timestamp_label) / torch.clamp(selected_timestamp_label, min=1.0))
                val_time_pred_total_cnt += torch.sum(batch_trace_len).item()

        val_acc = val_next_step_correct_cnt / max(1, val_next_step_total_cnt)
        val_mape = (val_time_pred_mape_sum / max(1, val_time_pred_total_cnt)).item()

        # Check for invalid validation metrics
        if math.isnan(val_acc) or math.isinf(val_acc) or val_acc < 0 or val_acc > 1:
            print(f"⚠️  Warning: Invalid val_acc detected: {val_acc}, setting to 0")
            val_acc = 0.0

        if math.isnan(val_mape) or math.isinf(val_mape) or val_mape < 0:
            print(f"⚠️  Warning: Invalid val_mape detected: {val_mape}, setting to large value")
            val_mape = 1000.0

        writer.add_scalar('val/next_step_acc', val_acc, epoch_id)
        writer.add_scalar('val/time_pred_mape', val_mape, epoch_id)
        if wb_enable:
            wandb.log({'val/next_step_acc': val_acc, 'val/time_pred_mape': val_mape, 'epoch': epoch_id + 1})

        # Track for Optuna
        validation_metrics.append({
            'epoch': epoch_id + 1,
            'val_acc': val_acc,
            'val_mape': val_mape
        })
        best_val_acc = max(best_val_acc, val_acc)
        
        # Report intermediate value to Optuna for pruning
        if optuna_trial is not None:
            optuna_trial.report(val_acc, epoch_id)
            # Check if trial should be pruned
            if optuna_trial.should_prune():
                logger.info(f"🔪 Trial pruned at epoch {epoch_id + 1} (val_acc={val_acc:.4f})")
                raise optuna.TrialPruned()
        
        model.train()

    torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
    logger.info(f'[distill] Saved best model to {save_dir}/best.pth')
    if wb_enable:
        wandb.save(os.path.join(save_dir, 'best.pth'))
        wandb.finish()
    
    # Return metrics for Optuna if requested
    if return_metrics:
        # Extract final learning rate (handle both tensor and float)
        if 'current_lr' in locals():
            final_lr = current_lr.item() if torch.is_tensor(current_lr) else current_lr
        else:
            final_lr = base_lr
            
        return {
            'best_val_acc': best_val_acc,
            'final_val_acc': validation_metrics[-1]['val_acc'] if validation_metrics else 0.0,
            'final_val_mape': validation_metrics[-1]['val_mape'] if validation_metrics else float('inf'),
            'validation_history': validation_metrics,
            'final_lr': final_lr
        }


if __name__ == '__main__':
    main()


