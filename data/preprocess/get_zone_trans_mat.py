from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=['Beijing'],
                       help='Dataset names to process (e.g., Beijing porto_hoser)')
    args = parser.parse_args()
    
    for dataset in args.datasets:
        print(f'Processing {dataset} dataset')

        road2zone = []
        with open(f'../{dataset}/road_network_partition', 'r') as file:
            for line in file:
                    road2zone.append(int(line.strip()))

        zone_cnt = max(road2zone) + 1
        zone_trans_mat = np.zeros((zone_cnt, zone_cnt), dtype=np.int64)

        traj = pd.read_csv(f'../{dataset}/train.csv')
        skipped_trajectories = 0
        
        for _, row in tqdm(traj.iterrows(), total=len(traj)):
                rid_list = eval(row['rid_list'])
                
                # Handle single-road trajectories (result of dead-end filtering)
                if isinstance(rid_list, int):
                    # Convert single integer to list for consistency
                    rid_list = [rid_list]
                
                # Skip trajectories with less than 2 roads (no transitions possible)
                if len(rid_list) < 2:
                    skipped_trajectories += 1
                    continue
                    
                zone_list = [road2zone[rid] for rid in rid_list]
                for prev_zone, next_zone in zip(zone_list[:-1], zone_list[1:]):
                    if prev_zone != next_zone:
                        zone_trans_mat[prev_zone, next_zone] += 1

        print(f'Skipped {skipped_trajectories} single-road trajectories')
        np.save(f'../{dataset}/zone_trans_mat.npy', zone_trans_mat)
