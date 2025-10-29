import os
import subprocess
import pandas as pd
import argparse

# KaHIP installation path
KAHIP_PATH = "/home/matt/Dev/KaHIP/build/kaffpa"


def main(args):
    current_directory = os.getcwd()
    for dataset in args.datasets:
        print(f"Processing {dataset} dataset")

        geo = pd.read_csv(f"../{dataset}/roadmap.geo")
        rel = pd.read_csv(f"../{dataset}/roadmap.rel")

        num_roads = len(geo)
        print(f"Number of roads: {num_roads}")

        # Build adjacency lists more efficiently without huge matrix
        adj_lists = {i: set() for i in range(num_roads)}

        print("Building adjacency lists from relations...")
        for _, row in rel.iterrows():
            origin_id = row["origin_id"]
            destination_id = row["destination_id"]
            adj_lists[origin_id].add(destination_id)
            adj_lists[destination_id].add(origin_id)

        # Count total edges (each undirected edge counted once)
        total_edges = sum(len(adj_list) for adj_list in adj_lists.values()) // 2
        print(f"Total edges: {total_edges}")

        print("Writing graph input file...")
        with open(f"../{dataset}/graph_input.tmp", "w") as file:
            file.write(f"{num_roads} {total_edges}\n")
            for rid in range(num_roads):
                adj_rid_list = sorted(list(adj_lists[rid]))
                # KaHIP expects 1-indexed node IDs
                file.write(" ".join([str(adj_rid + 1) for adj_rid in adj_rid_list]))
                file.write("\n")

        os.chdir(f"../{dataset}")
        result = subprocess.run(
            f"{KAHIP_PATH} ./graph_input.tmp --k 300 --seed 0 --preconfiguration=strong --output_filename=road_network_partition",
            shell=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(f"KaHIP stderr: {result.stderr}")
        os.chdir(current_directory)

        os.remove(f"../{dataset}/graph_input.tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=["Beijing"])
    args = parser.parse_args()
    main(args)
