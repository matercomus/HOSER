#!/usr/bin/env python3
import ast
from pathlib import Path
from critics.grid_mapper import GridMapper, GridConfig, map_roads_to_tokens
from tools.convert_to_lmtad_format import extract_road_centroids

EVAL_DIR = Path("hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732")
DATASET = "porto_hoser"
CSV = (
    EVAL_DIR
    / "gene_abnormal_lmtad_spatial"
    / DATASET
    / "seed42"
    / "vanilla_spatial_abnormal.csv"
)

print("CSV path:", CSV)
roadmap = Path("data") / DATASET / "roadmap.geo"
if not roadmap.exists():
    roadmap = Path(__file__).parent.parent / "data" / DATASET / "roadmap.geo"
print("roadmap exists:", roadmap.exists())
road_centroids, boundary = extract_road_centroids(roadmap)

grid_config = GridConfig(
    min_lat=boundary["min_lat"],
    max_lat=boundary["max_lat"],
    min_lng=boundary["min_lng"],
    max_lng=boundary["max_lng"],
    grid_size=0.001,
    downsample_factor=1,
)
mapper = GridMapper(
    boundary=grid_config, road_centroids=road_centroids, verify_hw=(46, 134)
)
road_to_token = mapper.map_all()

vocab_size = 6167

with open(CSV, "r") as f:
    header = f.readline().strip()
    print("header:", header)
    for i, line in enumerate(f):
        chunks = line.strip().split(",", 4)
        if len(chunks) < 5:
            continue
        gene_trace = chunks[4]
        try:
            road_list = ast.literal_eval(gene_trace)
        except Exception:
            road_list = []
        mapped, invalid_idxs = map_roads_to_tokens(road_list, road_to_token)
        invalid_tokens = [t for t in mapped if isinstance(t, int) and t >= vocab_size]
        mapped_ints = [t for t in mapped if isinstance(t, int)]
        dup_ratio = 1 - (len(set(mapped_ints)) / len(mapped_ints)) if mapped_ints else 0
        cons = sum(1 for a, b in zip(mapped_ints, mapped_ints[1:]) if a == b)
        print(
            f"Traj {i}: len={len(road_list):2d}, invalid_map={invalid_idxs}, invalid_tokens_count={len(invalid_tokens)}, dup_ratio={dup_ratio:.2f}, cons_dup={cons}, mapped_sample={mapped_ints[:10]}"
        )

print("done")
