# --- Standard and third-party imports (PEP 8: at top of file) ---
import os
import argparse
import polars as pl
import numpy as np


# --- Abnormality generation functions ---
def perturb_rids(rid_list, level, road_id_pool, rng):
    # Replace a number of road IDs with random others from the pool
    n = len(rid_list)
    n_perturb = {"low": 1, "medium": max(1, n // 10), "high": max(1, n // 5)}.get(
        level, max(1, n // 10)
    )
    if n < 2 or len(road_id_pool) < 2:
        return rid_list, []
    indices = rng.choice(range(n), size=n_perturb, replace=False)
    new_rid_list = rid_list.copy()
    perturbed = []
    for idx in indices:
        old = new_rid_list[idx]
        candidates = [r for r in road_id_pool if r != old]
        if not candidates:
            continue
        new = rng.choice(candidates)
        new_rid_list[idx] = new
        perturbed.append((idx, old, new))
    return new_rid_list, perturbed


def route_switch(rid_list, other_rid_list, level, rng):
    # Replace a segment of rid_list with a segment from other_rid_list
    # Level controls the length of the replaced segment
    n = len(rid_list)
    m = len(other_rid_list)
    if n < 4 or m < 4:
        return rid_list, None, None
    seg_len = {"low": 2, "medium": 3, "high": 4}.get(level, 3)
    if n <= seg_len or m <= seg_len:
        return rid_list, None, None
    start1 = rng.integers(1, n - seg_len)
    start2 = rng.integers(1, m - seg_len)
    seg2 = other_rid_list[start2 : start2 + seg_len]
    new_rid_list = rid_list[:start1] + seg2 + rid_list[start1 + seg_len :]
    return new_rid_list, (start1, start1 + seg_len), seg2


ABNORMALITY_TYPES = ["detour", "route_switch", "perturb"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate abnormal trajectories for HOSER datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with original CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for abnormal CSVs.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Splits to process.",
    )
    parser.add_argument(
        "--abnormality-types",
        nargs="*",
        default=["detour", "route_switch"],
        help="Abnormality types to generate.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="medium",
        help="Abnormality level (e.g., low, medium, high)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_csv_with_flexible_columns(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    cols = df.columns
    if cols[1] not in ["entity_id", "user_id"]:
        raise ValueError(f"Unexpected second column: {cols[1]}")
    return df


def add_abnormality_info_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.lit("normal").alias("abnormality_info")])


def save_with_abnormality_info(df: pl.DataFrame, out_path: str):
    # Ensure abnormality_info is last column
    cols = df.columns
    if cols[-1] != "abnormality_info":
        cols = [c for c in cols if c != "abnormality_info"] + ["abnormality_info"]
        df = df.select(cols)
    df.write_csv(out_path)


def insert_detour(rid_list, level, road_id_pool, rng):
    # Insert 1, 2, or 3 random road IDs depending on level
    n_insert = {"low": 1, "medium": 2, "high": 3}.get(level, 2)
    if len(road_id_pool) == 0:
        return rid_list, []
    detour_roads = rng.choice(road_id_pool, size=n_insert, replace=False).tolist()
    insert_pos = rng.integers(1, len(rid_list), size=n_insert)
    new_rid_list = rid_list.copy()
    for pos, road in sorted(zip(insert_pos, detour_roads)):
        new_rid_list.insert(pos, road)
    return new_rid_list, detour_roads


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        in_path = os.path.join(args.input_dir, f"{split}.csv")
        out_path = os.path.join(args.output_dir, f"{split}.csv")
        if not os.path.exists(in_path):
            print(f"Warning: {in_path} does not exist, skipping.")
            continue
        df = load_csv_with_flexible_columns(in_path)
        df = add_abnormality_info_column(df)

        # Prepare pool of all road IDs in the split
        all_rids = set()
        for rids in df["rid_list"].to_list():
            all_rids.update(map(int, rids.split(",")))
        road_id_pool = list(all_rids)

        abnormal_rows = []
        # Detour abnormality
        for row in df.iter_rows(named=True):
            rid_list = list(map(int, row["rid_list"].split(",")))
            new_rid_list, detour_roads = insert_detour(
                rid_list, args.level, road_id_pool, rng
            )
            if detour_roads:
                new_row = dict(row)
                new_row["rid_list"] = ",".join(map(str, new_rid_list))
                new_row["abnormality_info"] = (
                    f"type=detour|level={args.level}|inserted_roads={','.join(map(str, detour_roads))}"
                )
                abnormal_rows.append(new_row)

        # Route switch abnormality
        rows = list(df.iter_rows(named=True))
        for i, row in enumerate(rows):
            rid_list = list(map(int, row["rid_list"].split(",")))
            # Pick a different trajectory at random
            candidates = [j for j in range(len(rows)) if j != i]
            if not candidates:
                continue
            j = rng.choice(candidates)
            other_row = rows[j]
            other_rid_list = list(map(int, other_row["rid_list"].split(",")))
            new_rid_list, seg_range, seg2 = route_switch(
                rid_list, other_rid_list, args.level, rng
            )
            if seg_range and seg2:
                new_row = dict(row)
                new_row["rid_list"] = ",".join(map(str, new_rid_list))
                new_row["abnormality_info"] = (
                    f"type=route_switch|level={args.level}|from_traj={other_row['traj_id']}|segment={seg_range[0]}-{seg_range[1]}|inserted={','.join(map(str, seg2))}"
                )
                abnormal_rows.append(new_row)

        # Perturb abnormality
        for row in df.iter_rows(named=True):
            rid_list = list(map(int, row["rid_list"].split(",")))
            new_rid_list, perturbed = perturb_rids(
                rid_list, args.level, road_id_pool, rng
            )
            if perturbed:
                new_row = dict(row)
                new_row["rid_list"] = ",".join(map(str, new_rid_list))
                perturbed_str = ";".join(f"{i}:{o}->{n}" for i, o, n in perturbed)
                new_row["abnormality_info"] = (
                    f"type=perturb|level={args.level}|perturbed_indices={perturbed_str}"
                )
                abnormal_rows.append(new_row)

        if abnormal_rows:
            abnormal_df = pl.DataFrame(abnormal_rows)
            df = pl.concat([df, abnormal_df], how="vertical")

        save_with_abnormality_info(df, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
