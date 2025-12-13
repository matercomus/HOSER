# --- Standard and third-party imports (PEP 8: at top of file) ---
import os
import argparse
import csv
import logging
import polars as pl
import numpy as np
from typing import List, Sequence


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
    # Ensure that the function has the correct context
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


def route_switch_from_pool(rid_list, road_pool, level, rng):
    """Route-switch that draws a replacement segment from the global road pool.

    This variant is streaming-friendly (does not need another trajectory in memory).
    """
    n = len(rid_list)
    if n < 4 or len(road_pool) < 4:
        return rid_list, None, None
    seg_len = {"low": 2, "medium": 3, "high": 4}.get(level, 3)
    if n <= seg_len or len(road_pool) <= seg_len:
        return rid_list, None, None
    start1 = int(rng.integers(1, n - seg_len))
    # pick a contiguous slice from the pool (wrap if necessary)
    pool = list(road_pool)
    start2 = int(rng.integers(0, max(1, len(pool) - seg_len)))
    seg2 = pool[start2 : start2 + seg_len]
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


def build_road_pool_stream(path: str, rid_col: str) -> List[str]:
    """Build a deterministic, sorted road id pool by streaming the CSV once.

    Returns a sorted list of unique road ids as strings.
    """
    pool_set = set()
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if rid_col not in reader.fieldnames:
            raise ValueError(f"RID column '{rid_col}' not found in {path}")
        for row in reader:
            rids = row[rid_col].split(",") if row[rid_col] else []
            for r in rids:
                pool_set.add(r)
    # return a deterministic ordering
    return sorted(pool_set)


def process_split_streaming(
    input_path: str,
    output_path: str,
    seed: int,
    level: str,
    abnormal_types: Sequence[str],
):
    """Stream-process one CSV split deterministically and write output CSV.

    Two-pass approach: first pass builds the global road pool deterministically,
    second pass streams rows and writes originals + generated abnormal rows.
    """
    # detect rid column name by reading header
    with open(input_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    if "rid_list" in header:
        rid_col = "rid_list"
    else:
        rid_col = next((c for c in header if "rid" in c.lower()), "rid_list")

    logging.info("Building road pool for %s (rid_col=%s)", input_path, rid_col)
    road_pool = build_road_pool_stream(input_path, rid_col)
    logging.info("Road pool size=%d", len(road_pool))

    # open reader and writer for streaming second pass
    with (
        open(input_path, "r", newline="") as infh,
        open(output_path, "w", newline="") as outfh,
    ):
        reader = csv.DictReader(infh)
        fieldnames = list(reader.fieldnames) + ["abnormality_info"]
        writer = csv.DictWriter(outfh, fieldnames=fieldnames)
        writer.writeheader()

        global_seed = int(seed)
        for idx, row in enumerate(reader):
            if idx % 10000 == 0 and idx > 0:
                logging.info("Processed %d rows", idx)
            # write original row with 'normal' abnormality_info
            out_row = dict(row)
            out_row["abnormality_info"] = "normal"
            writer.writerow(out_row)

            # prepare common values
            rid_list = row.get(rid_col, "")
            rids = rid_list.split(",") if rid_list else []
            # per-row deterministic RNG
            rng = np.random.default_rng(global_seed + idx)

            for a_type in abnormal_types:
                if a_type == "detour":
                    new_rids, detour = insert_detour(rids, level, road_pool, rng)
                    if detour:
                        info = {"type": "detour", "level": level, "detour": detour}
                    else:
                        info = {"type": "detour", "level": level}
                elif a_type == "perturb":
                    new_rids, perturbed = perturb_rids(rids, level, road_pool, rng)
                    info = {"type": "perturb", "level": level, "perturbed": perturbed}
                elif a_type == "route_switch":
                    new_rids, seg_range, seg2 = route_switch_from_pool(
                        rids, road_pool, level, rng
                    )
                    info = {"type": "route_switch", "level": level}
                    if seg_range is not None:
                        info["seg_range"] = seg_range
                        info["seg2"] = seg2
                else:
                    continue

                # write abnormal row
                new_row = dict(row)
                new_row[rid_col] = ",".join(new_rids)
                # write abnormality_info as compact string
                new_row["abnormality_info"] = str(info)
                writer.writerow(new_row)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        in_path = os.path.join(args.input_dir, f"{split}.csv")
        out_path = os.path.join(args.output_dir, f"{split}.csv")
        if not os.path.exists(in_path):
            logging.warning("%s does not exist, skipping.", in_path)
            continue

        logging.info("Processing split=%s", split)
        process_split_streaming(
            input_path=in_path,
            output_path=out_path,
            seed=args.seed,
            level=args.level,
            abnormal_types=args.abnormality_types,
        )
        logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
