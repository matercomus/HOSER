import os
import argparse
import pandas as pd
import numpy as np

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
        default=["detour"],
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


def load_csv_with_flexible_columns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept either 'entity_id' or 'user_id' as the second column
    cols = list(df.columns)
    if cols[1] not in ["entity_id", "user_id"]:
        raise ValueError(f"Unexpected second column: {cols[1]}")
    return df


def add_abnormality_info_column(df: pd.DataFrame) -> pd.DataFrame:
    df["abnormality_info"] = "normal"
    return df


def save_with_abnormality_info(df: pd.DataFrame, out_path: str):
    # Ensure abnormality_info is last column
    cols = list(df.columns)
    if cols[-1] != "abnormality_info":
        cols = [c for c in cols if c != "abnormality_info"] + ["abnormality_info"]
    df = df[cols]
    df.to_csv(out_path, index=False)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        in_path = os.path.join(args.input_dir, f"{split}.csv")
        out_path = os.path.join(args.output_dir, f"{split}.csv")
        if not os.path.exists(in_path):
            print(f"Warning: {in_path} does not exist, skipping.")
            continue
        df = load_csv_with_flexible_columns(in_path)
        df = add_abnormality_info_column(df)
        save_with_abnormality_info(df, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
