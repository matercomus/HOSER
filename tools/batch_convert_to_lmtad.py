"""Batch convert HOSER CSVs to LM-TAD format (scriptable module)

This script imports the conversion function from
`tools.convert_to_lmtad_format` and runs it over a set of
pre-configured dataset directories (Porto, Beijing). It's intentionally
small and readable so you can tweak the top-level variables to change
behavior (vocab paths, batch size, splits, output folder name, etc.).

Usage (module):
  python -m tools.batch_convert_to_lmtad

You can also import `run_batch` from this file and call it from other
scripts or a notebook.
"""

from __future__ import annotations

from pathlib import Path
import logging
import json
from typing import Dict, Iterable

# Import converter module (it places repo root on sys.path when imported)
from tools import convert_to_lmtad_format as converter

logger = logging.getLogger(__name__)


# -----------------------------
# Easy-to-change configuration
# -----------------------------

# Per-dataset configuration. Keys are the `dataset` string expected by
# the converter (see `convert_to_lmtad_format.DATASET_CONFIGS`).
# `input_dir` is the directory that contains train/val/test CSVs.
# `roadmap` is the roadmap.geo used to extract centroids for mapping.
DATASET_RUNS = [
    {
        "dataset_key": "porto_hoser",
        "input_dir": Path("data/porto_hoser_abnormal"),
        "roadmap": Path("data/porto_hoser/roadmap.geo"),
    },
    {
        "dataset_key": "beijing_hoser_reference",
        "input_dir": Path("data/Beijing_abnormal"),
        "roadmap": Path("data/Beijing/roadmap.geo"),
    },
]

# Splits to process for each dataset. Change this list to control which
# files get processed (e.g., ['train'] to do only training set).
SPLITS = ["train", "val", "test"]

# Output folder name placed next to the input CSVs (relative to each
# input directory). The script writes trajectories_{split}.csv and a
# shared vocab.json in this folder.
OUTPUT_SUBDIR = "lmtad"

# Batch size passed to converter (change for memory/perf)
BATCH_SIZE = 10000

# Verbose logging to console
VERBOSE = False


def _ensure_paths_exist(paths: Iterable[Path]) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")


def run_batch(
    dataset_runs: Iterable[Dict],
    splits: Iterable[str] = SPLITS,
    output_subdir: str = OUTPUT_SUBDIR,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Run conversions for each configured dataset and split.

    This function is intentionally defensive: it checks that input files
    exist before invoking the converter and logs progress. It will raise
    on missing files to avoid silent failures.
    """

    for cfg in dataset_runs:
        dataset_key = cfg["dataset_key"]
        input_dir = Path(cfg["input_dir"]).resolve()
        roadmap = Path(cfg["roadmap"]).resolve()

        logger.info(f"Processing dataset: {dataset_key} (input: {input_dir})")

        # Validate roadmap and input directory
        _ensure_paths_exist([input_dir, roadmap])

        # prepare output directory (shared per dataset)
        out_dir = input_dir / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Shared vocab file for this dataset
        vocab_file = out_dir / "vocab.json"

        # collect mapping entries per dataset and write to out_dir/mapping.json
        mapping = {}

        for split in splits:
            input_file = input_dir / f"{split}.csv"
            entry = {
                "input": str(input_file),
                "output": None,
                "vocab": str(vocab_file),
                "status": None,
                "message": None,
            }

            if not input_file.exists():
                msg = f"Missing split file: {input_file}"
                logger.warning(msg)
                entry["status"] = "missing"
                entry["message"] = msg
                mapping[split] = entry
                continue

            # create output filename per split
            out_traj_file = out_dir / f"trajectories_{split}.csv"
            entry["output"] = str(out_traj_file)

            logger.info(
                f"Converting {input_file} -> {out_traj_file} (vocab: {vocab_file})"
            )

            try:
                converter.convert_hoser_to_lmtad_format(
                    trajectory_file=input_file,
                    roadmap_file=roadmap,
                    output_file=out_traj_file,
                    vocab_file=vocab_file,
                    dataset=dataset_key,
                    batch_size=batch_size,
                )
                entry["status"] = "ok"
            except Exception as e:
                logger.error(f"Conversion failed for {input_file}: {e}")
                entry["status"] = "failed"
                entry["message"] = str(e)

            mapping[split] = entry

        # write mapping file for this dataset
        try:
            mapping_path = out_dir / "mapping.json"
            with mapping_path.open("w") as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote mapping file: {mapping_path}")
        except Exception as e:
            logger.error(f"Failed to write mapping file to {out_dir}: {e}")
            raise


def main() -> None:
    logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
    run_batch(DATASET_RUNS)


if __name__ == "__main__":
    main()
