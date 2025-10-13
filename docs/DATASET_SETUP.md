# HOSER Dataset Setup Guide

This guide documents the complete process for setting up a new dataset for HOSER distillation training. Follow these steps to reproduce the setup for any HOSER-format dataset (Beijing, Porto, etc.).

## Prerequisites

### Required Software
- **Python 3.12+** with `uv` package manager
- **KaHIP** graph partitioning library
  - Installation path: `/home/matt/Dev/KaHIP/build/kaffpa`
  - Used for road network partitioning into zones
- **LM-TAD** teacher model trained on your dataset

### Required Dataset Files
Your dataset directory must contain these files in HOSER format:
- `roadmap.geo` - Road network geometry (coordinates, attributes)
- `roadmap.rel` - Road network topology (reachability relationships)
- `train.csv` - Training trajectories
- `val.csv` - Validation trajectories  
- `test.csv` - Test trajectories

## Setup Process Overview

1. Create dataset symlink in `data/` directory
2. Generate road network partition file
3. Generate zone transition matrix
4. Extract LM-TAD teacher weights
5. Create dataset configuration file
6. Verify setup and test

## Step-by-Step Instructions

### 1. Create Dataset Symlink

The HOSER codebase expects datasets at `data/<dataset_name>/`. Create a symlink pointing to your actual dataset directory.

```bash
cd /home/matt/Dev/HOSER/data
ln -s /path/to/your/dataset <dataset_name>

# Example for Porto:
ln -s /home/matt/Dev/HOSER-dataset-porto porto_hoser

# Verify symlink:
ls -la data/<dataset_name>/
# Should show: roadmap.geo, roadmap.rel, train.csv, val.csv, test.csv
```

**Note:** The symlink is gitignored, so it won't be tracked in version control.

### 2. Generate Road Network Partition

This creates a mapping from road segments to spatial zones (300 zones by default) using KaHIP graph partitioning.

```bash
cd /home/matt/Dev/HOSER/data/preprocess

uv run python partition_road_network.py --datasets <dataset_name>

# Example for Porto:
uv run python partition_road_network.py --datasets porto_hoser
```

**Output:** `data/<dataset_name>/road_network_partition`
- Text file with one zone ID per line (0-299)
- Line `i` contains the zone ID for road segment `i`

**Performance:** 
- Porto (11,024 roads): ~4 seconds
- Beijing (~40,000 roads): ~15-20 seconds

### 3. Generate Zone Transition Matrix

This creates a probability matrix of transitions between zones based on training trajectories.

```bash
cd /home/matt/Dev/HOSER/data/preprocess

uv run python get_zone_trans_mat.py --datasets <dataset_name>

# Example for Porto:
uv run python get_zone_trans_mat.py --datasets porto_hoser
```

**Output:** `data/<dataset_name>/zone_trans_mat.npy`
- NumPy array of shape (300, 300)
- Entry [i,j] = count of transitions from zone i to zone j

**Performance:**
- Porto (481,359 trajectories): ~67 seconds (~7,100 it/s)
- Beijing (~50,000 trajectories): ~10-15 seconds

### 4. Extract LM-TAD Teacher Weights

Convert the LM-TAD checkpoint to a weights-only format compatible with distillation.

**Find your LM-TAD checkpoint:**
```bash
find /home/matt/Dev/LMTAD/code/results/LMTAD/<dataset_name> -name "ckpt_best.pt"
```

**Determine grid dimensions:**
- Check LM-TAD's `train.sh` for your dataset
- Look for `grip_size` parameter
- Examples:
  - Beijing: `grip_size="205 252"` (51,660 vocab)
  - Porto: `grip_size="46 134"` (6,164 vocab)

**Extract weights:**
```bash
cd /home/matt/Dev/HOSER

uv run python tools/export_lmtad_weights.py \
  --repo /home/matt/Dev/LMTAD \
  --grip_size "<width> <height>" \
  --ckpt_in /path/to/LMTAD/checkpoint/ckpt_best.pt \
  --ckpt_out /path/to/LMTAD/checkpoint/weights_only.pt

# Example for Porto:
uv run python tools/export_lmtad_weights.py \
  --repo /home/matt/Dev/LMTAD \
  --grip_size "46 134" \
  --ckpt_in /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt \
  --ckpt_out /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/weights_only.pt
```

**Output:** `weights_only.pt` in the same directory as the original checkpoint

### 5. Create Dataset Configuration File

Copy the Beijing configuration and adapt it for your dataset.

```bash
cd /home/matt/Dev/HOSER
cp config/Beijing.yaml config/<dataset_name>.yaml
```

**Required changes:**

#### Update data directory:
```yaml
data_dir: /path/to/your/dataset
```

#### Update LM-TAD checkpoint path:
```yaml
distill:
  enable: true
  repo: /home/matt/Dev/LMTAD
  ckpt: /path/to/LMTAD/checkpoint/weights_only.pt
```

#### Update grid size comment:
```yaml
  grid_size: 0.001  # Based on LM-TAD training: grip_size="<width> <height>" (<dataset> grid)
```

#### Update WandB project and tags:
```yaml
wandb:
  enable: true
  project: hoser-distill-<dataset>
  run_name: ''
  tags: [<dataset>, distillation, optuna]
```

**Optional changes:**
- `batch_size` - Adjust based on dataset size and GPU memory
- `max_len` - Adjust based on trajectory lengths in your dataset
- `num_workers` - Adjust based on your system
- `n_trials` / `max_epochs` - Adjust based on time budget

### 6. Verify Setup

#### Check all files exist:
```bash
# Dataset files:
ls -lh data/<dataset_name>/roadmap.geo
ls -lh data/<dataset_name>/roadmap.rel
ls -lh data/<dataset_name>/train.csv
ls -lh data/<dataset_name>/val.csv
ls -lh data/<dataset_name>/test.csv

# Generated preprocessing files:
ls -lh data/<dataset_name>/road_network_partition
ls -lh data/<dataset_name>/zone_trans_mat.npy

# LM-TAD checkpoint:
ls -lh /path/to/LMTAD/checkpoint/weights_only.pt

# Config file:
ls -lh config/<dataset_name>.yaml
```

#### Quick test run:
```bash
cd /home/matt/Dev/HOSER

# Test with 1 trial, 2 epochs to verify everything loads correctly:
uv run python tune_hoser.py \
  --config config/<dataset_name>.yaml \
  --data_dir /path/to/your/dataset \
  --n_trials 1 \
  --max_epochs 2
```

This will:
- Load the configuration
- Create dataset cache (train_cache/, val_cache/)
- Initialize the model and distillation manager
- Run training for 2 epochs

If this completes without errors, your setup is correct!

## Dataset-Specific Notes

### Highway Type Format

Different datasets have different formats for the `highway` column in `roadmap.geo`:

**Beijing / San Francisco:**
- Format: String values or nested lists like `["primary", "secondary"]`
- Requires special parsing in `train_with_distill.py` line 327

**Porto:**
- Format: Integer codes (e.g., 7, 1, 5)
- No special parsing needed (direct label encoding)

If your dataset uses the nested list format, add it to the condition:
```python
if dataset_name in ["Beijing", "San_Francisco", "your_dataset"]:
    # Special highway type parsing
```

### Trajectory Length

Adjust `max_len` in your config based on typical trajectory lengths:
- **Beijing:** `max_len: 1024` (shorter trajectories)
- **Porto:** `max_len: 2000` (longer trajectories, ~500MB train.csv)

Check your dataset's trajectory length distribution before choosing.

### Memory Considerations

The dataset caching system will automatically decide whether to cache in RAM:
- Caches if dataset fits in 60% of available RAM
- Otherwise streams from disk

For large datasets:
- Reduce `batch_size` if you encounter OOM errors
- Increase `num_workers` for better disk I/O throughput
- Consider reducing `max_len` to limit memory per batch

## Full Usage Commands

### Standard Tuning Run
```bash
cd /home/matt/Dev/HOSER

uv run python tune_hoser.py \
  --config config/<dataset_name>.yaml \
  --data_dir /path/to/your/dataset
```

### Quick Test Run
```bash
uv run python tune_hoser.py \
  --config config/<dataset_name>.yaml \
  --data_dir /path/to/your/dataset \
  --n_trials 3 \
  --max_epochs 3
```

### Override Default Settings
```bash
uv run python tune_hoser.py \
  --config config/<dataset_name>.yaml \
  --data_dir /path/to/your/dataset \
  --n_trials 20 \
  --max_epochs 10
```

## Example: Porto Dataset Complete Setup

```bash
# 1. Create symlink
cd /home/matt/Dev/HOSER/data
ln -s /home/matt/Dev/HOSER-dataset-porto porto_hoser

# 2. Generate road network partition
cd /home/matt/Dev/HOSER/data/preprocess
uv run python partition_road_network.py --datasets porto_hoser

# 3. Generate zone transition matrix
uv run python get_zone_trans_mat.py --datasets porto_hoser

# 4. Extract LM-TAD teacher weights
cd /home/matt/Dev/HOSER
uv run python tools/export_lmtad_weights.py \
  --repo /home/matt/Dev/LMTAD \
  --grip_size "46 134" \
  --ckpt_in /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt \
  --ckpt_out /home/matt/Dev/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/weights_only.pt

# 5. Create config (manual edits required)
cp config/Beijing.yaml config/porto_hoser.yaml
# Edit config/porto_hoser.yaml with Porto-specific paths

# 6. Test setup
uv run python tune_hoser.py \
  --config config/porto_hoser.yaml \
  --data_dir /home/matt/Dev/HOSER-dataset-porto \
  --n_trials 1 \
  --max_epochs 2
```

## Troubleshooting

### KaHIP not found
**Error:** `partition_road_network.py` fails with "kaffpa not found"

**Solution:** Install KaHIP and update `KAHIP_PATH` in `data/preprocess/partition_road_network.py`

### Missing preprocessing files
**Error:** Training fails with "road_network_partition not found"

**Solution:** Run the preprocessing scripts in order (partition → zone trans)

### Checkpoint format error
**Error:** `export_lmtad_weights.py` fails with "Unrecognized checkpoint format"

**Solution:** Check that you're using `ckpt_best.pt` (not `final_model.pt` or iteration checkpoints)

### Grip size mismatch
**Error:** Distillation manager fails with grid dimension errors

**Solution:** Verify `--grip_size` matches your LM-TAD training configuration in `train.sh`

### Highway type parsing error
**Error:** Training fails during data loading with highway type errors

**Solution:** Check highway format in your `roadmap.geo` and add dataset to the conditional in `train_with_distill.py` if needed

### Out of memory
**Error:** CUDA OOM during training

**Root Cause:** Memory usage scales **quadratically** with trajectory length due to:
- `trace_distance_mat`: O(batch × T²) 
- `trace_time_interval_mat`: O(batch × T²)
- Transformer attention masks: O(batch × heads × T²)

**Example:** Porto trajectories are ~2x longer than Beijing (avg 8 vs 4.6 points), resulting in ~72% more memory usage despite smaller batch size.

**Solutions (in order of effectiveness):**
1. **Reduce `batch_size`** - Most direct fix for longer trajectories
2. **Enable `grad_checkpoint: true`** - Reduces activation memory at cost of ~20% speed
3. **Reduce `accum_steps`** - Adjust to maintain similar effective batch size
4. **Reduce `candidate_top_k`** (default: 64) - Caps candidates per timestep
5. **Reduce `max_len`** - Hard cap on trajectory length (last resort)
6. **Adjust dataloader** - Lower `prefetch_factor` to reduce staging memory

**Porto-specific adjustments** (see `config/porto_hoser.yaml`):
- `batch_size: 64` (vs Beijing's 128) - Compensates for longer trajectories
- `accum_steps: 4` (vs Beijing's 8) - Maintains reasonable effective batch size
- `grad_checkpoint: true` - Further reduces memory footprint
- `prefetch_factor: 8` (vs Beijing's 16) - Lower memory overhead

## File Structure After Setup

```
/home/matt/Dev/HOSER/
  data/
    <dataset_name> -> /path/to/your/dataset (symlink)
    preprocess/
      partition_road_network.py
      get_zone_trans_mat.py
  config/
    <dataset_name>.yaml
  tools/
    export_lmtad_weights.py

/path/to/your/dataset/
  roadmap.geo
  roadmap.rel
  train.csv
  val.csv
  test.csv
  road_network_partition (generated)
  zone_trans_mat.npy (generated)

/home/matt/Dev/LMTAD/code/results/LMTAD/<dataset_name>/.../
  ckpt_best.pt (original)
  weights_only.pt (generated)
```

## Summary Checklist

- [ ] Dataset symlink created at `data/<dataset_name>/`
- [ ] `road_network_partition` generated
- [ ] `zone_trans_mat.npy` generated
- [ ] LM-TAD `weights_only.pt` extracted with correct grip_size
- [ ] Config file created at `config/<dataset_name>.yaml`
- [ ] All paths updated in config (data_dir, ckpt, grip_size comment, wandb)
- [ ] Highway type handling verified (update code if needed)
- [ ] Test run completes without errors

Once all items are checked, you're ready to run the full hyperparameter tuning study!

