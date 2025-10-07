# WandB CLI Quick Reference

## Download Models Using WandB CLI

WandB has a built-in CLI that's often simpler than the Python API!

### Basic Download

```bash
# Download all files from a run
uv run wandb pull 0vw2ywd9 -p hoser-distill-optuna-6 -e matercomus
```

This will download all files to `./wandb/run-<timestamp>-<run_id>/files/`

### Download Specific Model

```bash
# Pull the entire run (includes model)
uv run wandb pull matercomus/hoser-distill-optuna-6/0vw2ywd9

# The model will be at:
# wandb/run-20250929_191519-0vw2ywd9/files/save/Beijing/seed42_distill/best.pth
```

### Find Run IDs

```bash
# List recent runs in a project
uv run wandb runs matercomus/hoser-distill-optuna-6

# Or use the web UI to get run IDs:
# https://wandb.ai/matercomus/hoser-distill-optuna-6
```

## Comparison: CLI vs Python API

### CLI Approach (Simpler!)
```bash
# One command to download everything
uv run wandb pull 0vw2ywd9 -p hoser-distill-optuna-6 -e matercomus

# Model is at: wandb/<run_dir>/files/save/Beijing/seed42_distill/best.pth
```

### Python API Approach (More Control)
```bash
# Our custom tool with search & filtering
uv run python tools/download_wandb_model.py --run_id 0vw2ywd9 -o my_model.pth

# Model is at: my_model.pth (cleaned up!)
```

## When to Use Which?

**Use WandB CLI** (`wandb pull`) when:
- You want ALL files from a run (logs, configs, artifacts)
- Quick one-off downloads
- You know the exact run ID

**Use our Python tool** (`tools/download_wandb_model.py`) when:
- You want to **search** for runs (by epochs, distill status, accuracy)
- You want just the model file (clean output path)
- You need to filter/compare multiple runs
- You want metadata displayed nicely

## Example Workflow

```bash
# 1. Search for vanilla 25-epoch runs
uv run python tools/download_wandb_model.py --search --distill false --epochs 25

# Output shows:
# ID              Name                           Distill    Epochs   Val Acc
# 0vw2ywd9        trial_000_vanilla              ❌ No      25       0.6957

# 2. Download the model (two ways):

# Option A: WandB CLI (gets everything)
uv run wandb pull 0vw2ywd9 -p hoser-distill-optuna-6 -e matercomus

# Option B: Our tool (just the model, clean path)
uv run python tools/download_wandb_model.py --run_id 0vw2ywd9 -o vanilla_25epoch.pth
```

## Pro Tips

1. **WandB Run Path Format**: `<entity>/<project>/<run_id>`
   - Example: `matercomus/hoser-distill-optuna-6/0vw2ywd9`

2. **Finding Run IDs**: 
   - From directory name: `run-20250929_191519-0vw2ywd9` → ID is `0vw2ywd9`
   - From WandB UI: Click run → Overview tab → See "Run path"

3. **Sync Local Runs**:
   ```bash
   # Upload local wandb runs to cloud
   uv run wandb sync wandb/run-<timestamp>-<id>
   ```

## Quick Download Commands

```bash
# Vanilla 25-epoch baseline (69.57% accuracy)
uv run wandb pull matercomus/hoser-distill-optuna-6/0vw2ywd9

# Copy model to clean location
cp wandb/run-20250929_191519-0vw2ywd9/files/save/Beijing/seed42_distill/best.pth \
   models_for_eval/vanilla_25epoch.pth
```
