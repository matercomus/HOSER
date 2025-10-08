# Python Evaluation Pipeline - Quick Reference

## Overview

This Python-based pipeline orchestrates trajectory generation and evaluation for HOSER distillation experiments. It handles all WandB logging centrally with background sync to eliminate upload delays.

## Key Features

- **Zero Upload Delays**: WandB runs in offline mode, sync happens in background
- **Separate WandB Runs**: Each model × OD source gets its own run for tracking
- **Graceful Interruption**: Ctrl+C properly handled with cleanup
- **YAML Configuration**: Centralized config with CLI overrides
- **Auto-Detection**: Automatically finds all models in `models/` directory

## Quick Start

```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6

# Run with default config (100 trajectories, train+test OD)
uv run python python_pipeline.py

# Full thesis evaluation (5000 trajectories)
uv run python python_pipeline.py --num-gene 5000

# Custom configuration
uv run python python_pipeline.py --config config/evaluation.yaml
```

## Configuration

Edit `config/evaluation.yaml` to change defaults:

```yaml
# Key settings
num_gene: 100          # Number of trajectories to generate
od_sources: [train, test]  # Evaluate on both train and test OD pairs
beam_width: 4          # Beam search width
grid_size: 0.001       # Grid size in degrees for OD matching
edr_eps: 100.0         # EDR threshold in meters

# WandB settings
wandb:
  enable: true
  project: hoser-distill-optuna-6
  background_sync: true  # Sync in background (non-blocking)
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config PATH` | YAML config file | `config/evaluation.yaml` |
| `--num-gene N` | Number of trajectories | 100 |
| `--skip-gene` | Skip generation (use existing) | false |
| `--skip-eval` | Skip evaluation | false |
| `--force` | Force regeneration | false |
| `--models M1,M2` | Specific models to run | auto-detect all |
| `--od-source train,test` | OD sources | train,test |
| `--cuda N` | CUDA device | 0 |
| `--no-wandb` | Disable WandB | false |
| `--verbose` | Verbose output | false |

## Pipeline Execution Flow

1. **Model Detection**: Auto-detects all `.pth` files in `models/`
   - vanilla, distilled, distilled_seed44, etc.

2. **For each model × OD source**:
   - **Generation**: Creates trajectories using beam search
     - WandB run: `gene_{model}_seed{seed}_{od}od`
   - **Evaluation**: Calculates metrics vs real data
     - WandB run: `eval_{model}_seed{seed}_{od}od`

3. **Background Sync**: Uploads all runs to WandB in parallel

## Expected WandB Runs

For 3 models (vanilla, distilled, distilled_seed44) × 2 OD sources (train, test):

**Generation Runs** (6 total):
- `gene_vanilla_seed42_trainod`
- `gene_vanilla_seed42_testod`
- `gene_distilled_seed42_trainod`
- `gene_distilled_seed42_testod`
- `gene_distilled_seed44_seed42_trainod`
- `gene_distilled_seed44_seed42_testod`

**Evaluation Runs** (6 total):
- `eval_vanilla_seed42_trainod`
- `eval_vanilla_seed42_testod`
- `eval_distilled_seed42_trainod`
- `eval_distilled_seed42_testod`
- `eval_distilled_seed44_seed42_trainod`
- `eval_distilled_seed44_seed42_testod`

**Total**: 12 WandB runs

## Metrics Logged

### Global Metrics (Distribution-level)
- `Distance_JSD`: Jensen-Shannon divergence for trip distance distributions
- `Duration_JSD`: JSD for per-segment duration distributions
- `Radius_JSD`: JSD for radius of gyration distributions

### Local Metrics (Trajectory-level)
- `Hausdorff_km`: Hausdorff distance in kilometers
- `DTW_km`: Dynamic Time Warping distance in kilometers
- `EDR`: Edit Distance on Real sequence (normalized)
- `matched_od_pairs`: Number of OD pairs matched for comparison
- `total_generated_od_pairs`: Total generated OD pairs

## Output Structure

```
hoser-distill-optuna-6/
├── gene/Beijing/seed42/          # Generated trajectories
│   ├── 2025-10-08_14-44-36.csv
│   ├── 2025-10-08_14-45-06.csv
│   └── ...
├── eval/                          # Evaluation results
│   └── 2025-10-08_XX-XX-XX/
│       └── results.json
├── wandb/                         # WandB offline runs
│   ├── offline-run-20251008_XXXXXX-XXXXXXXX/
│   └── ...
└── pipeline.log                   # Execution log
```

## Troubleshooting

### "No existing generated file found"
Run without `--skip-gene` to generate trajectories first.

### WandB not syncing
Check background sync status:
```bash
# Manual sync
wandb sync wandb/offline-run-*

# Check wandb status
wandb status
```

### Pipeline hangs
- Press Ctrl+C - it will cleanup gracefully
- Check `pipeline.log` for detailed errors
- Run with `--verbose` for debugging

### CUDA out of memory
Reduce beam width: `--config config/evaluation.yaml` and set `beam_width: 2`

## Examples

```bash
# Full evaluation with verbose output
uv run python python_pipeline.py --num-gene 5000 --verbose

# Only evaluate existing trajectories
uv run python python_pipeline.py --skip-gene

# Force regeneration of all files
uv run python python_pipeline.py --force

# Run specific models only
uv run python python_pipeline.py --models vanilla,distilled

# Evaluate only on test OD pairs
uv run python python_pipeline.py --od-source test

# Disable WandB (testing only)
uv run python python_pipeline.py --no-wandb
```

## Notes

- **Background Sync**: Pipeline exits immediately, WandB sync continues
- **Offline Mode**: Zero network delays during evaluation
- **Programmatic**: Imports `gene.py` and `evaluation.py` directly (no CLI overhead)
- **Bulletproof**: Extensive error handling and validation
- **Thesis-Ready**: Designed for reproducible research evaluation

