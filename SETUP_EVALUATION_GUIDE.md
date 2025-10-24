# Setup Evaluation Script Guide

## Overview

`setup_evaluation.py` automatically prepares a complete evaluation workspace for running `python_pipeline.py`.

## What It Does

1. **Finds trained models** in `./save/{dataset}/`
2. **Creates unique directory** with timestamp + UID (no overwriting!)
3. **Copies models** to proper naming format
4. **Creates config** from template with dataset settings
5. **Creates README** with usage instructions

## Quick Start

### Beijing Dataset (Default)

```bash
# Preview what will be created
uv run python setup_evaluation.py --dry-run

# Create evaluation workspace
uv run python setup_evaluation.py

# Run evaluation
cd hoser-distill-optuna-evaluation-*/
uv run python ../hoser-distill-optuna-6/python_pipeline.py
```

### Porto Dataset

```bash
uv run python setup_evaluation.py --dataset porto_hoser --name porto-eval
cd hoser-distill-optuna-porto-eval-*/
uv run python ../hoser-distill-optuna-6/python_pipeline.py
```

### Custom Name

```bash
uv run python setup_evaluation.py --name thesis-chapter5
cd hoser-distill-optuna-thesis-chapter5-*/
uv run python ../hoser-distill-optuna-6/python_pipeline.py
```

## Command Line Options

```bash
uv run python setup_evaluation.py [OPTIONS]

Options:
  --dataset DATASET     Dataset name (default: Beijing)
  --name NAME          Evaluation name (default: evaluation)
  --source-dir DIR     Source directory for models (default: ./save)
  --dry-run            Preview without creating files
```

## Directory Structure Created

```
hoser-distill-optuna-{name}-{uid}-{timestamp}/
├── models/                              # Trained model checkpoints
│   ├── vanilla_25epoch_seed42.pth
│   └── distilled_25epoch_seed42.pth
├── config/                              # Configuration
│   └── evaluation.yaml
├── gene/                                # Output: generated trajectories
│   └── {dataset}/
│       └── seed42/
├── eval/                                # Output: evaluation results
└── README.md                            # Usage instructions
```

## Model Naming Convention

The script automatically renames models to match pipeline expectations:

- `save/Beijing/seed42_vanilla/best.pth` → `vanilla_25epoch_seed42.pth`
- `save/Beijing/seed42_distill/best.pth` → `distilled_25epoch_seed42.pth`
- `save/Beijing/seed43_distill/best.pth` → `distilled_25epoch_seed43.pth`

## Error Messages

The script provides helpful error messages:

### Dataset Not Found

```
❌ Error: Dataset 'Beijing' not found in ./save
💡 Hint: Available datasets: porto_hoser, test_data
```

### No Models Found

```
❌ Error: No trained models found in ./save/Beijing
💡 Hint: Expected directories like: seed42_vanilla/, seed42_distill/
       Each with a best.pth checkpoint file
```

## Configuration

The created `config/evaluation.yaml` inherits all settings from the template:

- `num_gene: 5000` - Number of trajectories to generate
- `beam_width: 4` - Beam search width
- `od_sources: [train, test]` - Which OD pairs to evaluate
- `wandb.enable: true` - WandB logging enabled (offline mode)

You can edit this file or override with CLI arguments to `python_pipeline.py`.

## Testing

### Small Test Run

```bash
# Create workspace
uv run python setup_evaluation.py --name test

# Quick 10-trajectory test
cd hoser-distill-optuna-test-*/
uv run python ../hoser-distill-optuna-6/python_pipeline.py --num-gene 10
```

### Full Evaluation

```bash
# Create workspace
uv run python setup_evaluation.py

# Run full evaluation (5000 trajectories per model)
cd hoser-distill-optuna-evaluation-*/
uv run python ../hoser-distill-optuna-6/python_pipeline.py
```

## Tips

- **Always use `--dry-run` first** to preview what will be created
- **Unique directory names** mean you can run multiple evaluations in parallel
- **Original models are never modified** - only copies are used
- **Config can be edited** before running the pipeline
- **Clean separation** - each evaluation is completely independent

## Workflow Example

```bash
# 1. Preview
uv run python setup_evaluation.py --dry-run

# 2. Create workspace
uv run python setup_evaluation.py --name final-results

# 3. Navigate to workspace
cd hoser-distill-optuna-final-results-*/

# 4. Review/edit config if needed
nano config/evaluation.yaml

# 5. Run pipeline
uv run python ../hoser-distill-optuna-6/python_pipeline.py

# 6. Results are in eval/ directory
ls -la eval/
```

## Performance Profiling

The setup script integrates with the enhanced performance profiling features:

- Generation speed metrics automatically collected
- Per-trajectory timing statistics
- Model forward pass efficiency
- All metrics saved to `results.json` and WandB

See `PERFORMANCE_PROFILING_SUMMARY.md` for details on the metrics collected.

