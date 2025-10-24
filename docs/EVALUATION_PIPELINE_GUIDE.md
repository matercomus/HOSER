# HOSER Evaluation Pipeline Guide

## Overview

The HOSER evaluation pipeline provides a comprehensive framework for evaluating trajectory generation models. It consists of three main components:

1. **setup_evaluation.py** - Creates evaluation workspaces with models and configs
2. **python_pipeline.py** - Orchestrates generation and evaluation
3. **tools/analyze_scenarios.py** - Post-processing scenario analysis

## Directory Structure

```
HOSER/
├── python_pipeline.py          # Main evaluation script (NEW location)
├── setup_evaluation.py         # Workspace setup script
├── tools/
│   └── analyze_scenarios.py   # Scenario analysis tool
├── config/
│   ├── evaluation.yaml         # Evaluation config template
│   ├── scenarios_beijing.yaml  # Beijing scenario definitions
│   └── scenarios_porto.yaml    # Porto scenario definitions
└── save/
    └── Beijing/
        ├── seed42_vanilla/
        │   └── best.pth
        └── seed42_distill/
            └── best.pth
```

## Workflow

### Step 1: Setup Evaluation Workspace

```bash
# Create evaluation directory with models and configs
uv run python setup_evaluation.py --dataset Beijing --name baseline

# For Porto dataset
uv run python setup_evaluation.py --dataset porto_hoser --name porto-test
```

This creates a self-contained evaluation directory:
```
hoser-evaluation-baseline-abc123-20241024_123456/
├── models/
│   ├── vanilla_25epoch_seed42.pth
│   └── distilled_25epoch_seed42.pth
├── config/
│   ├── evaluation.yaml          # Customized for this dataset
│   └── scenarios_beijing.yaml   # Optional scenario config
├── gene/                        # Generated trajectories (created by pipeline)
├── eval/                        # Evaluation results (created by pipeline)
├── scenarios/                   # Scenario analysis (created by pipeline)
└── README.md                    # Quick start instructions
```

### Step 2: Run Evaluation Pipeline

```bash
# Navigate to evaluation directory
cd hoser-evaluation-baseline-abc123-20241024_123456

# Run full pipeline
uv run python ../python_pipeline.py

# Or run with custom options
uv run python ../python_pipeline.py \
    --num-gene 5000 \
    --models vanilla,distilled \
    --od-source test \
    --run-scenarios
```

You can also run from anywhere by specifying the eval directory:
```bash
uv run python python_pipeline.py --eval-dir path/to/eval/dir
```

### Step 3: Optional Scenario Analysis

If not run during the pipeline, you can run scenario analysis separately:

```bash
# From project root
uv run python tools/analyze_scenarios.py \
    --eval-dir hoser-evaluation-baseline-abc123 \
    --config config/scenarios_beijing.yaml
```

## Configuration

### evaluation.yaml

The main configuration file controls:
- Dataset and data paths
- Generation parameters (num_gene, beam_width)
- Evaluation settings (grid_size, edr_eps)
- Pipeline options (skip_gene, skip_eval)
- WandB settings
- Scenario analysis options

### Command-Line Override

All config options can be overridden via CLI:
```bash
uv run python ../python_pipeline.py \
    --seed 123 \
    --num-gene 1000 \
    --cuda 1 \
    --no-wandb \
    --force
```

## Output Structure

After running the pipeline:

```
evaluation_directory/
├── gene/Beijing/seed42/
│   ├── hoser_vanilla_testod_gene_20241024_123456.csv
│   └── hoser_distilled_testod_gene_20241024_123456.csv
├── eval/
│   └── results.json
├── scenarios/
│   ├── test/
│   │   ├── vanilla/
│   │   │   ├── scenario_analysis.json
│   │   │   └── visualizations...
│   │   └── distilled/
│   └── train/
└── wandb/                      # WandB offline runs
```

## Advanced Usage

### Running Specific Models

```bash
# Only run vanilla model
uv run python ../python_pipeline.py --models vanilla

# Run multiple specific models
uv run python ../python_pipeline.py --models vanilla,distilled_seed44
```

### Skip Phases

```bash
# Skip generation (use existing trajectories)
uv run python ../python_pipeline.py --skip-gene

# Skip evaluation (generation only)
uv run python ../python_pipeline.py --skip-eval
```

### Force Re-run

```bash
# Force re-run even if results exist
uv run python ../python_pipeline.py --force
```

### Scenario Analysis

```bash
# Run with scenario analysis
uv run python ../python_pipeline.py --run-scenarios

# Use custom scenario config
uv run python ../python_pipeline.py \
    --run-scenarios \
    --scenarios-config ../config/scenarios_beijing_custom.yaml
```

## Reproducibility

Each evaluation workspace is self-contained with:
- Exact model checkpoints used
- Configuration snapshot at runtime
- All results and intermediate files
- Unique directory naming prevents overwrites

## Troubleshooting

### Data Directory Not Found

Ensure the data path in `config/evaluation.yaml` is correct:
```yaml
data_dir: ../data/Beijing  # Relative to eval directory
```

### Model Detection Issues

Check that model files follow the naming pattern:
- `{model_type}_25epoch_seed{seed}.pth`
- Or ensure `setup_evaluation.py` created them correctly

### CUDA Out of Memory

Reduce batch size or number of trajectories:
```bash
uv run python ../python_pipeline.py --num-gene 100
```

### WandB Sync Issues

Disable WandB or use offline mode:
```bash
uv run python ../python_pipeline.py --no-wandb
```
