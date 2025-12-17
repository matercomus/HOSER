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

### LM‑TAD Spatial Abnormality (CI-friendly runs)

When running the LM‑TAD spatial abnormality evaluation you may want to reduce workload for CI runs or quick validation. The pipeline exposes several options to control the number of OD pairs, number of trajectories per OD, and duplicate‑trajectory checking.

- **Quick/CI example**: reduce OD pairs and trajectories per OD, and temporarily disable duplicate checks (useful to shorten runs while debugging):

```bash
cd /home/mka299/HOSER
uv run python python_pipeline.py \
    --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
    --run-lmtad-spatial \
    --only lmtad_spatial_abnormality \
    --lmtad-max-od-pairs 100 \
    --lmtad-num-trajectories-per-od 2 \
    --force \
    --lmtad-max-duplicate-ratio 1.0
```

-- **`--lmtad-max-duplicate-ratio`**: controls how tolerant the LM‑TAD trajectory validator is to consecutive-duplicate road segments. The validator will fail trajectories whose duplicate ratio (consecutive duplicated road ids) exceeds this threshold. Note: duplicate checking is disabled by default (`1.0`). Set the flag to a value < `1.0` (e.g., `0.1`) to enable duplicate checking.

- **Mapping note**: the pipeline maps HOSER road IDs to LM‑TAD token IDs before performing token-level validation. This prevents spurious "road ID >= vocab_size" errors that occur if raw road IDs are validated against the LM‑TAD vocabulary directly.

- **Seeded-model preference**: when multiple seeded variants of a model exist (for example `vanilla_seed42`, `vanilla_seed43`, ...), the pipeline and the aggregation step prefer the seeded variants and will ignore a plain base model name (e.g., `vanilla`) if seeded variants are present. This avoids mixing stale plain-model results with newer seeded evaluations.

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
