# HOSER Distillation Evaluation: hoser-distill-optuna-6

Complete trajectory generation and evaluation for comparing vanilla HOSER vs. distilled HOSER models.

## 📁 Directory Structure

```
hoser-distill-optuna-6/
├── models/                           # Model checkpoints
│   ├── vanilla_25epoch_seed42.pth   # Vanilla baseline (69.57% val_acc)
│   └── distilled_25epoch_seed42.pth # Distilled model (best Optuna params)
├── gene/                             # Generated trajectories
│   ├── vanilla_seed42/              # Vanilla generated trajectories
│   └── distilled_seed42/            # Distilled generated trajectories
├── eval/                             # Evaluation results
│   ├── vanilla_seed42/              # Vanilla evaluation metrics
│   └── distilled_seed42/            # Distilled evaluation metrics
├── data -> ../../../data             # Symlink to Beijing dataset
├── run_gene_eval_pipeline.sh        # Automated pipeline script
└── README.md                         # This file
```

## 🚀 Quick Start

### Run Now (Seed 42) - In Your Tmux Session!

```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6

# Default: Both models, seed 42 (uses cache if already run)
./run_gene_eval_pipeline.sh
```

**What it does**:
1. Generate 5000 trajectories using vanilla model (NetworkX A* + ML timing)
2. Evaluate vanilla trajectories against test set
3. Generate 5000 trajectories using distilled model
4. Evaluate distilled trajectories against test set

**Runtime**: ~1.5-3 hours total

All runs are logged to WandB project: `hoser-distill-optuna-6`

### 💡 Smart Caching (Important!)

The pipeline is **idempotent** and uses intelligent caching:

**First Run (Seed 42)**:
```bash
./run_gene_eval_pipeline.sh
```
- Generates all trajectories
- Evaluates all trajectories
- ~1.5-3 hours

**Second Run (Seed 42)**:
```bash
./run_gene_eval_pipeline.sh
```
- ✅ Finds existing generated files → skips generation
- ✅ Finds existing evaluation → skips evaluation  
- **Completes instantly!**

**Force Re-run** (if needed):
```bash
./run_gene_eval_pipeline.sh --force
```

### When Seeds 43 & 44 Finish Training

First, add the new models:
```bash
# Download or copy distilled models to:
models/distilled_25epoch_seed43.pth
models/distilled_25epoch_seed44.pth
```

Then run only the new seeds:
```bash
# Option A: Run individually (recommended)
./run_gene_eval_pipeline.sh --seed 43 --models distilled
./run_gene_eval_pipeline.sh --seed 44 --models distilled

# Option B: Batch run
./run_all_seeds.sh --seeds "43 44" --models distilled
```

The pipeline is smart enough to:
- ✅ Skip seed 42 (already done)
- ✅ Skip vanilla for 43 & 44 (vanilla only needs one seed)
- 🧬 Only generate & evaluate distilled for seeds 43 & 44

### Advanced Options

```bash
# Run all seeds (42, 43, 44) with both models
./run_all_seeds.sh

# Run specific seeds only
./run_all_seeds.sh --seeds "43 44"

# Run only distilled models for all seeds
./run_all_seeds.sh --models distilled

# Only seed 43, both models
./run_gene_eval_pipeline.sh --seed 43

# Only seed 44, distilled model
./run_gene_eval_pipeline.sh --seed 44 --models distilled

# Re-evaluate existing trajectories (skip generation)
./run_gene_eval_pipeline.sh --seed 43 --skip-gene

# Only generate (skip evaluation)
./run_gene_eval_pipeline.sh --seed 42 --skip-eval

# Force re-run everything (ignores cache)
./run_gene_eval_pipeline.sh --force
```

### Run Individual Steps

#### Generate Trajectories

```bash
# Vanilla
uv run python gene.py \
  --dataset Beijing \
  --seed 42 \
  --cuda 0 \
  --num_gene 5000 \
  --model_path hoser-distill-optuna-6/models/vanilla_25epoch_seed42.pth \
  --nx_astar \
  --wandb \
  --wandb_project hoser-distill-optuna-6 \
  --wandb_run_name gene_vanilla_seed42_nx_astar \
  --wandb_tags vanilla baseline 25epochs seed42 generation nx_astar

# Distilled
uv run python gene.py \
  --dataset Beijing \
  --seed 42 \
  --cuda 0 \
  --num_gene 5000 \
  --model_path hoser-distill-optuna-6/models/distilled_25epoch_seed42.pth \
  --nx_astar \
  --wandb \
  --wandb_project hoser-distill-optuna-6 \
  --wandb_run_name gene_distilled_seed42_nx_astar \
  --wandb_tags distilled final 25epochs seed42 generation nx_astar
```

#### Evaluate Trajectories

```bash
# Vanilla
uv run python evaluation.py \
  --run_dir hoser-distill-optuna-6/eval/.temp_vanilla_seed42 \
  --wandb \
  --wandb_project hoser-distill-optuna-6 \
  --wandb_run_name eval_vanilla_seed42 \
  --wandb_tags vanilla baseline 25epochs seed42 evaluation

# Distilled
uv run python evaluation.py \
  --run_dir hoser-distill-optuna-6/eval/.temp_distilled_seed42 \
  --wandb \
  --wandb_project hoser-distill-optuna-6 \
  --wandb_run_name eval_distilled_seed42 \
  --wandb_tags distilled final 25epochs seed42 evaluation
```

## 📊 Model Details

### Vanilla Baseline (25 epochs)
- **Path**: `models/vanilla_25epoch_seed42.pth`
- **Source**: WandB run `0vw2ywd9`
- **Val Accuracy**: **69.57%**
- **Training**: 25 epochs, seed 42, no distillation
- **Date**: Sep 29, 2025

### Distilled Final (25 epochs)
- **Path**: `models/distilled_25epoch_seed42.pth`
- **Source**: Optuna Phase 2 final run
- **Hyperparameters**:
  - λ (distill_lambda): 0.0014
  - Temperature: 4.37
  - Window: 7
- **Training**: 25 epochs, seed 42, best Optuna params
- **Date**: Oct 7, 2025

## 🧬 Generation Method

**NetworkX A* with ML Timing** (`--nx_astar`)
- Uses NetworkX A* algorithm for routing (fast, optimal paths)
- Uses HOSER model only for timestamp prediction
- Benefits:
  - Faster than full model-based search
  - Guaranteed shortest paths
  - Focuses evaluation on timing accuracy vs. routing quality

## 📈 Evaluation Metrics

### Global Metrics (Distribution Similarity)
- **Distance (JSD)**: Jensen-Shannon divergence of trajectory distances
- **Duration (JSD)**: Jensen-Shannon divergence of trajectory durations
- **Radius (JSD)**: Jensen-Shannon divergence of radius of gyration

### Local Metrics (Per-OD Comparison)
- **Hausdorff (km)**: Maximum deviation between generated and real trajectories
- **DTW (km)**: Dynamic Time Warping distance (shape similarity)

Lower is better for all metrics.

## 🔗 WandB Tracking

All runs log to: `https://wandb.ai/matercomus/hoser-distill-optuna-6`

### Run Traceability
- Generation runs store: model path, seed, search method, num_trajectories
- Evaluation runs store: training run ID, generation run ID, all metrics
- Easy to trace from evaluation → generation → training

### Tags for Filtering
- `vanilla` / `distilled`: Model type
- `baseline` / `final`: Training stage
- `25epochs`: Number of training epochs
- `seed42`: Random seed
- `generation` / `evaluation`: Pipeline stage
- `nx_astar`: Search method

## 📝 Results Location

After running the pipeline:

```
eval/
├── vanilla_seed42/
│   └── eval_<timestamp>/
│       └── results.json          # Vanilla metrics
└── distilled_seed42/
    └── eval_<timestamp>/
        └── results.json          # Distilled metrics
```

## 🔍 Quick Comparison

```bash
# View vanilla results
cat hoser-distill-optuna-6/eval/vanilla_seed42/eval_*/results.json

# View distilled results
cat hoser-distill-optuna-6/eval/distilled_seed42/eval_*/results.json

# Compare side-by-side
jq '.' hoser-distill-optuna-6/eval/vanilla_seed42/eval_*/results.json \
     hoser-distill-optuna-6/eval/distilled_seed42/eval_*/results.json
```

## ⚙️ Configuration

### Pipeline Options

Both `run_gene_eval_pipeline.sh` and `run_all_seeds.sh` support:

| Option | Description | Default |
|--------|-------------|---------|
| `--seed SEED` | Random seed | 42 |
| `--seeds "42 43 44"` | Multiple seeds (batch mode) | "42 43 44" |
| `--models MODEL1,MODEL2` | Models to run (vanilla, distilled) | vanilla,distilled |
| `--skip-gene` | Skip generation phase | false |
| `--skip-eval` | Skip evaluation phase | false |
| `--cuda DEVICE` | GPU device ID | 0 |
| `--num-gene N` | Number of trajectories | 5000 |

### When to Use Each Option

**`--skip-gene`**: When you already have generated trajectories and just want to re-evaluate
- Useful for trying different evaluation metrics
- Faster iteration on evaluation code

**`--skip-eval`**: When you only want to generate trajectories
- Useful for generating all trajectories first, then evaluating later
- Can evaluate multiple runs in batch

**`--models distilled`**: When you only want to process distilled models
- Useful after seed 43 and 44 training completes
- Don't need to regenerate vanilla trajectories

**`--seeds "43 44"`**: When you want to process specific seeds
- After training for seeds 43 and 44 completes
- Re-run failed seeds without touching completed ones

### Typical Workflows

#### Workflow 1: Initial Run (Seed 42)
```bash
# Generate and evaluate both models for seed 42
./run_gene_eval_pipeline.sh --seed 42
```

#### Workflow 2: After Seeds 43 & 44 Complete Training
```bash
# Option A: Run each seed individually
./run_gene_eval_pipeline.sh --seed 43 --models distilled
./run_gene_eval_pipeline.sh --seed 44 --models distilled

# Option B: Batch run all seeds at once
./run_all_seeds.sh --seeds "43 44" --models distilled

# Option C: Run all three seeds (if vanilla already done for 42)
./run_all_seeds.sh --models distilled
```

#### Workflow 3: Re-evaluate All Without Regenerating
```bash
# Re-evaluate all existing trajectories (faster)
./run_all_seeds.sh --skip-gene
```

#### Workflow 4: Two-Phase (Generate First, Evaluate Later)
```bash
# Phase 1: Generate all trajectories
./run_all_seeds.sh --skip-eval

# ... do other work ...

# Phase 2: Evaluate all trajectories
./run_all_seeds.sh --skip-gene
```

## 🎯 Expected Outcomes

**Hypothesis**: Distilled model should have:
- Similar or better global distribution metrics (JSD)
- Similar or better local trajectory similarity (Hausdorff, DTW)
- Demonstrates successful knowledge transfer from LM-TAD

**Validation Accuracy Baseline**:
- Vanilla: 69.57%
- Distilled: TBD (expected > 69.57% if distillation successful)

## 📚 References

- Training details: `optuna_results/hoser_tuning_20251003_162916/`
- Documentation: `LMTAD-Distillation.md`
- WandB models: Use `tools/download_wandb_model.py` to explore other runs

