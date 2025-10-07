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

### Run Complete Pipeline

```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6
./run_gene_eval_pipeline.sh
```

This will:
1. Generate 5000 trajectories using vanilla model (NetworkX A* + ML timing)
2. Evaluate vanilla trajectories against test set
3. Generate 5000 trajectories using distilled model
4. Evaluate distilled trajectories against test set

All runs are logged to WandB project: `hoser-distill-optuna-6`

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

Edit `run_gene_eval_pipeline.sh` to change:
- `WANDB_PROJECT`: WandB project name
- `DATASET`: Dataset name (Beijing, San_Francisco, etc.)
- `CUDA_DEVICE`: GPU device ID
- `NUM_GENE`: Number of trajectories to generate
- `SEED`: Random seed

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

