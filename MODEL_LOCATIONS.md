# Model Locations for Gene & Eval

This document lists all available models for generation and evaluation.

## 📦 Available Models (Seed 42)

### 1. **Vanilla Baseline (25 epochs)** ✅
**Location**: `models_for_eval/vanilla_25epoch_seed42.pth`

- **Source**: Downloaded from WandB (run ID: `0vw2ywd9`)
- **Training**: 25 epochs, seed 42
- **Config**: `distill.enable = false` (no distillation)
- **Final Val Acc**: **69.57%** 🎯
- **Date**: Sep 29, 2025
- **WandB Run**: https://wandb.ai/matercomus/hoser-distill-optuna/runs/0vw2ywd9

### 2. **Distilled Final Model (25 epochs)** ✅
**Location**: `models_for_eval/distilled_25epoch_seed42.pth`

- **Source**: `save/Beijing/seed42_distill/best.pth`
- **Training**: 25 epochs, seed 42
- **Config**: Best hyperparameters from Optuna Phase 1
  - `lambda`: 0.0014
  - `temperature`: 4.37
  - `window`: 7
- **Expected Val Acc**: TBD (check logs)
- **Date**: Oct 7, 2025
- **WandB Run**: Check `optuna_results/hoser_tuning_20251003_162916/final_config_seed42.yaml`

## 🚀 Running Gene & Eval

### Generate Predictions

```bash
# Vanilla baseline
uv run python gene.py \
  --dataset Beijing \
  --model_path models_for_eval/vanilla_25epoch_seed42.pth \
  --cuda 0

# Distilled model
uv run python gene.py \
  --dataset Beijing \
  --model_path models_for_eval/distilled_25epoch_seed42.pth \
  --cuda 0
```

### Evaluate

```bash
# Vanilla baseline
uv run python evaluation.py \
  --run_dir models_for_eval \
  --wandb \
  --wandb_project hoser-distill-final-eval \
  --wandb_run_name vanilla_25epoch_seed42_eval \
  --wandb_tags beijing vanilla baseline 25epochs

# Distilled model
uv run python evaluation.py \
  --run_dir models_for_eval \
  --wandb \
  --wandb_project hoser-distill-final-eval \
  --wandb_run_name distilled_25epoch_seed42_eval \
  --wandb_tags beijing distilled final 25epochs
```

## 📊 Model Comparison Summary

| Model | Epochs | Val Acc | Distillation | Seed | Date |
|-------|--------|---------|--------------|------|------|
| Vanilla Baseline | 25 | **69.57%** | ❌ No | 42 | Sep 29 |
| Distilled Final | 25 | TBD | ✅ Yes | 42 | Oct 7 |
| Distilled Final | 25 | In progress (epoch 7) | ✅ Yes | 43 | Oct 7 |
| Distilled Final | 25 | Awaiting | ✅ Yes | 44 | Oct 7 |

## 🔍 Finding Other Models

Use the WandB download tool to search for and download other runs:

```bash
# Search for vanilla runs with 25 epochs
uv run python tools/download_wandb_model.py --search --distill false --epochs 25

# Download a specific run by ID
uv run python tools/download_wandb_model.py --run_id <RUN_ID> --output my_model.pth
```

## 📝 Notes

1. **Fair Comparison**: Both models trained for 25 epochs with seed 42
2. **Vanilla Baseline**: This is a **strong baseline** at 69.57% (much higher than the 8-epoch Optuna baseline at 57.22%)
3. **Hardware**: All models trained on RTX 2080 Ti, 11GB VRAM
4. **Data**: Beijing taxi trajectories (629k train, 89k val)

## ⚠️ Important Discovery

The vanilla 25-epoch baseline (69.57%) is **significantly better** than the 8-epoch vanilla from Optuna Phase 1 (57.22%). This highlights the importance of training to convergence for proper comparison!
