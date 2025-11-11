# Model Checkpointing Strategy

This document explains the model checkpointing strategy used in HOSER training, including checkpoint frequency, selection criteria, retention policy, and file locations.

## Overview

HOSER uses two training modes with different checkpointing strategies:
1. **Vanilla training** (`train.py`): Standard supervised training
2. **Distillation training** (`train_with_distill.py`): Knowledge distillation from LM-TAD teacher

Both modes implement **per-epoch checkpointing** with **best model selection** based on validation accuracy.

## Checkpointing Frequency

### Per-Epoch Checkpointing

Both training modes save model checkpoints **after every epoch** (not per-batch or per-iteration):

- **When**: At the end of each validation phase
- **Why**: Balances training efficiency with recovery granularity
- **Trade-off**: Losing at most one epoch of training if interrupted

### Validation-Driven Checkpoints

Checkpoints are only created after validation completes, ensuring:
- Every checkpoint has an associated validation metric
- Model selection can be performed immediately after training
- No partial/incomplete checkpoints from training interruptions

## Model Selection Criteria

### Primary Metric: Validation Accuracy

The **best model** is selected using **validation next-step accuracy** (`val_next_step_acc`):

```python
# From train.py
metrics_list.append(val_next_step_correct_cnt / val_next_step_total_cnt)
best_epoch = np.argmax(metrics_list)
```

**Why this metric?**
- **Next-step accuracy** directly measures the model's core task: predicting the next road segment
- Validated on held-out data to prevent overfitting
- Simple, interpretable, and aligns with downstream evaluation

### Secondary Metric: Time Prediction MAPE

While not used for model selection, **time prediction MAPE** (Mean Absolute Percentage Error) is logged:

```python
val_time_pred_mape = val_time_pred_mape_sum / val_time_pred_total_cnt
```

**Why not used for selection?**
- Time prediction is a secondary task
- MAPE can be unstable with very short time intervals
- Next-step accuracy is the primary evaluation metric in HOSER papers

## Checkpoint Retention Policy

### Vanilla Training (`train.py`)

**Retention**: **All epoch checkpoints retained**

```
./save/{dataset}/seed{seed}/
├── epoch_1.pth
├── epoch_2.pth
├── ...
├── epoch_{N}.pth
└── best.pth  # Copy of best epoch model
```

**Rationale**:
- Enables post-hoc analysis of training dynamics
- Allows comparison of different epochs without retraining
- Storage cost is acceptable for typical training runs (25-50 epochs)
- Useful for debugging convergence issues

**Disk Usage**: ~500MB per checkpoint × N epochs ≈ 12-25GB for 25-50 epochs

### Distillation Training (`train_with_distill.py`)

**Retention**: **Single rolling checkpoint + best model**

```
./save/{dataset}/seed{seed}_distill/
├── checkpoint_latest.pth  # Overwritten each epoch
└── best.pth              # Saved at end of training
```

**Rationale**:
- **Resume capability**: Training can be interrupted and resumed from latest checkpoint
- **Space efficiency**: Only one checkpoint stored during training
- **Metadata validation**: Checkpoint includes seed, dataset, and distillation mode to prevent loading mismatched checkpoints

**Disk Usage**: ~500MB (single checkpoint during training) → ~1GB (checkpoint + best) at completion

### Checkpoint Cleanup

**Manual cleanup required** - no automatic deletion:

```bash
# Remove all epoch checkpoints except best (vanilla training)
cd ./save/{dataset}/seed{seed}/
ls epoch_*.pth | grep -v "epoch_$(best_epoch).pth" | xargs rm

# Or keep only best.pth
rm epoch_*.pth  # best.pth is separate, won't be deleted
```

## Early Stopping Criteria

### Vanilla Training: No Early Stopping

`train.py` does **not** implement early stopping:

```python
for epoch_id in range(config.optimizer_config.max_epoch):
    # Always runs all epochs
```

**Rationale**:
- Training typically converges within 25-50 epochs
- No risk of premature stopping
- Simpler training loop
- Consistent experimental setup (all models train same number of epochs)

### Distillation Training: Optuna Pruning (Optional)

`train_with_distill.py` supports **Optuna-based pruning** when used with hyperparameter tuning:

```python
if optuna_trial is not None:
    optuna_trial.report(val_acc, epoch_id)
    if optuna_trial.should_prune():
        raise optuna.TrialPruned()
```

**How it works**:
- Optuna's `HyperbandPruner` compares validation accuracy across trials
- Trials significantly underperforming others are pruned early
- Only active during hyperparameter search (not during final training)

**Pruning criteria** (Optuna HyperbandPruner):
- Compares `val_acc` at each epoch against median performance of completed trials
- Prunes if significantly below median (controlled by `reduction_factor` parameter)
- See [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html) for details

**Manual early stopping**: Monitor TensorBoard and manually stop if:
- Validation accuracy plateaus for 5+ epochs
- Training loss diverges (NaN or inf)
- Validation accuracy decreases for 5+ consecutive epochs (overfitting)

## Checkpoint File Locations

### Directory Structure

```
./save/
├── {dataset}/          # e.g., Beijing, San_Francisco
│   ├── seed{seed}/    # Vanilla training checkpoints
│   │   ├── epoch_1.pth
│   │   ├── epoch_2.pth
│   │   ├── ...
│   │   └── best.pth
│   └── seed{seed}_distill/  # Distillation training checkpoints
│       ├── checkpoint_latest.pth
│       └── best.pth
```

### Example Paths

```bash
# Vanilla training (seed 42, Beijing)
./save/Beijing/seed42/best.pth

# Distillation training (seed 42, Beijing)
./save/Beijing/seed42_distill/best.pth

# Per-epoch checkpoint (vanilla only)
./save/Beijing/seed42/epoch_10.pth
```

### Checkpoint Contents

#### Vanilla Training Checkpoints

**Per-epoch checkpoints** (`epoch_{N}.pth`):
```python
# State dict only
torch.load("epoch_10.pth")
# -> OrderedDict of model weights
```

**Best model checkpoint** (`best.pth`):
```python
# State dict only
torch.load("best.pth")
# -> OrderedDict of model weights
```

#### Distillation Training Checkpoints

**Rolling checkpoint** (`checkpoint_latest.pth`):
```python
checkpoint = torch.load("checkpoint_latest.pth")
# -> {
#     "epoch": 9,  # Last completed epoch (0-indexed)
#     "seed": 42,
#     "dataset": "Beijing",
#     "distill_enabled": True,
#     "model_state_dict": OrderedDict(...),
#     "optimizer_state_dict": {...},
#     "best_val_acc": 0.6832,
#     "validation_metrics": [...],
#     "wandb_run_id": "abc123xyz",  # For resuming WandB logging
# }
```

**Best model checkpoint** (`best.pth`):
```python
# State dict only
torch.load("best.pth")
# -> OrderedDict of model weights
```

### Loading Checkpoints

#### For Evaluation/Inference

```python
from models.hoser import HOSER

# Load best model for evaluation
model = HOSER(config, ...).to(device)
model.load_state_dict(torch.load("./save/Beijing/seed42/best.pth"))
model.eval()
```

#### Resuming Training (Distillation Only)

```python
# Automatic resume in train_with_distill.py
# Just run the same command - it detects and loads checkpoint_latest.pth
uv run python train_with_distill.py --dataset Beijing --seed 42 --cuda 0

# The script validates:
# - Seed matches
# - Dataset matches
# - Distillation mode matches
# If any mismatch, checkpoint is deleted and training restarts from scratch
```

## TensorBoard Logging

### Log Directory Structure

```
./tensorboard_log/
├── {dataset}/
│   ├── seed{seed}/          # Vanilla training logs
│   └── seed{seed}_distill/  # Distillation training logs
```

### Metrics Logged

**Training metrics** (per-batch):
- `loss_next_step`: Cross-entropy loss for next road prediction
- `loss_time_pred`: Time prediction loss (MAE)
- `loss_kl`: KL divergence loss (distillation only)
- `loss`: Total loss
- `learning_rate`: Current learning rate

**Validation metrics** (per-epoch):
- `val_next_step_acc`: Validation next-step accuracy
- `val_time_pred_mape`: Validation time prediction MAPE

### Viewing Logs

```bash
# Launch TensorBoard
tensorboard --logdir ./tensorboard_log/{dataset}/seed{seed}

# Compare vanilla vs distilled
tensorboard --logdir ./tensorboard_log/Beijing/
```

## Weights & Biases Integration

### WandB Logging (Distillation Training)

When `wandb.enable = true` in config:

```yaml
wandb:
  enable: true
  project: "hoser-distill"
  run_name: "Beijing_b128_acc4"  # Auto-generated or custom
  tags: ["beijing", "distillation", "optuna"]
```

**Logged data**:
- All TensorBoard metrics
- Full config YAML
- Best model artifact (optional)
- System metrics (GPU, CPU, memory)

**Resume support**:
- `wandb_run_id` saved in checkpoint
- Resuming training continues same WandB run
- Prevents duplicate runs from interruptions

### Downloading Models from WandB

```bash
# Search for models
uv run python tools/download_wandb_model.py \
  --search \
  --distill true \
  --epochs 25

# Download specific run
uv run python tools/download_wandb_model.py \
  --run_id abc123xyz \
  --output my_model.pth
```

## Best Practices

### Training from Scratch

**Vanilla training**:
```bash
uv run python train.py --dataset Beijing --seed 42 --cuda 0
```

**Distillation training**:
```bash
uv run python train_with_distill.py --dataset Beijing --seed 42 --cuda 0
```

### Resuming Interrupted Training

**Only distillation training supports resume**:

```bash
# Just re-run the same command
uv run python train_with_distill.py --dataset Beijing --seed 42 --cuda 0

# Check logs to verify resume:
# "✅ Valid checkpoint found: will resume from epoch 11"
# "✅ Resumed from checkpoint: epoch 11, best_val_acc=0.6832"
```

**Vanilla training does not support resume** - must restart from scratch.

### Selecting Best Model After Training

**The best model is automatically selected and saved**:

```python
# train.py does this automatically:
best_epoch = np.argmax(metrics_list)
logger.info(f"loading epoch_{best_epoch + 1}.pth")
best_model_state_dict = torch.load(f"epoch_{best_epoch + 1}.pth")
model.load_state_dict(best_model_state_dict)
torch.save(model.state_dict(), "best.pth")
```

**Always use `best.pth` for evaluation**, not the last epoch checkpoint.

### Monitoring Training Progress

**During training**:
```bash
# Watch validation accuracy in real-time
tail -f ./log/{dataset}/seed{seed}/*.log | grep "val_next_step_acc"

# Or use TensorBoard
tensorboard --logdir ./tensorboard_log/{dataset}/seed{seed}
```

**After training**:
```bash
# Find best epoch
grep "val_next_step_acc" ./log/{dataset}/seed{seed}/*.log

# Or check TensorBoard curves
```

### Disk Space Management

**Before training**:
```bash
# Check available space (need ~25GB for 25-50 epochs vanilla)
df -h ./save/

# Or use distillation mode for space-efficient checkpointing
```

**After training**:
```bash
# Keep only best model (vanilla)
cd ./save/{dataset}/seed{seed}/
rm epoch_*.pth  # Keeps best.pth

# Or archive old checkpoints
tar -czf epoch_checkpoints.tar.gz epoch_*.pth
rm epoch_*.pth
```

## Common Issues & Solutions

### Issue: "Checkpoint mismatch" warning

```
⚠️  Checkpoint mismatch (seed=42 vs 43, dataset=Beijing vs Porto)
⚠️  Deleting invalid checkpoint and starting fresh
```

**Cause**: Resuming with different seed or dataset than checkpoint was trained with

**Solution**: Either:
- Use the same seed/dataset as checkpoint: `--seed 42 --dataset Beijing`
- Or delete checkpoint to start fresh: `rm checkpoint_latest.pth`

### Issue: Out of disk space during training

```
OSError: [Errno 28] No space left on device
```

**Cause**: Per-epoch checkpoints consume 500MB each × N epochs

**Solution**:
- Use distillation training mode (only 1 checkpoint stored)
- Or delete old epoch checkpoints: `rm save/*/seed*/epoch_*.pth`
- Or increase disk space allocation

### Issue: Training resuming from wrong epoch

**Symptom**: Logs show "Resumed from epoch X" but validation accuracy starts from 0

**Cause**: Checkpoint file corrupted or model architecture changed

**Solution**:
```bash
# Delete corrupted checkpoint
rm ./save/{dataset}/seed{seed}_distill/checkpoint_latest.pth

# Restart training from scratch
uv run python train_with_distill.py --dataset Beijing --seed 42 --cuda 0
```

### Issue: Cannot find best.pth after training

**Cause**: Training crashed before final checkpoint save

**Solution**:
```bash
# Find best epoch from logs
grep "val_next_step_acc" ./log/{dataset}/seed{seed}/*.log | sort -t',' -k2 -rn | head -1

# Manually copy best epoch checkpoint
cp ./save/{dataset}/seed{seed}/epoch_{BEST}.pth ./save/{dataset}/seed{seed}/best.pth
```

## References

- [LMTAD-Distillation.md](LMTAD-Distillation.md) - Full distillation training guide
- [MODEL_LOCATIONS.md](reference/MODEL_LOCATIONS.md) - Available pre-trained models
- [Hyperparameter-Optimization.md](Hyperparameter-Optimization.md) - Optuna pruning setup
- [HOSER Paper](https://github.com/caoji2001/HOSER) - Original model architecture

## Summary

| Aspect | Vanilla Training | Distillation Training |
|--------|------------------|----------------------|
| **Frequency** | Per-epoch (after validation) | Per-epoch (after validation) |
| **Selection** | Best validation accuracy | Best validation accuracy |
| **Retention** | All epoch checkpoints + best | Single rolling + best |
| **Early Stop** | None (runs all epochs) | Optuna pruning (optional) |
| **Resume** | ❌ Not supported | ✅ Supported with validation |
| **Location** | `./save/{dataset}/seed{seed}/` | `./save/{dataset}/seed{seed}_distill/` |
| **Disk Usage** | 12-25GB (25-50 epochs) | ~1GB (checkpoint + best) |
| **WandB** | ❌ Not integrated | ✅ Full integration |

**Key takeaways**:
1. Checkpoints saved **per-epoch** after validation (not per-batch)
2. **Best model selected by validation accuracy** (not loss)
3. Vanilla: keeps all epochs; Distillation: keeps only latest + best
4. **No automatic early stopping** in vanilla mode
5. Distillation supports **Optuna pruning** during hyperparameter search
6. Always use **`best.pth`** for evaluation
