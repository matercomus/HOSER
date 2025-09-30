# Distilling LM‑TAD into HOSER at Training Time

## Executive Summary

We propose a training‑time distillation approach where a frozen LM‑TAD model (teacher) provides a “normality prior” for next‑step decisions, and HOSER (student) is trained to align its candidate distribution to the teacher while keeping HOSER’s original supervised objectives intact. This removes the need to co‑run LM‑TAD at inference time, keeping deployment simple and fast while capturing LM‑TAD’s behavioral knowledge.

- No runtime overhead: HOSER remains a single model at inference.
- Minimal code changes: a teacher wrapper, a road→grid token mapping, and a KL term in `train.py`.
- Tunable: temperature, KL weight, and sampling frequency control compute/quality trade‑offs.

## What’s Implemented (Current Code)

- `critics/lmtad_teacher.py`: Loads LM‑TAD from a weights‑only export (`{'state_dict', 'model_config'}`) and provides vectorized next‑token distributions with sliding window and AMP.
- `tools/export_lmtad_weights.py`: Utility to export LM‑TAD checkpoints into a pickle‑free, weights‑only format to avoid module import collisions.
- `critics/grid_mapper.py`: Vectorized road‑centroid → grid‑token mapper; precomputes `road_id → grid_token` using dataset bounds and grid params.
- `critics/distill_hook.py`: `DistillationManager` orchestrates teacher calls, mapping of candidates, renormalization on candidate set, and computes batched KL loss.
- `train_with_distill.py`: Standalone training script that wires in distillation, gradient accumulation, AMP, `torch.compile` (with CUDA graphs disabled), gradient checkpointing, dataloader tuning, per‑epoch validation, and full‑config WandB logging.
- `config/Beijing.yaml`: Single source of truth (config‑first). Contains `distill`, `training`, `dataloader`, `data`, `wandb`, and optimizer/model settings.
- Generation and evaluation:
  - `gene.py` supports `--model_path` to load the distilled checkpoint for generation.
  - `evaluation.py` computes global/local metrics (JSD, Hausdorff, DTW) and logs to WandB.
  - `tools/collect_distill_artifacts.py` converts generated trips to GeoJSON and bundles model/eval/WandB artifacts.

## Overview

- Teacher: a trained LM‑TAD model over grid tokens (SOT/EOT/PAD + grid indices). It outputs a probability distribution over the grid vocabulary given a history of grid tokens.
- Student: HOSER’s `Navigator` outputs logits over next‑road candidates at each step. We align to the teacher by mapping candidate roads to the teacher’s grid tokens and comparing distributions on the candidate subset.
- Distillation signal: add a KL divergence term between teacher and student next‑step distributions (temperature‑scaled) to HOSER’s original loss.

## Token Alignment (Road → Grid)

LM‑TAD tokens are grid cells produced from the road geometry’s centroid using the same grid config used during LM‑TAD preprocessing. Reuse LM‑TAD’s dataset_config (from the checkpoint) for:

- grid_size (degrees) and optional downsample_factor
- geographic boundaries (min/max lat/lng) used to compute indices

Mapping formula (centroid at $\text{lat},\text{lng}$):

$$
\begin{aligned}
g_i &= \left\lfloor \frac{\text{lat} - \text{min\_lat}}{\text{grid\_size}} \right\rfloor,\quad
g_j = \left\lfloor \frac{\text{lng} - \text{min\_lng}}{\text{grid\_size}} \right\rfloor \\
\text{token}(\text{road}) &= g_i \cdot N_{\text{lng}} + g_j\quad (\text{after downsampling and clamping})
\end{aligned}
$$

Precompute a vectorized `road_id -> grid_token` array once at startup to make training‑time mapping O(1).

## Loss Design

Let $C$ be the candidate set at a step, $z_k$ the teacher’s grid token for candidate $k$, $q$ the teacher’s distribution over the full grid vocab given history $H$, and $p$ the student’s candidate distribution from HOSER. With temperature $\tau>0$:

- Teacher on candidates (renormalized):
$$
q_C^{(\tau)}(k) = \frac{\left(q(z_k)\right)^{1/\tau}}{\sum\limits_{j\in C} \left(q(z_j)\right)^{1/\tau}}
$$

- Student on candidates:
$$
p_C^{(\tau)}(k) = \frac{\exp\left(\frac{\text{logits}_k}{\tau}\right)}{\sum\limits_{j\in C} \exp\left(\frac{\text{logits}_j}{\tau}\right)}
$$

- Distillation term (teacher→student):
$$
\mathcal{L}_{\text{KL}} = \mathrm{KL}\left(q_C^{(\tau)}\;\Vert\; p_C^{(\tau)}\right) = \sum_{k\in C} q_C^{(\tau)}(k)\,\left[\log q_C^{(\tau)}(k) - \log p_C^{(\tau)}(k)\right]
$$

- Total training objective per step (unchanged supervised terms):
$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{road}}}_{\text{cross‑entropy to label}}\; +\; \underbrace{\mathcal{L}_{\text{time}}}_{\text{MAPE time}}\; +\; \lambda_{\text{KL}}\,\mathcal{L}_{\text{KL}}
$$

Notes:
- Renormalization on $C$ makes the teacher comparable to the student (which only scores candidates).
- $\tau$ softens distributions; typical $\tau\in[1,3]$.

## Current Implementation Details

### 1) Teacher Wrapper (LM‑TAD)

- **Robust loading** via weights‑only artifacts to avoid pickle/import issues:
  - Export once using `tools/export_lmtad_weights.py` to a file with `{'state_dict', 'model_config'}`.
  - `critics/lmtad_teacher.py` constructs `LMTAD` from `model_config`, strips any `_orig_mod.` prefixes, and loads `state_dict`.
  - Automatically infers SOT (Start-Of-Trajectory) token ID from the model's vocabulary size: `sot_id = vocab_size - 1`.
  
- **Precision and performance**:
  - Uses `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` on RTX 20‑series GPUs.
  - Sliding window history with `config.distill.window` (default: 4 steps) capped by the model's block size.
  - Returns softmax distributions over the full grid vocabulary (51,663 tokens for Beijing).
  - **Batched inference**: All history sequences in a batch are processed simultaneously for maximum GPU utilization.
  
- **Special token handling**:
  - SOT token is prepended to all sequences during teacher inference.
  - The implementation correctly handles the vocabulary mismatch: LM-TAD uses grid cell tokens (0-51662) while HOSER uses road IDs (0-40059).
  - Grid tokens are precomputed and cached to avoid repeated computation during training.

### 2) Grid Mapping Utility

- **Implementation**: `critics/grid_mapper.py` with `tools/precompute_distill_tokens.py` for preprocessing.
- **Grid token computation**:
  - Uses road geometry centroids from `roadmap.geo` (mean of all coordinate points per road).
  - Grid parameters from config: `distill.grid_size` (0.001 degrees) and `distill.downsample` (1, no downsampling).
  - Dataset bounds automatically inferred from all road geometries (Beijing: lat 39.44-41.06, lng 115.42-117.50).
  - Grid dimensions: 205×252 cells = 51,663 total grid tokens (matches LM-TAD vocabulary).
  
- **Vectorized mapping**:
  - Formula: `token = floor((lat - min_lat) / grid_size) * n_cols + floor((lng - min_lng) / grid_size)`
  - Implemented as NumPy vectorized operations for all 40,060 roads at once.
  - Returns contiguous `np.int64` array for O(1) lookups: `road_to_token[road_id] → grid_token`.
  
- **Preprocessing optimization**:
  - Grid tokens are precomputed for all training/validation data using `tools/precompute_distill_tokens.py`.
  - Uses multiprocessing (all 64 CPU cores) to augment existing cache files with `trace_grid_token` and `candidate_grid_token` keys.
  - Processes ~2,000 cache files in parallel, avoiding per-batch computation overhead during training.

### 3) Distillation Hook and KL Computation

- **Implementation**: `critics/distill_hook.py` as `DistillationManager`.
- **Vectorized KL computation pipeline** (`compute_kl_for_batch`):
  
  1. **History preparation**:
     - Extract last `window` steps (default: 4) of grid tokens for each trajectory in the batch.
     - Prepend SOT token (ID: 51662) to each history sequence.
     - Pre-allocate `sot_tensor` once during initialization to avoid repeated tensor creation.
     - All operations kept on GPU to minimize CPU↔GPU transfers.
  
  2. **Teacher inference** (batched):
     - Construct batch of history sequences: `[batch_size × trace_len, history_window]`
     - Single batched forward pass through LM-TAD teacher model (no loop over samples).
     - Teacher outputs shape: `[batch_size × trace_len, vocab_size=51663]`
     - Uses FP16 autocast for memory efficiency (~2× faster than FP32).
  
  3. **Candidate mapping and renormalization**:
     - For each position, map candidate road IDs to grid tokens via precomputed `road_to_token` array.
     - Index teacher logits with candidate tokens: `teacher_logits[row, candidate_tokens]`
     - Renormalize to candidate set: `q_c = softmax(teacher_logits[candidates]) / sum(softmax(...))`
     - Handles edge cases: invalid probabilities (NaN/inf) are skipped.
  
  4. **Student distribution**:
     - HOSER outputs logits over candidates (not all roads): shape `[B, T, num_candidates]`
     - Temperature-scaled softmax: `p_tau = softmax(student_logits / temperature)`
     - Temperature τ ∈ [2.0, 4.0] softens both distributions for better knowledge transfer.
  
  5. **KL divergence computation**:
     - Forward KL: `KL(teacher || student) = Σ q_c(k) * [log(q_c(k)) - log(p_c(k))]`
     - Computed position-wise, then averaged over all valid positions in the batch.
     - Numerical stability: clamping and epsilon (1e-9) to prevent log(0).
  
- **Performance optimizations**:
  - **No caching**: Initial teacher output caching was removed—cache key creation overhead exceeded teacher forward pass time for diverse sequences.
  - **GPU-only operations**: All tensor operations (indexing, masking, softmax) done on GPU; only single scalar values converted to CPU.
  - **Batch sorting**: Sequences sorted by candidate count for slightly better memory access patterns.

### 4) Training Script and Loop

- **Implementation**: `train_with_distill.py` with full GPU optimization pipeline.
- **Configuration**: Config-first approach via `config/Beijing.yaml`; CLI can override `--dataset`, `--config`, `--data_dir`, `--seed`.

- **Training performance optimizations**:
  - **Mixed precision (AMP)**: `torch.amp.autocast` + `GradScaler` with correct accumulation semantics.
  - **Gradient accumulation**: `accum_steps=4` for effective batch size of 1024 (256 × 4).
  - **torch.compile**: Mode `max-autotune` for aggressive kernel fusion and optimization.
  - **CUDA graphs disabled**: `training.disable_cudagraphs: true` to avoid tensor reuse issues with dynamic shapes.
  - **Gradient checkpointing**: Optional in `TrajectoryEncoder` (disabled by default—sufficient VRAM available).
  - **TF32 tensor cores**: `allow_tf32: true` for ~2× matmul speedup on Ampere GPUs.
  - **cuDNN benchmarking**: `cudnn_benchmark: true` to select fastest convolution algorithms.

- **Dataloader tuning** (optimized for large batch size):
  - `num_workers: 16` for parallel data loading across 16 CPU cores.
  - `pin_memory: true` for async CPU→GPU transfer.
  - `prefetch_factor: 2` to keep 2 batches ready per worker.
  - `persistent_workers: true` to avoid worker respawn overhead between epochs.

- **Candidate filtering**:
  - `data.candidate_top_k: 0` (currently disabled) — would reduce candidates to top-k closest by distance.
  - When enabled, requires careful label remapping to handle filtered candidate indices.
  - Currently using full candidate sets (average ~3-4 candidates per position).

- **Performance profiling** (every 100 batches):
  - Tracks time breakdown: data transfer (0.2%), HOSER forward (1.6%), teacher KL (86.7%), backward (11.5%).
  - Logged to console and WandB for bottleneck identification.

- **Logging & validation**:
  - Full YAML config logged to WandB at run start for reproducibility.
  - Per-epoch validation: next-step accuracy, time-prediction MAPE.
  - Metrics logged to both TensorBoard and WandB.
  - Best model saved based on validation accuracy.

### 5) Configuration (YAML)

`config/Beijing.yaml` (current production settings):

```yaml
optimizer_config:
  max_epoch: 25
  batch_size: 256      # 8× increase from baseline (32), optimized for RTX 2080 Ti
  accum_steps: 4       # Effective batch size: 1024
  learning_rate: 0.001
  weight_decay: 0.1
  warmup_ratio: 0.1
  max_norm: 1.0

distill:
  enable: true
  repo: /home/matt/Dev/LMTAD
  ckpt: /home/matt/Dev/LMTAD/code/results/LMTAD/beijing_hoser_reference/.../weights_only.pt
  window: 4            # Reduced from 32 for faster teacher inference
  lambda: 0.01         # Tunable via Optuna (0.001-0.1 range)
  temperature: 2.0     # Tunable via Optuna (1.5-4.0 range)
  grid_size: 0.001     # Matches LM-TAD training: 205×252 grid
  downsample: 1        # No downsampling (LM-TAD Beijing model)

dataloader:
  num_workers: 16      # Full CPU parallelism (64-core system)
  pin_memory: true     # Async GPU transfer
  prefetch_factor: 2   # Balanced for large batch size
  persistent_workers: true

data:
  candidate_top_k: 0   # Disabled (label remapping complexity)

training:
  allow_tf32: true
  cudnn_benchmark: true
  torch_compile: true
  torch_compile_mode: max-autotune  # Aggressive optimization
  disable_cudagraphs: true          # Dynamic shapes require this

wandb:
  enable: true
  project: hoser-distill-optuna
  run_name: ''
  tags: [optuna, distill-tuning, beijing]
```

**Performance impact**:
- Batch size 256: 2.8× throughput improvement (77 → 218 samples/sec)
- Epoch time: ~49 minutes (down from ~2.7 hours with batch_size=32)
- Teacher inference: 857ms per batch (86.7% of total time), but amortized over 256 samples

## Simple Illustrative Example

Assume 3 candidates $C=\{A,B,C\}$ mapped to grid tokens $\{t_5,t_8,t_{12}\}$.

- Teacher (LM‑TAD) over full vocab assigns: $q(t_5)=0.7,\; q(t_8)=0.2,\; q(t_{12})=0.1$. After renormalizing to $C$: $q_C = [0.7, 0.2, 0.1]$.
- Student (HOSER) logits over candidates → $p_C = [0.4, 0.4, 0.2]$.

With $\tau=1$:
$$
\mathcal{L}_{\text{KL}} = 0.7\log\frac{0.7}{0.4}+0.2\log\frac{0.2}{0.4}+0.1\log\frac{0.1}{0.2}
$$
This term pushes the student to up‑weight \(A\) and down‑weight \(B\) and \(C\), aligning with teacher preferences while the main CE still anchors to the ground‑truth label.

## Implementation Plan

### 6) Hook‑Up Summary (Complete Pipeline)

1. **Preprocessing** (one-time):
   - Run `tools/precompute_distill_tokens.py` to augment cache files with grid tokens.
   - Uses all CPU cores to process ~2,000 cache files in parallel.

2. **Training initialization** (`train_with_distill.py`):
   - Load `config/Beijing.yaml` and construct HOSER model.
   - Initialize `DistillationManager` with:
     - `LMTADTeacher`: loads frozen LM-TAD weights, moves to GPU.
     - `GridMapper`: precomputes `road_to_token` mapping from road geometries.
   - Prepare dataloaders with optimized settings (16 workers, pin_memory, etc.).

3. **Training loop** (per batch):
   - Forward pass: HOSER predicts next-step logits and time estimates.
   - Supervised losses: `loss_road` (cross-entropy), `loss_time` (MAPE).
   - Distillation: `DistillationManager.compute_kl_for_batch(...)`:
     - Extract grid token histories from batch.
     - Batched teacher inference for all positions.
     - Map candidates to grid tokens, renormalize teacher distribution.
     - Compute KL divergence with temperature scaling.
   - **Total loss**: `loss_total = loss_road + loss_time + λ * kl_loss`
   - Backward pass with AMP gradient scaling.
   - Gradient accumulation (every `accum_steps` batches).
   - Optimizer step with gradient clipping (`max_norm=1.0`).

4. **Validation** (per epoch):
   - Compute next-step accuracy and time-prediction MAPE.
   - Log metrics to WandB and TensorBoard.
   - Save best model based on validation accuracy.

### 7) Performance Characteristics

**Measured timings** (batch_size=256, RTX 2080 Ti):
- Data transfer: 2.29ms (0.2%)
- HOSER forward: 15.64ms (1.6%)
- Teacher KL (LM-TAD): 856.71ms (86.7%)
- Backward pass: 113.91ms (11.5%)
- **Total**: 988.63ms per batch

**Throughput**: 218 samples/sec (256 samples / 0.989s)

**Bottleneck analysis**:
- Teacher inference dominates (87% of time) due to large vocabulary (51,663 tokens).
- Large batch size (256) amortizes this cost: 3.35ms per sample vs ~10ms with small batches.
- Further optimization challenging: teacher already uses FP16, batched inference, and all positions in parallel.

**Scalability**:
- Training set: 629,421 trajectories → 2,459 batches/epoch
- Epoch time: ~49 minutes (2,459 batches × 1.2s/batch)
- Full training (25 epochs): ~20 hours

## Alignment with LMTAD Code (train.sh, train_LMTAD.py, eval_porto.py)

- Checkpoint structure and loading
  - We export to weights‑only (`state_dict` + `model_config`) to eliminate pickle dependencies and module name collisions.
  - `_orig_mod.` prefixes are stripped before `load_state_dict`.

- Dtype and AMP policy
  - On RTX 2080 Ti, prefer `float16` autocast; set via config `distill.dtype: float16`.
  - Use `ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)` as in both `eval_porto.py` and `train_LMTAD.py` for teacher forward.

- Dictionary and special tokens
  - If needed, instantiate a lightweight `PortoDataset(dataset_config)` only to access `dictionary` (SOT/EOT/PAD). For next‑token scoring, EOT is not required; prepend SOT if your checkpoint used it.

- Grid dimensions (Beijing‑HOSER)
  - `train.sh` passes `--grip_size H W` (grid height, width) into `train_LMTAD.py`, stored in `dataset_config.grip_size` for Beijing runs.
  - Validate your computed grid dims against `dataset_config.grip_size` when `verify_grid_dims` is true.

- Efficient evaluation path
  - Follow `eval_porto.py` to compute probabilities efficiently: run the model once on the (windowed) history, take `logits[:, -1, :]`, apply `softmax`, then gather candidate token probabilities. No dataloader is needed for the teacher inside distillation.

## Key Implementation Challenges and Solutions

### 1. Vocabulary Mismatch (Grid Tokens vs Road IDs)
**Challenge**: LM-TAD operates on 51,663 grid cell tokens, while HOSER uses 40,060 road IDs.

**Solution**: 
- Precompute `road_to_token` mapping using road centroids and grid parameters.
- At training time: convert road IDs → grid tokens for teacher input, then index teacher's output distribution by candidate grid tokens.
- Many-to-one mapping is acceptable: multiple roads can share a grid cell; renormalization on candidates handles this.

### 2. SOT Token Handling
**Challenge**: LM-TAD requires SOT (Start-Of-Trajectory) token prepended to sequences.

**Solution**:
- Automatically infer SOT ID from vocabulary size: `sot_id = vocab_size - 1 = 51662`.
- Pre-allocate `sot_tensor` once during initialization to avoid repeated tensor creation.
- Prepend to all history sequences before teacher inference.

### 3. GPU Memory and Performance
**Challenge**: Teacher inference is expensive (87% of training time).

**Solutions implemented**:
- **Large batch size** (256): Amortizes teacher cost across more samples.
- **FP16 autocast**: Reduces memory and speeds up teacher by ~2×.
- **No caching**: Cache overhead exceeded forward pass time for diverse sequences.
- **GPU-only operations**: Minimize CPU↔GPU transfers; only convert final scalars.
- **Batched inference**: Process all positions simultaneously (no loops over samples).

### 4. Candidate Filtering and Label Remapping
**Challenge**: HOSER can filter candidates to top-k by distance, but labels index the full candidate list.

**Solution**: 
- Disabled `candidate_top_k` filtering (set to 0).
- Alternative: Implement vectorized label remapping using broadcasting to find where original labels appear in filtered candidates.
- Trade-off: Full candidate sets use more memory but avoid complex remapping logic.

### 5. Numerical Stability
**Solutions**:
- Clamp denominators with epsilon (1e-9) to prevent division by zero.
- Check for NaN/inf in teacher and student probabilities; skip invalid positions.
- Use log-softmax where possible to avoid exp() overflow.
- Temperature scaling (τ > 1) smooths distributions and reduces numerical issues.

## Practical Considerations

- **Many‑to‑one mapping**: Multiple roads can map to the same grid token. This is acceptable: the teacher supplies probability mass per grid cell, and renormalization on candidates makes the KL comparable.
- **Special tokens**: SOT is prepended as required by LM-TAD training; EOT is not needed for next‑step prediction.
- **Numerical stability**: Denominators clamped with epsilon (1e-9), NaN/inf checks on all distributions.
- **Ablations**: Set `lambda=0` to disable distillation, or sweep `temperature` ∈ [1.5, 4.0] for different knowledge transfer characteristics.

## Evaluation Plan

- Training metrics: Observe reduced `loss_next_step` and stable/declining `kl_loss` over epochs when $\lambda_{\text{KL}}>0$.
- Validation metrics: HOSER’s next‑step accuracy and time MAPE should stay the same or improve.
- Trajectory‑level: Compare DTW/EDR and distributional JSD vs real data; expect improvements similar to the runtime critic but at zero inference cost.

## Risks and Mitigations

- Teacher mismatch (tokens/grid): Ensure the mapping uses the LM‑TAD checkpoint’s dataset_config.
- Over‑regularization: Very high $\lambda_{\text{KL}}$ can hurt supervised metrics—sweep $\lambda_{\text{KL}}$ and $\tau$.
- Compute overhead: Limit sampled steps and use small windows; teacher runs in FP16 with no_grad.

## Deliverables

- `critics/lmtad_teacher.py` (LM‑TAD loader + next‑token distribution, weights‑only load)
- `critics/grid_mapper.py` (vectorized road→grid mapping)
- `critics/distill_hook.py` (batched KL computation over candidate set)
- `train_with_distill.py` (training loop integration, AMP/accumulation/compile/validation/WandB)
- `tools/export_lmtad_weights.py` (robust checkpoint exporter)

## Usage (example)

```bash
uv run python train_with_distill.py --dataset Beijing --cuda 0 \
  --config config/Beijing.yaml \
  --data_dir /path/to/hoser_format
```

### Generation and Evaluation

```bash
# Generate with distilled model
uv run python gene.py --dataset Beijing --model_path save/Beijing/seed0_distill/best.pth

# Evaluate and log to WandB
uv run python evaluation.py --run_dir <RUN_DIR> --wandb --wandb_project hoser-eval \
  --wandb_run_name <RUN_NAME>_eval --wandb_tags beijing distill eval

# Bundle artifacts and convert to GeoJSON
uv run python tools/collect_distill_artifacts.py \
  --run_name <RUN_NAME> \
  --run_dir <RUN_DIR> \
  --generated_csv <path/to/generated.csv> \
  --backup_root /mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil
```


