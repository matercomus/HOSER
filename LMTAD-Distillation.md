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

- Robust loading via weights‑only artifacts to avoid pickle/import issues:
  - Export once using `tools/export_lmtad_weights.py` to a file with `{'state_dict', 'model_config'}`.
  - `critics/lmtad_teacher.py` constructs `LMTAD` from `model_config`, strips any `_orig_mod.` prefixes, and loads `state_dict`.
- Precision and perf:
  - Uses `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` on RTX 20‑series.
  - Sliding window history with `config.distill.window` (e.g., 32) capped by the model’s block size.
  - Returns log‑softmax or softmax distributions over the full grid vocab; batched over histories for throughput.
- Tokens:
  - If special tokens exist in the checkpoint’s config, the wrapper handles them; current Beijing setup does not require an explicit SOT (logged as `SOT token id: None`).

### 2) Grid Mapping Utility

- Implemented in `critics/grid_mapper.py`.
- Computes `road_id → grid_token` once from `roadmap.geo` centroids using:
  - `distill.grid_size` (degrees) and optional `distill.downsample`.
  - Dataset bounds inferred from geometries.
- Mapping is NumPy‑vectorized and returns a contiguous `np.int64` array for O(1) lookups during training.

### 3) Distillation Hook and KL Computation

- Implemented in `critics/distill_hook.py` as `DistillationManager`.
- Core flow per batch (`compute_kl_for_batch`):
  1. Slice history per sample to the last `window` grid tokens via `GridMapper`.
  2. Batch teacher calls to get next‑token distributions for all histories.
  3. Map candidate road IDs at each distilled position to grid tokens; cap candidates to `data.candidate_top_k` for memory.
  4. Gather teacher probabilities for candidate tokens and renormalize to the candidate set $C$.
  5. Compute student candidate distribution with temperature and apply KL divergence.
  6. Return the mean KL loss for integration into the total loss.
- The computation is vectorized on GPU to maximize utilization.

### 4) Training Script and Loop

- Implemented in `train_with_distill.py`.
- Config‑first: loads `config/Beijing.yaml`; CLI can override `--dataset`, `--config`, `--data_dir`, `--seed`.
- Performance features:
  - AMP + `GradScaler` with correct accumulation semantics (`unscale_` only on optimizer step).
  - Gradient accumulation (`optimizer_config.accum_steps`).
  - `torch.compile(mode='reduce-overhead')` with CUDA graphs disabled (`training.disable_cudagraphs: true`) to avoid graph reuse issues.
  - Gradient checkpointing in `TrajectoryEncoder` (`trajectory_encoder_config.grad_checkpoint: true`).
  - Dataloader tuning: `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`.
  - Candidate capping: `data.candidate_top_k` reduces memory and HtoD traffic.
  - Cached road/zone embeddings once per epoch to avoid redundant recomputation.
- Logging & validation:
  - Full YAML config is logged to WandB at run start.
  - Per‑epoch validation of next‑step accuracy and time‑prediction MAPE (TensorBoard + WandB).

### 5) Configuration (YAML)

`config/Beijing.yaml` (key excerpts):

```yaml
distill:
  enable: true
  repo: /home/matt/Dev/LMTAD
  ckpt: /home/matt/Dev/LMTAD/code/results/LMTAD/beijing_hoser/ckpt_best_weights_only.pt
  window: 32
  lambda: 0.01
  temperature: 2.0
  grid_size: 0.001
  downsample: 4

dataloader:
  num_workers: 16
  pin_memory: false
  prefetch_factor: 2
  persistent_workers: true

data:
  candidate_top_k: 16

training:
  allow_tf32: true
  cudnn_benchmark: true
  torch_compile: true
  disable_cudagraphs: true

wandb:
  enable: true
  project: hoser-distill
  run_name: ''
  tags: [beijing, lmtad, distillation]
```

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

### 6) Hook‑Up Summary (How it fits together)

1. `train_with_distill.py` loads config and model, prepares dataloaders, and constructs `DistillationManager` with `LMTADTeacher` and `GridMapper`.
2. For each batch, the standard supervised losses are computed.
3. If `distill.enable`, `DistillationManager.compute_kl_for_batch(...)` is called to obtain `kl_loss` using the vectorized pipeline above.
4. The final loss is: `loss_total = loss_road + loss_time + distill.lambda * kl_loss`.
5. Gradients are accumulated per `accum_steps`, scaled/unscaled correctly, clipped (`max_norm`), and optimizer stepped.
6. Per‑epoch validation metrics are logged alongside training metrics to WandB/TensorBoard.

### 5) Performance Tips

- Use a small window (e.g., 64) for history.
- Distill only 1–2 steps per sequence per batch (`sample_steps_per_trace`) to bound compute.
- Batch teacher calls across sequences where possible. Follow LMTAD’s precision policy:
  - Use `float16` with autocast on CUDA (`train_LMTAD.py` default), or `bfloat16` on Ampere+.
  - Keep teacher on GPU; your student (HOSER) already runs on GPU.
- Cache `road_id -> grid_token` and LM‑TAD model on GPU; run student forward as before.

Additional optimizations implemented:
- Gradient accumulation (`optimizer_config.accum_steps`).
- `torch.compile` with CUDA graphs disabled to avoid tensor reuse errors.
- Gradient checkpointing in `TrajectoryEncoder`.
- Dataloader tuning and candidate capping to reduce memory pressure.

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

## Practical Considerations

- Many‑to‑one mapping: multiple roads can map to the same grid token. This is fine: the teacher supplies probability mass per grid cell, and renormalization on candidates makes the KL comparable.
- Special tokens: If LM‑TAD requires `SOT`, prepend it exactly as used in training; EOT is not needed for next‑step.
- Numerical stability: clamp denominators and add small epsilons in logs.
- Ablations: set `lambda_kl=0` to disable, or try different $\tau$ values.

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


