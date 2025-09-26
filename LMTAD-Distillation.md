# Distilling LM‑TAD into HOSER at Training Time

## Executive Summary

We propose a training‑time distillation approach where a frozen LM‑TAD model (teacher) provides a “normality prior” for next‑step decisions, and HOSER (student) is trained to align its candidate distribution to the teacher while keeping HOSER’s original supervised objectives intact. This removes the need to co‑run LM‑TAD at inference time, keeping deployment simple and fast while capturing LM‑TAD’s behavioral knowledge.

- No runtime overhead: HOSER remains a single model at inference.
- Minimal code changes: a teacher wrapper, a road→grid token mapping, and a KL term in `train.py`.
- Tunable: temperature, KL weight, and sampling frequency control compute/quality trade‑offs.

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

### 1) Teacher Wrapper (LM‑TAD)

- Load the trained LM‑TAD checkpoint using the same path as `eval_porto.py`.
- Expose: `predict_next_distribution(history_tokens: LongTensor[1,T]) -> Tensor[V]` returning a softmax over grid vocab.
- Use AMP/FP16, `torch.no_grad()`, and a sliding window (e.g., last 64 tokens). Cache KV state by the last K tokens if the LM‑TAD model exposes it.

Suggested module: `critics/lmtad_teacher.py`.

### 2) Grid Mapping Utility

- Build once: `road_id -> grid_token` using LM‑TAD’s dataset_config boundaries and grid parameters.
- Vectorize with NumPy for speed and store as a contiguous `np.int64` array. Move to GPU only if needed.

### 3) Training Config (YAML)

Add a `distill` section:

```yaml
distill:
  enabled: true
  repo_path: /home/matt/Dev/LMTAD
  ckpt_path: /home/matt/Dev/LMTAD/checkpoints/porto_lmtad.pt
  lambda_kl: 0.01
  temperature: 2.0
  window: 64
  sample_steps_per_trace: 1   # number of time steps per sequence to distill
  teacher_fp16: true
```

### 4) Hook in `train.py`

After computing HOSER logits/time (inside the training loop):

1. Choose which step(s) to distill per sequence (e.g., last step or random $M$ steps per trace to bound cost).
2. For each selected step:
   - Build the history as grid tokens (prepend SOT token if LM‑TAD expects it; omit EOT).
   - Map the candidate road IDs at that step to grid tokens; deduplicate identical grid tokens (sum later if many‑to‑one).
   - Call the teacher to get $q$; gather $q$ for the candidate tokens; renormalize to $q_C$.
   - Compute student $p_C$ with temperature; compute $\mathcal{L}_{\text{KL}}$ and add to the batch loss.

Pseudocode sketch:

```python
# logits: (B, T, Cmax), candidate_len: (B, T)
# Select positions idxs to distill
with torch.no_grad():
    q_list = []
    for b,t in idxs:
        hist_roads = batch_trace_road_id[b, :batch_trace_len[b]]  # to road IDs
        hist_tokens = road_to_grid[hist_roads.cpu().numpy()]      # to grid tokens
        hist_tokens = add_sot(hist_tokens)
        q_full = teacher.predict_next_distribution(hist_tokens)   # (V,)

        cand_roads = batch_candidate_road_id[b, t, :candidate_len[b, t]]
        cand_tokens = road_to_grid[cand_roads.cpu().numpy()]
        q_c = q_full[cand_tokens]
        q_c = q_c / q_c.sum().clamp_min(1e-9)
        q_list.append(q_c)

kl_loss = 0.0
for (b,t), q_c in zip(idxs, q_list):
    s_logits = logits[b, t, :candidate_len[b, t]]                # (C,)
    p_tau = torch.softmax(s_logits / T, dim=-1)
    q_tau = (q_c ** (1.0 / T))
    q_tau = q_tau / q_tau.sum().clamp_min(1e-9)
    kl = torch.sum(q_tau * (torch.log(q_tau + 1e-9) - torch.log(p_tau + 1e-9)))
    kl_loss = kl_loss + kl

loss = loss_next_step + loss_time_pred + lambda_kl * (kl_loss / len(idxs))
```

### 5) Performance Tips

- Use a small window (e.g., 64) for history.
- Distill only 1–2 steps per sequence per batch (`sample_steps_per_trace`) to bound compute.
- Batch teacher calls across sequences where possible.
- Cache `road_id -> grid_token` and LM‑TAD model on GPU; run student forward as before.

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

- `critics/lmtad_teacher.py` (LM‑TAD loader + next‑token distribution) 
- Grid mapping utility (reuse LMTAD preprocessing formula)
- `train.py` updates: config flags, KL computation path, logging for `kl_loss`

## Usage (example)

```bash
uv run python train.py --dataset Beijing --cuda 0 \
  --config config/Beijing.yaml  # distill.* keys enabled in YAML
```


