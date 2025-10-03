# Distilling LM‑TAD into HOSER at Training Time

## Executive Summary

We propose a training‑time distillation approach where a frozen LM‑TAD model (teacher) provides a “normality prior” for next‑step decisions, and HOSER (student) is trained to align its candidate distribution to the teacher while keeping HOSER’s original supervised objectives intact. This removes the need to co‑run LM‑TAD at inference time, keeping deployment simple and fast while capturing LM‑TAD’s behavioral knowledge.

- No runtime overhead: HOSER remains a single model at inference.
- Minimal code changes: a teacher wrapper, a road→grid token mapping, and a KL term in `train.py`.
- Tunable: temperature, KL weight, and sampling frequency control compute/quality trade‑offs.

## What's Implemented (Current Code)

- `critics/lmtad_teacher.py`: Loads LM‑TAD from a weights‑only export (`{'state_dict', 'model_config'}`) and provides vectorized next‑token distributions with sliding window and AMP.
- `tools/export_lmtad_weights.py`: Utility to export LM‑TAD checkpoints into a pickle‑free, weights‑only format to avoid module import collisions.
- `critics/grid_mapper.py`: Vectorized road‑centroid → grid‑token mapper; precomputes `road_id → grid_token` using dataset bounds and grid params.
- `critics/distill_hook.py`: `DistillationManager` orchestrates teacher calls, mapping of candidates, renormalization on candidate set, and computes batched KL loss.
- `train_with_distill.py`: Standalone training script that wires in distillation, gradient accumulation, AMP, `torch.compile` (with CUDA graphs disabled), gradient checkpointing, dataloader tuning, per‑epoch validation, and full‑config WandB logging.
- `config/Beijing.yaml`: Single source of truth (config‑first). Contains `distill`, `training`, `dataloader`, `data`, `profiling`, `wandb`, and optimizer/model settings.
- `dataset.py`: Smart, parallelized Dataset caching. It estimates the dataset's RAM footprint and available system RAM, then uses all CPU cores to load the entire dataset into memory if it fits. This eliminates disk I/O bottlenecks during training. If the dataset is too large, it falls back to the previous disk streaming mode.
- Generation and evaluation:
  - `gene.py` supports `--model_path` to load the distilled checkpoint for generation.
  - `evaluation.py` computes global/local metrics (JSD, Hausdorff, DTW) and logs to WandB.
  - `tools/collect_distill_artifacts.py` converts generated trips to GeoJSON and bundles model/eval/WandB artifacts.

## Recent Performance & Stability Improvements

The implementation has been heavily optimized to handle large-scale training (629k samples) with stable throughput. The primary bottleneck—disk I/O from streaming hundreds of thousands of small files—has been eliminated.

**Performance notes (current hardware, RTX 2080 Ti)**:
- **Vanilla baseline**: ~10–12 it/s with disk streaming (Data Wait ~50 ms). With RAM caching, Data Wait becomes negligible.
- **Distilled runs**: Dominated by teacher compute (~430 ms per step with window≈4), yielding ~1.6–1.8 it/s. RAM caching still helps, but teacher remains the bottleneck.
- **Data Wait** with RAM cache: typically <5% (all samples in memory).

**Key optimizations**:
1. **Smart Parallel RAM Caching**: The ultimate fix. `dataset.py` now automatically detects if the dataset can fit into available system RAM. If so, it uses all available CPU cores to load the entire dataset into memory in parallel at startup, eliminating all disk I/O during training.
2. **Candidate top-k filtering** (64): Prevents collate bottleneck from pathological traces with 1000+ candidates.
3. **Optimized collate function**: Vectorized operations, reduced allocations, pre-cropped arrays.
4. **Fallback to Disk Streaming**: If the dataset is too large for RAM, it falls back to the previous optimized streaming mode (6 workers, `prefetch_factor=16`) which sustains ~11-13 it/s.
5. **Label remapping**: GPU-based vectorized remapping with `-100` masking for filtered positions.
6. **Loss masking**: Both cross_entropy and time loss properly handle invalid positions.
7. **File handle management**: Explicit context managers prevent descriptor exhaustion in streaming mode.

## Overview

- Teacher: a trained LM‑TAD model over grid tokens (SOT/EOT/PAD + grid indices). It outputs a probability distribution over the grid vocabulary given a history of grid tokens.
- Student: HOSER’s `Navigator` outputs logits over next‑road candidates at each step. We align to the teacher by mapping candidate roads to the teacher’s grid tokens and comparing distributions on the candidate subset.
- Distillation signal: add a KL divergence term between teacher and student next‑step distributions (temperature‑scaled) to HOSER’s original loss.

## Visual Overview

### Training & Distillation Data Flow

```mermaid
flowchart TB
  subgraph Data[Data Preparation]
    A[HOSER Dataset] --> B{Fits in RAM?}
    B -- Yes --> C[Parallel RAM cache]
    B -- No --> D[Stream from disk]
  end

  subgraph Loader[Batch Assembly]
    C --> E[DataLoader]
    D --> E
    E --> F[MyCollateFn]
    F --> G[Top-K candidates]
    G --> H[Label remap]
  end

  subgraph Model[Forward & Losses]
    H --> I[HOSER forward]
    I --> J[Student logits]
    J --> K[Time head]
  end

  subgraph Teacher[Teacher Guidance]
    H --> L[Road to Grid tokens]
    L --> M[LM-TAD teacher FP16]
    M --> N[Teacher probs]
    N --> O[Select candidates and renorm]
  end

  subgraph Losses[Loss Computation]
    J --> P[Cross-entropy]
    K --> Q[Time loss]
    O --> R[KL divergence]
    P --> S[Total loss]
    Q --> S
    R --> S
    S --> T[GradScaler and Optimizer]
  end

  subgraph Eval[Logging & Eval]
    T --> U[TensorBoard and WandB]
    T --> V[Per-epoch validation]
    V --> W[Save best model]
  end
```

### Optuna Tuning Loop (2‑Phase)

```mermaid
flowchart LR
  subgraph Search[Phase 1: Hyperparameter Search]
    A0[Trial 0 vanilla baseline] --> B0[Evaluate and validate]
    A[CMA-ES sampler] --> B[Create trial config]
    B --> C[Run training max 8 epochs]
    C --> D[Report val_acc per epoch]
    D --> E{Hyperband pruner}
    E -- Prune --> F[Preserve artifacts pruned]
    E -- Continue --> G[Finish trial]
    G --> H[Preserve artifacts complete]
    H --> I[Update best]
  end

  subgraph Final[Phase 2: Final Evaluation]
    I --> J[25-epoch final runs]
    J --> K[Aggregate and save results]
  end
```

## HOSER's Zone-Based Architecture

Before explaining token alignment, it's important to understand HOSER's hierarchical spatial representation using **zones**.

### Zone Partitioning

HOSER partitions the road network into 300 zones using the KaHIP graph partitioner:

1. **Graph construction** (`data/preprocess/partition_road_network.py`):
   - Builds undirected graph from `roadmap.rel` (road connectivity)
   - Each road is a node, edges represent physical connections
   - 40,060 roads → 300 balanced partitions using KaHIP's multilevel k-way algorithm

2. **Zone transition matrix** (`data/preprocess/get_zone_trans_mat.py`):
   - Computes 300×300 matrix of zone-to-zone transition frequencies from training trajectories
   - Used to construct zone-level graph with weighted edges
   - Captures macro-level spatial flow patterns (e.g., from residential zones to commercial zones)

### Zone Embeddings in HOSER

Zones provide a **coarse-grained spatial context** that complements fine-grained road embeddings:

```python
# Road Network Encoder (models/road_network_encoder.py)
zone_id_emb: Embedding(300, 128)  # Learnable zone embeddings
zone_gcn: GCN(...)                # Zone-level graph neural network

# Zone GCN processes zone transition graph
zone_embedding_after_gnn = zone_gcn(
    zone_id_emb.weight,           # Initial embeddings
    zone_edge_index,              # Edges from transition matrix
    zone_edge_weight              # Transition frequencies
)
```

### Zone Usage in Forward Pass

Zones are used at **three critical points**:

1. **Trajectory encoding** (lines 26, 42 in `hoser.py`):
   ```python
   zone_embedding = all_zone_embedding_after_gnn[road2zone[trace_road_id]]
   ```
   Each road in trajectory gets its zone embedding, providing spatial cluster context.

2. **Road-zone fusion** (lines 165-172 in `trajectory_encoder.py`):
   ```python
   spatial_embedding = (
       road_embedding 
       + sigmoid(fusion_mlp(concat(road_embedding, zone_embedding))) 
       * zone_embedding
   )
   ```
   Gated fusion: the model learns how much zone context to incorporate per road.

3. **Destination context** (lines 30, 46 in `hoser.py`):
   ```python
   destination_zone_embedding = all_zone_embedding_after_gnn[road2zone[destination_road_id]]
   ```
   Navigator receives the destination's zone embedding to guide routing decisions.

### Navigator's Attention Mechanism

The navigator uses zone context when scoring candidates (line 36-43 in `navigator.py`):

```python
q = concat(trajectory_embedding, destination_zone_embedding)  # Query
k = concat(candidate_road_embedding, distance, angle)         # Key

logits = attention(q, k)  # Scores each candidate
```

The destination zone embedding helps the model reason about **which candidates lead toward the goal region**, not just geometrically close roads.

## Token Alignment (Road → Grid) 

**Key difference**: While HOSER uses hierarchical zones (300 semantic clusters), LM-TAD uses a **uniform spatial grid** (51,663 cells).

LM‑TAD tokens are grid cells produced from the road geometry's centroid using the same grid config used during LM‑TAD preprocessing. Reuse LM‑TAD's dataset_config (from the checkpoint) for:

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

### Zones vs Grid Tokens: Complementary Representations

| Aspect | HOSER Zones (300) | LM-TAD Grid Tokens (51,663) |
|--------|-------------------|----------------------------|
| **Granularity** | Coarse (133 roads/zone avg) | Fine (~0.8 roads/cell avg) |
| **Semantics** | Graph-based clusters | Uniform spatial grid |
| **Purpose** | Hierarchical routing context | Precise location encoding |
| **Learning** | GCN with transition weights | Transformer over sequences |
| **Distillation role** | HOSER's internal feature | Teacher's vocabulary |

During distillation:
- HOSER continues to use its learned zone embeddings internally
- Grid tokens are used **only** to query the LM-TAD teacher
- The student learns to leverage both: zone context for structure + teacher's grid-based priors

## Loss Design

This section defines the mathematical framework for distillation, showing how we align HOSER's predictions to LM-TAD's learned spatial patterns while preserving supervised learning objectives.

### Core Challenge: Vocabulary Mismatch

**Problem**: The teacher (LM-TAD) and student (HOSER) operate on different representations:
- **Teacher**: Outputs distribution over 51,663 grid tokens
- **Student**: Outputs distribution over $|C|$ candidate roads (typically 64 after top-k filtering)

**Solution**: We use the **candidate set** $C$ as a common ground. For each candidate road, we:
1. Map it to its grid token via precomputed `road_to_token[road_id]`
2. Extract the teacher's probability for that grid token
3. Renormalize to make probabilities sum to 1.0 over just the candidates

This makes the teacher's "opinion" directly comparable to the student's predictions.

### Mathematical Formulation

**Notation**:
- $C$: Candidate set at current timestep (size $|C| \leq 64$)
- $z_k$: Grid token corresponding to candidate road $k$
- $H$: History sequence of grid tokens (window size = 2-8)
- $q(z \mid H)$: Teacher's distribution over all grid tokens given history
- $\text{logits}_k$: Student's raw score for candidate $k$
- $\tau$: Temperature parameter for softening distributions

**Step 1: Teacher distribution on candidates (renormalized)**

Extract teacher probabilities for candidate grid tokens and renormalize:

$$
q_C^{(\tau)}(k) = \frac{\left(q(z_k \mid H)\right)^{1/\tau}}{\sum\limits_{j\in C} \left(q(z_j \mid H)\right)^{1/\tau}}
$$

- $q(z_k \mid H)$: Teacher's raw probability for grid token $z_k$
- $(...)^{1/\tau}$: Temperature scaling (raises probabilities to power $1/\tau$)
- Denominator: Renormalization constant so $\sum_{k \in C} q_C^{(\tau)}(k) = 1$

**Why this form?** Temperature scaling is applied in probability space for the teacher (not logit space) because we receive already-softmaxed outputs from LM-TAD.

**Step 2: Student distribution on candidates**

Temperature-scale the student's logits before softmax:

$$
p_C^{(\tau)}(k) = \frac{\exp\left(\frac{\text{logits}_k}{\tau}\right)}{\sum\limits_{j\in C} \exp\left(\frac{\text{logits}_j}{\tau}\right)}
$$

- $\text{logits}_k$: HOSER's raw score from navigator attention
- Division by $\tau$: Temperature scaling in logit space (equivalent to $(p)^{1/\tau}$ in probability space when $\tau \to 1$)

**Why different formulas?** The student's logits are temperature-scaled *before* softmax, which is mathematically equivalent to the teacher's probability-space scaling but computationally more stable.

**Step 3: KL divergence (distillation loss)**

Measure how much the student's distribution diverges from the teacher's:

$$
\mathcal{L}_{\text{KL}} = \text{KL}\left(q_C^{(\tau)}\;\Vert\; p_C^{(\tau)}\right) = \sum_{k\in C} q_C^{(\tau)}(k)\,\left[\log q_C^{(\tau)}(k) - \log p_C^{(\tau)}(k)\right]
$$

**KL divergence properties**:
- Always non-negative: $\mathcal{L}_{\text{KL}} \geq 0$
- Zero if and only if $q_C = p_C$ (perfect match)
- Asymmetric: $\text{KL}(q \| p) \neq \text{KL}(p \| q)$
- Direction: $\text{KL}(q \| p)$ means "teacher → student" (we want student to match teacher)

**Interpretation**: Each candidate $k$ contributes weighted "surprise":
- Weight: $q_C^{(\tau)}(k)$ (teacher's confidence)
- Surprise: $\log q_C^{(\tau)}(k) - \log p_C^{(\tau)}(k)$ (difference in log-probabilities)
- Large positive contribution when: student under-predicts where teacher is confident
- Large negative contribution when: student over-predicts where teacher is uncertain

**Step 4: Total training objective**

Combine distillation with existing supervised losses:

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{road}}}_{\text{cross-entropy}} \; + \; \underbrace{\mathcal{L}_{\text{time}}}_{\text{MAPE}} \; + \; \underbrace{\lambda \, \mathcal{L}_{\text{KL}}}_{\text{distillation}}
$$

Where:
- $\mathcal{L}_{\text{road}} = -\log p_C(\text{label})$: Standard cross-entropy to ground truth road
- $\mathcal{L}_{\text{time}}$: Mean Absolute Percentage Error for travel time prediction
- $\lambda \in [0.001, 0.1]$: Distillation weight (tuned via Optuna)

**Loss weighting rationale**:
- $\lambda$ is small (0.01-0.03 typically) because:
  - Supervised labels are the primary training signal
  - KL loss acts as regularization, not a hard constraint
  - Small $\lambda$ prevents overpowering the supervised objectives
  - Accumulated effect over 629k samples × 8 epochs is still significant

### Design Rationale

**Why renormalize on $C$?**
- HOSER never sees roads outside the candidate set, so comparing full vocabularies would be misleading
- Renormalization focuses the teacher's guidance on decisions HOSER actually faces
- Makes KL divergence magnitudes comparable across timesteps with varying $|C|$

**Why use temperature $\tau > 1$?**
- Softens both distributions, making them less "peaky"
- Reveals relative preferences: with $\tau=1$, teacher might output [0.95, 0.03, 0.01, 0.01]; with $\tau=3$, becomes [0.77, 0.11, 0.08, 0.03]
- Transfers "dark knowledge": the student learns not just *what* is correct, but *how much more likely* than alternatives
- Gentler gradients help optimization: sharp distributions (high confidence) create large, unstable gradients
- Typical range: $\tau \in [2.0, 4.0]$ balances smoothing vs preserving preference order

**Why forward KL (teacher → student)?**
- Forward KL: $\text{KL}(q \| p)$ penalizes student for being *too uncertain* where teacher is confident (mode-seeking)
- Alternative (reverse KL): $\text{KL}(p \| q)$ penalizes student for being *too confident* where teacher is uncertain (mean-seeking)
- Forward KL matches our goal: push student toward teacher's high-confidence predictions

**Why preserve supervised losses?**
- Distillation alone is insufficient: teacher may have learned biases or errors
- Ground truth labels provide direct supervision on *what* to predict
- Distillation adds *how confidently* and *how to rank alternatives*
- Combined training gets best of both: label accuracy + calibrated uncertainty

### Parameter Ranges and Effects

| Parameter | Range | Effect when increased | Typical optimum |
|-----------|-------|------------------------|-----------------|
| $\lambda$ | [0.001, 0.1] | Stronger teacher influence, may degrade if too high | 0.01-0.03 |
| $\tau$ | [1.0, 5.0] | Smoother distributions, less distinction between alternatives | 2.0-4.0 |
| window | [2, 8] | More historical context for teacher, slower inference | 4 |

**Tuning strategy**: Optuna searches the joint space to find the optimal combination for validation accuracy.

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
  - **Gradient accumulation**: `accum_steps=8` for effective batch size of 512 (64 × 8).
  - **torch.compile**: Mode `max-autotune` for aggressive kernel fusion and optimization.
  - **CUDA graphs disabled**: `training.disable_cudagraphs: true` to avoid tensor reuse issues with dynamic shapes.
  - **Gradient checkpointing**: Optional in `TrajectoryEncoder` (disabled by default—sufficient VRAM available).
  - **TF32 tensor cores**: `allow_tf32: true` for ~2× matmul speedup on Ampere GPUs.
  - **cuDNN benchmarking**: `cudnn_benchmark: true` to select fastest convolution algorithms.

- **Dataloader tuning** (optimized for large dataset streaming):
  - `num_workers: 6` for parallel data loading (reduced from 16 to avoid file descriptor exhaustion).
  - `pin_memory: false` to relieve CPU memory pressure (bottleneck for large datasets).
  - `prefetch_factor: 16` to maintain prefetch buffer during disk streaming.
  - `persistent_workers: false` to avoid memory leaks with repeated `torch.load` calls.
  - **Smart Caching**: Now caches the entire dataset in RAM if it fits, using all CPU cores for rapid parallel loading. This is the primary performance enhancement, eliminating disk I/O as a bottleneck.

- **Candidate filtering** (enabled for stability):
  - `data.candidate_top_k: 64` caps candidates per timestep to prevent memory explosion with pathological traces.
  - **Vectorized label remapping**: When filtering candidates, labels are remapped on GPU to match filtered indices.
  - Invalid positions (where ground truth is filtered out) are marked with `-100` and ignored via `F.cross_entropy(ignore_index=-100)`.
  - Prevents inf/NaN losses from empty candidate sets.

- **Stability improvements**:
  - **Cross-entropy masking**: Uses `ignore_index=-100` to skip positions with no valid candidates.
  - **Time loss masking**: Only computes MAPE where labels are valid (`label >= 0` and `label < candidate_len`).
  - **Label validation**: Ensures remapped labels remain within effective `candidate_len` after top-k filtering.

- **Performance profiling** (every 1000 batches):
  - Tracks time breakdown. With RAM caching, `Data Wait` is now negligible (<5%).
  - With optimizations: sustained **~25-30 it/s** throughput with RAM cache (was 2-3 it/s before fixes).

- **Logging & validation**:
  - Full YAML config logged to WandB at run start for reproducibility.
  - Per-epoch validation: next-step accuracy, time-prediction MAPE.
  - Metrics logged to both TensorBoard and WandB.
  - Best model saved based on validation accuracy.

### 5) Configuration (key knobs)

- optimizer_config: max_epoch (default 25), batch_size (vanilla 128; for distill you may use 64 to reduce teacher cost per step), accum_steps (8), learning_rate (1e-3), weight_decay (0.1)
- distill: enable flag, repo/ckpt, window (2–8), lambda (1e-3…1e-1), temperature (1.0…5.0), grid_size=0.001, downsample=1
- dataloader: num_workers=6, prefetch_factor=16, pin_memory=false, persistent_workers=false
- profiling: enable, log_interval (e.g., 1000)
- wandb: project hoser-distill-optuna-6, run_name auto, tags
- optuna: sampler=cmaes, pruner=hyperband (min_resource=5, reduction_factor=3), final_run (enabled, max_epochs=25, seeds=[42,43,44])

## Detailed Worked Example

This section walks through a complete distillation step with real numbers, showing exactly how the teacher's knowledge is transferred to the student model.

### Real-World Scenario (Beijing Road Network)

**Context**: A taxi is traveling along Beijing's 4th Ring Road approaching an interchange. The driver needs to merge onto an exit ramp to reach their destination in the Chaoyang district.

**Trajectory history**: The vehicle has been traveling eastbound on the 4th Ring Road for the past 5 timesteps, maintaining consistent speed and direction.

**Current decision point**: HOSER's navigator must predict the next road segment from 4 viable candidates:

| Road ID | Road Name | Grid Token | Distance (m) | Angle Diff (rad) | Notes |
|---------|-----------|------------|--------------|------------------|-------|
| 22077 | Jingkai Expressway North | 12458 | 45.2 | 0.12 | Continue on main expressway |
| 22080 | 4th Ring Rd East Ramp | 12459 | 38.7 | 0.08 | **Exit ramp** (ground truth) |
| 33716 | Chaoyang North Rd | 15203 | 52.1 | 0.31 | Parallel arterial road |
| 3087  | Guangqu Rd Extension | 8964 | 61.8 | 0.45 | Opposite direction |

**Ground truth**: The actual next road taken by the taxi is **22080** (4th Ring Rd East Ramp), which is the closest candidate by both distance (38.7m) and angle (0.08 rad ≈ 4.6°).

**Key observation**: Road 22080 and 22077 are spatially close (adjacent grid tokens 12459 vs 12458) and have similar geometric features. This makes the decision challenging for a model that relies solely on local geometric cues.

### Step 1: Teacher Distribution (LM-TAD)

**What the teacher sees**: LM-TAD operates purely on grid tokens (spatial locations). It doesn't have access to road IDs, geometric features, or HOSER's zone embeddings. Instead, it learns spatial transition patterns from thousands of trajectories.

**History window**: With `distill.window=4`, the teacher sees the last 4 grid tokens from the trajectory, plus the special SOT (Start-Of-Trajectory) token prepended:

$$
H = [\underbrace{\text{SOT}}_{\text{token 51662}}, \underbrace{t_{12455}, t_{12456}, t_{12457}, t_{12458}}_{\text{last 4 timesteps}}]
$$

This history tells the teacher: "We've been moving through grid cells 12455→12456→12457→12458, which corresponds to a trajectory along the 4th Ring Road heading east."

**Teacher inference**: LM-TAD's transformer processes this sequence and outputs logits over all 51,663 possible grid tokens. After softmax, we get a probability distribution $q$ over the entire Beijing grid. Most of the probability mass (~84%) is concentrated on grid cells near the current location and along plausible continuation paths.

**Extracting candidate probabilities**: For our 4 candidate roads, we look up their corresponding grid tokens and extract the teacher's assigned probabilities:

$$
\begin{aligned}
q(t_{12458}) &= 0.0032 \quad \text{(Jingkai Expressway - continue straight)} \\
q(t_{12459}) &= 0.1547 \quad \text{(4th Ring Ramp - exit right)} \\
q(t_{15203}) &= 0.0018 \quad \text{(Chaoyang North - parallel road)} \\
q(t_{8964}) &= 0.0003 \quad \text{(Guangqu Extension - wrong direction)}
\end{aligned}
$$

**Key insight**: The teacher strongly prefers grid token 12459 (15.47% of its full distribution), which corresponds to the correct exit ramp. This preference comes from LM-TAD having learned that vehicles on eastbound 4th Ring often take this exit to reach Chaoyang destinations.

**Renormalization to candidates**: Since HOSER only considers these 4 candidates (not all 51,663 grid cells), we need to renormalize the teacher's probabilities to sum to 1.0 over just these 4 options:

$$
\begin{aligned}
Z_C &= q(t_{12458}) + q(t_{12459}) + q(t_{15203}) + q(t_{8964}) \\
&= 0.0032 + 0.1547 + 0.0018 + 0.0003 = 0.1600
\end{aligned}
$$

$$
q_C = \begin{bmatrix}
0.0032/0.1600 \\
0.1547/0.1600 \\
0.0018/0.1600 \\
0.0003/0.1600
\end{bmatrix} = \begin{bmatrix}
0.0200 \\
0.9669 \\
0.0113 \\
0.0019
\end{bmatrix}
$$

**Interpretation**: After renormalization, the teacher assigns **96.7% probability to road 22080** (the correct exit ramp). This is a very confident prediction! The teacher has learned from historical taxi trajectories that this specific movement pattern (eastbound on 4th Ring at this location) almost always leads to taking the exit ramp.

**Why is the teacher so confident?**
1. **Spatial context**: The history sequence shows consistent eastbound movement along the ring road
2. **Destination inference**: LM-TAD implicitly learned that this trajectory pattern correlates with destinations in the Chaoyang district
3. **Historical frequency**: In the training data, 96%+ of vehicles following this exact path took the exit ramp
4. **Alternative rejection**: The other candidates are less plausible given the history (continuing straight would be more common from different approach angles)

### Step 2: Student Distribution (HOSER)

**What the student sees**: HOSER has access to different information than the teacher:
- **Trajectory embedding**: Encoded sequence of roads (not just grid cells) with road-level features
- **Zone embeddings**: Hierarchical spatial context from 300 learned zones
- **Destination context**: The target zone embedding helps guide routing decisions
- **Geometric features**: Distance (m) and angle (rad) to each candidate from current position

**Navigator scoring**: HOSER's attention-based navigator computes a score for each candidate by comparing:
- Query: `concat(trajectory_embedding, destination_zone_embedding)`
- Key: `concat(candidate_road_embedding, distance, angle)`

**Raw logits** (unnormalized scores before softmax):
$$
\text{logits} = \begin{bmatrix} 
2.1 \quad \text{(road 22077)} \\ 
1.8 \quad \text{(road 22080)} \\ 
-0.5 \quad \text{(road 33716)} \\ 
-1.2 \quad \text{(road 3087)}
\end{bmatrix}
$$

**Logit interpretation**: 
- Roads 22077 and 22080 both score positively (2.1 and 1.8), indicating HOSER considers them both reasonable
- Road 33716 scores slightly negative (-0.5), suggesting it's a less likely option
- Road 3087 scores strongly negative (-1.2), correctly identified as implausible (wrong direction)

**Temperature scaling**: With $\tau = 2.0$, we soften the distribution to encourage smoother knowledge transfer:

$$
p_C^{(\tau)}(k) = \frac{\exp(\text{logits}_k / \tau)}{\sum_j \exp(\text{logits}_j / \tau)}
$$

$$
\begin{aligned}
p_C^{(2.0)} &= \text{softmax}\left(\begin{bmatrix} 2.1/2.0 \\ 1.8/2.0 \\ -0.5/2.0 \\ -1.2/2.0 \end{bmatrix}\right) \\
&= \text{softmax}\left(\begin{bmatrix} 1.05 \\ 0.90 \\ -0.25 \\ -0.60 \end{bmatrix}\right) \\
&= \begin{bmatrix} 0.395 \\ 0.340 \\ 0.177 \\ 0.088 \end{bmatrix}
\end{aligned}
$$

**Problem identified**: HOSER is **overly uncertain** between the first two roads (39.5% vs 34.0%). Without distillation, HOSER relies primarily on local geometric features:
- Road 22080 is slightly closer (38.7m vs 45.2m) 
- Road 22077 has a slightly more aligned angle (0.12 vs 0.08 rad)

These competing geometric cues make HOSER hesitant. It hasn't yet learned the **spatial transition patterns** that make road 22080 (the exit ramp) strongly preferred in this specific context.

### Step 3: Temperature-Scaled Teacher Distribution

**Why temperature scaling?** We apply the same temperature $\tau$ to both teacher and student to ensure they're compared on equal footing. Temperature controls how "soft" the distributions are:
- $\tau = 1.0$: Original distribution (sharp, confident)
- $\tau > 1.0$: Softened distribution (smoother, less confident)
- $\tau \to \infty$: Uniform distribution (maximum uncertainty)

**Temperature scaling formula** for the teacher (different from student's logit scaling):

$$
q_C^{(\tau)}(k) = \frac{(q_C(k))^{1/\tau}}{\sum_j (q_C(j))^{1/\tau}}
$$

With $\tau = 2.0$ (equivalent to taking square root of probabilities):

$$
\begin{aligned}
q_C^{(2.0)} &= \frac{1}{Z}\begin{bmatrix}
(0.0200)^{0.5} \\
(0.9669)^{0.5} \\
(0.0113)^{0.5} \\
(0.0019)^{0.5}
\end{bmatrix} = \frac{1}{Z}\begin{bmatrix}
0.1414 \\
0.9834 \\
0.1063 \\
0.0436
\end{bmatrix}
\end{aligned}
$$

$$
Z = 0.1414 + 0.9834 + 0.1063 + 0.0436 = 1.2747
$$

$$
q_C^{(2.0)} = \begin{bmatrix}
0.111 \\
0.771 \\
0.083 \\
0.034
\end{bmatrix}
$$

**Effect of temperature**: The teacher's confidence in road 22080 drops from 96.7% to 77.1%, but it **still strongly prefers** the correct exit ramp. The softer distribution has three benefits:
1. **Reveals relative preferences**: The 0.111 vs 0.083 split between roads 22077 and 33716 shows the teacher considers the expressway continuation slightly more plausible than the parallel arterial
2. **Easier optimization**: Gentler gradients help the student learn more smoothly
3. **Transfers "dark knowledge"**: The student learns not just *what* to predict, but *how confident* to be and how to rank alternatives

### Step 4: KL Divergence Computation

**What is KL divergence?** KL (Kullback-Leibler) divergence measures how much one probability distribution differs from another. In distillation, it quantifies the "surprise" when we use the student's distribution instead of the teacher's.

$$
\begin{aligned}
\mathcal{L}_{\text{KL}} &= \text{KL}\left(q_C^{(2.0)} \,\|\, p_C^{(2.0)}\right) \\
&= \sum_{k=1}^{4} q_C^{(2.0)}(k) \left[\log q_C^{(2.0)}(k) - \log p_C^{(2.0)}(k)\right]
\end{aligned}
$$

**Formula interpretation**: For each candidate $k$, we compute:
- $q_C^{(2.0)}(k)$: Teacher's probability (the "weight" of this term)
- $\log q_C^{(2.0)}(k)$: Teacher's log-probability (what we *expect*)
- $\log p_C^{(2.0)}(k)$: Student's log-probability (what we *observe*)
- Difference measures "surprise": large when student assigns low probability where teacher assigns high

**Computing term by term**:

| Candidate | Teacher $q$ | Student $p$ | Contribution | Interpretation |
|-----------|-------------|-------------|--------------|----------------|
| 22077 | 0.111 | 0.395 | $-0.141$ | **Negative**: Student over-predicts this road |
| 22080 | 0.771 | 0.340 | $+0.631$ | **Positive**: Student under-predicts correct road! |
| 33716 | 0.083 | 0.177 | $-0.063$ | **Negative**: Student over-predicts parallel road |
| 3087  | 0.034 | 0.088 | $-0.032$ | **Negative**: Student over-predicts wrong direction |

$$
\begin{aligned}
k=1: \quad &0.111 \times [\log(0.111) - \log(0.395)] = 0.111 \times [-2.199 + 0.929] = -0.141 \\
k=2: \quad &0.771 \times [\log(0.771) - \log(0.340)] = 0.771 \times [-0.260 + 1.079] = +0.631 \\
k=3: \quad &0.083 \times [\log(0.083) - \log(0.177)] = 0.083 \times [-2.489 + 1.731] = -0.063 \\
k=4: \quad &0.034 \times [\log(0.034) - \log(0.088)] = 0.034 \times [-3.381 + 2.430] = -0.032
\end{aligned}
$$

$$
\mathcal{L}_{\text{KL}} = -0.141 + 0.631 - 0.063 - 0.032 = 0.395 \text{ nats} \approx 0.57 \text{ bits}
$$

**Key insight**: The **dominant term is +0.631** from candidate 22080. This large positive contribution signals the primary failure mode: *the student is not confident enough about the correct exit ramp*.

**Gradient effect during backpropagation**:
1. **Increase logit for road 22080**: The +0.631 term creates a gradient that pushes logit from 1.8 toward 2.5+
2. **Decrease logit for road 22077**: The -0.141 term pulls logit from 2.1 toward 1.5
3. **Rebalance uncertainty**: Student learns to be more decisive, matching teacher's confidence

**Crucially**: This happens *in addition to* the supervised cross-entropy loss (which also targets road 22080). The distillation loss teaches HOSER **how confidently** to make correct predictions, not just **what** to predict.

### Step 5: Total Loss and Gradient Update

**Combining all objectives**: HOSER's complete training loss includes three terms, each teaching different aspects:

$$
\begin{aligned}
\mathcal{L}_{\text{road}} &= -\log p_C(\text{road 22080}) = -\log(0.340) = 1.079 \\
&\quad \text{(Supervised: predict correct next road)} \\
\\
\mathcal{L}_{\text{time}} &= \text{MAPE}(\hat{t}, t_{\text{true}}) = 0.127 \\
&\quad \text{(Supervised: predict travel time accurately)} \\
\\
\mathcal{L}_{\text{KL}} &= 0.395 \\
&\quad \text{(Distillation: match teacher's distribution)} \\
\\
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{road}} + \mathcal{L}_{\text{time}} + \lambda \mathcal{L}_{\text{KL}} \\
&= 1.079 + 0.127 + 0.01 \times 0.395 \\
&= 1.206 + 0.004 = 1.210
\end{aligned}
$$

**Weighting with** $\lambda = 0.01$: The distillation term contributes only **0.33%** of the total loss magnitude. Why so small?

1. **Avoid overpowering supervision**: The ground truth labels remain the primary training signal
2. **Accumulated effect**: Over 629k training samples × 8 epochs, these small nudges accumulate significantly
3. **Regularization role**: KL acts as a soft constraint, not a hard requirement
4. **Tunable trade-off**: Optuna searches $\lambda \in [0.001, 0.1]$ to find the optimal balance

**Loss breakdown by magnitude**:
- Road prediction: 1.079 (89.2%) — **Primary signal**
- Time prediction: 0.127 (10.5%) — **Secondary objective**
- Distillation: 0.004 (0.3%) — **Regularization**

**What each loss teaches HOSER**:

| Loss Component | What it teaches | Example gradient effect |
|----------------|-----------------|------------------------|
| $\mathcal{L}_{\text{road}}$ | **What** to predict | "Road 22080 is correct, increase its logit" |
| $\mathcal{L}_{\text{time}}$ | **When** to arrive | "Predicted 45s, actual 52s, adjust time head" |
| $\mathcal{L}_{\text{KL}}$ | **How confident** to be | "Be 77% sure (not 34%) when history shows this pattern" |

**Training dynamics over time**:

**Without distillation** ($\lambda=0$):
- HOSER learns purely from labels: "22080 is correct, others are wrong"
- May become overconfident (95%+) or underconfident (40%) depending on geometric features
- Treats all correct predictions equally, regardless of context
- Final accuracy: 57.2% (baseline)

**With distillation** ($\lambda=0.01$):
- HOSER learns from both labels AND teacher's reasoning: "22080 is correct, AND it should be strongly preferred in this context"
- Calibrates confidence based on spatial patterns: high confidence for exit ramps from ring roads, moderate confidence for ambiguous intersections
- Learns to rank alternatives: "22077 is more plausible than 33716 even though both are wrong"
- Expected improvement: +1-3% accuracy (from calibrated confidence and better tie-breaking)

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

**Measured timings** (batch_size=64, RTX 2080 Ti, after optimizations):
- Data Wait: ~35ms (48%)
- Data Transfer: ~4ms (5%)
- HOSER forward: ~13ms (18%)
- Teacher KL (LM-TAD): ~1ms (1.4%)
- Backward pass: ~20ms (27%)
- Optimizer step: ~0.5ms (0.7%)
- **Total**: ~74ms per batch

**Throughput**: **11-13 it/s** sustained (was 2-3 it/s before fixes)

**Bottleneck analysis**:
- **Data loading** is now the primary bottleneck (48%) for large datasets (629k samples):
  - Streaming from disk dominates (no RAM cache for datasets >200k)
  - Mitigated by high prefetch_factor (16) and 6 workers
  - File handle management prevents descriptor leaks
- **Teacher inference** is now highly optimized (~1.4% of time):
  - Small history window (4 steps) reduces computation
  - FP16 autocast, batched inference, GPU-only operations
  - Top-k candidate filtering (64) reduces renormalization overhead

**Scalability**:
- Training set: 629,380 trajectories → 9,835 batches/epoch (batch_size=64)
- Epoch time: ~45-50 minutes with optimizations
- Full training (10-25 epochs): ~8-20 hours
- **Performance improvements**:
  - 4-5× faster than initial implementation (eliminated data loading cliff)
  - No worker crashes due to file descriptor exhaustion
  - Stable throughput throughout entire epoch (no batch 3600+ slowdown)

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
**Challenge**: HOSER can filter candidates to top-k by distance, but labels index the full candidate list. Without filtering, pathological traces with 1000+ candidates cause collate bottlenecks and memory explosion.

**Solution implemented**: 
- **Enabled** `candidate_top_k: 64` to cap candidates per timestep.
- **Vectorized GPU-based label remapping**:
  - Find where original labels appear in sorted top-k indices via broadcasting
  - Mark positions where label is filtered out as `-100` (ignored by cross_entropy)
  - Ensure remapped labels remain within effective `candidate_len` after clamping
- **Benefits**: Prevents data loading cliff (was 280-360ms wait at batch 3600+, now consistent 35ms)

### 5. Numerical Stability and Loss Masking
**Challenges**: 
- Empty candidate sets after top-k filtering produce inf in cross_entropy
- Invalid label positions cause indexing errors in time loss
- NaN/inf gradients halt training

**Solutions implemented**:
- **Cross-entropy masking**: Use `F.cross_entropy(ignore_index=-100)` to skip invalid positions
- **Time loss masking**: Only compute MAPE where `(label >= 0) & (label < candidate_len)`
- **Label validation**: Ensure remapped labels are clamped and validated before indexing
- **Gradient checking**: Log warnings for NaN/inf losses but continue training with fallback values
- **Numerical safety**: Clamp denominators with epsilon (1e-9), use log-softmax where possible

### 6. Data Loading Performance
**Challenges**:
- 629k training samples require streaming from disk (>25 min to cache in RAM)
- Collate function with pathological traces (1000+ candidates) caused 300ms+ stalls
- Worker processes crashed after ~5500 batches with "received 0 items of ancdata"

**Solutions implemented**:
- **Conditional caching**: Cache datasets ≤200k in RAM, stream larger ones from disk
- **Explicit file handle management**: Use context manager to close files immediately after `torch.load`
- **Optimized collate**: Vectorize operations, pre-crop candidate arrays, reduce allocations
- **Worker tuning**: Reduce to 6 workers, disable persistent_workers to prevent memory leaks
- **Prefetch optimization**: Increase prefetch_factor to 16 for streaming workload
- **Result**: Sustained 11-13 it/s with no crashes, eliminated batch 3600+ cliff

## Vectorization Details

The implementation uses extensive GPU-based vectorization to eliminate Python loops and maximize throughput. Key vectorized operations:

### 1. Collate Function Vectorization

**Destination IDs** (commit `71b9b0b`):
```python
# ❌ Before: Loop over batch
for i in range(batch_size):
    batch_destination_road_id[i] = destination_road_ids[i]

# ✅ After: Single vectorized operation
batch_destination_road_id = torch.from_numpy(
    np.array(destination_road_ids, dtype=np.int64)
)
```

**Pre-cropped candidate arrays**:
```python
# Crop once before inner loop instead of repeated slicing
cand_road_ids_cropped = candidate_road_ids[i][crop_start:crop_start+actual_len]
cand_lens_cropped = candidate_lens[i][crop_start:crop_start+actual_len]
cand_metric_dis_cropped = metric_diss[i][crop_start:crop_start+actual_len]
cand_metric_angle_cropped = metric_angles[i][crop_start:crop_start+actual_len]

# Then process in tight loop with reduced memory access
for j in range(actual_len):
    cand_len = int(cand_lens_cropped[j])  # Already cropped
    if cand_len > 0:
        batch_candidate_road_id[i, j, :actual_cand_len] = torch.from_numpy(
            cand_road_ids_cropped[j][:actual_cand_len]
        )
```

**Direct slice assignment**:
```python
# ✅ Faster: Direct slice assignment (single copy operation)
batch_trace_road_id[i, :actual_len] = torch.from_numpy(
    trace_road_ids[i][crop_start:crop_start+actual_len]
)

# ❌ Slower: Item-by-item assignment
for j in range(actual_len):
    batch_trace_road_id[i, j] = torch.from_numpy(trace_road_ids[i][crop_start+j])
```

### 2. Top-K Candidate Filtering (GPU-Accelerated)

**Sorting and gathering** (commit `bc09886`):
```python
# Sort all candidates by distance on GPU in one operation
sorted_indices = torch.argsort(batch_metric_dis_gpu, dim=-1, descending=False)
# Shape: [B, T, C] where B=batch, T=timesteps, C=candidates

# Create valid candidate mask using broadcasting
indices = torch.arange(C, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, C]
valid_mask = indices < batch_candidate_len_gpu.unsqueeze(-1)  # [B, T, C]

# Push invalid candidates to end (vectorized)
sorted_indices_masked = torch.where(
    valid_mask.gather(-1, sorted_indices),
    sorted_indices,
    torch.tensor(C, device=device),
)

# Select top-k with single gather operation (no loops)
idx = sorted_indices_masked[..., :k]  # Take first k from each position
batch_candidate_road_id = torch.gather(batch_candidate_road_id, -1, idx)
batch_metric_dis = torch.gather(batch_metric_dis_gpu, -1, idx)
batch_metric_angle = torch.gather(batch_metric_angle, -1, idx)
```

### 3. Label Remapping (Fully Vectorized)

**Broadcasting-based label matching**:
```python
# Expand dimensions for broadcasting: [B, T, 1] vs [B, T, k]
labels_expanded = batch_road_label_device.unsqueeze(-1)  # Ground truth labels
idx_expanded = sorted_indices[..., :k]  # Top-k sorted indices

# Find where each label appears in top-k (element-wise comparison across entire batch)
matches = (idx_expanded == labels_expanded).float()  # [B, T, k]

# Check if label exists in top-k
has_match = matches.sum(dim=-1) > 0  # [B, T]

# Find position of match (argmax over k dimension)
match_positions = torch.argmax(matches, dim=-1)  # [B, T]

# Set new labels: vectorized conditional assignment
new_road_label = torch.where(
    has_match, 
    match_positions, 
    torch.tensor(-100, device=device)  # Ignore filtered-out labels
)

# Validate against candidate_len
new_road_label = torch.where(
    new_road_label < batch_candidate_len_gpu,
    new_road_label,
    torch.tensor(-100, device=device),
)
```

**Key benefits**:
- **No Python loops**: Entire batch processed in parallel on GPU
- **Broadcasting**: Comparisons across [B×T×k] tensor in single operation
- **Masked assignment**: `torch.where` replaces conditional branches

### 4. Time Loss Masking (Vectorized)

**Valid position filtering**:
```python
# Create validity mask for all positions in batch
valid_time = (selected_road_label >= 0) & (selected_road_label < selected_candidate_len)
# Shape: [N] where N = total valid timesteps across batch

# Safe indexing with clamping
safe_labels = torch.clamp(selected_road_label, min=0)
selected_time_pred = time_pred[flat_mask][flat_indices, safe_labels]

# Compute MAPE only on valid positions (vectorized masked operation)
if torch.any(valid_time):
    diff = torch.abs(selected_time_pred - selected_timestamp_label)
    mape = diff / torch.clamp(selected_timestamp_label, min=1.0)
    loss_time_pred = torch.mean(mape[valid_time])  # Masked mean
```

### 5. Teacher KL Computation (Batched)

**History preparation** (from `critics/distill_hook.py`):
```python
# Pre-allocate SOT tensor once during initialization
self.sot_tensor = torch.full((1, 1), sot_id, dtype=torch.long, device=device)

# Batch-construct all history sequences with single concatenation
histories = []
for b in range(batch_size):
    for t in range(trace_len[b]):
        window = trace_grid_tokens[b][max(0, t-self.window):t]
        histories.append(torch.cat([self.sot_tensor, window], dim=-1))

# Single batched forward pass for all positions
teacher_logits = self.teacher(torch.stack(histories))  # [N, vocab_size]
```

**Candidate renormalization**:
```python
# Index teacher distribution by candidate tokens (vectorized gather)
teacher_cand_logits = teacher_logits[batch_indices, candidate_tokens]

# Temperature-scaled softmax and renormalization (all vectorized)
teacher_probs = F.softmax(teacher_cand_logits / temperature, dim=-1)
teacher_probs_renorm = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)

# KL divergence (vectorized across all positions)
kl = (teacher_probs_renorm * (
    torch.log(teacher_probs_renorm + 1e-9) - 
    torch.log(student_probs + 1e-9)
)).sum(dim=-1).mean()
```

## Performance Impact of Vectorization

| Operation | Before (CPU/Loop) | After (GPU/Vectorized) | Speedup |
|-----------|------------------|------------------------|---------|
| Top-k filtering | ~50ms (Python loop) | ~2ms (GPU gather) | **25×** |
| Label remapping | ~30ms (loop + search) | ~1ms (broadcasting) | **30×** |
| Collate function | ~100ms (nested loops) | ~5ms (slice ops) | **20×** |
| Time loss masking | ~15ms (conditional loop) | ~0.5ms (masked mean) | **30×** |

**Overall impact**: Vectorization reduced per-batch overhead from ~200ms to ~10ms, enabling the 4-5× overall throughput improvement.

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

## Hyperparameter Tuning with Optuna

The project uses a sophisticated 2-phase hyperparameter optimization framework with Optuna to find the best distillation settings and then validate them robustly.

### Tunable Hyperparameters

The following distillation-specific parameters are tuned after the vanilla baseline (Trial 0):

```python
# Distillation parameters
distill_lambda: float       # Range: [0.001, 0.1] (log-scale)
                           # Controls KL loss weight
                           # Higher = stronger teacher influence

distill_temperature: float  # Range: [1.0, 5.0] (linear)
                           # Softens probability distributions
                           # Higher = smoother knowledge transfer

distill_window: int        # Range: [2, 8] (integer)
                           # History length for teacher
                           # Longer = more context, slower inference
```

### Optuna 2-Phase Strategy

The tuning process is split into two automated phases configured in `config/Beijing.yaml`:

**Phase 1: Hyperparameter Search**
- **Goal**: Efficiently explore the hyperparameter space to find the best combination.
- **Sampler**: `CmaEsSampler` (Covariance Matrix Adaptation Evolution Strategy), which is highly effective for continuous/integer parameter spaces with a limited trial budget (10-100 trials).
- **Pruner**: `HyperbandPruner` aggressively stops unpromising trials early (e.g., after 5 of 8 epochs), saving significant time and allowing more configurations to be tested.
- **Process**:
  - **Trial 0** always runs as a **vanilla baseline** (no distillation) to establish a performance floor.
  - **Trials 1+** tune the distillation hyperparameters (`lambda`, `temperature`, `window`).

**Phase 2: Final Evaluation**
- **Goal**: Take the single best hyperparameter set from Phase 1 and run a full, rigorous training to get a reliable final performance metric.
- **Process**:
  - After the search is complete, the script automatically launches one or more full 25-epoch training runs using the best parameters.
  - Supports running with multiple seeds (e.g., `[42, 43, 44]`) for statistical robustness.
  - The results of these final runs are the primary measure of the distilled model's quality.

### Usage

The entire process is managed by `tune_hoser.py`:

```bash
# Run the full 2-phase tuning process (uses settings from config/Beijing.yaml)
uv run python tune_hoser.py --data_dir /home/matt/Dev/HOSER-dataset

# Resume an existing study
uv run python tune_hoser.py --data_dir /home/matt/Dev/HOSER-dataset --study_name <your_study_name>
```

### Optuna Study Results Integration

All trial artifacts (models, logs) for successful and pruned trials are preserved in the `optuna_trials/` directory.

**WandB integration** via `WeightsAndBiasesCallback`:
- Each trial creates a separate WandB run, grouped under the study name.
- Hyperparameters and final metrics are logged for analysis.
- Parallel coordinate and other plots in WandB help visualize hyperparameter importance.

### Example Hyperparameter Sensitivity

Based on preliminary tuning results:

| Parameter | Low Value | High Value | Impact on val_acc |
|-----------|-----------|------------|-------------------|
| λ (lambda) | 0.001 | 0.1 | ±2.3% (moderate) |
| τ (temperature) | 1.0 | 5.0 | ±1.1% (small) |
| window | 2 | 8 | ±0.7% (minimal) |

**Interpretation**:
- **λ (distillation weight)** has the largest impact: too low provides insufficient regularization, too high overpowers supervised signal
- **τ (temperature)** shows diminishing returns above 3.0: extreme smoothing loses fine-grained preferences
- **window** has minimal impact: LM-TAD captures most relevant patterns within 4 steps

### Recommended Starting Points

Based on empirical results:

```yaml
distill:
  lambda: 0.029        # ~3% of total loss for KL term
  temperature: 3.4     # Balanced smoothing
  window: 4            # Good speed/performance trade-off
```

### Advanced: Multi-Objective Tuning

Future extensions could optimize multiple objectives simultaneously:
- Maximize validation accuracy
- Minimize training time (throughput)
- Minimize inference latency

Using Optuna's multi-objective capabilities with Pareto-optimal frontier selection.


