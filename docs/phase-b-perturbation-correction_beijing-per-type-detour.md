# Phase B (Perturbation Correction) Results — Beijing per-type detour

**Workspace:** `hoser-perturbed-beijing-pert-type-detour-eval/`  
**Dataset:** `Beijing_per_type_detour`  
**Generated:** 2026-01-12

This document analyzes the Phase B `perturbation_correction` outputs in the evaluation workspace above. It is written to support **faithful reproduction** of the metrics, tables, and plots.

---

## Contents

- [What Phase B measures](#what-phase-b-measures)
- [Inputs, configuration, and artifacts](#inputs-configuration-and-artifacts)
- [How to reproduce end-to-end](#how-to-reproduce-end-to-end)
- [Plots (embedded)](#plots-embedded)
- [Results tables](#results-tables)
- [Interpretation](#interpretation)
- [Outliers and failure modes](#outliers-and-failure-modes)
- [Notes and caveats](#notes-and-caveats)

---

## What Phase B measures

Phase B evaluates whether a model’s generated trajectory is **closer to the clean (ground-truth) trajectory** than to the perturbed (dirty) trajectory.

For each abnormal sample:

- Let `pred` be the model-generated road-id sequence.
- Let `clean` be the clean reference road-id sequence.
- Let `dirty` be the perturbed reference road-id sequence.
- Compute DTW distances in kilometers:
  - `dtw_to_clean_km = DTW(pred, clean)`
  - `dtw_to_dirty_km = DTW(pred, dirty)`
- A sample is marked **corrected** iff:

  `dtw_to_clean_km < dtw_to_dirty_km` (strict inequality)

The primary metric is:

- **RSR (correction rate)** = `corrected / valid`.

Validity rules (from `tools/perturbation_correction.py`):

- DTW uses `fastdtw` with `haversine` distance over per-road **centroid GPS** points.
- If either DTW is non-finite for a sample, that sample is counted as `invalid` and does not contribute to `valid`.

---

## Inputs, configuration, and artifacts

### Configuration used for this eval workspace

The Phase B configuration is stored in:

- `hoser-perturbed-beijing-pert-type-detour-eval/config/evaluation.yaml`

Key Phase B fields in that file (verbatim):

```yaml
# Phase B: correction/triangulation
perturbation_source_csv: /local/data/mka299/hoser/data/_per_type/Beijing_per_type_detour/train.csv
perturbation_od_source: train
perturbation_max_entries: 1000
perturbation_seed: 0
perturbation_use_astar: true

# Optional LM-TAD teacher scoring
perturbation_lmtad_checkpoint: /home/mka299/LMTAD/code/results/LMTAD/beijing_hoser_reference/run_20250928_202718/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt
perturbation_lmtad_repo: /home/mka299/LMTAD/
perturbation_lmtad_batch_size: 128
```

Important disambiguation:

- `perturbation_seed` is the Phase B **sampling/generation seed** (here: `0`).
- The model names include `seed42/43/44`, which refer to the **training seed / checkpoint variant**, not the Phase B sampling seed.

### Data contract and how “clean” is recovered

Phase B expects a perturbation CSV (here: `train.csv`) containing abnormal rows with:

- `rid_list`: the **dirty** road-id list
- `abnormality_info`: a Python-literal dict string; for abnormal rows it contains a `real` field holding the **clean** road-id list

See `tools/perturbation_correction.py` for the precise contract.

### Phase B outputs (on disk)

For each model, Phase B writes:

- `hoser-perturbed-beijing-pert-type-detour-eval/perturbation_correction/<model>/summary.json`
- `hoser-perturbed-beijing-pert-type-detour-eval/perturbation_correction/<model>/rows.jsonl`

Additionally, Phase B generates (or reuses) trajectories at:

- `hoser-perturbed-beijing-pert-type-detour-eval/gene_perturbation/<dataset>/seed<perturbation_seed>/perturb_<hash>_<model>_<od_source>.csv`

The `<hash>` (“sample_id”) is computed as:

- `sha256(f"{perturbation_source_csv}|{max_entries}|{seed}|{search_method}")[:10]`

So changing any of those fields changes the generated filename.

---

## How to reproduce end-to-end

All commands assume you run from the repository root.

### 1) Ensure environment

Project policy is to use `uv`.

- Sync the locked environment:

```bash
uv sync
```

Phase B runtime dependencies include (already present in this repo’s environment in typical setups):

- `fastdtw`, `haversine` (DTW computation)
- `numpy`

For plotting (used by the visualization script):

- `matplotlib`, `pandas`, `seaborn`

If plotting deps are missing, add them with:

```bash
uv add pandas seaborn matplotlib
```

### 2) Run Phase B via the pipeline

The pipeline phase is implemented in `python_pipeline.py` as `@phase("perturbation_correction")`.

Run only Phase B:

```bash
uv run python python_pipeline.py \
  --eval-dir hoser-perturbed-beijing-pert-type-detour-eval \
  --only perturbation_correction
```

If you want to override config at runtime (these override `eval-dir/config/evaluation.yaml`):

```bash
uv run python python_pipeline.py \
  --eval-dir hoser-perturbed-beijing-pert-type-detour-eval \
  --only perturbation_correction \
  --perturbation-max-entries 1000 \
  --perturbation-seed 0 \
  --perturbation-od-source train \
  --perturbation-use-astar
```

To force regeneration instead of reusing `gene_perturbation` CSVs, pass `--force`.

### 3) Generate the plots

Plots are produced by:

- `tools/visualize_perturbation_correction_results.py`

Command (uses the default output dir under the eval workspace):

```bash
uv run python tools/visualize_perturbation_correction_results.py \
  --eval-dir hoser-perturbed-beijing-pert-type-detour-eval
```

This reads the saved Phase B outputs and writes:

- `hoser-perturbed-beijing-pert-type-detour-eval/figures/perturbation_correction/`

---

## Plots (embedded)

### Correction Rate (RSR)

![RSR by model](../hoser-perturbed-beijing-pert-type-detour-eval/figures/perturbation_correction/rsr_by_model.png)

### Mean DTW gap (dirty − clean)

![DTW gap by model](../hoser-perturbed-beijing-pert-type-detour-eval/figures/perturbation_correction/dtw_gap_by_model.png)

### Per-sample DTW delta distribution

![DTW delta boxplot](../hoser-perturbed-beijing-pert-type-detour-eval/figures/perturbation_correction/dtw_delta_boxplot.png)

---

## Results tables

### Table 1 — Per-model Phase B metrics (from saved outputs)

Definitions:

- Mean gap (km) = `mean_to_dirty_km - mean_to_clean_km`
- Per-sample Δ (km) = `dtw_to_dirty_km - dtw_to_clean_km`
- Neg frac = fraction of samples with Δ < 0
- Tie frac = fraction of samples with Δ ≈ 0 (absolute value < 1e-9)

| Model | Variant | Seed | RSR | Corrected | Valid | Mean DTW to clean (km) | Mean DTW to dirty (km) | Mean gap (dirty-clean) (km) | Neg frac | Tie frac | Median Δ (km) | P90 Δ | P95 Δ | P99 Δ | Max Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilled_l0p001_seed42 | distilled (λ=0.001) | 42 | 0.945 | 945 | 1000 | 16.594 | 17.684 | 1.090 | 0.0500 | 0.0080 | 0.474 | 2.722 | 4.535 | 8.934 | 18.861 |
| distilled_l0p001_seed43 | distilled (λ=0.001) | 43 | 0.947 | 947 | 1000 | 16.254 | 17.357 | 1.103 | 0.0470 | 0.0080 | 0.445 | 2.732 | 4.436 | 8.969 | 23.625 |
| distilled_l0p001_seed44 | distilled (λ=0.001) | 44 | 0.935 | 935 | 1000 | 15.266 | 16.296 | 1.030 | 0.0580 | 0.0100 | 0.451 | 2.708 | 4.067 | 8.238 | 14.086 |
| distilled_l0p5_seed42 | distilled (λ=0.5) | 42 | 0.940 | 940 | 1000 | 15.420 | 16.426 | 1.006 | 0.0570 | 0.0050 | 0.430 | 2.460 | 4.264 | 8.829 | 23.858 |
| distilled_l0p5_seed43 | distilled (λ=0.5) | 43 | 0.937 | 937 | 1000 | 17.603 | 18.626 | 1.023 | 0.0550 | 0.0100 | 0.451 | 2.568 | 4.332 | 8.312 | 23.625 |
| distilled_l0p5_seed44 | distilled (λ=0.5) | 44 | 0.941 | 941 | 1000 | 16.156 | 17.187 | 1.031 | 0.0530 | 0.0100 | 0.445 | 2.580 | 4.223 | 7.412 | 14.229 |
| distilled_l1_seed42 | distilled (λ=1.0) | 42 | 0.945 | 945 | 1000 | 15.827 | 16.845 | 1.018 | 0.0500 | 0.0060 | 0.441 | 2.639 | 4.176 | 7.412 | 13.960 |
| distilled_l1_seed43 | distilled (λ=1.0) | 43 | 0.935 | 935 | 1000 | 18.255 | 19.342 | 1.087 | 0.0550 | 0.0130 | 0.448 | 2.804 | 4.487 | 8.374 | 23.625 |
| distilled_l1_seed44 | distilled (λ=1.0) | 44 | 0.943 | 943 | 1000 | 17.219 | 18.277 | 1.059 | 0.0460 | 0.0140 | 0.451 | 2.515 | 4.098 | 8.835 | 38.279 |
| vanilla_seed42 | vanilla | 42 | 0.936 | 936 | 1000 | 17.281 | 18.327 | 1.046 | 0.0550 | 0.0120 | 0.441 | 2.525 | 4.325 | 8.294 | 20.042 |
| vanilla_seed43 | vanilla | 43 | 0.949 | 949 | 1000 | 15.903 | 16.971 | 1.068 | 0.0450 | 0.0080 | 0.460 | 2.580 | 4.112 | 9.030 | 22.520 |
| vanilla_seed44 | vanilla | 44 | 0.947 | 947 | 1000 | 16.082 | 17.145 | 1.064 | 0.0480 | 0.0080 | 0.459 | 2.600 | 4.325 | 9.916 | 14.005 |

### Table 2 — Group averages across seeds (variant-level)

These values summarize Table 1 grouped by model variant.

| Variant | N runs | Mean RSR | Std RSR | Mean gap km | Std gap km | Mean neg frac |
|---|---:|---:|---:|---:|---:|---:|
| distilled (λ=0.001) | 3 | 0.9423 | 0.0052 | 1.0743 | 0.0317 | 0.0517 |
| distilled (λ=0.5) | 3 | 0.9393 | 0.0017 | 1.0198 | 0.0104 | 0.0550 |
| distilled (λ=1.0) | 3 | 0.9410 | 0.0043 | 1.0544 | 0.0285 | 0.0503 |
| vanilla | 3 | 0.9440 | 0.0057 | 1.0596 | 0.0096 | 0.0493 |

### Table 3 — RSR split by perturbation strength (per model)

Here, `strong` is exactly rows where `ab_strength == "strong"`; everything else is grouped into `other`.

| Model | Variant | Seed | Strong n | Strong RSR | Other n | Other RSR |
|---|---:|---:|---:|---:|---:|---:|
| distilled_l0p001_seed42 | distilled (λ=0.001) | 42 | 369 | 0.9593 | 631 | 0.9366 |
| distilled_l0p001_seed43 | distilled (λ=0.001) | 43 | 369 | 0.9593 | 631 | 0.9398 |
| distilled_l0p001_seed44 | distilled (λ=0.001) | 44 | 369 | 0.9404 | 631 | 0.9319 |
| distilled_l0p5_seed42 | distilled (λ=0.5) | 42 | 369 | 0.9485 | 631 | 0.9350 |
| distilled_l0p5_seed43 | distilled (λ=0.5) | 43 | 369 | 0.9295 | 631 | 0.9414 |
| distilled_l0p5_seed44 | distilled (λ=0.5) | 44 | 369 | 0.9431 | 631 | 0.9398 |
| distilled_l1_seed42 | distilled (λ=1.0) | 42 | 369 | 0.9539 | 631 | 0.9398 |
| distilled_l1_seed43 | distilled (λ=1.0) | 43 | 369 | 0.9458 | 631 | 0.9287 |
| distilled_l1_seed44 | distilled (λ=1.0) | 44 | 369 | 0.9512 | 631 | 0.9382 |
| vanilla_seed42 | vanilla | 42 | 369 | 0.9512 | 631 | 0.9271 |
| vanilla_seed43 | vanilla | 43 | 369 | 0.9648 | 631 | 0.9398 |
| vanilla_seed44 | vanilla | 44 | 369 | 0.9377 | 631 | 0.9525 |

### Table 4 — Strength-stratified aggregates (variant-level)

This table pools rows across the 3 seeds per variant.

| Variant | Strength bucket | N rows | RSR | Mean Δ (km) | Median Δ (km) | Neg frac |
|---|---|---:|---:|---:|---:|---:|
| distilled (λ=0.001) | strong | 1107 | 0.9530 | 1.9535 | 1.2798 | 0.0470 |
| distilled (λ=0.001) | other | 1893 | 0.9361 | 0.5602 | 0.2888 | 0.0544 |
| distilled (λ=0.5) | strong | 1107 | 0.9404 | 1.8285 | 1.1689 | 0.0596 |
| distilled (λ=0.5) | other | 1893 | 0.9387 | 0.5468 | 0.2918 | 0.0523 |
| distilled (λ=1.0) | strong | 1107 | 0.9503 | 1.9188 | 1.2399 | 0.0497 |
| distilled (λ=1.0) | other | 1893 | 0.9356 | 0.5489 | 0.2996 | 0.0507 |
| vanilla | strong | 1107 | 0.9512 | 1.9200 | 1.1522 | 0.0488 |
| vanilla | other | 1893 | 0.9398 | 0.5564 | 0.3010 | 0.0497 |

### Table 5 — Optional LM-TAD teacher perplexity triangulation (from summaries)

If the Phase B config includes LM-TAD (`perturbation_lmtad_checkpoint` + `perturbation_lmtad_repo`), Phase B also computes teacher perplexities for:

- Generated trajectories (`pred_ppl`)
- Clean reference trajectories (`clean_ppl`)
- Dirty reference trajectories (`dirty_ppl`)

It then defines **triangulation_rate** as:

- fraction of samples where `abs(logppl_gen - logppl_clean) < abs(logppl_gen - logppl_dirty)`

Per-model values:

| Model | Variant | Seed | Mean log ppl (gen) | Mean log ppl (clean) | Mean log ppl (dirty) | Triangulation rate |
|---|---:|---:|---:|---:|---:|---:|
| distilled_l0p001_seed42 | distilled (λ=0.001) | 42 | 1.1369 | 0.7960 | 1.3046 | 0.4820 |
| distilled_l0p001_seed43 | distilled (λ=0.001) | 43 | 1.1458 | 0.7960 | 1.3046 | 0.4780 |
| distilled_l0p001_seed44 | distilled (λ=0.001) | 44 | 1.1192 | 0.7960 | 1.3046 | 0.4820 |
| distilled_l0p5_seed42 | distilled (λ=0.5) | 42 | 1.1254 | 0.7960 | 1.3046 | 0.4780 |
| distilled_l0p5_seed43 | distilled (λ=0.5) | 43 | 1.1663 | 0.7960 | 1.3046 | 0.4420 |
| distilled_l0p5_seed44 | distilled (λ=0.5) | 44 | 1.1394 | 0.7960 | 1.3046 | 0.4750 |
| distilled_l1_seed42 | distilled (λ=1.0) | 42 | 1.1325 | 0.7960 | 1.3046 | 0.4740 |
| distilled_l1_seed43 | distilled (λ=1.0) | 43 | 1.1674 | 0.7960 | 1.3046 | 0.4500 |
| distilled_l1_seed44 | distilled (λ=1.0) | 44 | 1.1717 | 0.7960 | 1.3046 | 0.4450 |
| vanilla_seed42 | vanilla | 42 | 1.1431 | 0.7960 | 1.3046 | 0.4620 |
| vanilla_seed43 | vanilla | 43 | 1.1472 | 0.7960 | 1.3046 | 0.4560 |
| vanilla_seed44 | vanilla | 44 | 1.1241 | 0.7960 | 1.3046 | 0.4620 |

Variant-level averages:

| Variant | N | Mean triangulation | Mean log ppl gen | Mean log ppl clean | Mean log ppl dirty |
|---|---:|---:|---:|---:|---:|
| distilled (λ=0.001) | 3 | 0.4807 | 1.1340 | 0.7960 | 1.3046 |
| distilled (λ=0.5) | 3 | 0.4650 | 1.1437 | 0.7960 | 1.3046 |
| distilled (λ=1.0) | 3 | 0.4563 | 1.1572 | 0.7960 | 1.3046 |
| vanilla | 3 | 0.4600 | 1.1381 | 0.7960 | 1.3046 |

---

## Interpretation

### 1) Correction performance is high and tightly clustered

Across all 12 runs, RSR falls in a narrow band (~0.935–0.949). Variant-level differences are small:

- Vanilla has the highest mean RSR (0.944), but its across-seed std (0.0057) is comparable to the gap vs the distilled variants.
- Distillation setting λ=0.5 has the lowest mean RSR (0.939), but again within small absolute margins.

In practical terms for this eval setup, Phase B RSR does **not** strongly separate the model variants.

### 2) “Not corrected” is mostly near-ties, not strongly wrong predictions

Neg frac (Δ < 0) is ~4.5–5.8%, while 1−RSR is ~5–6.5%. The difference is explained by **ties/near-ties** where `dtw_to_clean_km ≈ dtw_to_dirty_km` and thus fail the strict inequality.

This matters for interpretation: many “failures” are not cases where the model clearly matches the dirty trajectory; they’re cases where clean-vs-dirty distances are extremely close under this DTW metric.

### 3) Strong perturbations are often *easier* under this metric

Strength stratification shows that `ab_strength == "strong"` rows have much larger separation between dirty and clean under DTW:

- Strong rows have mean Δ ≈ 1.83–1.95 km depending on variant.
- Other rows have mean Δ ≈ 0.55–0.56 km.

This is consistent with the idea that “strong” detours are more distinct from clean, making the “closer to clean than dirty” decision easier.

### 4) Teacher (LM-TAD) triangulation is qualitatively different from DTW RSR

Teacher triangulation_rate is ~0.44–0.48, far below the DTW-based RSR (~0.94). This is expected if LM-TAD perplexity is sensitive to different aspects of the trajectory than centroid-DTW (e.g., tokenization, language-model plausibility, or different distance geometry).

Also note: `mean_log_perplexity_clean` and `mean_log_perplexity_dirty` are constant across models here because they are computed on the same fixed set of clean/dirty references; only the generated trajectories differ.

---

## Outliers and failure modes

### Most negative DTW deltas (generated closer to dirty)

| Rank | Model | traj_id | od | ab_strength | DTW clean (km) | DTW dirty (km) | Δ (dirty-clean) km | corrected |
|---:|---|---:|---|---|---:|---:|---:|---|
| 1 | distilled_l1_seed43 | 175 | [3057, 33361] | strong | 11.445 | 1.033 | -10.412 | False |
| 2 | distilled_l0p001_seed42 | 285 | [16357, 16364] | strong | 23.903 | 17.639 | -6.264 | False |
| 3 | distilled_l0p001_seed43 | 93 | [8634, 33873] | strong | 52.085 | 46.642 | -5.443 | False |
| 4 | distilled_l0p001_seed42 | 172 | [34285, 21872] | strong | 116.953 | 112.603 | -4.350 | False |
| 5 | vanilla_seed43 | 33 | [23063, 14079] | None | 136.686 | 132.508 | -4.177 | False |
| 6 | vanilla_seed44 | 172 | [34285, 21872] | strong | 30.703 | 26.697 | -4.005 | False |
| 7 | distilled_l0p5_seed43 | 49 | [15333, 26419] | strong | 69.786 | 65.950 | -3.836 | False |
| 8 | distilled_l1_seed42 | 172 | [34285, 21872] | strong | 36.736 | 32.950 | -3.786 | False |
| 9 | distilled_l0p5_seed43 | 33 | [23063, 14079] | None | 190.062 | 186.365 | -3.697 | False |
| 10 | distilled_l1_seed44 | 172 | [34285, 21872] | strong | 93.986 | 90.334 | -3.652 | False |

### Most positive DTW deltas (dirty much farther than clean)

| Rank | Model | traj_id | od | ab_strength | DTW clean (km) | DTW dirty (km) | Δ (dirty-clean) km | corrected |
|---:|---|---:|---|---|---:|---:|---:|---|
| 1 | distilled_l1_seed44 | 117 | [39387, 24819] | strong | 219.695 | 257.973 | 38.279 | True |
| 2 | distilled_l0p5_seed42 | 117 | [39387, 24819] | strong | 91.855 | 115.713 | 23.858 | True |
| 3 | distilled_l1_seed43 | 117 | [39387, 24819] | strong | 103.605 | 127.230 | 23.625 | True |
| 4 | distilled_l0p5_seed43 | 117 | [39387, 24819] | strong | 103.605 | 127.230 | 23.625 | True |
| 5 | distilled_l0p001_seed43 | 117 | [39387, 24819] | strong | 103.605 | 127.230 | 23.625 | True |
| 6 | vanilla_seed43 | 153 | [5607, 1570] | strong | 99.952 | 122.472 | 22.520 | True |
| 7 | vanilla_seed42 | 153 | [5607, 1570] | strong | 87.070 | 107.112 | 20.042 | True |
| 8 | vanilla_seed42 | 117 | [39387, 24819] | strong | 81.628 | 100.490 | 18.861 | True |
| 9 | distilled_l0p001_seed42 | 117 | [39387, 24819] | strong | 82.317 | 101.179 | 18.861 | True |
| 10 | distilled_l1_seed43 | 153 | [5607, 1570] | strong | 114.176 | 132.601 | 18.424 | True |

Practical suggestion for deeper analysis:

- Consider inspecting shared failing ODs (e.g., `[34285, 21872]` appears repeatedly in extreme negatives across variants). Those may indicate a specific region/network structure where DTW-to-clean is systematically harder.

---

## Notes and caveats

- DTW is computed in GPS space using per-road centroids derived from `roadmap.geo`. This is a useful, consistent proxy, but it is not the same as path-length DTW on the full polyline geometry.
- `fastdtw` is an approximation; results are typically stable for this use, but it is not guaranteed identical to exact DTW.
- The “corrected” criterion is strict (`<`), so ties count as not corrected.
- Teacher triangulation is based on LM-TAD log perplexity and is not expected to numerically align with DTW-based RSR.
