# GitHub Issue: Add Phase B “Perturbation Correction” Evaluation Phase to `python_pipeline.py`

## What you get (Outcome)
Add a new pipeline phase `perturbation_correction` that you can run inside any evaluation directory (old or new) to answer:

> Given a perturbed dataset row that contains both a dirty trajectory and its embedded clean original, does the model’s generation look closer to the clean route or the dirty route?

This yields:
- Per-model **RSR** (Restoration Success Rate): fraction of cases where $DTW(\hat{T}, T_{clean}) < DTW(\hat{T}, T_{dirty})$.
- Per-entry diagnostics (DTW-to-clean, DTW-to-dirty, corrected flag).

This is intended to be implemented and validated **before** you train any models on perturbed data, by running Phase B against your existing (clean-trained) models/eval dirs.

---

## Fast path: verify Phase B works on current models (before training)
1) Ensure you have a perturbed CSV generated with existing tooling (or use an existing one):
   - Example: `data/Beijing_abnormal_3/train.csv`
2) Pick an existing eval dir containing models (clean-trained is fine):
   - Example: `hoser-distill-beijing/` or any older eval dir with `models/`
3) Run only the new phase, pointing at the same perturbed CSV for every eval dir:

```bash
cd /path/to/eval-dir
uv run python ../python_pipeline.py \
  --only perturbation_correction \
  --no-wandb
```

Implementation requirement for the pipeline wiring: if the eval dir’s config doesn’t contain the new keys (older eval dirs), you must be able to provide `perturbation_source_csv` via CLI override (or a documented default lookup). This is what makes “run the same phase across many eval dirs” practical.

Config/CLI rule (must follow):
- The phase reads defaults from the eval directory’s snapshotted config under `eval-dir/config/`.
- CLI flags override the YAML values.

---

## Data contract (use perturbed `train.csv` directly)
No new manifest file is required.

### Canonical input
- `perturbation_source_csv`: a perturbed split CSV created by `generate_hoser_abnormalities.py`.
  - Example: `data/Beijing_abnormal_3/train.csv`

### How Phase B extracts references
Stream the CSV and filter to rows where `abnormality_info != "normal"`.

For each abnormal row:
- **dirty** $T_{dirty}$: the row’s `rid_list` (comma-separated IDs)
- **clean** $T_{clean}$: `ast.literal_eval(abnormality_info)["real"]["rid_list"]` (comma-separated IDs)
- **entry id**: `output_row_index` (the row number in the perturbed CSV excluding header, i.e. the streaming counter)
- **origin_time** (optional): first timestamp in `ast.literal_eval(abnormality_info)["real"]["time_list"]`

Important constraint:
- For detours, the dirty `rid_list` length can differ from the clean length while `time_list` stays the original length. Treat timestamps as trajectory-level metadata only (use `origin_time`), never as per-road aligned.

---

## How the pieces connect (Data flow)
Phase B is intentionally built as two layers:

1) `tools/evaluate_perturbation_correction.py` (core logic)
   - Reads the perturbed CSV, extracts (clean, dirty, origin_time, id)
   - Optionally generates $\hat{T}$ for each entry for a given model
   - Computes DTW($\hat{T}$, clean) and DTW($\hat{T}$, dirty)
   - Writes per-entry and per-model summaries

2) `python_pipeline.py` phase `perturbation_correction` (orchestration)
   - Runs the core evaluator for each detected model checkpoint in the eval dir
   - Handles config/CLI overrides and output directory layout

This separation is deliberate: you can unit test the evaluator without touching pipeline wiring, and you can smoke-test the phase against existing eval dirs without training anything.

---

## Phase definition (metrics)
Per entry:
- $d_{truth} = DTW(\hat{T}, T_{clean})$
- $d_{noise} = DTW(\hat{T}, T_{dirty})$
- `corrected = (d_truth < d_noise)`

Per model:
- $RSR = \frac{1}{N}\sum_i \mathbb{I}[d_{truth}<d_{noise}]$

Optional teacher signal:
- Compute LM‑TAD log-perplexity for $\hat{T}$ (and optionally clean/dirty) using `simple_evaluate_with_lmtad.evaluate_trajectories_direct`.

---

## Where it runs (compatibility with old eval dirs)
Requirement: Phase B must run in any eval dir that contains `models/`, including eval dirs for models trained on clean data.

Design rule:
- Eval dir is only for **model discovery** and **writing outputs**.
- Perturbed reference data is always provided externally via `perturbation_source_csv`.

---

## Configuration (what the phase needs)
Add these keys to the eval dir config (snapshotted under `eval-dir/config/`, typically `eval-dir/config/evaluation.yaml`).
CLI must override these values when provided.

Keys:
- `perturbation_source_csv`: path to perturbed CSV (absolute, or clearly defined relative behavior)
- `perturbation_max_entries`: optional cap for quick smoke tests
- `perturbation_seed`: optional sampling seed
- `perturbation_use_astar`: bool; make Phase B deterministic if possible
- `perturbation_lmtad_checkpoint`: optional; enable teacher scoring

Backwards-compat:
- If the eval dir config file exists but does not contain these keys, use CLI overrides when provided.
- If neither config nor CLI provides `perturbation_source_csv`, skip Phase B with a clear error message.

## Implementation quality requirements (non-negotiable)
- Modular and readable: separate parsing, generation, metric computation, and IO.
- SOLID: keep responsibilities tight; avoid "god functions".
- Use objects to avoid long parameter lists:
  - e.g. a `PerturbationCorrectionConfig` object for all config values
  - a `PerturbationDataset`/`AbnormalRowIterator` object for streaming abnormal rows
  - a `PerturbationCorrectionEvaluator` object for running the computation for a given model
- Logging: use the stdlib `logging` module, include counts (rows scanned, abnormal rows, sampled rows), and log file paths written.
- Fail fast: validate inputs up front (CSV exists, required columns exist, abnormality payload parseable, roadmap available), and raise clear exceptions (don’t silently continue on corrupt data).

---

## Implementation plan (ordered, with dependencies)

### Task 1 — Implement core evaluator (`tools/evaluate_perturbation_correction.py`)
Goal: a standalone script/module that can run on one model + one perturbed CSV and write outputs.

Why first: pipeline wiring is trivial once the evaluator exists.

Must do:
- Stream/filter abnormal rows, parse `abnormality_info` with `ast.literal_eval`.
- Derive clean/dirty sequences and entry ids.
- Compute DTW-to-clean vs DTW-to-dirty using the same road centroid mapping approach as `evaluation.py`.
- Output:
  - `per_entry.csv` (one row per entry)
  - `results.json` (aggregate, includes RSR)

### Task 2 — Add tests for triangulation correctness
Goal: tiny synthetic sequences + mock road centroids where DTW ordering is predictable.

Depends on: Task 1.

### Task 3 — Wire into `python_pipeline.py` as phase `perturbation_correction`
Goal: run Task 1 across all detected models in the eval dir.

Depends on: Task 1.

Must do:
- Register phase + add to phase ordering.
- Read config from the eval dir snapshot when present.
- Support older eval dirs by allowing CLI override for at least `perturbation_source_csv` (otherwise the phase should skip cleanly).

### Task 4 — (Optional) timestamp control for better faithfulness
Goal: condition generation on `origin_time` from the perturbed CSV.

Depends on: Task 1 + Task 3.

Notes:
- This may require extending `generate_trajectories_programmatic` to accept explicit per-entry start times.
- You can defer this while you’re just smoke-testing the evaluation machinery.

### Task 5 — Runbook + comparison workflow
Goal: make it easy to run the same phase across many eval dirs.

Depends on: Task 3.

---

## Acceptance criteria
- `perturbation_correction` runs in an eval dir and produces `results.json` + `per_entry.csv`.
- The phase can be run on older eval dirs (clean-trained models) by pointing to the same `perturbation_source_csv`.
- Deterministic behavior when `perturbation_seed` is fixed and deterministic search is enabled.

---

## Example commands (intended)
```bash
# Create perturbed dataset (existing tooling)
uv run python generate_hoser_abnormalities.py \
  --input-dir data/Beijing \
  --output-dir data/Beijing_abnormal_3 \
  --level high \
  --seed 42

# Run Phase B inside one eval dir
cd hoser-distill-beijing
uv run python ../python_pipeline.py --only perturbation_correction --no-wandb

# Override config from CLI (should override eval-dir/config/evaluation.yaml)
uv run python ../python_pipeline.py \
  --only perturbation_correction \
  --no-wandb \
  --perturbation-source-csv /home/mka299/HOSER/data/Beijing_abnormal_3/train.csv \
  --perturbation-max-entries 2000 \
  --perturbation-seed 0

# Run Phase B across many eval dirs (same perturbation source, comparable results)
for d in hoser-distill-beijing hoser-distill-beijing-lambda-1 hoser-distill-beijing-and-l1-normal; do
  echo "== $d =="
  (cd "$d" && uv run python ../python_pipeline.py --only perturbation_correction --no-wandb)
done
```
