# Phase B: Perturbation Correction (DTW Triangulation)

This is the canonical, end-to-end description of Phase B as implemented in:

- `tools/perturbation_correction.py` (core evaluator)
- `python_pipeline.py` (pipeline phase `perturbation_correction`)
- `tools/visualize_perturbation_correction_results.py` (plots)

It replaces/absorbs older “issue-style” notes and the separate plotting doc.

## Goal (what Phase B answers)

Given a perturbed (abnormal) dataset row that contains both:

- a **dirty** trajectory (the perturbed one)
- and the embedded **clean** original

Phase B asks:

> Is the model’s generated trajectory closer (in GPS DTW distance) to the clean route than to the dirty route?

## Data contract (inputs)

### 1) Perturbed split CSV (required)

Phase B reads one perturbed CSV (typically `train.csv`) produced by the repository’s perturbation tooling (e.g. `generate_hoser_abnormalities.py`).

Required columns:

- `traj_id`
- `rid_list` — dirty road-id sequence
- `abnormality_info` — a string parseable by `tools.abnormality_metadata.parse_abnormality_info`.

For abnormal rows, `abnormality_info` must contain a `real` dict with:

- `real.rid_list` — clean road-id sequence

Rows that are not abnormal are skipped (`parse_abnormality_info(...)` returns `None`).

### 2) Road network centroids (required)

DTW is computed in GPS space using per-road centroids loaded from:

- `data/<dataset>/roadmap.geo`

Phase B loads this using `evaluation.load_road_network(...)` and expects a usable `center_gps` per road.

Important details of centroid handling:

- Road IDs without centroids are skipped when converting a trajectory to coordinates.
- If either trajectory ends up with fewer than 2 coordinate points, DTW is treated as infinite (invalid).

### 3) An evaluation directory with models (required)

Phase B is meant to run inside an evaluation directory created by `setup_evaluation.py`, i.e. a directory that contains at least:

- `models/` (model checkpoints to evaluate)
- `config/evaluation.yaml` (snapshotted config)

The perturbed CSV itself is *not* expected to live in the eval dir; it is typically external and passed via config or CLI.

## Methodology (what Phase B computes)

### Reference trajectories

For each abnormal row:

- Dirty reference $T_{dirty}$ comes from the row’s `rid_list`.
- Clean reference $T_{clean}$ comes from `abnormality_info["real"]["rid_list"]`.

### Prediction generation (OD-only)

For each selected abnormal row, Phase B derives an OD pair from the clean trajectory:

$$OD = (T_{clean}[0], T_{clean}[-1])$$

and generates a model prediction $\hat{T}$ for that OD pair using `gene.generate_trajectories_programmatic`.

Notes:

- Only the OD endpoints are used for generation; Phase B does not currently condition generation on the row’s timestamps.
- Dirty/clean trajectory length mismatches are fine because DTW does not require equal lengths.

### Sampling strategy

Phase B streams the perturbed CSV and keeps only abnormal rows.

- If `perturbation_max_entries` is set, it uses reservoir sampling with `perturbation_seed`.
- Otherwise it uses all abnormal rows.

This design lets you run Phase B on very large perturbed CSVs without loading them fully into RAM.

### Distance metric: GPS DTW (km)

Phase B computes DTW over coordinate sequences using:

- `fastdtw.fastdtw`
- `haversine.haversine` as the per-point distance

Two DTW variants are computed per entry:

- Raw DTW in km:
  - $d_{clean} = DTW(\hat{T}, T_{clean})$
  - $d_{dirty} = DTW(\hat{T}, T_{dirty})$
- Normalized DTW (for diagnostics only):

$$DTW_{norm}(A,B) = \frac{DTW(A,B)}{(|A|+|B|)/2}$$

Validity rule (used for counting):

- An entry is **valid** if both raw DTWs are finite.

### Correction decision and headline metric (RSR)

Per-entry “corrected” flag:

$$corrected = (d_{clean} < d_{dirty})$$

Headline metric:

$$RSR = \frac{\#corrected}{\#valid}$$

Where “valid” means both DTW distances were finite.

### Optional teacher signal: LM-TAD perplexity triangulation

If configured, Phase B additionally computes LM‑TAD log-perplexity for:

- generated trajectories $\hat{T}$
- clean trajectories $T_{clean}$
- dirty trajectories $T_{dirty}$

and reports a separate, *teacher-space* triangulation rate:

$$triangulated = (|ppl(\hat{T}) - ppl(T_{clean})| < |ppl(\hat{T}) - ppl(T_{dirty})|)$$

This is independent from DTW and is meant as an auxiliary signal.

Implementation notes:

- Teacher scoring is performed by `simple_evaluate_with_lmtad.evaluate_trajectories_direct`.
- You must supply `perturbation_lmtad_repo` pointing to the LM‑TAD repository root that contains `code/models/LMTAD.py`.

## How to run (reproducible runbook)

### 0) Dependencies

Phase B core evaluation requires:

- `fastdtw`
- `haversine`

If missing in your environment:

```bash
uv add fastdtw haversine
```

Plotting requires:

- `matplotlib`
- `pandas`
- `seaborn`

If missing:

```bash
uv add pandas seaborn
```

### 1) Prepare inputs

You need:

1) A perturbed CSV, e.g. `data/Beijing_abnormal_3/train.csv`.
2) An eval directory containing models, e.g. `hoser-distill-beijing/`.

### 2) Run Phase B in one eval directory (recommended)

From inside the eval dir:

```bash
uv run python ../python_pipeline.py \
  --only perturbation_correction \
  --no-wandb \
  --perturbation-source-csv ../data/Beijing_abnormal_3/train.csv \
  --perturbation-max-entries 200 \
  --perturbation-seed 0 \
  --perturbation-use-astar
```

From repo root:

```bash
uv run python python_pipeline.py \
  --eval-dir hoser-distill-beijing \
  --only perturbation_correction \
  --no-wandb \
  --perturbation-source-csv data/Beijing_abnormal_3/train.csv
```

### 3) Run Phase B across many eval dirs (same perturbation source)

```bash
for d in hoser-distill-beijing hoser-distill-beijing-and-l1-normal; do
  echo "== $d =="
  (
    cd "$d" \
    && uv run python ../python_pipeline.py \
      --only perturbation_correction \
      --no-wandb \
      --perturbation-source-csv ../data/Beijing_abnormal_3/train.csv \
      --perturbation-max-entries 500 \
      --perturbation-seed 0 \
      --perturbation-use-astar
  )
done
```

## Configuration and CLI overrides

Phase B reads defaults from `eval_dir/config/evaluation.yaml` and allows CLI overrides.

YAML keys (eval-dir config):

```yaml
# Phase B: perturbation correction
perturbation_source_csv: ../data/Beijing_abnormal_3/train.csv
perturbation_od_source: train
perturbation_max_entries: 200
perturbation_seed: 0
perturbation_use_astar: true

# Optional LM-TAD teacher signal
perturbation_lmtad_checkpoint: null
perturbation_lmtad_repo: null
perturbation_lmtad_batch_size: 128
```

CLI flags (override YAML):

- `--perturbation-source-csv`
- `--perturbation-od-source`
- `--perturbation-max-entries`
- `--perturbation-seed`
- `--perturbation-use-astar`
- `--perturbation-lmtad-checkpoint`
- `--perturbation-lmtad-repo`
- `--perturbation-lmtad-batch-size`

Path resolution:

- If `perturbation_source_csv` is a relative path, the pipeline resolves it relative to the eval dir.

Skip/fail behavior:

- If `perturbation_source_csv` is missing, Phase B is skipped.
- If you run `--only perturbation_correction` without configuring the CSV, the pipeline raises a clear error.

## Outputs (file layout + schemas)

Per model, Phase B writes to:

- `eval_dir/perturbation_correction/<model_type>/summary.json`
- `eval_dir/perturbation_correction/<model_type>/rows.jsonl`

Additionally, it caches the generated trajectories used for Phase B under:

- `eval_dir/gene_perturbation/<dataset>/seed<seed>/perturb_<hash>_<model_type>_<od_source>.csv`

This file is reused on subsequent runs unless `--force` is set.

### rows.jsonl

One JSON object per sampled abnormal entry:

- `i`: index in the sampled list (0..N-1)
- `traj_id`
- `od`: `[origin_road_id, dest_road_id]` (derived from clean trajectory endpoints)
- `ab_type`, `ab_level`, `ab_strength` (from `abnormality_info`, if present)
- `dtw_to_clean_km`, `dtw_to_dirty_km`
- `dtw_to_clean_norm`, `dtw_to_dirty_norm`
- `corrected`: `true|false|null` (`null` indicates invalid DTW)

### summary.json

Top-level fields include:

- `rsr`
- `counts.total`, `counts.valid`, `counts.invalid`, `counts.corrected`
- `dtw_km.mean_to_clean`, `dtw_km.mean_to_dirty`
- `artifacts.generated_csv`, `artifacts.rows_jsonl`

If teacher scoring is enabled, summary also includes:

- `teacher.mean_log_perplexity_generated`
- `teacher.mean_log_perplexity_clean`
- `teacher.mean_log_perplexity_dirty`
- `teacher.triangulation_rate`

## Plotting Phase B results

Use `tools/visualize_perturbation_correction_results.py` to generate summary plots.

It reads:

- `eval_dir/perturbation_correction/{model}/summary.json`
- `eval_dir/perturbation_correction/{model}/rows.jsonl`

and uses model display names/colors from `tools/model_detection.py`.

Run from repo root:

```bash
uv run python tools/visualize_perturbation_correction_results.py \
  --eval-dir hoser-distill-beijing \
  --output-dir hoser-distill-beijing/figures/perturbation_correction
```

If you omit `--output-dir`, it defaults to:

- `<eval-dir>/figures/perturbation_correction`

Optional title override:

```bash
uv run python tools/visualize_perturbation_correction_results.py \
  --eval-dir hoser-distill-beijing \
  --title "Beijing Phase B (Smoke)"
```

Plots produced (PNG + SVG):

- `rsr_by_model` — bar chart of RSR per model
- `dtw_gap_by_model` — mean DTW gap $(dirty - clean)$ per model
- `dtw_delta_boxplot` — per-sample DTW delta distribution

## Determinism and reproducibility notes

To make Phase B as reproducible as possible:

- Fix `--perturbation-seed`.
- Fix `--perturbation-max-entries` (or omit it to use all abnormal rows).
- Prefer `--perturbation-use-astar` to reduce search stochasticity.

Phase B also caches the generated CSV per (source_csv, max_entries, seed, search_method, model_type, od_source), so re-runs will reuse identical generations unless `--force` is set.

## Troubleshooting

- **Phase skipped / error about missing CSV**: set `perturbation_source_csv` in `config/evaluation.yaml` or pass `--perturbation-source-csv`.
- **Road network not found**: Phase B requires `data/<dataset>/roadmap.geo` in the repo root.
- **Many invalid rows**: indicates missing road centroids or unknown road IDs in the references/predictions; DTW becomes infinite when fewer than 2 centroid points are available.
- **Teacher repo error**: `perturbation_lmtad_repo` must point at the LM‑TAD repo root containing `code/models/LMTAD.py`.

## Known limitations (current behavior)

- Phase B currently generates trajectories from OD endpoints only; it does not use `origin_time` from the perturbed CSV.
- The per-row `i` field is the sampled index, not the original CSV row number.

