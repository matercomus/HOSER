# Phase B: Perturbation Correction (DTW Triangulation)

This doc covers the Phase B evaluator implemented in `tools/perturbation_correction.py` and exposed via the `perturbation_correction` phase in `python_pipeline.py`.

## What it measures

Given an abnormal (perturbed) training row:
- **Dirty reference**: the row’s `rid_list`
- **Clean reference**: `abnormality_info['real']['rid_list']`

For each sampled abnormal example, Phase B generates a prediction and computes DTW distances (in km) between:
- prediction → clean
- prediction → dirty

A sample is counted as **corrected** when:

$$\mathrm{DTW}(pred, clean) < \mathrm{DTW}(pred, dirty)$$

The headline metric is **RSR** (repair / corrected rate):

$$\mathrm{RSR} = \frac{\#corrected}{\#valid}$$

(“valid” means both DTW distances were finite, i.e. enough road centroids were available.)

## Inputs

### Required
- A perturbed CSV (typically `train.csv`) that contains abnormal rows with:
  - `traj_id`
  - `rid_list` (dirty trajectory)
  - `abnormality_info` (Python-literal dict string) containing `real.rid_list` (clean trajectory)

### Road centroids
DTW is computed in GPS space using road centroids loaded from:
- `data/<dataset>/roadmap.geo`

## How to run

Phase B is designed to run inside an evaluation directory created by `setup_evaluation.py`.

### Via pipeline (recommended)

From inside the eval dir:
```bash
uv run python ../python_pipeline.py --only perturbation_correction \
  --perturbation-source-csv ../data/Beijing_abnormal_3/train.csv \
  --perturbation-max-entries 200 \
  --perturbation-seed 0 \
  --perturbation-use-astar
```

Or from repo root:
```bash
uv run python python_pipeline.py --eval-dir hoser-distill-beijing --only perturbation_correction \
  --perturbation-source-csv ../data/Beijing_abnormal_3/train.csv \
  --perturbation-max-entries 200 \
  --perturbation-seed 0 \
  --perturbation-use-astar
```

### Configuration keys (eval-dir/config/evaluation.yaml)

Phase B reads these keys from `config/evaluation.yaml` in the eval dir (CLI flags override):
- `perturbation_source_csv` (required to run the phase)
- `perturbation_od_source` (currently used for timestamp selection; typically `train`)
- `perturbation_max_entries` (optional sampling cap)
- `perturbation_seed`
- `perturbation_use_astar` (defaults to `true` in the template)
- Optional teacher settings:
  - `perturbation_lmtad_checkpoint`
  - `perturbation_lmtad_repo`
  - `perturbation_lmtad_batch_size`

CLI overrides in `python_pipeline.py`:
- `--perturbation-source-csv`
- `--perturbation-od-source`
- `--perturbation-max-entries`
- `--perturbation-seed`
- `--perturbation-use-astar`
- `--perturbation-lmtad-checkpoint`
- `--perturbation-lmtad-repo`
- `--perturbation-lmtad-batch-size`

## Outputs

Per model, outputs are written under:
- `eval_dir/perturbation_correction/<model_type>/summary.json`
- `eval_dir/perturbation_correction/<model_type>/rows.jsonl`

### `summary.json`
Contains aggregate metrics such as:
- `rsr`
- counts: `total`, `valid`, `corrected`
- DTW means: `mean_to_clean`, `mean_to_dirty`

### `rows.jsonl`
One JSON record per sampled abnormal example, typically including:
- identifiers (e.g. `traj_id`, `od`)
- `dtw_to_clean_km`, `dtw_to_dirty_km`, `corrected`

## Notes / gotchas

- Phase B will **skip** automatically unless `perturbation_source_csv` is set; if you run `--only perturbation_correction`, it will fail fast with a clear error.
- DTW requires valid road centroids; missing/unknown road IDs can reduce `valid`.
- Generation for Phase B is intended to use **A\*** (the config template defaults `perturbation_use_astar: true`).
- Optional LM-TAD teacher scoring is enabled only when `perturbation_lmtad_checkpoint` is provided (and requires `perturbation_lmtad_repo`).
