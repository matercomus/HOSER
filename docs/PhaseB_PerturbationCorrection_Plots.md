# Phase B: Plotting Perturbation Correction Results

This doc covers `tools/visualize_perturbation_correction_results.py`, which plots the outputs produced by the Phase B evaluator (`perturbation_correction` pipeline phase).

## What it reads

The script expects Phase B outputs in:
- `eval_dir/perturbation_correction/{model}/summary.json`
- `eval_dir/perturbation_correction/{model}/rows.jsonl`

Model display names and colors are taken from `tools/model_detection.py` via:
- `get_display_name(model)`
- `get_model_color(model)`

## Plots produced

By default it writes PNG+SVG versions of:
- `rsr_by_model.(png|svg)` — bar chart of RSR (corrected rate) per model
- `dtw_gap_by_model.(png|svg)` — mean DTW gap per model: $(dirty - clean)$
- `dtw_delta_boxplot.(png|svg)` — distribution of per-sample DTW delta: $(dirty - clean)$

## How to run

From repo root (recommended):
```bash
uv run python /home/mka299/HOSER/tools/visualize_perturbation_correction_results.py \
  --eval-dir /home/mka299/HOSER/hoser-distill-beijing \
  --output-dir /home/mka299/HOSER/hoser-distill-beijing/figures/perturbation_correction
```

If you omit `--output-dir`, it defaults to:
- `<eval-dir>/figures/perturbation_correction`

### Title override
```bash
uv run python /home/mka299/HOSER/tools/visualize_perturbation_correction_results.py \
  --eval-dir /home/mka299/HOSER/hoser-distill-beijing \
  --title "Beijing Phase B (Smoke)"
```

## Dependencies

The plotting script requires:
- `matplotlib`
- `pandas`
- `seaborn`

If `pandas`/`seaborn` are missing, the script raises a helpful error message suggesting:
- `uv add pandas`
- `uv add seaborn`

## Troubleshooting

- If you see `No Phase B outputs found`, confirm you have run Phase B and that the directory `eval_dir/perturbation_correction/` exists.
- If a model folder is missing `summary.json`, it will be skipped with a warning.
