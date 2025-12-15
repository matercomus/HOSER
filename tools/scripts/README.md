Diagnosis scripts for LM-TAD detection vs injected anomalies

Usage (reproducible via `uv`):

- Run the diagnostic script for a dataset (writes CSVs and PNGs into the eval dir):

```bash
uv run python tools/scripts/diagnose_detection.py \
  --eval-dir tools_eval_lmtad/Beijing_abnormal_2 \
  --data-dir data/Beijing_abnormal_2 \
  --out-dir tools_eval_lmtad/Beijing_abnormal_2
```

- Requirements (use `uv` to add packages):

```bash
uv venv
uv add pandas matplotlib scikit-learn scipy
uv sync
```

Notes:
- Do NOT import `tools.scripts.diagnose_detection` from the `tools` package in a REPL: importing `tools` may trigger other package imports (e.g., plotting libs) that are not in the minimal requirements. Run the script as shown above.
- The script attempts to match evaluation entries to injection bookkeeping by `row_index`/`idx`/`trajectory_id`. If the evaluation used sampling, matching may fail; the script will warn in that case.
- Output files written to `--out-dir`:
  - `detection_at_injected_rate.csv`
  - `detection_at_injected_rate_<split>.png`
  - `pr_data_<split>.csv`, `pr_curve_<split>.png`, `pr_summary.csv`
  - `score_distribution_stats.csv`, `score_distributions_<split>.png`

If you want, I can run the script now for Beijing and Porto and attach the outputs to the report.
