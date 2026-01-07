# LM-TAD decision benchmarking report (Beijing) — per-type abnormality isolation

Date: 2026-01-07

## Executive summary

This report covers:

1. Earlier LM-TAD decision benchmarks (Jan 03 overnight).
2. Balancing sanity checks.
3. Per-type abnormality isolation benchmarks (detour-only vs route_switch-only) and how results vary across decision boundaries.

Headline findings:

- The earlier route_switch benchmark was uninformative due to data sparsity: the `Beijing_abnormal_3_detectable_route_switch` balanced set was empty on the `train` split, so AUROC/AUPRC and operating-point metrics are undefined.
- Per-type isolation fixes comparability: both per-type datasets have enough abnormal rows to build balanced eval sets.
- On balanced `val`, LM-TAD separates detour much better than route_switch:
  - Detour: AUROC=0.8468, AUPRC=0.8072
  - Route_switch: AUROC=0.6743, AUPRC=0.6505
- Decision boundaries should be interpreted as operating-point choices: changing `q`/`k` changes TP/FP/TN/FN and precision/recall/FPR, but does not change AUROC/AUPRC.

## Runs and artifacts referenced

- Balancing sanity check:
  - `research_runs/_balanced_checks/balanced_data/bj_detectable_dr_bal1/balanced_manifest.json`
- Earlier benchmark (Jan 03 overnight):
  - `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260103_123758/report.md`
- Per-type isolation benchmark (Jan 07 val):
  - `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_val_qgrid_20260107/analysis/summary.md`
  - `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_val_qgrid_20260107/analysis/Beijing_per_type_detour/report.md`
  - `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_val_qgrid_20260107/analysis/Beijing_per_type_route_switch/report.md`
- Per-type isolation benchmark (Jan 07 train, still running at time of writing):
  - `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_train_qgrid_20260107.log`

## Balancing sanity check: what was verified

Balanced sets are constructed as:

- include all abnormal rows in the split
- sample normals to match (here: 1 normal per abnormal)
- bucket by trajectory length to better match length distributions

From `research_runs/_balanced_checks/balanced_data/bj_detectable_dr_bal1/balanced_manifest.json`:

- `normal_per_abnormal = 1`
- `length_bucket = 5`
- `seed = 42`
- `allow_replacement = false`
- output size: 3,446 total = 1,723 abnormal + 1,723 normal

This confirms the intended class balance and deterministic behavior under a fixed seed.

## Earlier benchmark (Jan 03 overnight): what we learned

Summary from `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260103_123758/report.md`:

| Dataset | N | Pos | AUROC | AUPRC | Key note |
|---|---:|---:|---:|---:|---|
| Beijing_abnormal_3_detectable | 212 | 106 | 0.9475 | 0.9302 | Very strong separation; strict baseline q can be overly conservative |
| Beijing_abnormal_3_detectable_dr | 3446 | 1723 | 0.8146 | 0.7818 | Moderate separation; q and matched top-k behave similarly |
| Beijing_abnormal_3_detectable_route_switch | 0 | 0 | NA | NA | Balanced set empty (0 abnormal rows in `train` split) |

Interpretation:

- The `route_switch` dataset result was **not a negative model finding**; it was a **coverage problem** (no abnormal examples in the benchmarked split), making the benchmark incapable of estimating AUROC/AUPRC or threshold metrics.
- The overnight run also demonstrated a general phenomenon:
  - with strong separation, an overly strict baseline-calibrated quantile (e.g., `q=0.99`) can yield very low or zero recall, even if the model ranks abnormals clearly higher.

## Per-type isolation benchmarks (Jan 07 val): primary findings

Run summary:

- `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_val_qgrid_20260107/analysis/summary.md`

Per-type results (balanced val):

| Dataset | N | Pos | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Beijing_per_type_detour | 26,972 | 13,486 | 0.8468 | 0.8072 |
| Beijing_per_type_route_switch | 26,972 | 13,486 | 0.6743 | 0.6505 |

Per-dataset reports (contain full q-grid tables):

- Detour: `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_val_qgrid_20260107/analysis/Beijing_per_type_detour/report.md`
- Route_switch: `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_val_qgrid_20260107/analysis/Beijing_per_type_route_switch/report.md`

### Interpretation of the AUROC/AUPRC gap

- **Detour (AUROC 0.85)**: LM-TAD log-perplexity meaningfully distinguishes detours from normals in the balanced val set.
- **Route_switch (AUROC 0.67)**: separation exists but is substantially weaker; at many operating points you should expect low recall unless you accept a high false positive rate.

This is consistent with the intuition that detours are "distributional" anomalies (longer or off-manifold routing), while route_switch can be more subtle (still plausible paths but different route choices), making score separation harder.

## Decision boundaries in the per-type val run (how to read the tables)

The per-type reports include two tables per dataset:

- `baseline_quantile`:
  - threshold is taken from a baseline calibration distribution at quantile `q`.
  - the resulting **flag rate on the target set is not guaranteed to equal $1-q$**.
- `topk_matched`:
  - fixes the alert volume: flag exactly $k = \lceil (1-q)N \rceil$ items.

### A concrete operating-point comparison (val)

Detour (`Beijing_per_type_detour`):

- At `q=0.95`:
  - baseline_quantile: recall=0.143, precision=0.885, flag_rate=0.081
  - topk_matched: recall=0.091, precision=0.908, flag_rate=0.050

Interpretation:

- Here baseline_quantile is **more permissive** on the target set than the implied $(1-q)$ budget (8.1% vs 5%). It yields higher recall but at the cost of flagging more total examples.
- top-k provides a more controlled alert rate, but (as expected) recall drops at the smaller budget.

Route_switch (`Beijing_per_type_route_switch`):

- At `q=0.95`:
  - baseline_quantile: recall=0.065, precision=0.748, flag_rate=0.044
  - topk_matched: recall=0.075, precision=0.749, flag_rate=0.050

Interpretation:

- The two decision rules are fairly similar at this operating point, suggesting the main limitation is **score separability**, not the rule.

### Boundary-dependent outputs (recommended for comparing "policy")

For each dataset, the following are generated from `metrics.json` without any re-scoring:

- Tables and summaries:
  - `analysis/<dataset>/plots/decision/tables/*.csv`
  - `analysis/<dataset>/plots/decision/summary/*.md`
- Confusion matrix grids across all q:
  - `analysis/<dataset>/plots/decision/confusion/*_grid.png`
- Metric curves vs q:
  - `analysis/<dataset>/plots/decision/curves/*_metrics_vs_q.png`

These are the right artifacts to compare "different boundaries" because they change with `q`.

## Per-type train benchmark status (Jan 07)

Train run:

- Log: `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_train_qgrid_20260107.log`
- Output directory: `research_runs/_benchmarks/bj_pertype_detour_vs_routeswitch_train_qgrid_20260107/`

At time of writing, train evaluation is still running (processing a large balanced set) and has not yet emitted the per-dataset analysis outputs for both targets.

## Conclusion

1. The Jan 03 overnight benchmark established that LM-TAD can strongly detect some abnormality regimes, but also exposed a critical pitfall: **a benchmark can be invalid if the split contains too few abnormal examples** (route_switch empty balanced set).
2. The Jan 07 per-type isolation benchmarks corrected the dataset comparability issue and produced a clean head-to-head comparison:
   - **Detour is substantially more detectable than route_switch on balanced val** by AUROC/AUPRC.
3. Decision boundaries must be interpreted as **operating-point choices**:
   - AUROC/AUPRC answer whether detection is possible in principle (ranking quality).
   - The chosen `q` / `k` answers whether the chosen alert budget meets the operational goal (recall vs precision vs FPR).

Practical recommendation:

- Use the per-type AUROC/AUPRC to decide which mechanisms are promising for LM-TAD-based detection.
- Use the `plots/decision/` artifacts to pick an operating point for deployment (or for fair policy comparisons), since that is where boundary selection meaningfully changes outcomes.
