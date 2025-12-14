**LM‑TAD Evaluation: Investigation Report**

Date: 2025-12-14

Author: GitHub Copilot (working in-repo)

---

**Executive Summary**

- I investigated why LM‑TAD evaluation outputs and the abnormality generator outputs did not align as expected.
- I modified and ran a confusion-matrix / analysis tool (`tools/confusion_matrix_lmtad_dataset_eval.py`) to compare evaluation outputs (in `tools_eval_lmtad/*`) against ground-truth `abnormality_info` stored in our generated CSVs.
- I discovered two important causes for the mismatch:
  1. The LMTAD `get_metrics` implementation (from the external LMTAD repo) uses a 3‑sigma rule applied to non‑outlier set AND (importantly) rounds scores to 1 decimal before comparing and uses a strict `>` comparator. It also passed binary predictions into `average_precision_score` (bug). Rounding + `>` produced ties that suppressed flags (notably in `BJUT_Beijing`).
  2. The generator we wrote (`generate_hoser_abnormalities.py`) and the original conversion script in LMTAD (`convert_HOSER_to_LMTAD.py`) implement different generation strategies and operate at different representations (road-id vs grid-token). These differences reduce overlap between "injected" anomalies and what the LM considers high‑perplexity.

This report summarizes what I changed, empirical results from runs, a comparison of generators, and recommended follow-ups.

---

**Files changed / added (work done in this investigation)**

- Modified: `tools/confusion_matrix_lmtad_dataset_eval.py`
  - Added per-dataset and global CSV summaries (if not present) — `confusion_summary.csv` and `confusion_summary_all.csv`.
  - If `evaluation_results.json` contains per-row scores (`log_perplexity_values`), compute a Porto-style 3σ threshold and collect porto TP/FP/FN/TN/precision/recall/F1/accuracy + PR metrics.
  - Added a multi-rule experiment comparing detection rules:
    - `pct95_gt` (95th percentile, `>`), `pct95_ge` (95th percentile, `>=`)
    - `porto_3sigma` (mean_non_outlier + 3*std)
    - `porto_2sigma` (mean_non_outlier + 2*std)
    - `top_k_5pct` (top-k by score, k=ceil(5% * n))
  - For each rule compute confusion metrics and save tidy `rule_comparison.csv` and a barplot `rule_comparison.png` under each `tools_eval_lmtad/<dataset>` directory when scores exist.

- Created: `docs/lmtad_evaluation_report.md` (this file) — summary of findings and next steps.


---

**What I ran and where (representative commands you executed earlier)**

- Evaluation runs (examples):

```bash
uv run python tools/evaluate_dataset_with_lmtad.py \
  --dataset Beijing \
  --data-dir data/Beijing \
  --roadmap data/Beijing/roadmap.geo \
  --lmtad-checkpoint /path/to/ckpt_best.pt \
  --sample-frac 0.01
```

- Plotted results with `tools/plot_lmtad_results.py` (produced per-split PNGs and abnormality CSVs).

- Ran the confusion summary / rule comparison script I added:

```bash
uv run python tools/confusion_matrix_lmtad_dataset_eval.py
```

Outputs were written to `tools_eval_lmtad/<dataset>/confusion_summary.csv`, `rule_comparison.csv`, `rule_comparison.png`, and `tools_eval_lmtad/confusion_summary_all.csv`.

---

**Key empirical observations**

1. Original evaluator behavior (95th percentile):
   - For large datasets (Beijing, Porto), the evaluator produced ~5% reported_outlier_rate (expected).
   - For `BJUT_Beijing` the reported outlier rate was much smaller (≈0.3–0.7%). We confirmed this is caused by ties at the 95th percentile combined with a strict `>` comparator.

2. Porto 3‑sigma detector behavior:
   - The Porto-style threshold (mean_non_outlier + 3*std) is often above the observed maximum score for these datasets (distributions are tight), so it detects zero rows.
   - 3σ is therefore too conservative for these particular score distributions.

3. Overlap between injected anomalies and model detections:
   - For `_abnormal` datasets the generator injected ≈4.8–5.3% abnormal rows (close to config 5%).
   - Model-detected outliers and generator-injected anomalies overlap poorly: TP counts were small while FP and FN were large → very low precision and recall (single-digit percent in many cases).

4. Metrics and statistical bugs discovered in the LMTAD `get_metrics` implementation (from your LMTAD code):
   - Rounds the metric to 1 decimal before comparing, causing quantization and ties.
   - Uses strict `>` comparator — ties are not flagged.
   - Calls `average_precision_score(y_true, y_pred_binary)` which is incorrect — it must receive continuous scores.

These issues (rounding + strict `>`) explain the anomalously low reported rates for `BJUT_Beijing` and distort AP/PR measures.

---

**Comparison: `convert_HOSER_to_LMTAD.py` vs `generate_hoser_abnormalities.py`**

Similarities
- Both implement detour, route_switch, and perturb (perturb/perturbed) types.
- Both can be deterministic when run with fixed seeds.

Differences (major, practical)
- Representation:
  - `convert_HOSER_to_LMTAD.py` operates on LMTAD grid tokens (after mapping road IDs to grid cells).
  - `generate_hoser_abnormalities.py` operates on HOSER road ID sequences (strings in `rid_list`) and writes `abnormality_info` to a CSV.
- Selection and production:
  - convert: selects a set of indices (np.random.seed) and generates separate outlier sets written to files — batch/collection oriented.
  - generate_hoser_abnormalities: streaming-friendly; supports probabilistic per-row Bernoulli generation using `default_rng(global_seed + idx)` and writes augmented CSV inline.
- Perturbation strength and semantics:
  - convert's `_perturb_point` changes grid tokens by offsets in grid space (potentially strong, localized perturbations). It may also produce route-switches by sampling existing trajectories.
  - generate's `perturb_rids` samples road IDs from a global road pool; `route_switch_from_pool` picks contiguous slices from a deterministic pool (which may be less realistic than sampling other trajectories).

Implication: Differences in representation and generator strength likely cause a large part of the observed mismatch between injected anomalies and LM‑TAD high‑perplexity detections.

---

**What we changed in the confusion script and why it helps**

- The script now consumes per-row `log_perplexity_values` (if available) and runs multiple detection rules to compare performance. This lets us:
  - Directly measure whether model scores have discriminative power (PR AUC / average precision reported for continuous scores).
  - Compare practical detection rules side-by-side to decide which rule best matches the generator or yields better precision/recall.
- The per-dataset `rule_comparison.csv` provides a tidy table you can use to produce plots or to tune thresholds programmatically.

---

**Short recommendations (next steps)**

1. Fix LMTAD `metrics.get_metrics` bugs (high priority):
   - Remove rounding of scores when comparing.
   - Use `average_precision_score(y_true, continuous_scores)` instead of passing binary predictions.
   - Consider making comparator configurable (use `>=` option) or adopt percentile rule consistently.

2. Run per-row discriminative diagnostics (for example `Beijing_abnormal/train`):
   - Compute ROC AUC and PR AUC using per-row continuous scores vs `abnormality_info`.
   - If AUC ≈ 0.5, the model cannot separate the injected anomalies and generator changes need to be stronger or more realistic.

3. Align generator representation with evaluation:
   - Either convert generated abnormal CSVs into LMTAD token space (using the same conversion pipeline) or modify the generation to operate in token space so that evaluation measures the real detectability of injections.
   - Consider stronger or different perturbation heuristics (larger grid offsets, route-switch using other trajectories, more intrusive changes) to ensure injected anomalies are detectable.

4. If the goal is to benchmark detection performance, create a reproducible pipeline:
   - Deterministic selection of indices (seeded) → deterministic abnormality injection → convert to evaluation token space → evaluate with consistent rule (e.g., percentile or configurable rule).

---

**Useful commands to reproduce my current analysis**

- Re-run confusion + rule comparison I added:

```bash
uv run python tools/confusion_matrix_lmtad_dataset_eval.py
# inspect outputs
ls -la tools_eval_lmtad/*/rule_comparison.*
```

- Plotting code (already available):

```bash
uv run tools/plot_lmtad_results.py --eval-dir tools_eval_lmtad/Beijing --out tools_eval_lmtad/Beijing
```

- If you want me to apply fixes to LMTAD `metrics.py` (remove rounding; fix average_precision_score): tell me which file path in your LMTAD repo to patch (example from earlier: `/home/matt/Dev/LMTAD/code/metrics.py`). I can patch and then re-run `tools/evaluate_dataset_with_lmtad.py` on a sampled split.

---

**Suggested immediate action (my recommendation)**

1. Apply the two quick metric fixes in LMTAD (`remove rounding`, `average_precision_score` fix) and re-evaluate one sampled dataset (e.g., `Beijing_abnormal/train`). This will remove quantization-induced ties and produce correct AP values.
2. Run the `rule_comparison` outputs we created and inspect `rule_comparison.csv` for that dataset. Use PR AUC to decide whether to prefer percentile or a sigma rule, or to tune sigma (2σ vs 3σ).

If you approve, I will:
- Patch LMTAD `metrics.py` with the two fixes and re-run evaluation on `Beijing_abnormal/train` (sampled) and regenerate confusion + rule comparison outputs.

---

If you want anything else summarized, or want me to proceed with the metric fixes and re-run now, say so and I'll take care of it and report back with the new results and plots.
