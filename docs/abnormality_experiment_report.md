# Abnormality Injection & Detection Report

Date: 2025-12-15

This report summarizes the experiments where we injected ~15% HOSER-format abnormalities into existing datasets, evaluated them with the LM‑TAD teacher, and ran the confusion / rule-comparison analysis. It collects the key numbers, embedded diagnostic plots produced by the evaluation tooling, and the main observations and next steps.

**Datasets evaluated**
- `Beijing` (original)
- `Beijing_abnormal_2` (generated with --abnormality-rate 0.15, level=high)
- `porto_hoser` (original)
- `porto_hoser_abnormal_2` (generated with --abnormality-rate 0.15, level=high)

**Where to find evaluation outputs**
- Per-dataset evaluation JSON: `tools_eval_lmtad/<dataset>/evaluation_results.json`
- Confusion / rule comparison CSV: `tools_eval_lmtad/<dataset>/confusion_summary.csv`
- Plots (boxplots / abnormality hist): `tools_eval_lmtad/<dataset>/lmtad_eval_*.png`

**Embedded Plots**

Beijing — Normal (left) vs Abnormal_2 (right)

![Beijing abnormality summary comparison](../tools_eval_lmtad/Beijing/comparison_lmtad_eval_abnormality.png)

![Beijing perplexity boxplot comparison](../tools_eval_lmtad/Beijing/comparison_lmtad_eval_boxplot.png)

Porto — Normal (left) vs Abnormal_2 (right)

![Porto abnormality summary comparison](../tools_eval_lmtad/porto_hoser/comparison_lmtad_eval_abnormality.png)

![Porto perplexity boxplot comparison](../tools_eval_lmtad/porto_hoser/comparison_lmtad_eval_boxplot.png)

**Key numeric findings**

- Injection bookkeeping (train split):
  - `data/Beijing_abnormal_2/train.csv.injected_indices.jsonl`: 94,253 injected entries
  - `data/Beijing_abnormal_2/train.csv`: 723,622 total rows
  - Observed injection fraction ≈ 94,253 / 723,622 = 0.12995 (~13.0%) — close to requested 15% (streaming sampling / rounding explains the gap).

**Key numeric findings**
  - Beijing (original): mean_log_perplexity = 11.9433, outlier_rate ≈ 5.00%
  - Beijing_abnormal_2: mean_log_perplexity = 12.0030 (+0.0598), outlier_rate ≈ 5.01%
  - porto_hoser (original): mean_log_perplexity ≈ 10.2813
  - porto_hoser_abnormal_2: mean_log_perplexity ≈ 10.3347 (+0.0534), outlier_rate ≈ 5.00%

Injection bookkeeping (train split):

| Dataset | Split | Injected entries | Total rows | Observed injection fraction |
|---|---:|---:|---:|---:|
| `Beijing_abnormal_2` | train | 94,253 | 723,622 | 0.12995 (~13.0%) |
- Confusion-summary (evaluator's default 95th‑percentile outliers; reported as `confusion_summary.csv`) — selected rows (train):

  - Beijing (original): compared=6279, TP=0, FP=314, FN=0, TN=5965, injected_rate=0.0, reported_outlier_rate=0.0500

  - Beijing_abnormal_2 (train):
    - compared=7203, TP=53, FP=308, FN=883, TN=5959
LM‑TAD aggregated stats (train split):

| Dataset | Split | mean_log_perplexity | outlier_rate |
|---|---:|---:|---:|
| `Beijing` (original) | train | 11.9433 | 0.0500 |
| `Beijing_abnormal_2` | train | 12.0030 (+0.0598) | 0.05012 |
| `porto_hoser` (original) | train | 10.2813 | 0.0500 |
| `porto_hoser_abnormal_2` | train | 10.3347 (+0.0534) | 0.0500 |
    - precision ≈ 0.147, recall ≈ 0.0566
    - injected_rate ≈ 0.12995, reported_outlier_rate ≈ 0.05012

  - porto_hoser (original): compared=4796, TP=0, FP=240, FN=0, TN=4556, injected_rate=0.0, reported_outlier_rate≈0.05004

  - porto_hoser_abnormal_2 (train):
    - compared=5520, TP=32, FP=244, FN=686, TN=4558
    - precision ≈ 0.116, recall ≈ 0.0446
    - injected_rate ≈ 0.13007, reported_outlier_rate ≈ 0.05

Notes: full per-dataset confusion CSVs were written to `tools_eval_lmtad/<dataset>/confusion_summary.csv`.

**Why reported outlier rate stays ≈5%**

- The LM‑TAD evaluation code (`evaluate_trajectories_direct` in `simple_evaluate_with_lmtad.py`) turns the continuous per-trajectory log-perplexity scores into binary outliers by selecting the 95th percentile threshold:

Confusion-summary (evaluator's default 95th‑percentile outliers)

| Dataset | Compared | TP | FP | FN | TN | Precision | Recall | Injected rate | Reported outlier rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Beijing` (original) | 6279 | 0 | 314 | 0 | 5965 | 0.000 | 0.000 | 0.0 | 0.0500 |
| `Beijing_abnormal_2` (train) | 7203 | 53 | 308 | 883 | 5959 | 0.147 | 0.0566 | 0.12995 | 0.05012 |
| `porto_hoser` (original) | 4796 | 0 | 240 | 0 | 4556 | 0.000 | 0.000 | 0.0 | 0.05004 |
| `porto_hoser_abnormal_2` (train) | 5520 | 32 | 244 | 686 | 4558 | 0.116 | 0.0446 | 0.13007 | 0.0500 |
  threshold = np.percentile(all_outlier_scores, 95)
  outliers = score > threshold

- By definition this yields approximately a 5% reported outlier rate regardless of how many anomalies you injected. That is the main reason the aggregated `outlier_rate` in the evaluation JSON stays ~5% even though the generator injected ~13% anomalies.

**What the confusion results tell us**

- The injections demonstrably increase mean log-perplexity (Beijing +0.06, Porto +0.05), so the teacher finds the injected trajectories somewhat harder to predict.
- Using the evaluator's default 95th‑percentile rule produces low recall (≈4–6%), because the threshold only marks the top 5% as outliers while the injected fraction is larger (~13%).
- Precision is low-to-moderate (Beijing ≈ 0.15, Porto ≈ 0.12) under the default rule: some injected anomalies are being detected, but many are below the 95th percentile threshold.

**Additional diagnostics already produced**

- Per-dataset PR-style diagnostics and plots were computed by the rule-comparison tool and saved under:
  - `tools_eval_lmtad/<dataset>/rule_comparison.csv` (if present)
  - The tool also updated `tools_eval_lmtad/confusion_summary_all.csv` for cross-dataset overview.

  **Diagnostic Script Outputs (quick summary)**

  - Per-dataset diagnostics (detection-at-injected-rate, PR summary, score-distribution stats) were generated and written to:
    - `tools_eval_lmtad/<dataset>/detection_at_injected_rate.csv`
    - `tools_eval_lmtad/<dataset>/pr_summary.csv` and `tools_eval_lmtad/<dataset>/pr_curve_<split>.png`
    - `tools_eval_lmtad/<dataset>/score_distribution_stats.csv` and `tools_eval_lmtad/<dataset>/score_distributions_<split>.png`
  - How to interpret them:
    - Use `detection_at_injected_rate.csv` to see precision/recall when thresholded at the measured injection budget (top-13% / top-15%).
    - Use `pr_summary.csv` and `pr_curve_<split>.png` to inspect average-precision (AP) and the full precision–recall tradeoff — AP indicates how well the continuous LM‑TAD scores separate injected vs non-injected examples.
    - Use `score_distribution_stats.csv` and the distribution PNGs to visually and statistically evaluate overlap between injected and non-injected score distributions (KS / Mann–Whitney / Cohen's d reported).

  Short practical takeaway: check `detection_at_injected_rate.csv` first — if recall improves substantially when thresholded to the injected fraction, the issue is mainly the 95%-by-design threshold; if AP is low in `pr_summary.csv` and distributions overlap heavily, then the injected anomalies are not strongly separable by LM‑TAD and further generator tuning or feature combinations will be needed.

  **One-line detection summary (train split, top-13% threshold used to match injected budget)**

  - `Beijing_abnormal_2` (train, top-13%): precision=0.1633, recall=0.1411, AP=0.15285
  - `porto_hoser_abnormal_2` (train, top-13%): precision=0.1407, recall=0.1227, AP=0.14686

  **Per-split summaries with thumbnails**

  - `Beijing_abnormal_2`:
    - train: precision=0.1633, recall=0.1411, AP=0.15285  ![PR train](../tools_eval_lmtad/Beijing_abnormal_2/pr_curve_train.png) ![Dist train](../tools_eval_lmtad/Beijing_abnormal_2/score_distributions_train.png)
    - val:   precision=0.1176, recall=0.1127, AP=0.13488  ![PR val](../tools_eval_lmtad/Beijing_abnormal_2/pr_curve_val.png) ![Dist val](../tools_eval_lmtad/Beijing_abnormal_2/score_distributions_val.png)
    - test:  precision=0.1358, recall=0.1188, AP=0.15261  ![PR test](../tools_eval_lmtad/Beijing_abnormal_2/pr_curve_test.png) ![Dist test](../tools_eval_lmtad/Beijing_abnormal_2/score_distributions_test.png)

  - `porto_hoser_abnormal_2`:
    - train: precision=0.1407, recall=0.1227, AP=0.14686  ![PR train](../tools_eval_lmtad/porto_hoser_abnormal_2/pr_curve_train.png) ![Dist train](../tools_eval_lmtad/porto_hoser_abnormal_2/score_distributions_train.png)
    - val:   precision=0.2039, recall=0.1826, AP=0.17913  ![PR val](../tools_eval_lmtad/porto_hoser_abnormal_2/pr_curve_val.png) ![Dist val](../tools_eval_lmtad/porto_hoser_abnormal_2/score_distributions_val.png)
    - test:  precision=0.1527, recall=0.1342, AP=0.14943  ![PR test](../tools_eval_lmtad/porto_hoser_abnormal_2/pr_curve_test.png) ![Dist test](../tools_eval_lmtad/porto_hoser_abnormal_2/score_distributions_test.png)

---

## Baseline-calibrated threshold check (train)

The 95th-percentile rule used by `simple_evaluate_with_lmtad.evaluate_trajectories_direct()` is computed **within** the evaluated set, so it cannot answer the question “how abnormal is this dataset relative to a normal baseline?”.

To get a baseline-comparable number, we compute the threshold on the **normal** dataset and apply that fixed threshold to the **abnormal** dataset using the saved evaluation outputs:

- Script: `tools/analyze_lmtad_baseline_threshold.py`

### Beijing (baseline = `Beijing`, target = `Beijing_abnormal_2`)

| Baseline quantile | Threshold (from baseline) | Baseline outlier rate | Target outlier rate |
|---:|---:|---:|---:|
| 0.95 | 12.974031 | 0.0500 | 0.0539 |
| 0.99 | 13.286913 | 0.0100 | 0.0107 |

### Porto (baseline = `porto_hoser`, target = `porto_hoser_abnormal_2`)

| Baseline quantile | Threshold (from baseline) | Baseline outlier rate | Target outlier rate |
|---:|---:|---:|---:|
| 0.95 | 11.633315 | 0.0500 | 0.0531 |
| 0.99 | 12.134086 | 0.0100 | 0.0082 |

Interpretation:
- The injected datasets do shift scores upward slightly, but the shift is small: applying a baseline 95th-percentile threshold increases outlier rate from 5.00% → ~5.3–5.4%.
- This is consistent with the low AP (~0.15) reported by the PR analysis: the score distributions for injected vs non-injected overlap heavily.

---

## Reference-run comparison (LM-TAD native eval)

We also have LM‑TAD’s own evaluation runs (native pipeline) under `results/LMTAD/.../eval/`.
These runs show *very strong* separation of injected outliers using the same model family/checkpoints.

- Script: `tools/summarize_lmtad_reference_eval.py`

| Dataset | Run | Non-outlier mean | Route-switch mean | Detour mean | AP | PR-AUC | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| beijing_hoser_reference | run_20250928_202718 | 0.5325 | 3.5049 | 5.0796 | 0.7162 | 0.9654 | 0.9966 |
| porto_hoser | run_20251010_212829 | 0.3822 | 7.0265 | 8.4132 | 0.8366 | 0.9999 | 0.7571 |

Important observation: the score *scale* differs dramatically between the reference eval and the HOSER-side on-the-fly evaluation.

- Reference non-outlier means are ~0.38–0.53.
- HOSER-side `tools_eval_lmtad/*` non-outlier means are ~10–12.

That mismatch strongly suggests our current HOSER→LM‑TAD conversion/mapping in `tools/evaluate_dataset_with_lmtad.py` is not aligned with the tokenization/boundaries used in LM‑TAD’s native pipeline, which can destroy separability even if the model itself is good.

---

## Clamp diagnostic (train)

`simple_evaluate_with_lmtad.evaluate_trajectories_direct()` clamps per-step probabilities using `min_prob=1e-6`, which corresponds to a maximum possible log-perplexity of $-\log(10^{-6}) \approx 13.8155$.

We checked how often trajectories hit this clamp exactly (using the saved `evaluation_results.json`):

- `Beijing` (train): 3 / 6279 (~0.0478%) at 13.8155
- `Beijing_abnormal_2` (train): 4 / 7203 (~0.0555%) at 13.8155
- `porto_hoser` (train): 0 / 4796 (0.0%) at 13.8155
- `porto_hoser_abnormal_2` (train): 0 / 5520 (0.0%) at 13.8155

So “everything is clamped” is *not* the main driver of poor performance here; the overlap is happening even without widespread hard-clipping.

**Recommended next steps**

1. Validate and align HOSER→LM‑TAD conversion with the reference pipeline.
  - Goal: bring “normal” score scale closer to reference runs and recover separability.
  - Concretely: reuse the boundary/centroid logic from `tools/convert_to_lmtad_format.py` (which is explicitly designed to match LM‑TAD preprocessing) rather than deriving boundaries from centroids in `tools/evaluate_dataset_with_lmtad.py`.
2. Keep the baseline-comparable reporting.
  - Use baseline-calibrated thresholds (computed on normal, applied to abnormal) for a stable, interpretable “outlier rate vs baseline”.
  - Script is now in-repo: `tools/analyze_lmtad_baseline_threshold.py`.
3. Only after mapping alignment: reassess anomaly strength.
  - If AP remains low even with aligned conversion, then the generator needs to enforce “token-space abnormality” (e.g., guarantee large grid-token jumps or connectivity violations).
4. Keep using AP/PR curves as the main metric.
  - Percentile-thresholded outlier rates are operating-point choices; AP/PR summarize separability without committing to a threshold.

**Where I put things**

- Per-dataset evaluation outputs (JSON / PNG / CSV) are in `tools_eval_lmtad/<dataset>/`.
- Injection logs (exact indices and injected info) were written to the generated CSV directory as `<split>.csv.injected_indices.jsonl` (for example `data/Beijing_abnormal_2/train.csv.injected_indices.jsonl`) — use these for exact alignment if you want to compute synthetic ground-truth labels for different samplers.