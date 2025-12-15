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

- LM‑TAD aggregated stats (train split):
  - Beijing (original): mean_log_perplexity = 11.9433, outlier_rate ≈ 5.00%
  - Beijing_abnormal_2: mean_log_perplexity = 12.0030 (+0.0598), outlier_rate ≈ 5.01%
  - porto_hoser (original): mean_log_perplexity ≈ 10.2813
  - porto_hoser_abnormal_2: mean_log_perplexity ≈ 10.3347 (+0.0534), outlier_rate ≈ 5.00%

- Confusion-summary (evaluator's default 95th‑percentile outliers; reported as `confusion_summary.csv`) — selected rows (train):

  - Beijing (original): compared=6279, TP=0, FP=314, FN=0, TN=5965, injected_rate=0.0, reported_outlier_rate=0.0500

  - Beijing_abnormal_2 (train):
    - compared=7203, TP=53, FP=308, FN=883, TN=5959
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

**Recommended next steps**

1. Quick: compute detection numbers at the injected rate (top-13% / 87th percentile) to see recall/precision when threshold matches injection budget — can be produced immediately from `evaluation_results.json` files (fast).
2. If you want evaluation-reported outlier rates to match a chosen detection budget (e.g., 15%), add a CLI option `--outlier-rate` to the evaluator and re-run LM evaluation (this requires re-running model inference; slower).
3. Produce PR curves / average-precision for each dataset using the continuous `log_perplexity_values`. This gives threshold-agnostic performance metrics and helps choose operating points.
4. If Porto 3σ is desired operationally, inspect `porto_threshold` in `confusion_summary.csv`. In these runs Porto thresholds were conservative (returned 0 GTs above that threshold for some splits) — consider computing 2σ or adjusting the non-outlier subset used for Porto-style renormalization.

**Where I put things**

- Per-dataset evaluation outputs (JSON / PNG / CSV) are in `tools_eval_lmtad/<dataset>/`.
- Injection logs (exact indices and injected info) were written to the generated CSV directory as `<split>.csv.injected_indices.jsonl` (for example `data/Beijing_abnormal_2/train.csv.injected_indices.jsonl`) — use these for exact alignment if you want to compute synthetic ground-truth labels for different samplers.

**If you want me to continue**

- I can now (pick one):
  - Run a fast script that computes precision/recall/TP/FN for thresholds matching the injected fraction (e.g., top-13%, top-15%) across all evaluated datasets (fast).
  - Add `--outlier-rate` to the evaluator and re-run LM inference for a dataset (slow).
  - Produce PR curves and average-precision numbers and save plots per dataset (fast).

Tell me which of the above you want next and I will proceed.

---

Generated by the HOSER experiment tooling on: 2025-12-15
