# Abnormality Injection & Detection Report

Date: 2025-12-15

This report summarizes the experiments where we injected ~15% HOSER-format abnormalities into existing datasets, evaluated them with the LM‑TAD teacher, and ran the confusion / rule-comparison analysis. It collects the key numbers, embedded diagnostic plots produced by the evaluation tooling, and the main observations and next steps.

## TL;DR

- Score scale and separability are now strong after conversion alignment.
- Best default operating-point method: **baseline-calibrated quantile** with `q=0.95` (≈5% baseline FPR).
- If you want a tighter alert budget: use `q=0.99` (≈1% baseline FPR).
- Across methods (quantile / zscore / mad_z), **AUROC/AP are essentially unchanged** because they evaluate ranking; the main difference is how each method maps “one knob” (q or z) to an actual baseline FPR.

## Inputs

**Datasets evaluated**
- `Beijing` (baseline)
- `Beijing_abnormal_2` (injected; requested rate 0.15)
- `porto_hoser` (baseline)
- `porto_hoser_abnormal_2` (injected; requested rate 0.15)

**Injection bookkeeping (train split, full dataset)**
- `data/Beijing_abnormal_2/train.csv.injected_indices.jsonl`: 94,253 injected entries
- `data/Beijing_abnormal_2/train.csv`: 723,622 total rows
- Observed injection fraction ≈ 94,253 / 723,622 = 0.12995 (~13.0%)

## Where to find outputs

- Per-dataset evaluation JSON: `tools_eval_lmtad/<dataset>/evaluation_results.json`
- Sampled CSV used for evaluation (labels come from `abnormality_info`): `tools_eval_lmtad/<dataset>/<split>_sampled.csv`
- Baseline calibration file (written by `tools/evaluate_dataset_with_lmtad.py --write-baseline`): `tools_eval_lmtad/<baseline>/baseline_eval.json`
- Baseline-threshold multi-method summaries + plots:
  - `tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/summary.md`
  - `tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/summary.md`
  - Plots: `tools_eval_lmtad/_baseline_threshold_plots/<pair>/<split>/<method>/*_hist.png` and `*_pr.png`

## 1) LM-TAD aggregated stats (sampled splits)

Score scale is now in the expected range (~0.4–1.3 means on these splits).

| Dataset | Split | n | mean_log_perplexity | outlier_rate (within-split 95th) |
|---|---:|---:|---:|---:|
| `Beijing` | train | 6279 | 0.805477 | 0.05001 |
| `Beijing` | val | 901 | 0.810412 | 0.04994 |
| `Beijing` | test | 1775 | 0.816866 | 0.05014 |
| `Beijing_abnormal_2` | train | 7203 | 1.330323 | 0.05012 |
| `Beijing_abnormal_2` | val | 1041 | 1.335913 | 0.04995 |
| `Beijing_abnormal_2` | test | 2038 | 1.323207 | 0.05005 |
| `porto_hoser` | train | 4796 | 0.446920 | 0.05004 |
| `porto_hoser` | val | 689 | 0.431454 | 0.05080 |
| `porto_hoser` | test | 1348 | 0.441382 | 0.05045 |
| `porto_hoser_abnormal_2` | train | 5520 | 0.815243 | 0.05000 |
| `porto_hoser_abnormal_2` | val | 791 | 0.768835 | 0.05057 |
| `porto_hoser_abnormal_2` | test | 1561 | 0.825648 | 0.04997 |

## 2) Default evaluator rule (within-split 95th percentile)

The default evaluator marks outliers by thresholding at the 95th percentile *within the evaluated split*:

```python
threshold = np.percentile(all_outlier_scores, 95)
outliers = score > threshold
```

So the outlier rate stays ≈5% by construction.

### Confusion vs injected labels (sampled splits)

These numbers use the evaluator's within-split thresholding rule (top 5% of each evaluated split), compared against `abnormality_info != normal` in the corresponding `*_sampled.csv`.

| Dataset | Split | n | threshold (95th) | outlier_rate | injected_rate | precision | recall | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Beijing_abnormal_2` | train | 7203 | 5.055740 | 0.0501 | 0.1283 | 0.9806 | 0.3831 | 354 | 7 | 570 | 6272 |
| `Beijing_abnormal_2` | val | 1041 | 4.992525 | 0.0500 | 0.1287 | 1.0000 | 0.3881 | 52 | 0 | 82 | 907 |
| `Beijing_abnormal_2` | test | 2038 | 5.183321 | 0.0500 | 0.1266 | 1.0000 | 0.3953 | 102 | 0 | 156 | 1780 |
| `porto_hoser_abnormal_2` | train | 5520 | 3.669760 | 0.0500 | 0.1223 | 0.9964 | 0.4074 | 275 | 1 | 400 | 4844 |
| `porto_hoser_abnormal_2` | val | 791 | 3.198837 | 0.0506 | 0.1239 | 1.0000 | 0.4082 | 40 | 0 | 58 | 693 |
| `porto_hoser_abnormal_2` | test | 1561 | 3.600372 | 0.0500 | 0.1268 | 1.0000 | 0.3939 | 78 | 0 | 120 | 1363 |

**What this tells us**

- Mean log-perplexity shifts upward on injected datasets.
- Precision is very high here, but recall is capped because only ~5% can be flagged.

---

## 3) Baseline-calibrated thresholding (recommended)

This answers: “How abnormal is this dataset relative to a normal baseline?”

- Compute threshold from the **baseline** split.
- Apply the same threshold to the **target** split.

### Multi-method testing summary (train + val)

Methods evaluated:
- `quantile`: threshold = baseline quantile q
- `zscore`: threshold = mean + z * std
- `mad_z`: threshold = median + z * 1.4826 * MAD

**Findings**
- Ranking metrics are consistently strong across methods:
  - Beijing: val AUROC = 0.9804, val AP = 0.9057
  - Porto: val AUROC = 0.9966, val AP = 0.9792
- The main practical difference is **baseline FPR control**:
  - `quantile` directly targets a baseline FPR budget (e.g., `q=0.95` ≈ 5%).
  - `zscore`/`mad_z` produce baseline FPRs that can drift by split and dataset.

**Recommended defaults**
- Default: `quantile q=0.95` (≈5% baseline FPR) for a balanced precision/recall operating point.
- Tighter budget: `quantile q=0.99` (≈1% baseline FPR) for higher precision.
- If you want an approximate “mid-budget” without choosing q: `zscore z=3` often lands near a few-percent baseline FPR, but it’s less directly controllable.

**Uncertainty (bootstrap CIs)**

Bootstrap 95% confidence intervals are included in the per-pair summaries (generated with `--bootstrap 1000`, `--seed 0`):
- `tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/summary.md`
- `tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/summary.md`

- Script: `tools/analyze_lmtad_baseline_threshold.py`
- Full outputs (train + val):
  - `tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/summary.md`
  - `tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/summary.md`

### Recommended operating points (val)

These are compact “best-of” rows; full grids (train + val) are in the `summary.md` files listed above.

#### Beijing vs `Beijing_abnormal_2`

| Method | Setting | Baseline outlier rate | Target outlier rate | Precision | Recall | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| quantile | q=0.95 | 0.0499 | 0.1527 | 0.7044 | 0.8358 | 0.9804 | 0.9057 |
| quantile | q=0.99 | 0.0100 | 0.0903 | 0.9468 | 0.6642 | 0.9804 | 0.9057 |
| zscore | z=3 | 0.0222 | 0.1153 | 0.8417 | 0.7537 | 0.9804 | 0.9057 |
| mad_z | z=3 | 0.1121 | 0.2392 | 0.5301 | 0.9851 | 0.9804 | 0.9057 |

**Val plots (examples)**

![Beijing val quantile q=0.95 histogram](../tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/val/quantile/quantile_q0p95_thr2p0909_hist.png)

![Beijing val quantile q=0.95 PR](../tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/val/quantile/quantile_q0p95_thr2p0909_pr.png)

![Beijing val zscore z=3 histogram](../tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/val/zscore/zscore_z3_thr2p608605_hist.png)

![Beijing val zscore z=3 PR](../tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/val/zscore/zscore_z3_thr2p608605_pr.png)

![Beijing val mad_z z=3 histogram](../tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/val/mad_z/mad_z_z3_thr1p451509_hist.png)

![Beijing val mad_z z=3 PR](../tools_eval_lmtad/_baseline_threshold_plots/Beijing_vs_abnormal2/val/mad_z/mad_z_z3_thr1p451509_pr.png)

#### Porto vs `porto_hoser_abnormal_2`

| Method | Setting | Baseline outlier rate | Target outlier rate | Precision | Recall | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| quantile | q=0.95 | 0.0508 | 0.1631 | 0.7442 | 0.9796 | 0.9966 | 0.9792 |
| quantile | q=0.99 | 0.0102 | 0.1239 | 0.9082 | 0.9082 | 0.9966 | 0.9792 |
| zscore | z=3 | 0.0247 | 0.1391 | 0.8636 | 0.9694 | 0.9966 | 0.9792 |
| mad_z | z=3 | 0.0668 | 0.1719 | 0.7132 | 0.9898 | 0.9966 | 0.9792 |

**Val plots (examples)**

![Porto val quantile q=0.95 histogram](../tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/val/quantile/quantile_q0p95_thr0p965512_hist.png)

![Porto val quantile q=0.95 PR](../tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/val/quantile/quantile_q0p95_thr0p965512_pr.png)

![Porto val zscore z=3 histogram](../tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/val/zscore/zscore_z3_thr1p217137_hist.png)

![Porto val zscore z=3 PR](../tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/val/zscore/zscore_z3_thr1p217137_pr.png)

![Porto val mad_z z=3 histogram](../tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/val/mad_z/mad_z_z3_thr0p829314_hist.png)

![Porto val mad_z z=3 PR](../tools_eval_lmtad/_baseline_threshold_plots/porto_vs_abnormal2/val/mad_z/mad_z_z3_thr0p829314_pr.png)

---

## 4) Quick baseline-calibrated run outputs (train + val only)

These numbers are from `tools/evaluate_dataset_with_lmtad.py` using the baseline written by `--write-baseline` and then applying baseline-calibrated quantile outliers (`q=0.95`).

Note: test split intentionally ignored.

| Dataset | Split | n | outlier_method | outlier_rate (baseline q=0.95) |
|---|---:|---:|---|---:|
| `porto_hoser_abnormal_2` | train | 5520 | baseline_quantile | 0.1540 |
| `porto_hoser_abnormal_2` | val | 791 | baseline_quantile | 0.1631 |

**Porto plots (from the latest run)**

![Porto abnormality overview](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_abnormality.png)

![Porto boxplot](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_boxplot.png)

![Porto train plot](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_train.png)

![Porto train roc](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_train_roc.png)

![Porto train pr](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_train_pr.png)

![Porto val plot](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_val.png)

![Porto val roc](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_val_roc.png)

![Porto val pr](../tools_eval_lmtad/porto_hoser_abnormal_2/lmtad_eval_val_pr.png)

---

**Beijing plots (from the latest run)**

![Beijing abnormality overview](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_abnormality.png)

![Beijing boxplot](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_boxplot.png)

![Beijing train plot](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_train.png)

![Beijing train roc](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_train_roc.png)

![Beijing train pr](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_train_pr.png)

![Beijing val plot](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_val.png)

![Beijing val roc](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_val_roc.png)

![Beijing val pr](../tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_val_pr.png)

---