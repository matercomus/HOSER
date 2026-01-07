# LM-TAD abnormality detection benchmarks — Beijing (self-contained report)

Date: 2026-01-07

## 1. Purpose and reader assumptions

This report is written for a reader with minimal context.

Goal:

- Measure how well **LM-TAD** (a pretrained trajectory model) can detect trajectory abnormalities using its anomaly score (log-perplexity).
- Compare abnormality mechanisms **in isolation** (detour-only vs route_switch-only).
- Explain how different **decision boundaries** (threshold choices) change results.

Why we did this:

- Earlier benchmarking for route switching was not informative because the dataset/split contained too few abnormal rows to evaluate properly.
- We needed a controlled setup where each mechanism has enough examples and where comparisons are fair.

## 2. What is being detected (abnormality mechanisms)

We focus on two abnormality mechanisms:

- **Detour**: trajectories that deviate (take a longer/off-route path) relative to the expected behavior.
- **Route_switch**: trajectories that plausibly travel between the same endpoints but take an alternate route (a more subtle change than a detour).

The working hypothesis is that detours are more “distribution-shifting” (easier to detect), while route_switch can remain plausible and therefore may be harder.

## 3. The LM-TAD score and what the metrics mean

LM-TAD outputs a **continuous anomaly score** for each trajectory:

- Higher log-perplexity ⇒ less likely under the model ⇒ more anomalous.

We report two kinds of metrics:

1. **Ranking quality** (threshold-free):
   - AUROC
   - AUPRC

   These answer: “Do abnormal examples tend to have higher scores than normal examples?”

2. **Operating-point metrics** (threshold-dependent):
   - precision, recall
   - false positive rate (FPR)
   - flag rate
   - confusion matrix counts (TP/FP/TN/FN)

   These answer: “If we choose a particular boundary, how many alerts do we get and how many are correct?”

Important distinction:

- Changing the decision boundary changes operating-point metrics (TP/FP/TN/FN),
- but does **not** change AUROC/AUPRC (which summarize ranking across all thresholds).

## 4. Evaluation design: balanced sets and why they matter

To make comparisons reliable, evaluations are performed on **balanced sets**:

- Include all abnormal rows available in the split.
- Sample normal rows to match the abnormal count (1 normal per abnormal).
- Use length bucketing to better match trajectory length distributions.

Balancing configuration that was sanity-checked:

| Setting | Value |
|---|---:|
| normal_per_abnormal | 1 |
| length_bucket | 5 |
| seed | 42 |
| allow_replacement | false |
| output rows | 3,446 |
| abnormal rows | 1,723 |
| normal rows | 1,723 |

This sanity check confirms the balancing pipeline produces deterministic, exactly balanced datasets under the chosen parameters.

## 5. Earlier benchmark (Jan 03): what happened and what we learned

### 5.1 Summary table

The Jan 03 benchmark evaluated three targets (train split):

| Dataset | N | Pos | AUROC | AUPRC | Notes |
|---|---:|---:|---:|---:|---|
| Beijing_abnormal_3_detectable | 212 | 106 | 0.9475 | 0.9302 | Strong separation |
| Beijing_abnormal_3_detectable_dr | 3,446 | 1,723 | 0.8146 | 0.7818 | Moderate separation |
| Beijing_abnormal_3_detectable_route_switch | 0 | 0 | NA | NA | Empty balanced set (no abnormal rows in that split) |

Key lesson:

- The route_switch result was **not** evidence that LM-TAD fails.
- It was a **data coverage problem**: with zero abnormal rows, there is nothing to score/evaluate.

### 5.2 Visual evidence (score distributions)

Beijing_abnormal_3_detectable:

![](figures/pertype_lmtad_report_20260107/Beijing_abnormal_3_detectable.score_hist.png)

![](figures/pertype_lmtad_report_20260107/Beijing_abnormal_3_detectable.score_by_type_box.png)

Beijing_abnormal_3_detectable_dr:

![](figures/pertype_lmtad_report_20260107/Beijing_abnormal_3_detectable_dr.score_hist.png)

![](figures/pertype_lmtad_report_20260107/Beijing_abnormal_3_detectable_dr.score_by_type_box.png)

### 5.3 Decision boundary behavior: an important phenomenon

The earlier benchmark also showed a general phenomenon:

- Even with strong separation, an extremely strict operating point (e.g., very high quantile) can yield very low or even zero recall.

This motivated the more careful analysis of decision boundaries described below.

Concrete examples from the Jan 03 benchmark (headline operating points):

Beijing_abnormal_3_detectable:

| q | Method | Flag rate | Recall | Precision | FPR | TP | FP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.95 | baseline-quantile | 0.028 | 0.057 | 1.000 | 0.000 | 6 | 0 |
| 0.95 | top-k matched | 0.052 | 0.104 | 1.000 | 0.000 | 11 | 0 |
| 0.99 | baseline-quantile | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| 0.99 | top-k matched | 0.014 | 0.028 | 1.000 | 0.000 | 3 | 0 |

Beijing_abnormal_3_detectable_dr:

| q | Method | Flag rate | Recall | Precision | FPR | TP | FP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.95 | baseline-quantile | 0.050 | 0.088 | 0.879 | 0.012 | 152 | 21 |
| 0.95 | top-k matched | 0.050 | 0.088 | 0.879 | 0.012 | 152 | 21 |
| 0.99 | baseline-quantile | 0.010 | 0.017 | 0.853 | 0.003 | 29 | 5 |
| 0.99 | top-k matched | 0.010 | 0.017 | 0.829 | 0.003 | 29 | 6 |

## 6. Per-type isolation benchmark (Jan 07, validation split): primary findings

### 6.1 Why per-type isolation

Per-type isolation addresses two problems at once:

1. **Coverage**: each mechanism has enough abnormal examples.
2. **Interpretability**: the measured performance corresponds to a specific mechanism rather than a mixture of effects.

### 6.2 Overall ranking-quality results (balanced validation)

| Dataset | N | Pos | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Beijing_per_type_detour | 26,972 | 13,486 | 0.8468 | 0.8072 |
| Beijing_per_type_route_switch | 26,972 | 13,486 | 0.6743 | 0.6505 |

Interpretation:

- Detour is substantially more detectable than route_switch by AUROC/AUPRC.
- Route_switch still shows separation above random, but much weaker.

### 6.3 Visual evidence (score distributions and ROC/PR)

Detour:

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.score_hist.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.score_by_type_box.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.roc.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.pr.png)

Route_switch:

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.score_hist.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.score_by_type_box.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.roc.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.pr.png)

## 7. Decision boundaries: how we chose operating points

We compare two decision policies:

- **baseline_quantile**: choose a score threshold corresponding to baseline quantile `q`.
  - This aims to control false positives relative to a baseline calibration distribution.
  - The resulting flag rate on the target set is not guaranteed to be exactly $1-q$.

- **topk_matched**: choose $k = \lceil (1-q)N \rceil$ and flag the top-k scored trajectories.
  - This controls alert volume exactly.

You can think of these as answering two different operational questions:

- baseline_quantile: “What if I want a threshold tied to baseline rarity?”
- topk_matched: “What if I can only handle k alerts?”

## 8. Boundary-dependent findings (Jan 07 validation)

This section contains the full boundary grid results for both per-type datasets.

### 8.1 Detour: boundary behavior

Metric-vs-q and confusion-grid summaries:

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.baseline_quantile_metrics_vs_q.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.topk_matched_metrics_vs_q.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.baseline_quantile_confusion_grid.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_detour.topk_matched_confusion_grid.png)

Full per-q tables (detour):

Baseline-quantile thresholds:

| q | thr | flag_rate | recall | precision | FPR | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.587692 | 0.672 | 0.939 | 0.698 | 0.406 | 12666 | 5471 | 820 | 8015 |
| 0.60 | 0.676987 | 0.586 | 0.870 | 0.742 | 0.302 | 11731 | 4069 | 1755 | 9417 |
| 0.70 | 0.826965 | 0.458 | 0.716 | 0.782 | 0.200 | 9655 | 2696 | 3831 | 10790 |
| 0.80 | 1.087722 | 0.298 | 0.479 | 0.805 | 0.116 | 6465 | 1567 | 7021 | 11919 |
| 0.85 | 1.289099 | 0.221 | 0.363 | 0.820 | 0.079 | 4889 | 1071 | 8597 | 12415 |
| 0.90 | 1.551051 | 0.152 | 0.257 | 0.845 | 0.047 | 3468 | 635 | 10018 | 12851 |
| 0.92 | 1.723711 | 0.120 | 0.207 | 0.866 | 0.032 | 2798 | 432 | 10688 | 13054 |
| 0.94 | 1.890461 | 0.096 | 0.168 | 0.878 | 0.023 | 2261 | 315 | 11225 | 13171 |
| 0.95 | 2.019887 | 0.081 | 0.143 | 0.885 | 0.019 | 1935 | 252 | 11551 | 13234 |
| 0.96 | 2.168889 | 0.066 | 0.118 | 0.894 | 0.014 | 1588 | 189 | 11898 | 13297 |
| 0.97 | 2.381984 | 0.050 | 0.090 | 0.907 | 0.009 | 1212 | 124 | 12274 | 13362 |
| 0.98 | 2.625441 | 0.036 | 0.065 | 0.919 | 0.006 | 880 | 78 | 12606 | 13408 |
| 0.99 | 3.176166 | 0.016 | 0.030 | 0.933 | 0.002 | 405 | 29 | 13081 | 13457 |

Top-k matched to q:

| q_match | k | cutoff | flag_rate | recall | precision | FPR | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 13486 | 0.775226 | 0.500 | 0.771 | 0.771 | 0.229 | 10391 | 3095 | 3095 | 10391 |
| 0.60 | 10789 | 0.907518 | 0.400 | 0.633 | 0.791 | 0.167 | 8534 | 2255 | 4952 | 11231 |
| 0.70 | 8092 | 1.083197 | 0.300 | 0.483 | 0.804 | 0.117 | 6510 | 1582 | 6976 | 11904 |
| 0.80 | 5395 | 1.356865 | 0.200 | 0.330 | 0.825 | 0.070 | 4449 | 946 | 9037 | 12540 |
| 0.85 | 4046 | 1.562172 | 0.150 | 0.254 | 0.846 | 0.046 | 3423 | 623 | 10063 | 12863 |
| 0.90 | 2698 | 1.852818 | 0.100 | 0.175 | 0.876 | 0.025 | 2364 | 334 | 11122 | 13152 |
| 0.92 | 2158 | 2.028170 | 0.080 | 0.142 | 0.885 | 0.018 | 1909 | 249 | 11577 | 13237 |
| 0.94 | 1619 | 2.235853 | 0.060 | 0.108 | 0.899 | 0.012 | 1455 | 164 | 12031 | 13322 |
| 0.95 | 1349 | 2.372380 | 0.050 | 0.091 | 0.908 | 0.009 | 1225 | 124 | 12261 | 13362 |
| 0.96 | 1079 | 2.545624 | 0.040 | 0.074 | 0.920 | 0.006 | 993 | 86 | 12493 | 13400 |
| 0.97 | 810 | 2.734490 | 0.030 | 0.055 | 0.919 | 0.005 | 744 | 66 | 12742 | 13420 |
| 0.98 | 540 | 3.016274 | 0.020 | 0.037 | 0.933 | 0.003 | 504 | 36 | 12982 | 13450 |
| 0.99 | 270 | 3.508024 | 0.010 | 0.019 | 0.926 | 0.001 | 250 | 20 | 13236 | 13466 |

### 8.2 Route_switch: boundary behavior

Metric-vs-q and confusion-grid summaries:

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.baseline_quantile_metrics_vs_q.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.topk_matched_metrics_vs_q.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.baseline_quantile_confusion_grid.png)

![](figures/pertype_lmtad_report_20260107/Beijing_per_type_route_switch.topk_matched_confusion_grid.png)

Full per-q tables (route_switch):

Baseline-quantile thresholds:

| q | thr | flag_rate | recall | precision | FPR | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.587692 | 0.541 | 0.674 | 0.623 | 0.408 | 9094 | 5498 | 4392 | 7988 |
| 0.60 | 0.676987 | 0.432 | 0.561 | 0.649 | 0.303 | 7565 | 4088 | 5921 | 9398 |
| 0.70 | 0.826965 | 0.306 | 0.410 | 0.669 | 0.203 | 5531 | 2732 | 7955 | 10754 |
| 0.80 | 1.087722 | 0.187 | 0.254 | 0.681 | 0.119 | 3431 | 1607 | 10055 | 11879 |
| 0.85 | 1.289099 | 0.133 | 0.185 | 0.696 | 0.081 | 2495 | 1092 | 10991 | 12394 |
| 0.90 | 1.551051 | 0.088 | 0.127 | 0.723 | 0.049 | 1714 | 658 | 11772 | 12828 |
| 0.92 | 1.723711 | 0.068 | 0.100 | 0.738 | 0.036 | 1351 | 480 | 12135 | 13006 |
| 0.94 | 1.890461 | 0.053 | 0.079 | 0.747 | 0.027 | 1070 | 362 | 12416 | 13124 |
| 0.95 | 2.019887 | 0.044 | 0.065 | 0.748 | 0.022 | 881 | 297 | 12605 | 13189 |
| 0.96 | 2.168889 | 0.036 | 0.056 | 0.772 | 0.017 | 759 | 224 | 12727 | 13262 |
| 0.97 | 2.381984 | 0.027 | 0.043 | 0.791 | 0.011 | 579 | 153 | 12907 | 13333 |
| 0.98 | 2.625441 | 0.020 | 0.032 | 0.817 | 0.007 | 432 | 97 | 13054 | 13389 |
| 0.99 | 3.176166 | 0.010 | 0.016 | 0.838 | 0.003 | 222 | 43 | 13264 | 13443 |

Top-k matched to q:

| q_match | k | cutoff | flag_rate | recall | precision | FPR | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 13486 | 0.618010 | 0.500 | 0.632 | 0.632 | 0.368 | 8520 | 4966 | 4966 | 8520 |
| 0.60 | 10789 | 0.709302 | 0.400 | 0.524 | 0.655 | 0.276 | 7067 | 3722 | 6419 | 9764 |
| 0.70 | 8092 | 0.836733 | 0.300 | 0.402 | 0.670 | 0.198 | 5422 | 2670 | 8064 | 10816 |
| 0.80 | 5395 | 1.047875 | 0.200 | 0.273 | 0.681 | 0.128 | 3676 | 1722 | 9810 | 11764 |
| 0.85 | 4046 | 1.215375 | 0.150 | 0.207 | 0.688 | 0.094 | 2785 | 1261 | 10701 | 12225 |
| 0.90 | 2698 | 1.462352 | 0.100 | 0.142 | 0.711 | 0.058 | 1917 | 781 | 11569 | 12705 |
| 0.92 | 2158 | 1.605828 | 0.080 | 0.116 | 0.727 | 0.044 | 1568 | 590 | 11918 | 12896 |
| 0.94 | 1619 | 1.806131 | 0.060 | 0.090 | 0.747 | 0.030 | 1209 | 410 | 12277 | 13076 |
| 0.95 | 1349 | 1.923906 | 0.050 | 0.075 | 0.749 | 0.025 | 1010 | 339 | 12476 | 13147 |
| 0.96 | 1079 | 2.088273 | 0.040 | 0.061 | 0.757 | 0.019 | 817 | 262 | 12669 | 13224 |
| 0.97 | 810 | 2.308275 | 0.030 | 0.047 | 0.785 | 0.013 | 636 | 174 | 12850 | 13312 |
| 0.98 | 540 | 2.615308 | 0.020 | 0.033 | 0.819 | 0.007 | 442 | 98 | 13044 | 13388 |
| 0.99 | 270 | 3.159629 | 0.010 | 0.017 | 0.837 | 0.003 | 226 | 44 | 13260 | 13442 |

## 9. Per-type train benchmark status

A corresponding training-split run was launched on Jan 07 and is still running at the time of writing, so this report is limited to validation-split results.

## 10. Conclusions

1. Data coverage matters: a benchmark can be invalid if the chosen split contains too few abnormal rows (route_switch was empty in the earlier run).
2. Per-type isolation fixes comparability and yields interpretable results.
3. Detour is substantially more detectable than route_switch by AUROC/AUPRC on balanced validation.
4. Decision boundaries must be treated as operating-point choices; selecting `q` or `k` is effectively selecting an alert budget and false-positive tolerance.

Practical recommendation:

- Use AUROC/AUPRC to decide whether a mechanism is detectable in principle.
- Use the full boundary tables and confusion grids above to pick an operating point aligned with the operational constraint (fixed FPR vs fixed alert volume).
