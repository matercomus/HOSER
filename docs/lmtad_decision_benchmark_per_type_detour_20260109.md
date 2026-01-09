# LM-TAD decision benchmark — per-type detour (2026-01-09)

## Abstract

We evaluate LM-TAD as an abnormal-trajectory scoring function on two per-type detour datasets (Beijing, Porto) using a 1% sampled subset of the `train` split.
We report both (i) **score separation** (AUROC/AUPRC; threshold-free) and (ii) **decision performance** across a sweep of alert-volume operating points (quantile $q$).

## Key takeaways (high level)

- LM-TAD shows **moderate separability** on both datasets (AUROC ~0.77–0.80; AUPRC ~0.27–0.31 at ~12% prevalence), i.e., useful ranking signal but substantial overlap.
- The teacher’s **top-score tail is enriched but not “clean”**: Precision@top1% is ~0.30 (Beijing) / ~0.38 (Porto), about **2.4–3.0×** the base prevalence.
- That enrichment comes with **low coverage at tiny review budgets**: Recall@top1% is only ~2–3% (you miss ~97–98% of abnormals when reviewing 1%).
- Moving to **larger review budgets** increases coverage materially: at ~10% review, Recall@top10% is ~0.21 (Beijing) / ~0.29 (Porto) while Precision@top10% remains ~0.27 / ~0.36.
- In the q-sweep, this appears as the expected trade-off: $q=0.99$ (~1% flagged) gives high precision/very low recall; $q=0.90$–$0.95$ yields much higher recall at higher FP volume.
- Baseline-quantile is generally **slightly higher recall** at fixed $q$; top-k matched often **tracks similar precision** but can flag slightly fewer trajectories at the same nominal $q$.

This report is intended to be readable; for reproducibility details and full artifacts, see the References section.

## Methods

### Experimental setup

- Data: `data/_per_type/Beijing_per_type_detour` and `data/_per_type/porto_hoser_per_type_detour`
- Split: `train`
- Sampling: Bernoulli sampling at `sample_frac=0.01` with `sample_seed=42`
- Decision sweep: quantiles $q \in \{0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99\}$
- Decision rules:
  - **Baseline-quantile thresholding**: threshold calibrated on baseline at each $q$
  - **Top-k matched**: choose k to match baseline’s flagged volume at each $q$

## Results

### Results overview (1% sample)

| Dataset | N | Abnormal fraction | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Beijing_per_type_detour | 7206 | 12.68% | 0.7748 | 0.2684 |
| porto_hoser_per_type_detour | 5524 | 12.42% | 0.8006 | 0.3128 |

Interpretation:

- AUROC/AUPRC measure **ranking quality** (how well higher scores correspond to abnormality), independent of a specific threshold.
- The q-sweep evaluates **decision rules** at matched operating points. Larger $q$ means lower alert volume (more conservative).

What this suggests in practice:

- **AUROC ~0.77–0.80** means LM-TAD tends to rank abnormals above normals, but there is still substantial overlap.
- **AUPRC should be read relative to prevalence**: with ~12% abnormality, a random ranker has AUPRC ~0.12. Here AUPRC ~0.27–0.31 indicates meaningful enrichment, but not “near-perfect” separability.
- For operational use, the main question becomes **how much recall you can buy** for a given alert volume (the q-sweep / decision curves).

Important caveats:

- These metrics are computed on a **1% Bernoulli sample** of `train`, not a held-out test set.
- AUPRC is sensitive to the abnormal prevalence (~12% here), so cross-dataset comparison should consider prevalence.

### Teacher separability summary (score as a continuous signal)

This table quantifies whether LM-TAD scores are *systematically higher* for synthetic abnormal trajectories than for normal ones.
Values are point estimates with **95% bootstrap CIs** (stratified resampling, 500 replicates).

| Dataset | N | Prev. | AUROC | Cliff’s δ | Cohen’s d | W1 | Recall@top1% | Recall@top5% | Recall@top10% | Precision@top1% | Precision@top5% | Precision@top10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Beijing_per_type_detour | 7206 | 12.68% | 0.7748 [0.7616, 0.7882] | 0.5496 [0.5231, 0.5764] | 0.8374 [0.7541, 0.9248] | 0.5086 [0.4556, 0.5628] | 0.0241 [0.0164, 0.0328] | 0.1193 [0.1018, 0.1373] | 0.2123 [0.1882, 0.2374] | 0.3014 [0.2055, 0.4110] | 0.3019 [0.2576, 0.3477] | 0.2691 [0.2386, 0.3010] |
| porto_hoser_per_type_detour | 5524 | 12.42% | 0.8006 [0.7844, 0.8159] | 0.6013 [0.5688, 0.6317] | 0.9541 [0.8469, 1.0808] | 0.2800 [0.2493, 0.3108] | 0.0306 [0.0190, 0.0423] | 0.1356 [0.1122, 0.1574] | 0.2915 [0.2638, 0.3273] | 0.3750 [0.2321, 0.5179] | 0.3357 [0.2780, 0.3899] | 0.3617 [0.3273, 0.4060] |

How to read these:

- **AUROC**: probability an abnormal scores higher than a normal.
- **Cliff’s δ**: effect size derived from AUROC ($\delta = 2\,\text{AUROC}-1$); 0 means no separation.
- **Cohen’s d** and **W1**: complementary measures of distribution separation (scale depends on score units).
- **Recall@top-k%**: fraction of abnormals captured by reviewing only the top-scoring k% of trajectories.
- **Precision@top-k%**: abnormal fraction within the reviewed top-scoring k% ("purity" of the tail).

Key findings from the tail metrics (why they matter):

- **Top 1% review is “high purity, low coverage.”** Precision@top1% is ~0.30 (Beijing) / ~0.38 (Porto), which is roughly a **2.4–3.0× enrichment** over the base prevalence (~0.12). But Recall@top1% is only ~0.024–0.031, meaning you capture **~2–3% of all abnormals**.
- **Top 10% review increases coverage more than it degrades purity.** Recall@top10% rises to ~0.21 (Beijing) / ~0.29 (Porto), while Precision@top10% remains ~0.27 (Beijing) / ~0.36 (Porto). This indicates that the teacher signal has a useful ordering, but not a clean separation.

Implication for distillation-style use:

- If you use LM-TAD as a **continuous weighting/teacher signal**, these results support that it contains information, but it is **noisy**: many normal samples still appear in the high-score tail.
- If you instead use it for **hard filtering** (e.g., keep only the top-k%), you’ll trade off “clean tail” versus “enough abnormal mass” very sharply at small k.

Plots (with 95% bootstrap CIs):

![Teacher separability metric summary](../research_runs/_benchmarks/lmtad_teacher_separability_plots_20260109/teacher_separability_metrics.png)

![Teacher separability recall curve](../research_runs/_benchmarks/lmtad_teacher_separability_plots_20260109/teacher_separability_recall_curve.png)

![Teacher separability precision curve](../research_runs/_benchmarks/lmtad_teacher_separability_plots_20260109/teacher_separability_precision_curve.png)

### Per-dataset results (selected operating points)

#### Beijing — `Beijing_per_type_detour` (1% sample)

- Sample size: `N=7206`
- Abnormal fraction in sample: `12.68%` (914 / 7206)
- Baseline-calibrated thresholding (q = 0.99):
  - Baseline threshold: `3.176166`
  - Baseline outlier rate: `1.00%`
  - Target outlier rate: `1.11%`
- Mean log-perplexity (sample): `0.8709`

Interpretation of the operating point:

- At $q=0.99$, the rule is tuned to flag roughly ~1% of trajectories (very low review volume).
- The resulting recall is expected to be low unless abnormal and normal score distributions are strongly separated.

Decision rules @ `q=0.99` (very low alert volume):

- Baseline-quantile: Precision `0.3125`, Recall `0.0274`, F1 `0.0503`, FPR `0.0087` (TP=25, FP=55)
- Top-k matched: k=73, Precision `0.3014`, Recall `0.0241`, F1 `0.0446`, FPR `0.0081` (TP=22, FP=51)

Decision rules at other operating points (from `metrics.json`):

| q | Method | Alert rate | Precision | Recall | F1 | FPR |
|---:|---|---:|---:|---:|---:|---:|
| 0.90 | baseline-quantile | 0.1253 | 0.2780 | 0.2746 | 0.2763 | 0.1036 |
| 0.90 | top-k matched (k=721) | 0.1001 | 0.2691 | 0.2123 | 0.2373 | 0.0838 |
| 0.95 | baseline-quantile | 0.0623 | 0.2940 | 0.1444 | 0.1937 | 0.0504 |
| 0.95 | top-k matched (k=361) | 0.0501 | 0.3019 | 0.1193 | 0.1710 | 0.0401 |

Interpretation (Beijing):

- Moving from $q=0.99 \rightarrow 0.95$ increases recall substantially (from ~2–3% to ~12–14%), but also increases false positives.
- At $q=0.90$, both rules reach ~21–27% recall, which may be more appropriate for discovery/triage workflows.
- The gap between baseline-quantile and top-k matched suggests the candidate-matching constraint can reduce flagged volume; whether that is desirable depends on downstream review capacity.

What to take away:

- If your budget is **~1% review**, LM-TAD is best seen as a **prioritization signal** (good for “top of queue”), not as a detector.
- If you can review **~5–12%**, you start to get into a regime where recall is non-trivial and comparisons between decision rules become meaningful.

#### Porto — `porto_hoser_per_type_detour` (1% sample)

- Sample size: `N=5524`
- Abnormal fraction in sample: `12.42%` (686 / 5524)
- Baseline-calibrated thresholding (q = 0.99):
  - Baseline threshold: `1.731729`
  - Baseline outlier rate: `1.00%`
  - Target outlier rate: `0.94%`
- Mean log-perplexity (sample): `0.4646`

Interpretation of the operating point:

- At $q=0.99$, we again target ~1% flagged; this is a stringent setting intended for “only the most suspicious” trajectories.

Decision rules @ `q=0.99` (very low alert volume):

- Baseline-quantile: Precision `0.4038`, Recall `0.0306`, F1 `0.0569`, FPR `0.0064` (TP=21, FP=31)
- Top-k matched: k=56, Precision `0.3750`, Recall `0.0306`, F1 `0.0566`, FPR `0.0072` (TP=21, FP=35)

Decision rules at other operating points (from `metrics.json`):

| q | Method | Alert rate | Precision | Recall | F1 | FPR |
|---:|---|---:|---:|---:|---:|---:|
| 0.90 | baseline-quantile | 0.1271 | 0.3718 | 0.3805 | 0.3761 | 0.0912 |
| 0.90 | top-k matched (k=553) | 0.1001 | 0.3617 | 0.2915 | 0.3228 | 0.0730 |
| 0.95 | baseline-quantile | 0.0523 | 0.3322 | 0.1399 | 0.1969 | 0.0399 |
| 0.95 | top-k matched (k=277) | 0.0501 | 0.3357 | 0.1356 | 0.1931 | 0.0380 |

Interpretation (Porto):

- Porto shows similar trade-offs but slightly stronger separation than Beijing (consistent with higher AUROC/AUPRC).
- At $q=0.90$, precision and recall are both ~0.37–0.38 for baseline-quantile, indicating a more usable mid-volume operating point.
- At $q=0.99$, precision is relatively high (~0.38–0.40) but recall remains very low (~3%).

What to take away:

- Porto’s higher tail precision and recall curves suggest the teacher signal may be **more reliable** (less overlap) in this setting; it’s still far from a clean separator.

## Plots and qualitative interpretation

These plots are meant for quick sanity-checking score separation and per-type score shifts.

### Beijing — `Beijing_per_type_detour`

![Beijing score histogram](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/score_hist.png)

![Beijing score by type boxplot](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/score_by_type_box.png)

Additional plots (from `tools/plot_lmtad_results.py`, train split):

![Beijing normal vs abnormal density](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/lmtad_results/density/lmtad_eval_train_density.png)

![Beijing ROC curve](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/lmtad_results/roc/lmtad_eval_train_roc.png)

![Beijing PR curve](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/lmtad_results/pr/lmtad_eval_train_pr.png)

Qualitative read (Beijing):

- The density plot should show whether abnormal scores are shifted right (higher log-perplexity) relative to normal.
- Substantial overlap implies that high-$q$ thresholds will only capture the extreme tail of abnormal trajectories (low recall at $q=0.99$), which matches the decision metrics.
- The PR curve is the most informative for imbalanced detection; look for how quickly precision drops as recall increases.

Concrete interpretation to align with the tables:

- The density overlap you see here is the visual reason why **high-$q$ (low-volume) thresholding** has low recall: only a small fraction of abnormals live in the extreme tail.
- The PR curve and the tail-precision plot tell a consistent story: the top tail is enriched (precision > prevalence), but enrichment is not extreme.

Decision plots (from `metrics.json`):

![Beijing baseline-quantile confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/lmtad_results/decision/confusion/baseline_quantile_grid.png)

![Beijing top-k matched confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/lmtad_results/decision/confusion/topk_matched_grid.png)

![Beijing baseline-quantile metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/lmtad_results/decision/curves/baseline_quantile_metrics_vs_q.png)

![Beijing top-k matched metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/plots/lmtad_results/decision/curves/topk_matched_metrics_vs_q.png)

How to use the decision plots:

- The confusion-grid panels visualize the transition from high recall / higher FPR (lower $q$) to low recall / low FPR (higher $q$).
- The metrics-vs-q curves summarize this trade-off. A practical choice of $q$ is typically the smallest $q$ that keeps FPR and flag rate within operational constraints.

Practical reading tip:

- Treat $q$ as a **review-budget knob**. Pick a budget (e.g., 1%, 5%, 10%), then read off the achievable precision/recall at the corresponding flagged rate; don’t over-index on any single threshold-free metric.

### Porto — `porto_hoser_per_type_detour`

![Porto score histogram](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/score_hist.png)

![Porto score by type boxplot](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/score_by_type_box.png)

Additional plots (from `tools/plot_lmtad_results.py`, train split):

![Porto normal vs abnormal density](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/lmtad_results/density/lmtad_eval_train_density.png)

![Porto ROC curve](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/lmtad_results/roc/lmtad_eval_train_roc.png)

![Porto PR curve](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/lmtad_results/pr/lmtad_eval_train_pr.png)

Qualitative read (Porto):

- If the abnormal density is more cleanly right-shifted (less overlap), that supports the higher AUROC/AUPRC and the stronger $q=0.90$ operating point.
- Compare PR curves across datasets only cautiously: they are prevalence-dependent, but shape differences still indicate relative separability.

Concrete interpretation to align with the tables:

- Porto’s density/PR curves are consistent with the higher AUROC/AUPRC and the fact that **Precision@top-k% stays higher** (especially at 10%) than Beijing.

Decision plots (from `metrics.json`):

![Porto baseline-quantile confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/lmtad_results/decision/confusion/baseline_quantile_grid.png)

![Porto top-k matched confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/lmtad_results/decision/confusion/topk_matched_grid.png)

![Porto baseline-quantile metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/lmtad_results/decision/curves/baseline_quantile_metrics_vs_q.png)

![Porto top-k matched metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/plots/lmtad_results/decision/curves/topk_matched_metrics_vs_q.png)

## Discussion

### What these results suggest

- LM-TAD’s score is informative (better than random ranking), but not cleanly separable: the model can prioritize suspicious trajectories, yet thresholding at very low alert rates will miss most abnormal cases.
- For workflows that require breadth (higher recall), operating points like $q=0.90$–$0.95$ appear materially more useful, at the expense of higher false positive volume.

Relevance to distillation (teacher-signal framing):

- The separability metrics support using LM-TAD as a **soft teacher** (continuous target/weight) rather than a strict binary labeler.
- If the student is trained to match or be guided by this signal, expect **label noise in the tail**; techniques like temperature scaling, per-type normalization, or down-weighting near the decision boundary may help if training becomes unstable.

### Limitations

- Split choice: these results are on `train` (sampled). A more rigorous statement requires `val`/`test`.
- Sampling variance: 1% sampling makes runs fast, but increases uncertainty, especially for extreme-$q$ behavior.
- Label noise: abnormality labels may be heterogeneous across types; per-type calibration might improve decision quality.
- Synthetic abnormality realism: per-type detours are controlled perturbations and may not reflect the full distribution of real-world anomalies.

### Suggested next experiments

- Re-run on `val` (or a held-out `test`) with the same sampling seed, and compare AUROC/AUPRC + decision curves.
- Add confidence intervals (e.g., bootstrap on the sampled subset) for AUROC/AUPRC and for selected $q$ operating points.
- Produce per-type PR curves or per-type recall at fixed flag rates (more actionable than aggregate).

## References (artifacts and reproducibility)

### Code

- Launcher: [scripts/run_lmtad_decision_benchmark_overnight.sh](../scripts/run_lmtad_decision_benchmark_overnight.sh)
- Benchmark runner: [tools/run_lmtad_decision_benchmark.py](../tools/run_lmtad_decision_benchmark.py)
- Plotter (density/ROC/PR + decision artifacts): [tools/plot_lmtad_results.py](../tools/plot_lmtad_results.py)

### Run directories

- Output root: [research_runs](../research_runs)
- Beijing run: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing`
- Porto run: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto`

### Key files

- Beijing:
  - Summary: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/summary.json`
  - Dataset report: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/report.md`
  - Metrics: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/analysis/Beijing_per_type_detour/metrics.json`
  - Scores: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/eval/Beijing_per_type_detour/evaluation_results.json`

- Porto:
  - Summary: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/summary.json`
  - Dataset report: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/report.md`
  - Metrics: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/analysis/porto_hoser_per_type_detour/metrics.json`
  - Scores: `research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/eval/porto_hoser_per_type_detour/evaluation_results.json`

### Storage note

On this cluster, `research_runs/` may be a symlink into `/local/...` (node-local). If you don’t see runs on another node, that can be expected.
