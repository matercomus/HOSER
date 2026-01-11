# LM-TAD decision benchmark — abnormal-3 (2026-01-11)

## Abstract

We evaluate LM-TAD as an abnormal-trajectory scoring function on two “abnormal-3” datasets (Beijing, Porto) using a 1% Bernoulli-sampled subset of the `train` split.
We report both (i) **ranking / separability** (AUROC/AUPRC and continuous-signal separability metrics) and (ii) **decision performance** across a sweep of alert-volume operating points (quantile $q$).

## Key takeaways (high level)

- LM-TAD shows **moderate separability** on abnormal-3 for both datasets (Beijing AUROC 0.733, Porto AUROC 0.760).
- The teacher’s top-score tail is **meaningfully enriched**: Precision@top1% is ~0.315 (Beijing) / ~0.393 (Porto), roughly **2.5–3.2×** the base prevalence (~12–13%).
- However, at very low review budgets the teacher only captures a small fraction of abnormals: Recall@top1% is ~2.5–3.2%.
- Increasing the alert volume (e.g., $q=0.90$–$0.95$) yields **much higher recall** at the cost of more false positives; the decision curves make this trade-off explicit.
- Baseline-quantile typically yields **slightly higher recall** at fixed $q$, while top-k matched often **tracks similar precision** but can flag slightly fewer trajectories at the same nominal $q$.

This report is intended to be readable; for reproducibility details and full artifacts, see the References section.

## Methods

### Experimental setup

- Data: `data/Beijing_abnormal_3` and `data/porto_hoser_abnormal_3`
- Split: `train`
- Sampling: Bernoulli sampling at `sample_frac=0.01` with `sample_seed=42`
- Decision sweep: quantiles $q \in \{0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99\}$
- Decision rules:
  - **Baseline-quantile thresholding**: threshold calibrated on baseline at each $q$
  - **Top-k matched**: choose k to match baseline’s flagged volume at each $q$

Notes:

- Abnormal-3 datasets are synthetic anomaly mixtures (e.g., detours + route switches) with the label in `abnormality_info`.
- All results below are on a sampled subset of `train` and should be treated as *development* measurements.

## Results

### Results overview (1% sample)

| Dataset | N | Abnormal fraction | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Beijing_abnormal_3 | 7206 | 12.71% | 0.7333 | 0.2470 |
| porto_hoser_abnormal_3 | 5524 | 12.38% | 0.7599 | 0.2935 |

Interpretation:

- AUROC/AUPRC measure **ranking quality** (how well higher scores correspond to abnormality), independent of a specific threshold.
- AUPRC is prevalence-sensitive; here prevalence is ~12%, so a random ranker would have AUPRC $\approx 0.12$.
- The q-sweep evaluates **decision rules** at matched operating points. Larger $q$ means lower alert volume (more conservative).

### Teacher separability summary (score as a continuous signal)

This table quantifies whether LM-TAD scores are *systematically higher* for synthetic abnormal trajectories than for normal ones.
Values are point estimates with **95% bootstrap CIs** (stratified resampling, 500 replicates).

| Dataset | N | Prev. | AUROC | Cliff’s δ | Cohen’s d | W1 | Recall@top1% | Recall@top5% | Recall@top10% | Precision@top1% | Precision@top5% | Precision@top10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Beijing_abnormal_3 | 7206 | 12.71% | 0.7333 [0.7177, 0.7498] | 0.4667 [0.4354, 0.4995] | 0.7165 [0.6249, 0.8000] | 0.4407 [0.3823, 0.4932] | 0.0251 [0.0186, 0.0349] | 0.1037 [0.0862, 0.1201] | 0.1987 [0.1758, 0.2205] | 0.3151 [0.2329, 0.4384] | 0.2632 [0.2188, 0.3047] | 0.2524 [0.2233, 0.2802] |
| porto_hoser_abnormal_3 | 5524 | 12.38% | 0.7599 [0.7411, 0.7793] | 0.5198 [0.4823, 0.5587] | 0.8752 [0.7761, 0.9942] | 0.2608 [0.2305, 0.2936] | 0.0322 [0.0219, 0.0424] | 0.1374 [0.1170, 0.1608] | 0.2851 [0.2573, 0.3129] | 0.3929 [0.2679, 0.5179] | 0.3394 [0.2888, 0.3971] | 0.3526 [0.3183, 0.3870] |

How to read these:

- **AUROC**: probability an abnormal scores higher than a normal.
- **Cliff’s δ**: effect size derived from AUROC ($\delta = 2\,\text{AUROC}-1$); 0 means no separation.
- **Cohen’s d** and **W1**: complementary measures of distribution separation (scale depends on score units).
- **Recall@top-k%**: fraction of abnormals captured by reviewing only the top-scoring k% of trajectories.
- **Precision@top-k%**: abnormal fraction within the reviewed top-scoring k% ("purity" of the tail).

Plots (with 95% bootstrap CIs):

![Teacher separability metric summary](../research_runs/_benchmarks/lmtad_teacher_separability_plots_abnormal3_20260111/teacher_separability_metrics.png)

![Teacher separability recall curve](../research_runs/_benchmarks/lmtad_teacher_separability_plots_abnormal3_20260111/teacher_separability_recall_curve.png)

![Teacher separability precision curve](../research_runs/_benchmarks/lmtad_teacher_separability_plots_abnormal3_20260111/teacher_separability_precision_curve.png)

Key tail-metric takeaway:

- The top tail is enriched (precision > prevalence), but at small review budgets (1%) the **coverage is low**.
- Moving from 1% to 10% review increases recall substantially, while precision remains meaningfully above prevalence.

### Per-dataset results (selected operating points)

#### Beijing — `Beijing_abnormal_3` (1% sample)

- Sample size: `N=7206`
- Abnormal fraction in sample: `12.71%` (916 / 7206)

Baseline-calibrated thresholding (q = 0.99):

- Baseline threshold: `3.176166`
- Baseline outlier rate: `1.00%`
- Target outlier rate: `1.14%`

Mean log-perplexity (sample): `0.8622`

Interpretation of the operating point:

- At $q=0.99$, the rule is tuned to flag roughly ~1% of trajectories (very low review volume).
- The resulting recall is expected to be low unless abnormal and normal score distributions are strongly separated.

Decision rules @ `q=0.99` (very low alert volume):

- Baseline-quantile: Alert rate `0.0114`, Precision `0.3293`, Recall `0.0295`, F1 `0.0541`, FPR `0.0087` (TP=27, FP=55)
- Top-k matched: k=73, Alert rate `0.0101`, Precision `0.3151`, Recall `0.0251`, F1 `0.0465`, FPR `0.0079` (TP=23, FP=50)

Decision rules at other operating points (from `metrics.json`):

| q | Method | Alert rate | Precision | Recall | F1 | FPR |
|---:|---|---:|---:|---:|---:|---:|
| 0.90 | baseline-quantile | 0.1210 | 0.2534 | 0.2413 | 0.2472 | 0.1035 |
| 0.90 | top-k matched (k=721) | 0.1001 | 0.2524 | 0.1987 | 0.2223 | 0.0857 |
| 0.95 | baseline-quantile | 0.0597 | 0.2651 | 0.1245 | 0.1695 | 0.0502 |
| 0.95 | top-k matched (k=361) | 0.0501 | 0.2632 | 0.1037 | 0.1488 | 0.0423 |

Interpretation (Beijing):

- $q=0.99$ is a strict “top of queue” setting: precision is high relative to prevalence, but recall is only ~3%.
- $q=0.90$–$0.95$ provides much higher recall, at higher alert volumes.

Baseline-quantile vs top-k matched (Beijing):

- At matched $q$, top-k matched typically flags slightly fewer trajectories (lower alert rate), which tends to reduce recall modestly.
- Precision is usually similar between methods at these operating points; the primary difference is the small volume/coverage shift.

What to take away:

- If your budget is **~1% review**, LM-TAD is best treated as a **prioritization signal** (good for “top of queue”), not as a detector.
- If you can review **~5–12%**, recall becomes non-trivial and the baseline-vs-topk trade-offs become practically meaningful.

#### Porto — `porto_hoser_abnormal_3` (1% sample)

- Sample size: `N=5524`
- Abnormal fraction in sample: `12.38%` (684 / 5524)

Baseline-calibrated thresholding (q = 0.99):

- Baseline threshold: `1.731729`
- Baseline outlier rate: `1.00%`
- Target outlier rate: `0.94%`

Mean log-perplexity (sample): `0.4619`

Interpretation of the operating point:

- At $q=0.99$, we again target ~1% flagged; this is a stringent setting intended for “only the most suspicious” trajectories.
- If score overlap is substantial, this setting yields high precision but very low recall.

Decision rules @ `q=0.99` (very low alert volume):

- Baseline-quantile: Alert rate `0.0094`, Precision `0.4038`, Recall `0.0307`, F1 `0.0569`, FPR `0.0064` (TP=21, FP=31)
- Top-k matched: k=56, Alert rate `0.0101`, Precision `0.3929`, Recall `0.0322`, F1 `0.0595`, FPR `0.0070` (TP=22, FP=34)

Decision rules at other operating points (from `metrics.json`):

| q | Method | Alert rate | Precision | Recall | F1 | FPR |
|---:|---|---:|---:|---:|---:|---:|
| 0.90 | baseline-quantile | 0.1236 | 0.3558 | 0.3553 | 0.3555 | 0.0909 |
| 0.90 | top-k matched (k=553) | 0.1001 | 0.3526 | 0.2851 | 0.3153 | 0.0740 |
| 0.95 | baseline-quantile | 0.0525 | 0.3345 | 0.1418 | 0.1991 | 0.0399 |
| 0.95 | top-k matched (k=277) | 0.0501 | 0.3394 | 0.1374 | 0.1956 | 0.0378 |

Interpretation (Porto):

- Porto shows slightly stronger separability than Beijing (higher AUROC/AUPRC, higher tail precision).
- At $q=0.90$, baseline-quantile reaches ~0.36 precision and ~0.36 recall, a more usable mid-volume operating point.

Baseline-quantile vs top-k matched (Porto):

- At fixed $q$, top-k matched again tends to flag slightly fewer trajectories and therefore slightly reduces recall.
- In this dataset the precision difference between methods is small; the main decision is the review budget ($q$), not the rule choice.

What to take away:

- Porto’s higher tail precision and recall curves suggest the teacher signal may be **more reliable** here than Beijing, but it is still far from a clean separator.
- For workflows that require breadth (higher recall), operating points like $q=0.90$–$0.95$ are materially more useful than $q=0.99$.

## Plots and qualitative interpretation

These plots are meant for quick sanity-checking score separation and per-type score shifts.

### Beijing — `Beijing_abnormal_3`

![Beijing score histogram](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/score_hist.png)

![Beijing score by type boxplot](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/score_by_type_box.png)

Additional plots (from `tools/plot_lmtad_results.py`, train split):

![Beijing normal vs abnormal density](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/lmtad_results/density/lmtad_eval_train_density.png)

![Beijing ROC curve](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/lmtad_results/roc/lmtad_eval_train_roc.png)

![Beijing PR curve](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/lmtad_results/pr/lmtad_eval_train_pr.png)

Qualitative read (Beijing):

- The density plot visualizes score overlap between normal and abnormal; overlap implies low recall at strict thresholds.
- The PR curve is the most informative for imbalanced detection; compare its shape to the prevalence baseline.

Concrete interpretation to align with the tables:

- The density overlap is the visual reason why high-$q$ (low-volume) thresholding has low recall: only a small fraction of abnormals live in the extreme right tail.
- The PR curve and the tail precision plot should tell a consistent story: the top tail is enriched (precision > prevalence), but enrichment is not extreme.

Decision plots (from `metrics.json`):

![Beijing baseline-quantile confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/lmtad_results/decision/confusion/baseline_quantile_grid.png)

![Beijing top-k matched confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/lmtad_results/decision/confusion/topk_matched_grid.png)

![Beijing baseline-quantile metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/lmtad_results/decision/curves/baseline_quantile_metrics_vs_q.png)

![Beijing top-k matched metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/plots/lmtad_results/decision/curves/topk_matched_metrics_vs_q.png)

### Porto — `porto_hoser_abnormal_3`

![Porto score histogram](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/score_hist.png)

![Porto score by type boxplot](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/score_by_type_box.png)

Additional plots (from `tools/plot_lmtad_results.py`, train split):

![Porto normal vs abnormal density](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/lmtad_results/density/lmtad_eval_train_density.png)

![Porto ROC curve](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/lmtad_results/roc/lmtad_eval_train_roc.png)

![Porto PR curve](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/lmtad_results/pr/lmtad_eval_train_pr.png)

Qualitative read (Porto):

- Porto’s density/PR curves are consistent with higher AUROC/AUPRC and stronger tail precision than Beijing.

Concrete interpretation to align with the tables:

- A cleaner right shift (less overlap) supports Porto’s higher AUROC/AUPRC and the stronger mid-volume operating point around $q=0.90$.
- Tail precision staying higher at top-k% is the curve-level view of “enrichment,” but recall remains limited at very small k.

Decision plots (from `metrics.json`):

![Porto baseline-quantile confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/lmtad_results/decision/confusion/baseline_quantile_grid.png)

![Porto top-k matched confusion grid across q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/lmtad_results/decision/confusion/topk_matched_grid.png)

![Porto baseline-quantile metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/lmtad_results/decision/curves/baseline_quantile_metrics_vs_q.png)

![Porto top-k matched metrics vs q](../research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/plots/lmtad_results/decision/curves/topk_matched_metrics_vs_q.png)

## Discussion

### What these results suggest

- LM-TAD is informative (better than random ranking) on abnormal-3, but not cleanly separable: it can prioritize suspicious trajectories, yet strict thresholds miss most abnormal cases.
- For review workflows, the key decision is choosing a review budget (or $q$). The tail precision/recall curves quantify how much abnormal mass concentrates in the top-scoring tail.

Baseline-quantile vs top-k matched (rule choice):

- In these runs, the two rules usually have similar precision at comparable operating points; baseline-quantile often yields slightly higher recall because it can end up flagging slightly more trajectories at the same nominal $q$.
- In practice, $q$ (review budget) is the dominant knob; the rule choice mostly shifts volume/coverage modestly.

### Limitations

- Split choice: these results are on `train` (sampled). A more rigorous statement requires `val`/`test`.
- Sampling variance: 1% sampling makes runs fast, but increases uncertainty, especially for extreme tails (top 1%).
- Synthetic abnormality realism: abnormal-3 is a controlled perturbation mixture and may not reflect the distribution of real-world anomalies.

### Suggested next experiments

- Re-run on `val` (or held-out `test`) with the same sampling seed and compare AUROC/AUPRC + decision curves.
- Stratify by abnormal type (detour vs route switch) and report per-type tail metrics.
- Compare abnormal-3 against per-type detour to understand which anomaly types produce stronger teacher separability.

## References (artifacts and reproducibility)

### Code

- Benchmark runner: [tools/run_lmtad_decision_benchmark.py](../tools/run_lmtad_decision_benchmark.py)
- Evaluator: [tools/evaluate_dataset_with_lmtad.py](../tools/evaluate_dataset_with_lmtad.py)
- Plotter (density/ROC/PR + decision artifacts): [tools/plot_lmtad_results.py](../tools/plot_lmtad_results.py)
- Teacher separability: [tools/teacher_separability.py](../tools/teacher_separability.py)
- Teacher separability plots: [tools/plot_teacher_separability.py](../tools/plot_teacher_separability.py)

### Run directories

- Output root: [research_runs](../research_runs)
- Beijing run: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing`
- Porto run: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto`

### Key files

- Beijing:
  - Dataset report: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/report.md`
  - Metrics: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/analysis/Beijing_abnormal_3/metrics.json`
  - Scores: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/eval/Beijing_abnormal_3/evaluation_results.json`
  - Sampled labels: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_beijing/sampled_data/Beijing_abnormal_3/train.csv`

- Porto:
  - Dataset report: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/report.md`
  - Metrics: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/analysis/porto_hoser_abnormal_3/metrics.json`
  - Scores: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/eval/porto_hoser_abnormal_3/evaluation_results.json`
  - Sampled labels: `research_runs/_benchmarks/lmtad_decision_bench_abnormal3_20260111_095914_porto/sampled_data/porto_hoser_abnormal_3/train.csv`

- Teacher separability artifacts:
  - Table: `research_runs/_benchmarks/lmtad_teacher_separability_abnormal3_20260111.md`
  - Plots: `research_runs/_benchmarks/lmtad_teacher_separability_plots_abnormal3_20260111/`

### Storage note

On this cluster, `research_runs/` may be a symlink into `/local/...` (node-local). If you don’t see runs on another node, that can be expected.
