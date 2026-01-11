# Teacher separability (LM-TAD score analysis)

This document is a methodology reference for [tools/teacher_separability.py](../tools/teacher_separability.py). It is intended to be detailed enough to reproduce the reported measurements and to understand exactly what statistical tests/estimators are being used.

## Purpose and framing

The goal is to evaluate LM-TAD as a continuous teacher signal (log-perplexity / log-perplexity-like score), not primarily as a binary classifier.

Given:
- An evaluation output JSON containing per-split `log_perplexity_values` (higher means more “surprising” / more abnormal under the teacher).
- The exact sampled CSV used for that evaluation, containing a label column (default `abnormality_info`).

The script quantifies whether abnormal trajectories tend to score higher than normal ones and how concentrated abnormals are in the high-score tail.

This answers two research questions:

1) **Separation / ranking quality**: do abnormal trajectories systematically receive higher teacher scores than normal trajectories?
2) **Operational concentration**: if we only look at the top-scoring tail (e.g., top 1%, 5%, 10%), how many abnormals do we capture (recall), and how “pure” is that tail (precision)?

## Outputs

The script prints a Markdown table to stdout and optionally writes it to disk with `--out-md`.

Metrics reported per dataset:
- AUROC (Mann–Whitney U; handles ties)
- Cliff’s δ (effect size derived from AUROC: δ = 2·AUROC − 1)
- Cohen’s d (standardized mean separation)
- Wasserstein-1D distance (distribution shift)
- Recall@top{1,5,10}% (coverage of positives when reviewing the top-scoring tail)
- Precision@top{1,5,10}% (purity of the reviewed top-scoring tail)

Confidence intervals:
- 95% bootstrap CIs by default (`--ci 0.95`)
- Stratified bootstrap (positives and negatives resampled separately) to keep prevalence stable

## Inputs and preprocessing (reproducibility-critical)

### Score input: evaluation_results.json

The script reads `log_perplexity_values` from `evaluation_results.json` for a requested split (default `train`). Formally, you provide a sequence of real-valued scores:

- $s_i \in \mathbb{R}$ for each trajectory $i$.

Only finite scores are retained (`np.isfinite`).

Important assumption:

- **Higher score = more abnormal**. All tail metrics select the top scores by sorting descending.

### Label input: sampled CSV

The script reads a boolean label per row from the sampled CSV. It assumes the CSV row order matches the score order from the evaluation output.

Label parsing is intentionally tolerant and maps raw strings to a boolean “abnormal” indicator $y_i \in \{0,1\}$:

- Normal if (case-insensitive) in `{ "", "nan", "none", "null", "normal" }` or equals `--normal-value` (default `normal`).
- Abnormal otherwise.

Reproducibility invariant:

- The CSV must be the **exact** sampled CSV used for evaluation (same length and ordering). The script checks `len(scores) == len(labels)` and errors otherwise.

### Typical producers of the evaluation JSON

The evaluation output JSON is typically produced by:

- [tools/evaluate_dataset_with_lmtad.py](../tools/evaluate_dataset_with_lmtad.py)
- [tools/run_lmtad_decision_benchmark.py](../tools/run_lmtad_decision_benchmark.py)

## Methodology: estimators and measurements

Let $\{(s_i, y_i)\}_{i=1}^{n}$ be the filtered dataset of finite scores and labels, with $y_i=1$ for abnormal and $y_i=0$ for normal.

### AUROC (Mann–Whitney U)

AUROC is computed using the Mann–Whitney U statistic, with **average ranks for ties** (equivalent to `scipy.stats.rankdata(..., method="average")`, but implemented without SciPy).

Procedure:

1) Assign ranks $r_i$ (1-based) to scores $s_i$, where tied scores receive the average of their rank positions.
2) Let $n_1$ be the number of positives and $n_0$ the number of negatives.
3) Compute the positive rank sum $R_1 = \sum_{i:y_i=1} r_i$.
4) Compute $U = R_1 - \frac{n_1(n_1+1)}{2}$.
5) Report $\text{AUROC} = \frac{U}{n_1 n_0}$.

Interpretation:

- AUROC equals $\Pr(s_\text{pos} > s_\text{neg})$ plus half the probability of ties.

### Cliff’s delta (from AUROC)

Cliff’s delta is derived directly from AUROC:

$$
\delta = 2\,\text{AUROC} - 1.
$$

This is the standardized probability of superiority:

- $\delta=0$ means no separation; $\delta=1$ means all positives score above all negatives.

### Cohen’s d

Let $\mu_1, \sigma_1^2$ be the mean and (sample) variance of positive scores, and $\mu_0, \sigma_0^2$ for negative scores.

The script reports the pooled-standard-deviation effect size:

$$
d = \frac{\mu_1 - \mu_0}{s_p},\quad
s_p = \sqrt{\frac{(n_1-1)\sigma_1^2 + (n_0-1)\sigma_0^2}{n_1+n_0-2}}.
$$

Edge handling:

- If the pooled variance is zero (or degrees of freedom are insufficient), the implementation returns 0.

### Wasserstein-1D distance

The 1D Wasserstein distance (earth mover distance) is computed by integrating the absolute CDF difference over the joint support:

1) Sort positive sample $x$ and negative sample $y$.
2) Construct a sorted support grid as the unique values of $x \cup y$.
3) Evaluate empirical CDFs of $x$ and $y$ on that grid.
4) Integrate $|\mathrm{CDF}_x - \mathrm{CDF}_y|$ via trapezoidal rule.

This yields a nonnegative quantity in the same units as the score.

### Tail metrics: Recall@top-k% and Precision@top-k%

For a fraction $f \in (0,1]$, define:

- $k = \lceil f\,n \rceil$ (clipped to $[1, n]$).
- Let $T_f$ be the set of indices of the top-$k$ scores (sorted descending).
- Let $\mathrm{TP}_f = \sum_{i \in T_f} y_i$.
- Let $n_1 = \sum_i y_i$.

Then:

$$
\mathrm{Recall@top}f = \frac{\mathrm{TP}_f}{n_1},
\quad
\mathrm{Precision@top}f = \frac{\mathrm{TP}_f}{k}.
$$

Notes:

- The top set is chosen by sorting scores descending (highest = most abnormal).
- Because $k$ is a ceiling, top-1% means “at least 1%”, especially for small $n$.

### Confidence intervals: stratified bootstrap

All confidence intervals are computed via a stratified bootstrap over the two score groups.

Inputs:

- Positive score vector $x$ of length $n_1$.
- Negative score vector $z$ of length $n_0$.

For each bootstrap replicate $b=1..B$:

1) Resample $x^{(b)}$ by sampling $n_1$ elements from $x$ with replacement.
2) Resample $z^{(b)}$ by sampling $n_0$ elements from $z$ with replacement.
3) Concatenate scores $s^{(b)} = [x^{(b)}, z^{(b)}]$ and labels $y^{(b)} = [1..1, 0..0]$.
4) Apply a random permutation to avoid artifacts from concatenation order.
5) Compute AUROC, Cliff’s δ, Cohen’s d, Wasserstein-1D, and all requested tail metrics on $(s^{(b)}, y^{(b)})$.

The script then reports a two-sided quantile interval at level `--ci`:

- Let $\alpha = (1-\text{ci})/2$. The CI is $[Q_{\alpha}, Q_{1-\alpha}]$ computed via `np.quantile`.

Implementation details relevant for reproduction:

- RNG: `numpy.random.default_rng(seed)`.
- Bootstrap keeps $(n_1, n_0)$ fixed in every replicate (prevalence stable).
- The reported tail metric bands in plots are computed across replicates as pointwise quantiles over $f$.

## Reproducibility checklist

To reproduce results exactly:

1) Use the same `evaluation_results.json` and the **exact** sampled CSV used in evaluation.
2) Use the same split name (`--split`).
3) Use the same bootstrap count (`--bootstrap`) and seed (`--seed`).
4) Use the same `--label-col` and `--normal-value`.
5) Ensure you run with the project environment (policy: `uv run ...`).

Recommended practice:

- Record the command used and store the emitted Markdown table via `--out-md` alongside the evaluation artifacts.

## CLI usage

### Basic usage (one dataset)

```bash
uv run python tools/teacher_separability.py \
  --name Beijing_per_type_detour \
  --eval-json research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/eval/Beijing_per_type_detour/evaluation_results.json \
  --labels-csv research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/sampled_data/Beijing_per_type_detour/train.csv \
  --split train \
  --bootstrap 500 \
  --seed 0 \
  --ci 0.95
```

### Multiple datasets in one call (repeatable flags)

Repeat `--name`, `--eval-json`, and `--labels-csv` the same number of times.

```bash
uv run python tools/teacher_separability.py \
  --name Beijing_per_type_detour \
  --eval-json research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/eval/Beijing_per_type_detour/evaluation_results.json \
  --labels-csv research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_beijing/sampled_data/Beijing_per_type_detour/train.csv \
  --name porto_hoser_per_type_detour \
  --eval-json research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/eval/porto_hoser_per_type_detour/evaluation_results.json \
  --labels-csv research_runs/_benchmarks/lmtad_decision_bench_overnight_20260109_100132_porto/sampled_data/porto_hoser_per_type_detour/train.csv \
  --split train \
  --bootstrap 500 \
  --seed 0 \
  --ci 0.95 \
  --out-md research_runs/_benchmarks/lmtad_teacher_separability_20260109.md
```

### Changing label conventions

If your CSV uses a different column name or different normal sentinel value:

```bash
uv run python tools/teacher_separability.py \
  --name MyDataset \
  --eval-json path/to/evaluation_results.json \
  --labels-csv path/to/train.csv \
  --label-col my_label_column \
  --normal-value NORMAL \
  --split train
```

### Disabling bootstrap (point estimates only)

```bash
uv run python tools/teacher_separability.py \
  --name MyDataset \
  --eval-json path/to/evaluation_results.json \
  --labels-csv path/to/train.csv \
  --bootstrap 0
```

## How to interpret the key metrics

- AUROC: probability that a randomly drawn abnormal has a higher score than a randomly drawn normal. AUROC=0.5 is random ranking; AUROC=1.0 is perfect separation.
- Precision@top-k%: if you review the top k% highest-scoring trajectories, this is the fraction that are abnormal (purity).
- Recall@top-k%: if you review the top k% highest-scoring trajectories, this is the fraction of all abnormals you capture (coverage).

A common workflow interpretation:
- Small k (1%) answers “is the extreme tail worth prioritizing?”
- Larger k (5–10%) answers “how much abnormal mass can I capture at moderate review budgets?”

## Limitations and common pitfalls

- **Sampling variance**: if your evaluation used Bernoulli sampling (e.g., 1%), extreme-tail quantities (top 1%) can have noticeable variance.
- **Prevalence dependence**: precision and PR-curve summaries depend on abnormal prevalence; compare across datasets with care.
- **Label semantics**: the CSV label column is treated as ground truth here; if labels are weak or heterogeneous by type, all metrics reflect that.
- **Score direction**: this script assumes higher score means more abnormal; if you change the score definition, you must adjust the ordering.

## Programmatic use (importing functions)

The module is pure-Python and can be imported in notebooks/scripts for custom analysis. Common entry points:
- `auroc(scores, labels)`
- `recall_at_top_frac(scores, labels, frac)`
- `precision_at_top_frac(scores, labels, frac)`
- `bootstrap_stratified(pos=..., neg=..., bootstrap=..., seed=..., top_fracs=...)`

If you need plots, use [tools/plot_teacher_separability.py](../tools/plot_teacher_separability.py), which consumes the same inputs and produces PNGs.
