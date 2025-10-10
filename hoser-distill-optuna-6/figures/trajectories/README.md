# Trajectory Visualizations

This directory contains trajectory visualizations for the HOSER distillation evaluation.

## Directory Structure

- `separate/`: Individual trajectory plots (short, medium, long for each model/OD type)
- `overlaid/`: Overlaid plots showing all three length categories together
- `cross_model/`: **NEW** - Cross-model comparisons showing vanilla, distilled (seed 42), distilled (seed 44), and **real trajectories** for the same OD pairs

## Models Visualized

1. **distilled** (seed 42)
2. **distilled_seed44** (seed 44)
3. **vanilla** (baseline)

Each model has visualizations for both:
- **Train OD**: Trajectories generated from training origin-destination pairs
- **Test OD**: Trajectories generated from test origin-destination pairs

## Sampling Strategy

Trajectories are sampled using the **length-based** strategy:
- **Short**: 25th percentile trajectory length
- **Medium**: 50th percentile trajectory length (median)
- **Long**: 75th percentile trajectory length

## File Naming Convention

**Separate plots:**
```
{model}_{od_type}_{length}.{pdf,png}
```
Example: `distilled_train_short.pdf`

**Overlaid plots:**
```
{model}_{od_type}_all.{pdf,png}
```
Example: `distilled_train_all.pdf`

## Generation

**Separate and overlaid plots:**
```bash
uv run python visualize_trajectories.py --sample_strategy length_based
```

**Cross-model comparisons (same OD pairs with real data):**
```bash
uv run python visualize_trajectories.py --cross_model --no_separate --no_overlaid
```

## Key Observations

- **Distilled models** generate significantly longer, more realistic trajectories compared to vanilla
- **Vanilla model** trajectories are consistently shorter across all length categories
- Both train and test OD show similar patterns, indicating good generalization

## File Count

- **36 files** in `separate/` (6 models × 3 lengths × 2 formats)
- **12 files** in `overlaid/` (6 models × 2 formats)
- **48 total files** (24 PDF + 24 PNG)

