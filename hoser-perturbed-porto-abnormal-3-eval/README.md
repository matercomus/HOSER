# HOSER Perturbed Eval — Porto (abnormal_3)

This evaluation workspace targets the *perturbed-train / clean-test* protocol:
- Train set is perturbed: `../data/porto_hoser_abnormal_3/train.csv`
- Val/Test remain clean via symlinks from `data/porto_hoser/`

## Run (robustness)
From this directory:

```bash
uv run python ../python_pipeline.py --only generation,base_eval
```

## Run (correction / triangulation)

```bash
uv run python ../python_pipeline.py --only perturbation_correction
```

## Plot Phase B

```bash
uv run python ../tools/visualize_perturbation_correction_results.py \
  --eval-dir . \
  --output-dir figures/perturbation_correction
```

Notes:
- `models/` is currently empty; copy trained `.pth` checkpoints here before running generation/eval.
- `config/evaluation.yaml` is pre-wired to the abnormal train split.
