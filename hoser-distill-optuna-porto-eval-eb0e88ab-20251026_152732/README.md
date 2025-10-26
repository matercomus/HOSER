# HOSER Evaluation: porto-eval

**Created**: 2025-10-26 15:27:32
**Dataset**: porto_hoser
**Models**: distill, distill_seed43, distill_seed44, vanilla, vanilla_seed43, vanilla_seed44

## Quick Start

```bash
cd hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732
uv run python ../python_pipeline.py
```

## Test Run

```bash
cd hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732
uv run python ../python_pipeline.py --num-gene 10
```

## With Scenario Analysis

```bash
cd hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732
uv run python ../python_pipeline.py --run-scenarios
```

## Configuration

Edit `config/evaluation.yaml` to change settings.
CLI arguments override config values.

## Output

- `gene/porto_hoser/seed42/` - Generated trajectories
- `eval/` - Evaluation results
- `scenarios/` - Scenario-based analysis (if enabled)
