# Quick Agent Guide — HOSER repository

This file gives concise, repository-specific guidance for AI coding agents (Copilot-style) to be immediately productive when editing, testing, or extending HOSER.

## What this project is
- **Purpose**: Research code for distilling a large transformer (LM‑TAD teacher) into a compact, fast student (HOSER) for trajectory prediction. See `docs/LMTAD-Distillation.md` for the full rationale.
- **Main components**: model & training (`train.py`, `train_with_distill.py`), data loader (`dataset.py`), evaluation pipeline (`python_pipeline.py`, `setup_evaluation.py`), distillation utilities in `critics/`, and analysis tools in `tools/`.

## Developer environment & commands (must-follow)
- **Use the `uv` tool** (project policy): e.g. `uv venv`, `uv add <package>` (use `-D` for dev deps), `uv sync` to sync from the lockfile, `uv run python train.py --config config/Beijing.yaml`.
-  Avoid `uv pip install` for adding packages; prefer `uv add` so dependencies are recorded in the project's lockfile.
- **Tests**: `pytest tests/` or targeted `pytest tests/test_trajectory_validation.py::test_x`.
- **Pre-commit**: `pre-commit run --all-files` before commits/PRs.

## Core idea: LM‑TAD → HOSER distillation (practical summary)
- Teacher (LM‑TAD) is a frozen large transformer that outputs grid-cell probability distributions; student (HOSER) predicts road-segment candidates.
- Key problem: vocabulary mismatch (grid tokens vs road IDs). The project solves this with a precomputed `road_id -> grid_token` mapping (`critics/grid_mapper.py`) and renormalizes teacher probabilities over HOSER's candidate set.
- Orchestration: `critics/lmtad_teacher.py` wraps teacher inference (AMP, sliding window), `critics/distill_hook.py` maps teacher outputs to candidate roads and computes the KL distillation loss, and `train_with_distill.py` wires these into the training loop.
- Practical constraints: teacher inference is expensive (dominates distilled training throughput); `dataset.py` implements smart RAM-caching to reduce I/O bottlenecks for both vanilla and distilled runs.

Files to inspect for distillation internals: `critics/lmtad_teacher.py`, `critics/grid_mapper.py`, `critics/distill_hook.py`, `train_with_distill.py`, and `docs/LMTAD-Distillation.md`.

## Pipeline phases (explicit summary from `python_pipeline.py`)
- The evaluation pipeline is phase-based. Phases registered in `PipelineConfig.phases` include:
  - `generation` (critical)
  - `base_eval` (critical)
  - `paired_analysis`
  - `cross_dataset`
  - `road_network_translate`
  - `abnormal`, `abnormal_od_extract`, `abnormal_od_generate`, `abnormal_od_evaluate`
  - `wang_abnormality`, `lmtad_spatial_abnormality`
  - `scenarios`
- Phase control is exposed via CLI flags: `--only` (run only listed phases), `--skip` (skip listed phases), and compatibility flags `--skip-gene` / `--skip-eval` (backwards compatible).
- Important: `generation` and `base_eval` are marked critical in code (pipeline stops on failure). The pipeline auto-detects models from `models/` and operates inside an evaluation directory created by `setup_evaluation.py`.

Examples:
```bash
# From an evaluation directory created by `setup_evaluation.py`
uv run python ../python_pipeline.py               # run full pipeline
uv run python ../python_pipeline.py --only generation,base_eval  # run only generation+base eval
uv run python ../python_pipeline.py --skip scenarios,abnormal  # skip heavier analysis phases
uv run python ../python_pipeline.py --use-astar  # force A* search instead of beam search
```

## Evaluation workspace layout and run pattern
- Create workspace: `uv run python setup_evaluation.py --dataset Beijing --name my-eval`
- Typical eval dir structure: `models/`, `config/` (snapshotted), `gene/` (generated trajectories), `eval/` (results), `scenarios/`.
- Run pipeline from inside eval dir (preferred) to avoid path issues: `uv run python ../python_pipeline.py`.

Note: two canonical evaluation directories present in this repo and useful as examples for `--eval-dir` are:
- `hoser-distill-beijing/` — baseline distilled Beijing eval workspace
- `hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/` — Optuna Porto evaluation workspace (contains `models/`, `gene/`, `eval/`, `scenarios/`).

Example running pipeline using one of these directories:
```bash
uv run python python_pipeline.py --eval-dir hoser-distill-beijing
uv run python python_pipeline.py --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732
```

## Repo-specific conventions & quick pointers
- **Config-first (preferred)**: Put static settings and hyperparameters in YAML files under `config/` (e.g., `config/Beijing.yaml`, `config/evaluation.yaml`). These files are snapshotted into evaluation workspaces for reproducibility. Use CLI arguments for dynamic, short-term overrides only (examples: `--seed`, `--cuda`, `--force`, `--num-gene`, `--no-wandb`). Avoid encoding long‑term hyperparameter changes only via CLI — prefer updating the YAML and re-running so experiments are fully reproducible.
- **Checkpoint names**: follow `tools/model_detection.py` naming patterns (e.g. `vanilla_25epoch_seed42.pth`).
- **WandB**: enabled by default; disable with `--no-wandb` or config `enable_wandb: false` for offline runs.
- **Optuna outputs**: `optuna_results/`, `optuna_trials*/` contain hyperparameter search artifacts.

- **Git commit style**: Use emoji prefixes in commit messages to make change types obvious at a glance. Examples: `✨ feat:`, `🔧 fix:`, `📝 docs:`, `✅ test:`, `♻️ refactor:`, `📈 perf:`, `🚀 chore:`. Example full message: `✨ feat: add vocabulary mapping validation`.

 
## Where to start when asked to change behavior
- Small change (model code): run unit test(s) touching the file and a smoke `uv run python train.py --config config/Beijing.yaml` on a reduced dataset.
- Distillation changes: test `critics/distill_hook.py` behavior in a short `train_with_distill.py` run with `--distill.window=2` and small `--epochs=1` to validate mapping/renormalization logic.
- Pipeline changes: run `python_pipeline.py` with `--only` and `--verbose` to exercise the modified phases locally in an eval workspace.
