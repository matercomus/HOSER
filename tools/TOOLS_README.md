# HOSER Tools - Programmatic Interfaces

This directory contains tools for trajectory analysis, evaluation, and workflow orchestration. All tools provide both CLI and programmatic interfaces.

## Core Workflow Tools

### Abnormal OD Workflow

Complete pipeline for analyzing model performance on abnormal origin-destination pairs.

#### `run_abnormal_od_workflow.py` - Workflow Orchestrator

**Programmatic:**
```python
from pathlib import Path
from tools.run_abnormal_od_workflow import run_abnormal_od_workflow

analysis_dir = run_abnormal_od_workflow(
    eval_dir=Path("hoser-distill-optuna-6"),
    dataset="Beijing",
    real_data_dir=Path("data/Beijing"),
    num_trajectories=50,
    max_pairs_per_category=20,
    seed=42,
    skip_detection=True
)
```

**CLI:**
```bash
uv run python tools/run_abnormal_od_workflow.py \
  --eval-dir hoser-distill-optuna-6 \
  --dataset Beijing \
  --real-data-dir data/Beijing \
  --skip-detection \
  --num-traj 50 \
  --max-pairs 20 \
  --seed 42
```

#### `analyze_abnormal.py` - Wang Statistical Detection

**Programmatic:**
```python
from pathlib import Path
from tools.analyze_abnormal import run_abnormal_analysis

run_abnormal_analysis(
    real_file=Path("data/Beijing/train.csv"),
    dataset="Beijing",
    config_path=Path("config/abnormal_detection_statistical.yaml"),
    output_dir=Path("abnormal/Beijing/train/real_data")
)
```

**CLI:**
```bash
uv run python tools/analyze_abnormal.py \
  --real_file data/Beijing/train.csv \
  --dataset Beijing \
  --config config/abnormal_detection_statistical.yaml \
  --output_dir abnormal/Beijing/train/real_data
```

#### `extract_abnormal_od_pairs.py` - Extract OD Pairs

**Programmatic:**
```python
from pathlib import Path
from tools.extract_abnormal_od_pairs import extract_and_save_abnormal_od_pairs

extract_and_save_abnormal_od_pairs(
    detection_results_files=[
        Path("abnormal/train/real_data/detection_results.json"),
        Path("abnormal/test/real_data/detection_results.json")
    ],
    real_data_files=[
        Path("data/train.csv"),
        Path("data/test.csv")
    ],
    dataset_name="Beijing",
    output_file=Path("abnormal_od_pairs.json")
)
```

**CLI:**
```bash
uv run python tools/extract_abnormal_od_pairs.py \
  --detection-results abnormal/train/real_data/detection_results.json \
                      abnormal/test/real_data/detection_results.json \
  --real-data data/train.csv data/test.csv \
  --dataset Beijing \
  --output abnormal_od_pairs.json
```

#### `generate_abnormal_od.py` - Generate Trajectories

**Programmatic:**
```python
from pathlib import Path
from tools.generate_abnormal_od import generate_abnormal_od_trajectories

generate_abnormal_od_trajectories(
    od_pairs_file=Path("abnormal_od_pairs.json"),
    model_dir=Path("models"),
    output_dir=Path("gene_abnormal/Beijing/seed42"),
    dataset="Beijing",
    num_traj_per_od=50,
    max_pairs_per_category=20,
    seed=42
)
```

**CLI:**
```bash
uv run python tools/generate_abnormal_od.py \
  --od-pairs abnormal_od_pairs.json \
  --model-dir models \
  --output-dir gene_abnormal/Beijing/seed42 \
  --num-traj 50 \
  --max-pairs 20 \
  --seed 42
```

#### `evaluate_abnormal_od.py` - Evaluate Performance

**Programmatic:**
```python
from pathlib import Path
from tools.evaluate_abnormal_od import evaluate_abnormal_od

evaluate_abnormal_od(
    generated_dir=Path("gene_abnormal/Beijing/seed42"),
    real_abnormal_file=Path("data/Beijing/train.csv"),
    abnormal_od_pairs_file=Path("abnormal_od_pairs.json"),
    output_dir=Path("eval_abnormal/Beijing"),
    dataset="Beijing"
)
```

**CLI:**
```bash
uv run python tools/evaluate_abnormal_od.py \
  --generated-dir gene_abnormal/Beijing/seed42 \
  --real-abnormal-file data/Beijing/train.csv \
  --abnormal-od-pairs abnormal_od_pairs.json \
  --output-dir eval_abnormal/Beijing
```

### Wang Analysis Tools

#### `analyze_wang_results.py` - Aggregate Results

**Programmatic:**
```python
from pathlib import Path
from tools.analyze_wang_results import analyze_wang_results

wang_results_file = analyze_wang_results(
    eval_dirs=[Path("hoser-distill-optuna-6")],
    output_file=Path("wang_results_aggregated.json")
)
```

**CLI:**
```bash
uv run python tools/analyze_wang_results.py \
  --eval-dir hoser-distill-optuna-6 \
  --output wang_results_aggregated.json
```

#### `visualize_wang_results.py` - Generate Plots

**Programmatic:**
```python
from pathlib import Path
from tools.visualize_wang_results import generate_wang_visualizations

generate_wang_visualizations(
    results_file=Path("wang_results_aggregated.json"),
    output_dir=Path("figures/wang_abnormality")
)
```

**CLI:**
```bash
uv run python tools/visualize_wang_results.py \
  --input wang_results_aggregated.json \
  --output-dir figures/wang_abnormality
```

### Scenario Analysis

#### `analyze_scenarios.py` - Scenario Analysis

**Programmatic:**
```python
from pathlib import Path
from tools.analyze_scenarios import analyze_all_scenarios

analyze_all_scenarios(
    eval_dir=Path("hoser-evaluation-baseline-abc123"),
    dataset="Beijing"
)
```

**CLI:**
```bash
uv run python tools/analyze_scenarios.py \
  --eval-dir hoser-evaluation-baseline-abc123 \
  --dataset Beijing
```

### Paired Statistical Tests

#### `compare_models_paired_analysis.py` - Statistical Comparison

**Programmatic:**
```python
from pathlib import Path
from tools.compare_models_paired_analysis import run_paired_comparison

run_paired_comparison(
    real_file=Path("data/Beijing/test.csv"),
    generated_dir=Path("gene/Beijing/testod"),
    output_dir=Path("paired_analysis")
)
```

**CLI:**
```bash
uv run python tools/compare_models_paired_analysis.py \
  --real-file data/Beijing/test.csv \
  --generated-dir gene/Beijing/testod \
  --output-dir paired_analysis
```

## Design Principles

All tools in this directory follow these principles:

### 1. Dual Interface

Every tool provides both CLI and programmatic interfaces:
- **CLI**: For manual execution and shell scripts
- **Programmatic**: For integration into Python workflows

### 2. Fail-Fast Validation

All programmatic functions use `assert` statements for input validation:
```python
assert input_file.exists(), f"File not found: {input_file}"
assert len(data) > 0, "No data to process"
```

### 3. Return Path Objects

Functions return `Path` objects for easy chaining:
```python
od_pairs_file = extract_and_save_abnormal_od_pairs(...)
gene_dir = generate_abnormal_od_trajectories(od_pairs_file=od_pairs_file, ...)
eval_dir = evaluate_abnormal_od(generated_dir=gene_dir, ...)
```

### 4. Comprehensive Docstrings

All functions have docstrings with:
- Description
- Args with types
- Returns with type
- Example usage

### 5. No Subprocess Calls

All tools can be imported and called as modules - no subprocess calls within workflows.

## Testing

Tests are located in `tests/` directory:

```bash
# Run all tool tests
python3 -m pytest tests/test_wang_analysis_tools.py -v

# Run specific test
python3 -m pytest tests/test_wang_analysis_tools.py::TestAnalyzeWangResultsProgrammaticInterface -v
```

## Documentation

- **Comprehensive Guide**: `docs/ABNORMAL_OD_WORKFLOW_GUIDE.md`
- **Evaluation Pipeline**: `docs/EVALUATION_PIPELINE_GUIDE.md`
- **Visualization Guide**: `docs/VISUALIZATION_GUIDE.md`
- **Scenario Analysis**: `docs/SCENARIO_ANALYSIS_GUIDE.md`

## Common Patterns

### Chaining Operations

```python
from pathlib import Path
from tools.extract_abnormal_od_pairs import extract_and_save_abnormal_od_pairs
from tools.generate_abnormal_od import generate_abnormal_od_trajectories
from tools.evaluate_abnormal_od import evaluate_abnormal_od

# Chain phases together
od_pairs_file = extract_and_save_abnormal_od_pairs(
    detection_results_files=[...],
    real_data_files=[...],
    dataset_name="Beijing",
    output_file=Path("abnormal_od_pairs.json")
)

gene_dir = generate_abnormal_od_trajectories(
    od_pairs_file=od_pairs_file,
    model_dir=Path("models"),
    output_dir=Path("gene_abnormal/Beijing/seed42"),
    dataset="Beijing",
    num_traj_per_od=50,
    seed=42
)

eval_dir = evaluate_abnormal_od(
    generated_dir=gene_dir,
    real_abnormal_file=Path("data/Beijing/train.csv"),
    abnormal_od_pairs_file=od_pairs_file,
    output_dir=Path("eval_abnormal/Beijing"),
    dataset="Beijing"
)
```

### Error Handling

```python
from pathlib import Path
from tools.run_abnormal_od_workflow import run_abnormal_od_workflow

try:
    analysis_dir = run_abnormal_od_workflow(
        eval_dir=Path("my-eval"),
        dataset="Beijing",
        real_data_dir=Path("data/Beijing"),
        skip_detection=True
    )
    print(f"Success! Results in: {analysis_dir}")
except AssertionError as e:
    print(f"Validation failed: {e}")
except Exception as e:
    print(f"Workflow failed: {e}")
```

### Parallel Processing

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tools.run_abnormal_od_workflow import run_abnormal_od_workflow

datasets = ["Beijing", "porto_hoser", "BJUT_Beijing"]

def run_for_dataset(dataset):
    return run_abnormal_od_workflow(
        eval_dir=Path("hoser-eval"),
        dataset=dataset,
        real_data_dir=Path(f"data/{dataset}"),
        skip_detection=True
    )

# Run in parallel
with ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(run_for_dataset, datasets))
```

## Contributing

When adding new tools:

1. **Add programmatic interface** with comprehensive docstring
2. **Add CLI interface** using argparse
3. **Use assert for validation** (fail-fast)
4. **Return Path objects** for chaining
5. **Add tests** in `tests/`
6. **Update this README** with usage examples
7. **Update relevant docs** in `docs/`

## Support

For issues or questions:
1. Check function docstrings: `help(function_name)`
2. Review documentation in `docs/`
3. Run with `--help` for CLI usage
4. Check tests in `tests/` for examples
5. Open GitHub issue if needed
