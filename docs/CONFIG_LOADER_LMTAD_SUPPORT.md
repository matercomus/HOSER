# Config Loader LM-TAD Support Documentation

## Overview

The `tools/config_loader.py` module now supports LM-TAD teacher baseline evaluation through new configuration fields and helper methods. This allows the same evaluation workflows (e.g., abnormal OD analysis) to be used for both HOSER student models and LM-TAD teacher models.

## Key Features

### 1. New Configuration Fields

Three new fields added to `EvaluationConfig`:

```python
lmtad_evaluation: bool = False      # Enable LM-TAD teacher evaluation
lmtad_repo: Optional[Path] = None   # Path to LM-TAD repository root
lmtad_checkpoint: Optional[Path] = None  # Path to checkpoint (.pt file)
lmtad_real_data_dir: Optional[Path] = None  # Real data directory (optional)
```

### 2. Helper Methods

#### `get_lmtad_checkpoint() -> Optional[Path]`

Resolves and validates the LM-TAD checkpoint path.

**Features:**
- Returns `None` if `lmtad_evaluation=False`
- Smart path resolution (searches: repo root, eval dir, CWD)
- Validates checkpoint file exists
- Clear error messages if misconfigured

**Example:**
```python
config = load_evaluation_config(eval_dir=Path("hoser-eval"))
if config.lmtad_evaluation:
    ckpt = config.get_lmtad_checkpoint()
    print(f"Teacher checkpoint: {ckpt}")
```

#### `get_lmtad_real_data_path() -> Optional[Path]`

Gets the real data directory for teacher evaluation.

**Features:**
- Returns `None` if `lmtad_evaluation=False`
- Priority order:
  1. Explicit `lmtad_real_data_dir`
  2. Fall back to `data_dir` (if LM-TAD format)
  3. Auto-detect from `lmtad_repo + dataset`
- Validates directory exists (warns if missing)

**Example:**
```python
config = load_evaluation_config(eval_dir=Path("hoser-eval"))
if config.lmtad_evaluation:
    real_data = config.get_lmtad_real_data_path()
    train_csv = real_data / "train.csv"
```

### 3. YAML Configuration Format

#### Minimal LM-TAD Configuration

```yaml
dataset: Beijing
lmtad_evaluation: true
lmtad_repo: /home/matt/Dev/LMTAD
lmtad_checkpoint: code/results/LMTAD/beijing_hoser_reference/.../weights_only.pt
```

#### Full LM-TAD Configuration

```yaml
# Dataset
dataset: Beijing
data_dir: /home/matt/Dev/LMTAD/data/beijing_hoser_reference

# LM-TAD settings
lmtad_evaluation: true
lmtad_repo: /home/matt/Dev/LMTAD
lmtad_checkpoint: code/results/LMTAD/beijing_hoser_reference/.../weights_only.pt
lmtad_real_data_dir: /home/matt/Dev/LMTAD/data/beijing_hoser_reference

# Evaluation settings
num_gene: 5000
beam_width: 4
seed: 42
grid_size: 0.001
edr_eps: 100.0
```

## Backward Compatibility

**All new fields are optional and backward compatible:**
- Existing configs work without modification
- Default behavior: HOSER student evaluation (`lmtad_evaluation=False`)
- All new fields default to safe values
- Legacy field names supported (e.g., `cross_dataset_name`)

## Path Resolution Logic

### Checkpoint Path Resolution

When `lmtad_checkpoint` is specified as a relative path:

1. Try `lmtad_repo / checkpoint_path`
2. Try `eval_dir / checkpoint_path`
3. Try `CWD / checkpoint_path`
4. Raise `FileNotFoundError` if not found

### Real Data Path Resolution

When `lmtad_real_data_dir` is not specified:

1. Use `data_dir` if provided
2. Auto-detect: `lmtad_repo / "data" / f"{dataset}_hoser_reference"`
3. Raise `ValueError` if cannot be determined

## Usage Examples

### Example 1: Standard HOSER Evaluation (No Changes Required)

```python
# Existing config (no LM-TAD fields)
config = load_evaluation_config(eval_dir=Path("hoser-eval"))
assert config.lmtad_evaluation == False  # Default
assert config.get_lmtad_checkpoint() is None
```

### Example 2: LM-TAD Teacher Evaluation

```python
# LM-TAD config with all fields
config = load_evaluation_config(eval_dir=Path("lmtad-eval"))
if config.lmtad_evaluation:
    # Get checkpoint path
    ckpt = config.get_lmtad_checkpoint()

    # Get real data directory
    real_data = config.get_lmtad_real_data_path()
    train_csv = real_data / "train.csv"
    test_csv = real_data / "test.csv"
```

### Example 3: Auto-Detection

```yaml
# Minimal config - auto-detects real data path
dataset: Beijing
lmtad_evaluation: true
lmtad_repo: /home/matt/Dev/LMTAD
lmtad_checkpoint: code/results/LMTAD/beijing_hoser_reference/.../weights_only.pt
# lmtad_real_data_dir auto-detected: /home/matt/Dev/LMTAD/data/beijing_hoser_reference
```

## Configuration Template

A complete example configuration file is provided:
- **Location**: `config/evaluation_lmtad_example.yaml`
- **Purpose**: Template for LM-TAD teacher evaluation
- **Usage**: Copy to eval directory and customize paths

## Testing

The implementation includes comprehensive tests covering:
1. Backward compatibility with minimal configs
2. LM-TAD evaluation configuration
3. Helper method defaults
4. Path resolution logic

Run tests:
```bash
uv run python3 -c "import tests..."  # See implementation
```

## Data Format Differences

**Teacher (LM-TAD) vs Student (HOSER):**

| Aspect | Teacher (LM-TAD) | Student (HOSER) |
|--------|------------------|-----------------|
| Data Format | Grid-based tokens | Road network graphs |
| Vocabulary | Spatial grid cells | Road IDs |
| Input | Token sequences | Road sequences |
| Data Directory | `lmtad_real_data_dir` | `data_dir` |

Both formats can be generated from the same raw trajectory data using appropriate preprocessing.

## Error Handling

### Clear Error Messages

```python
# Missing checkpoint when lmtad_evaluation=True
ValueError: lmtad_evaluation=True but lmtad_checkpoint not specified in config

# Checkpoint file not found
FileNotFoundError: LM-TAD checkpoint not found: /path/to/checkpoint.pt
Original config value: code/results/...
lmtad_repo: /home/matt/Dev/LMTAD

# Cannot determine real data path
ValueError: Cannot determine LM-TAD real data path. Please specify one of:
  - lmtad_real_data_dir (preferred)
  - data_dir (if it's LM-TAD format)
  - lmtad_repo + dataset (for auto-detection)
```

## Integration with Workflows

This configuration system integrates with:
- `run_abnormal_od_workflow.py` - Abnormal OD analysis
- `python_pipeline.py` - General evaluation pipeline
- Custom evaluation scripts using `config_loader`

**Workflow Compatibility:**
- Same workflow can evaluate teacher and student models
- Configuration determines which model to evaluate
- Metrics and analysis remain consistent across models

## Future Enhancements

Potential future additions:
- LM-TAD model configuration (layers, heads, embedding dim)
- Teacher-specific evaluation metrics
- Distillation configuration integration
- Multi-teacher support for ensemble evaluation

## References

- **Implementation**: `tools/config_loader.py`
- **Example Config**: `config/evaluation_lmtad_example.yaml`
- **Teacher Results**: `docs/results/TEACHER_BASELINE_COMPARISON.md`
- **Workflow Guide**: `docs/ABNORMAL_OD_WORKFLOW_GUIDE.md`
