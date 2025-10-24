# Performance Profiling Implementation Summary

## Overview
Added comprehensive inference performance profiling to HOSER trajectory generation and evaluation pipeline. The system now tracks detailed timing metrics for trajectory generation, including per-trajectory statistics, model forward pass efficiency, and throughput measurements.

## Implementation Details

### 1. Searcher Class Timing (`gene.py`)
**Commit:** `b4d4690` - ✨ feat: add performance profiling to Searcher class

- Added CUDA-aware timing using `torch.cuda.Event()` for GPU precision
- Fallback to `time.perf_counter()` for CPU timing
- Modified `search()` and `beam_search()` to return performance metrics:
  - `total_time`: Total trajectory generation time
  - `forward_time_total`: Cumulative model forward pass time
  - `forward_count`: Number of model forward passes
  - `forward_time_avg`: Average time per forward pass
  - `beam_width`: Beam search configuration (beam_search only)

**Key Features:**
- Zero overhead using PyTorch best practices
- Accurate async operation timing with CUDA events
- Per-trajectory granular metrics

### 2. Trajectory Generation Metrics (`gene.py`)
**Commit:** `0a5cd61` - ✨ feat: add performance metrics collection to trajectory generation

- Updated `generate_trajectories_programmatic()` to collect timing stats
- Calculate aggregate statistics across all trajectories:
  - Mean/std/min/max/median/95th percentile for total time
  - Forward pass efficiency metrics
  - Throughput (trajectories per second)
  - Total generation time
- Modified return type from `str` to `dict` with:
  - `output_file`: Path to generated CSV
  - `num_generated`: Number of trajectories
  - `performance`: Dict with timing statistics
- Automatic WandB logging of performance metrics

### 3. Evaluation Integration (`evaluation.py`)
**Commit:** `4812f3b` - ✨ feat: integrate performance metrics into evaluation

- Added `generation_performance` parameter to `evaluate_trajectories_programmatic()`
- Include performance metrics in results dict and JSON output
- Enhanced console output with performance summary section showing:
  - Throughput (traj/s)
  - Mean/min/max/std/median/p95 timing
  - Forward passes per trajectory
  - Forward time per step
  - Device and configuration info
- Automatic WandB logging under `generation/` namespace

### 4. Pipeline Integration (`hoser-distill-optuna-6/python_pipeline.py`)
**Commit:** `9395695` - ✨ feat: integrate performance metrics into evaluation pipeline

- Updated `TrajectoryGenerator.generate_trajectories()` to return `(path, performance)` tuple
- Updated `TrajectoryEvaluator.evaluate_trajectories()` to accept `generation_performance`
- Pass metrics from generation to evaluation in pipeline
- Log performance metrics to WandB with `perf/` namespace
- Display performance summary in pipeline logs

## Metrics Collected

### Per-Trajectory Metrics
- `total_time`: End-to-end generation time (seconds)
- `forward_time_total`: Model inference time (seconds)
- `forward_count`: Number of model forward passes
- `forward_time_avg`: Average forward pass time (seconds)

### Aggregate Statistics
- `total_time_mean/std/min/max/median/p95`: Distribution statistics
- `forward_time_mean/std`: Forward pass distribution
- `forward_count_mean/std`: Forward pass count distribution
- `throughput_traj_per_sec`: Trajectories per second
- `total_generation_time`: Total wall-clock time
- `forward_time_per_step_mean`: Average model inference per step

### Configuration Info
- `beam_search_enabled`: Boolean
- `beam_width`: Int
- `device`: String (e.g., "cuda:0")

## Usage

### Programmatic API

```python
from gene import generate_trajectories_programmatic
from evaluation import evaluate_trajectories_programmatic

# Generate trajectories with performance profiling
result = generate_trajectories_programmatic(
    dataset='Beijing',
    model_path='./save/Beijing/seed42_distill/best.pth',
    od_source='test',
    seed=42,
    num_gene=100,
    cuda_device=0,
    beam_search=True,
    beam_width=4
)

# Extract results
output_file = result['output_file']
num_generated = result['num_generated']
performance = result['performance']

print(f"Generated {num_generated} trajectories")
print(f"Throughput: {performance['throughput_traj_per_sec']:.2f} traj/s")
print(f"Mean time: {performance['total_time_mean']:.3f}s")

# Evaluate with performance metrics
eval_results = evaluate_trajectories_programmatic(
    generated_file=output_file,
    dataset='Beijing',
    od_source='test',
    generation_performance=performance  # Include generation metrics
)

# Results JSON now includes 'generation_performance' section
```

### Pipeline Usage

```bash
cd hoser-distill-optuna-6
uv run python python_pipeline.py --num-gene 100
```

Performance metrics automatically included in:
- Console output
- `results.json` files
- WandB logs (under `perf/` and `generation/` namespaces)

## Example Output

### Console Output
```
--- Generation Performance ---
Throughput                1.91 traj/s
Mean time/trajectory      0.523s
Min time                  0.312s
Max time                  1.245s
Std dev                   0.142s
Median time               0.498s
95th percentile           0.789s
Forward passes/traj       12.5
Forward time/step         33.0ms
Beam width                4
Device                    cuda:0
Total generation time     52.3s
```

### JSON Output
```json
{
    "Distance_JSD": 0.1234,
    "Duration_JSD": 0.2345,
    "generation_performance": {
        "total_time_mean": 0.523,
        "total_time_std": 0.142,
        "throughput_traj_per_sec": 1.91,
        "forward_time_mean": 0.412,
        "forward_count_mean": 12.5,
        "beam_width": 4,
        "device": "cuda:0"
    }
}
```

## Testing

To test the implementation:

```bash
# Test generation only (small batch)
cd hoser-distill-optuna-6
uv run python -c "
import sys
sys.path.insert(0, '..')
from gene import generate_trajectories_programmatic

result = generate_trajectories_programmatic(
    dataset='Beijing',
    model_path='./models/distilled_25epoch_seed42.pth',
    od_source='test',
    seed=42,
    num_gene=10,  # Small batch for testing
    cuda_device=0,
    beam_search=True,
    beam_width=4
)

print(f\"Generated: {result['num_generated']} trajectories\")
print(f\"Performance: {result['performance']['throughput_traj_per_sec']:.2f} traj/s\")
"

# Test full pipeline
uv run python python_pipeline.py --num-gene 20 --models distilled --od-source test
```

## Benefits

1. **Performance Profiling**: Understand model inference speed and bottlenecks
2. **Bottleneck Identification**: Identify slow trajectories or operations
3. **Device Comparison**: Compare GPU vs CPU performance
4. **Configuration Tuning**: Optimize beam width vs speed tradeoff
5. **Reproducibility**: Track inference performance across runs
6. **WandB Integration**: Visualize performance trends across experiments

## Technical Details

### PyTorch Best Practices
- Uses `torch.cuda.Event()` for accurate GPU timing
- Calls `torch.cuda.synchronize()` before and after timing
- Minimal overhead (CUDA events are lightweight)
- Fallback to `time.perf_counter()` for CPU

### Backward Compatibility
- Generation performance parameter is optional in evaluation
- Existing code continues to work without modifications
- Performance metrics gracefully omitted if not provided

## Branch
All changes are in the `feat/speed-eval` branch.

## Commits
1. `b4d4690` - Add performance profiling to Searcher class
2. `0a5cd61` - Add performance metrics collection to trajectory generation
3. `4812f3b` - Integrate performance metrics into evaluation
4. `9395695` - Integrate performance metrics into evaluation pipeline
