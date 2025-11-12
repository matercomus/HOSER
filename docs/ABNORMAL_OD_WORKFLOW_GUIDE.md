# Abnormal OD Workflow Guide

## Overview

The Abnormal OD Workflow analyzes how well trajectory generation models handle edge cases and abnormal patterns. It executes a complete pipeline from detection through evaluation and visualization.

**Key Components:**
- **Phase 0**: Wang statistical abnormality detection on real data
- **Phase 3**: Extract abnormal OD pairs from detection results
- **Phase 4**: Generate trajectories for abnormal OD pairs
- **Phase 5**: Evaluate model performance on abnormal patterns
- **Analysis**: Aggregate results and generate visualizations

## Quick Start

### Using the Workflow Orchestrator (Recommended)

The easiest way to run the complete workflow is using the orchestrator module:

```bash
# Programmatic usage
python3 -c "
from pathlib import Path
from tools.run_abnormal_od_workflow import run_abnormal_od_workflow

analysis_dir = run_abnormal_od_workflow(
    eval_dir=Path('hoser-distill-optuna-6'),
    dataset='Beijing',
    real_data_dir=Path('data/Beijing'),
    num_trajectories=50,
    max_pairs_per_category=20,
    seed=42,
    skip_detection=True  # If detection already exists
)
print(f'Analysis complete: {analysis_dir}')
"

# Or via CLI
uv run python tools/run_abnormal_od_workflow.py \
  --eval-dir hoser-distill-optuna-6 \
  --dataset Beijing \
  --real-data-dir data/Beijing \
  --skip-detection \
  --num-traj 50 \
  --max-pairs 20 \
  --seed 42
```

## Workflow Phases

### Phase 0: Wang Statistical Detection

Identifies abnormal trajectories in real data using Wang et al. 2018 statistical methods.

**When to run:**
- First time analyzing a dataset
- After updating detection thresholds

**When to skip:**
- Detection results already exist
- Rerunning generation/evaluation only

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

**Outputs:**
- `detection_results.json`: Full detection results with abnormal indices
- `statistics_by_category.json`: Summary statistics

### Phase 3: Extract Abnormal OD Pairs

Extracts origin-destination pairs from trajectories marked as abnormal.

**Programmatic:**
```python
from pathlib import Path
from tools.extract_abnormal_od_pairs import extract_and_save_abnormal_od_pairs

extract_and_save_abnormal_od_pairs(
    detection_results_files=[
        Path("abnormal/Beijing/train/real_data/detection_results.json"),
        Path("abnormal/Beijing/test/real_data/detection_results.json")
    ],
    real_data_files=[
        Path("data/Beijing/train.csv"),
        Path("data/Beijing/test.csv")
    ],
    dataset_name="Beijing",
    output_file=Path("abnormal_od_pairs.json")
)
```

**CLI:**
```bash
uv run python tools/extract_abnormal_od_pairs.py \
  --detection-results abnormal/Beijing/train/real_data/detection_results.json \
                      abnormal/Beijing/test/real_data/detection_results.json \
  --real-data data/Beijing/train.csv \
              data/Beijing/test.csv \
  --dataset Beijing \
  --output abnormal_od_pairs.json
```

**Output:**
```json
{
  "dataset": "Beijing",
  "total_abnormal_trajectories": 1234,
  "total_unique_od_pairs": 456,
  "od_pairs_by_category": {
    "speeding": [[100, 200], [150, 250], ...],
    "detour": [[300, 400], ...],
    "suspicious_stops": [...],
    "circuitous": [...],
    "unusual_duration": [...]
  }
}
```

### Phase 4: Generate Trajectories for Abnormal ODs

Generates trajectories using each model for the identified abnormal OD pairs.

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
    max_pairs_per_category=20,  # Limit for faster execution
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

**Parameters:**
- `num_traj_per_od`: Trajectories per OD pair (50-100 recommended)
- `max_pairs_per_category`: Limit pairs per category (20 for testing, None for all)
- `seed`: Random seed for reproducibility

**Output:**
```
gene_abnormal/Beijing/seed42/
├── vanilla_seed42_abnormal_od.csv
├── distilled_seed42_abnormal_od.csv
├── distill_phase1_seed42_abnormal_od.csv
└── ...
```

**Runtime:** 2-4 hours (depends on number of OD pairs and models)

### Phase 5: Evaluate Models on Abnormal ODs

Evaluates how well generated trajectories reproduce abnormal patterns.

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

**Output:**
```
eval_abnormal/Beijing/
├── vanilla_seed42/
│   └── abnormal_od_evaluation.json
├── distilled_seed42/
│   └── abnormal_od_evaluation.json
└── comparison_report.json
```

**Metrics Evaluated:**
- **Abnormality Reproduction Rate**: % of generated trajectories marked abnormal
- **Similarity Metrics**: EDR, DTW, Hausdorff distance vs real abnormal trajectories
- **Category Distribution**: Breakdown by abnormality type

### Analysis & Visualization

Aggregates results and generates publication-quality visualizations.

**Programmatic:**
```python
from pathlib import Path
from tools.analyze_wang_results import analyze_wang_results
from tools.visualize_wang_results import generate_wang_visualizations

# Aggregate Wang detection results
wang_results_file = analyze_wang_results(
    eval_dirs=[Path("hoser-distill-optuna-6")],
    output_file=Path("wang_results_aggregated.json")
)

# Generate visualizations
generate_wang_visualizations(
    results_file=wang_results_file,
    output_dir=Path("figures/wang_abnormality")
)
```

## Output Directory Structure

Complete workflow generates:

```
<eval-dir>/
├── abnormal_od_pairs_<dataset>.json          # Phase 3 output
├── gene_abnormal/<dataset>/seed<N>/          # Phase 4 output
│   ├── model1_abnormal_od.csv
│   ├── model2_abnormal_od.csv
│   └── ...
├── eval_abnormal/<dataset>/                  # Phase 5 output
│   ├── model1/
│   │   └── abnormal_od_evaluation.json
│   ├── model2/
│   │   └── abnormal_od_evaluation.json
│   └── comparison_report.json
├── analysis_abnormal/<dataset>/              # Analysis output
│   ├── wang_results_aggregated.json
│   └── workflow_summary.json
└── figures/
    ├── abnormal_od/<dataset>/                # Abnormal OD plots
    │   ├── abnormality_reproduction_rates.png/svg
    │   ├── similarity_metrics_comparison.png/svg
    │   ├── abnormality_by_category.png/svg
    │   └── metrics_heatmap.png/svg
    └── wang_abnormality/<dataset>/           # Wang detection plots
        ├── abnormality_rates_<dataset>.png/svg
        ├── pattern_distribution_<dataset>.png/svg
        ├── model_rankings_<dataset>_test.png/svg
        └── statistical_significance_<dataset>_test.png/svg
```

## Visualization Outputs

### Abnormal OD Evaluation Plots

**1. Abnormality Reproduction Rates**
- Horizontal bar chart comparing models
- Shows what % of generated trajectories reproduce abnormal patterns
- Color-coded by model type (distilled vs vanilla)
- Labels show effect size (green=small/good, red=large/poor)

**2. Similarity Metrics Comparison**
- Grouped bar chart for EDR, DTW, Hausdorff metrics
- Lower is better (closer to real abnormal trajectories)
- Helps identify which models best match abnormal patterns

**3. Abnormality by Category**
- Stacked bar chart showing distribution across categories
- Categories: speeding, detour, suspicious_stops, circuitous, unusual_duration
- Identifies which abnormal patterns each model reproduces

**4. Metrics Comparison Heatmap**
- Color-coded heatmap (red=worse, green=better)
- Normalized by column for fair comparison
- Shows actual metric values in cells

### Wang Detection Plots

**1. Abnormality Rates**
- Compares real vs generated abnormality rates
- 95% confidence intervals shown
- Effect size color-coding

**2. Pattern Distribution**
- Distribution of Abp1-4 patterns (normal, temporal delay, route deviation, both)
- Separate plots per dataset

**3. Model Rankings**
- Top 6 models by realism (closest to real data)
- Separate plots for test/train splits
- With confidence intervals

**4. Statistical Significance**
- -log10(p-value) for each comparison
- Color-coded by effect size
- Cohen's h annotated on bars

## Standalone Plotting

The plotting functionality has been decoupled from the workflow orchestrator and can be run independently on existing data.

### Evaluation Plots (Generated Trajectories)

Generate plots for model evaluation results without running the full workflow:

```bash
uv run python tools/plot_abnormal_evaluation.py \
  --comparison-report eval_abnormal/Beijing/comparison_report.json \
  --output-dir figures/abnormal_od/Beijing \
  --dataset Beijing
```

**Use Cases:**
- Regenerating plots after modifying visualization code
- Creating plots for evaluation results from previous runs
- Comparing results across different evaluation directories

### Analysis Plots (Real Abnormal Data)

Generate analysis plots for real abnormal trajectories without trajectory generation:

```bash
uv run python tools/plot_abnormal_analysis.py \
  --abnormal-od-pairs abnormal_od_pairs_Beijing.json \
  --real-data-dir data/Beijing \
  --detection-results-dir abnormal/Beijing \
  --samples-dir abnormal/Beijing \
  --output-dir figures/abnormal_od/Beijing \
  --dataset Beijing \
  --include-normal
```

**Use Cases:**
- Analyzing real abnormal patterns without generating trajectories
- Comparing abnormal vs normal OD patterns
- Creating publication-ready visualizations from existing detection results

### Normal OD Heatmap

The `--include-normal` flag enables generation of normal OD heatmaps for comparison:

**What it does:**
1. Extracts all normal trajectories (excluding those marked as abnormal)
2. Computes OD pair frequencies from normal trajectories
3. Uses the same top N origins/destinations as the abnormal heatmap for direct comparison
4. Generates side-by-side comparison plots

**Interpretation:**
- **Red areas in abnormal heatmap, blue in normal**: OD pairs over-represented in abnormal trajectories
- **Red in both**: Common OD pairs that can appear in both normal and abnormal contexts
- **Blue in normal only**: OD pairs that are common in normal trajectories but rare in abnormal ones

This comparison helps identify:
- Spatial patterns that distinguish abnormal from normal behavior
- Which OD pairs are most indicative of abnormal patterns
- Whether certain road segments are more prone to abnormal trajectories

## Use Cases

### Within-Dataset Analysis

Test models on abnormal OD pairs from the same dataset they were trained on.

```bash
uv run python tools/run_abnormal_od_workflow.py \
  --eval-dir hoser-distill-optuna-porto-eval \
  --dataset porto_hoser \
  --real-data-dir data/porto_hoser \
  --skip-detection \
  --num-traj 50 \
  --max-pairs 20 \
  --seed 42
```

**Research question**: How well do models handle edge cases in their training domain?

### Cross-Dataset Analysis

Test models trained on one dataset using abnormal OD pairs from another.

```bash
# Train on Porto, test on BJUT Beijing
uv run python tools/run_abnormal_od_workflow.py \
  --eval-dir hoser-distill-optuna-porto-eval \
  --dataset BJUT_Beijing \
  --real-data-dir data/BJUT_Beijing \
  --detection-config config/abnormal_detection_statistical.yaml \
  --num-traj 50 \
  --max-pairs 20 \
  --seed 42
```

**Research question**: Do models generalize abnormal pattern reproduction to unseen road networks?

**Note**: Cross-dataset requires road network translation (see `tools/map_road_networks.py`)

## Parameter Tuning

### Number of Trajectories (`--num-traj`)

- **10-20**: Quick testing
- **50**: Standard testing (recommended)
- **100**: Thorough analysis
- **200+**: Publication-quality (slow)

### Max Pairs per Category (`--max-pairs`)

- **5-10**: Quick prototype
- **20**: Balanced testing (recommended)
- **None/0**: All pairs (thorough but slow)

### Considerations

- More trajectories = more statistical power, longer runtime
- More OD pairs = broader coverage, longer runtime
- Trade-off: 50 trajectories × 20 pairs = ~1000 trajectories per model
- Runtime: ~1-2 minutes per model per 1000 trajectories on GPU

## Troubleshooting

### "Detection results not found"

**Solution**: Run detection first or provide `--detection-config`:
```bash
uv run python tools/run_abnormal_od_workflow.py \
  --eval-dir path/to/eval \
  --dataset Beijing \
  --real-data-dir data/Beijing \
  --detection-config config/abnormal_detection_statistical.yaml
```

### "No model files found"

**Solution**: Ensure `<eval-dir>/models/` contains `.pth` files:
```bash
ls -la <eval-dir>/models/*.pth
```

### High abnormality rates (>30%)

**Possible causes:**
1. Detection thresholds too sensitive
2. Models genuinely produce abnormal patterns
3. Cross-dataset without road network translation

**Check:**
- Review `config/abnormal_detection_statistical.yaml` thresholds
- Compare within-dataset vs cross-dataset results
- Verify road network translation quality

### "CUDA out of memory"

**Solution**: Reduce batch size or number of trajectories:
```python
run_abnormal_od_workflow(
    ...,
    num_trajectories=25,  # Reduced from 50
    max_pairs_per_category=10  # Reduced from 20
)
```

## Interpreting Results

### Good Performance Indicators

- **Abnormality rate**: 0-12% (similar to real data)
- **Similarity metrics**: Low EDR/DTW/Hausdorff (close to real abnormal patterns)
- **Category distribution**: Similar to real data distribution
- **Statistical significance**: Small effect sizes (Cohen's h < 0.2)

### Areas for Improvement

- **High abnormality rate** (>20%): Models may be generating unrealistic patterns
- **High similarity metrics**: Generated abnormal trajectories don't match real ones well
- **Skewed category distribution**: Models reproduce some abnormality types but not others
- **Large effect sizes**: Significant deviation from real data

### Cross-Dataset Expectations

Expect slightly higher abnormality rates and metrics in cross-dataset analysis:
- Within-dataset: 0-8% abnormal
- Cross-dataset: 5-15% abnormal (still acceptable)
- Large gap (>20 points) indicates poor generalization

## Integration with Other Tools

### With Scenario Analysis

```python
# First run abnormal OD workflow
from tools.run_abnormal_od_workflow import run_abnormal_od_workflow
analysis_dir = run_abnormal_od_workflow(...)

# Then analyze scenarios
from tools.analyze_scenarios import analyze_all_scenarios
analyze_all_scenarios(eval_dir=eval_dir, dataset=dataset)
```

### With Paired Statistical Tests

```python
# After abnormal OD evaluation
from tools.compare_models_paired_analysis import run_paired_comparison

run_paired_comparison(
    real_file=Path("data/Beijing/test.csv"),
    generated_dir=Path("gene_abnormal/Beijing/seed42"),
    output_dir=Path("paired_analysis_abnormal")
)
```

## References

- **Wang et al. 2018**: Statistical abnormality detection methodology
- **docs/archive/ABNORMAL_CROSS_DATASET.md**: Cross-dataset analysis considerations
- **docs/EVALUATION_PIPELINE_GUIDE.md**: Standard evaluation workflow
- **docs/VISUALIZATION_GUIDE.md**: Visualization best practices

## API Reference

### Main Workflow Function

```python
def run_abnormal_od_workflow(
    eval_dir: Path,
    dataset: str,
    real_data_dir: Path,
    num_trajectories: int = 50,
    max_pairs_per_category: Optional[int] = 20,
    seed: int = 42,
    skip_detection: bool = False,
    detection_config: Optional[Path] = None,
) -> Path:
    """
    Run complete abnormal OD workflow.
    
    Returns:
        Path to analysis output directory
    """
```

### Individual Phase Functions

```python
# Phase 0
def run_abnormal_analysis(
    real_file: Path,
    dataset: str,
    config_path: Path,
    output_dir: Path
) -> None

# Phase 3
def extract_and_save_abnormal_od_pairs(
    detection_results_files: List[Path],
    real_data_files: List[Path],
    dataset_name: str,
    output_file: Path
) -> Path

# Phase 4
def generate_abnormal_od_trajectories(
    od_pairs_file: Path,
    model_dir: Path,
    output_dir: Path,
    dataset: str,
    num_traj_per_od: int = 50,
    max_pairs_per_category: Optional[int] = None,
    seed: int = 42,
    cuda_device: int = 0
) -> Path

# Phase 5
def evaluate_abnormal_od(
    generated_dir: Path,
    real_abnormal_file: Path,
    abnormal_od_pairs_file: Path,
    output_dir: Path,
    dataset: str = "Beijing"
) -> Path

# Analysis
def analyze_wang_results(
    eval_dirs: Optional[List[Path]] = None,
    output_file: Optional[Path] = None
) -> Path

def generate_wang_visualizations(
    results_file: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> None
```

## Best Practices

1. **Start with small tests**: Use `--num-traj 10 --max-pairs 5` for quick validation
2. **Skip detection when possible**: Use `--skip-detection` if results exist
3. **Use consistent seeds**: Enables reproducible comparisons across runs
4. **Check intermediate outputs**: Verify each phase before proceeding
5. **Monitor GPU memory**: Reduce batch sizes if needed
6. **Save plots as SVG**: Better for publications (`figures/` contains both PNG and SVG)
7. **Document parameters**: Keep track of `num_traj`, `max_pairs`, and `seed` used

## Example: Complete Porto Workflow

```bash
# Within-dataset analysis
uv run python tools/run_abnormal_od_workflow.py \
  --eval-dir hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732 \
  --dataset porto_hoser \
  --real-data-dir data/porto_hoser \
  --skip-detection \
  --num-traj 50 \
  --max-pairs 20 \
  --seed 42

# Results in:
# - hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/figures/abnormal_od/porto_hoser/
# - hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/analysis_abnormal/porto_hoser/
```

## Support

For questions or issues:
1. Check troubleshooting section above
2. Review example outputs in `docs/archive/ABNORMAL_ANALYSIS_RESULTS_BEIJING.md`
3. Open an issue on GitHub with:
   - Command used
   - Error message
   - `workflow_summary.json` contents
