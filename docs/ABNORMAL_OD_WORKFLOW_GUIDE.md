... [Previous content remains unchanged until Phase 6 section]

### Phase 6: LM-TAD Teacher Evaluation (Optional)

Evaluate the LM-TAD teacher model on both real and generated trajectories to enable direct performance comparison with student models.

**When to run:**
- Establishing teacher baseline performance
- Direct teacher-student comparison on abnormal patterns
- Validating compression-performance tradeoff

**Configuration:**
```yaml
# config/evaluation.yaml
lmtad:
  # Required for Phase 6
  enabled: true  # Enable LM-TAD teacher evaluation
  repo: /home/matt/Dev/LMTAD  # Path to LM-TAD repository
  checkpoint: /path/to/weights_only.pt  # Teacher checkpoint
  grid_size: 0.002  # Grid size (Beijing: 0.002, Porto: 0.001)
  device: cuda:0  # Device for evaluation

evaluation:
  # Abnormal OD parameters
  max_pairs_per_category: 20
  num_trajectories_per_od: 50
  seed: 42
```

**Programmatic Usage:**
```python
from pathlib import Path
from tools.run_abnormal_od_workflow import run_abnormal_od_workflow

# Run complete workflow including Phase 6
analysis_dir = run_abnormal_od_workflow(
    eval_dir=Path("abnormal/Beijing"),
    dataset="Beijing",
    real_data_dir=Path("data/Beijing"),
    num_trajectories=50,
    max_pairs_per_category=20,
    skip_detection=True  # If detection already exists
)
```

**CLI:**
```bash
uv run python tools/run_abnormal_od_workflow.py \\
  --eval-dir abnormal/Beijing \\
  --dataset Beijing \\
  --real-data-dir data/Beijing \\
  --skip-detection \\
  --num-traj 50 \\
  --max-pairs 20
```

**Output Structure:**
```
eval_lmtad/Beijing/
├── real_data/
│   ├── evaluation_results.tsv   # Per-trajectory perplexity scores
│   └── outlier_stats.json      # Summary statistics (mean, std, threshold)
├── generated/
│   ├── model1/
│   │   ├── trajectories_lmtad_format.csv  # Grid-tokenized trajectories
│   │   ├── evaluation_results.tsv         # Perplexity scores
│   │   └── outlier_stats.json            # Summary statistics
│   └── model2/...
└── comparison_summary.json     # Teacher vs student comparison
```

**Metrics Evaluated:**
1. **Real Data Baseline:**
   - Mean perplexity and standard deviation
   - Outlier threshold (auto-computed)
   - Outlier rate in real trajectories

2. **Generated Trajectories:**
   - Per-model perplexity distributions
   - Outlier rates compared to real baseline
   - Performance retention vs teacher

3. **Teacher vs Student Comparison:**
   - Detection F1 score (teacher)
   - Abnormality reproduction rate (student)
   - Compression-performance tradeoff metrics

**Required Files:**
1. **LM-TAD Repository:**
   - Located at path specified in config
   - Contains model implementation and utils

2. **Teacher Checkpoint:**
   - Trained LM-TAD model weights
   - Format: PyTorch state dict (.pt)

3. **Pre-converted Real Data:**
   - Grid-tokenized real trajectories
   - Matching vocabulary files
   - Used for baseline evaluation

**Integration with Previous Phases:**
1. Uses abnormal OD pairs from Phase 3
2. Evaluates trajectories generated in Phase 4
3. Complements Wang analysis from Phase 5

**Error Resolution:**
1. "LM-TAD real data not found":
   - Check pre-converted data paths
   - Ensure grid tokenization matches teacher

2. "Checkpoint not found":
   - Verify checkpoint path in config
   - Check file permissions

3. "CUDA out of memory":
   ```yaml
   # Reduce batch size in config
   lmtad:
     batch_size: 64  # Default: 128
   ```

**Best Practices:**
1. **Grid Size Consistency:**
   - Use same grid as teacher training
   - Beijing: 0.002, Porto: 0.001

2. **Resource Management:**
   - Enable AMP (automatic mixed precision)
   - Monitor GPU memory usage
   - Adjust batch sizes if needed

3. **Result Validation:**
   - Compare with published baselines
   - Check perplexity distributions
   - Validate outlier thresholds

**Visualization:**
Results can be visualized using the plotting module:

```bash
uv run python tools/plot_lmtad_evaluation.py \\
  --eval-dir eval_lmtad/Beijing \\
  --output-dir figures/lmtad \\
  --dataset Beijing
```

Generated plots include:
- Perplexity distribution comparison
- Outlier rate analysis
- Teacher-student performance gap
- Compression ratio visualization

See `docs/results/ABNORMAL_OD_TEACHER_STUDENT_BRIDGE.md` for interpretation guidelines.

[Rest of the document remains unchanged]