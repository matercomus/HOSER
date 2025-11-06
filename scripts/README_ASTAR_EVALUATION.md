# A* Search Evaluation Scripts

Automated scripts for running complete A* search evaluation pipeline with backup, logging, and monitoring.

## Quick Start

### Launch Both Datasets (Recommended)

```bash
cd /home/matt/Dev/HOSER/scripts
./launch_both_astar_evaluations.sh
```

This will:
- Start Beijing A* evaluation in tmux session `beijing-astar`
- Start Porto A* evaluation in tmux session `porto-astar`
- Both run in parallel
- Expected completion: 12-18 hours

### Run Individual Dataset

```bash
cd /home/matt/Dev/HOSER/scripts

# Beijing only
./run_astar_evaluation.sh beijing

# Porto only
./run_astar_evaluation.sh porto
```

## What the Scripts Do

### 1. Backup
- Creates timestamped backup: `eval.backup.astar_YYYYMMDD_HHMMSS`
- Backs up existing `paired_analysis` directory
- Safe to re-run - never overwrites data

### 2. Generation
- Generates trajectories using A* search (`--use-astar`)
- Creates files with `_astar` suffix

### 3. Evaluation
- Evaluates all generated trajectories
- Creates `trajectory_metrics.json` for paired tests
- Runs paired statistical analysis
- Runs cross-dataset, abnormality, scenario analyses

### 4. Logging
- Main log: `logs/astar_evaluation_TIMESTAMP.log`
- Pipeline log: `logs/pipeline_TIMESTAMP.log`
- Colored console output with timestamps

## Monitoring

### Check Status

```bash
# List running sessions
tmux ls

# Attach to Beijing
tmux attach -t beijing-astar

# Attach to Porto
tmux attach -t porto-astar

# Detach from session: Ctrl+b, then d
```

### View Logs

```bash
# Beijing logs
tail -f ~/Dev/HOSER/hoser-distill-optuna-6/logs/astar_evaluation_*.log
tail -f ~/Dev/HOSER/hoser-distill-optuna-6/logs/pipeline_*.log

# Porto logs
tail -f ~/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/logs/astar_evaluation_*.log
tail -f ~/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/logs/pipeline_*.log
```

### Check Progress

```bash
# Count generated A* files
find ~/Dev/HOSER/hoser-distill-optuna-6/gene -name "*_astar.csv" | wc -l
find ~/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/gene -name "*_astar.csv" | wc -l

# Check latest eval results
ls -lt ~/Dev/HOSER/hoser-distill-optuna-6/eval/ | head -5
ls -lt ~/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/eval/ | head -5
```

## Expected Results

After completion, each dataset will have:

```
eval/
  └── TIMESTAMP/
      ├── results.json              # Aggregate metrics
      └── trajectory_metrics.json   # Per-trajectory metrics

paired_analysis/
  ├── train/
  │   ├── distilled_vs_vanilla/
  │   │   ├── paired_comparison.json
  │   │   └── paired_comparison.md
  │   └── ...
  └── test/
      └── ...

gene/ (or generated/)
  └── dataset/seed42/
      ├── *_astar.csv              # A* search trajectories
      └── *_astar_perf.json        # Performance stats

logs/
  ├── astar_evaluation_*.log       # Main script log
  └── pipeline_*.log               # Pipeline output

eval.backup.astar_TIMESTAMP/       # Backup of previous eval
```

## Troubleshooting

### Script Not Found

```bash
# Ensure scripts are executable
chmod +x /home/matt/Dev/HOSER/scripts/run_astar_evaluation.sh
chmod +x /home/matt/Dev/HOSER/scripts/launch_both_astar_evaluations.sh
```

### Session Already Exists

```bash
# Kill existing session
tmux kill-session -t beijing-astar
tmux kill-session -t porto-astar

# Then relaunch
./launch_both_astar_evaluations.sh
```

### Check for Errors

```bash
# View full logs
cat ~/Dev/HOSER/hoser-distill-optuna-6/logs/astar_evaluation_*.log
grep ERROR ~/Dev/HOSER/hoser-distill-optuna-6/logs/astar_evaluation_*.log
```

### Disk Space

```bash
# Check available space
df -h /home/matt/Dev/HOSER

# Each dataset evaluation uses ~10-20GB
```

## Timeline

**Parallel execution (both datasets):**
- A* Generation: 8-12 hours each
- Evaluation: 2-4 hours each
- Paired Analysis: 5-10 minutes each
- **Total**: ~12-18 hours

**Both run simultaneously, so total clock time is the same as one dataset**

## After Completion

Once tmux sessions exit (check with `tmux ls`):

1. **Verify Results**:
   ```bash
   # Check paired analysis was created
   ls -la ~/Dev/HOSER/hoser-distill-optuna-6/paired_analysis/
   ls -la ~/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/paired_analysis/
   
   # Validate JSON files
   find ~/Dev/HOSER/hoser-distill-optuna-6/paired_analysis -name "*.json" -exec python3 -c "import json; json.load(open('{}'))" \; && echo "✓ All valid"
   ```

2. **Review Summary**:
   ```bash
   # Read final summary from logs
   tail -50 ~/Dev/HOSER/hoser-distill-optuna-6/logs/astar_evaluation_*.log
   tail -50 ~/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/logs/astar_evaluation_*.log
   ```

3. **Compare A* vs Beam Search**:
   - You'll have both A* and Beam results
   - Can compare which search method performs better
   - See `docs/SEARCH_METHOD_GUIDANCE.md` for interpretation

## Safety Features

- ✅ Automatic backups before running
- ✅ Comprehensive logging
- ✅ Error handling and validation
- ✅ Status reporting
- ✅ Safe to disconnect laptop (runs in tmux)
- ✅ Can re-run without data loss

## Notes

- Scripts use `uv run python` for dependency management
- Pipeline respects all config files
- WandB logging is enabled (offline mode works)
- Can run on remote server and disconnect
- Results are saved progressively (safe to interrupt)

---

**Related Documentation**:
- Issue #51: Paired Statistical Tests Implementation
- `docs/PAIRED_STATISTICAL_TESTS_GUIDE.md`
- `docs/SEARCH_METHOD_GUIDANCE.md`
- `docs/BEAM_ABLATION_STUDY.md`

