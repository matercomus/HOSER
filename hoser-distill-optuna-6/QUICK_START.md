# 🚀 Quick Start: Running Gene & Eval for Seed 42

## Step 1: Navigate to Directory

```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6
```

## Step 2: Run Pipeline (In Your Tmux Session!)

```bash
./run_gene_eval_pipeline.sh
```

That's it! This will:
1. Generate 5000 trajectories with vanilla model (seed 42)
2. Evaluate vanilla trajectories
3. Generate 5000 trajectories with distilled model (seed 42)
4. Evaluate distilled trajectories

**Runtime**: ~1.5-3 hours total

---

## ✨ Smart Caching Features

The pipeline is **idempotent** and uses intelligent caching:

### First Run (Seed 42)
```bash
./run_gene_eval_pipeline.sh
```
- Generates all trajectories
- Evaluates all trajectories
- ~1.5-3 hours

### Second Run (Seed 42)
```bash
./run_gene_eval_pipeline.sh
```
- ✅ Finds existing generated files → skips generation
- ✅ Finds existing evaluation → skips evaluation  
- **Completes instantly!**

### Adding Seeds 43 & 44 Later
```bash
# Option A: Run individually (recommended)
./run_gene_eval_pipeline.sh --seed 43 --models distilled
./run_gene_eval_pipeline.sh --seed 44 --models distilled

# Option B: Batch run
./run_all_seeds.sh --seeds "43 44" --models distilled
```
- Only runs new seeds (42 already cached)
- Only runs distilled (vanilla already cached for all seeds)

---

## 🔄 Force Re-run

If you want to regenerate everything:

```bash
./run_gene_eval_pipeline.sh --force
```

Or re-evaluate with new metrics (keeps trajectories):

```bash
./run_gene_eval_pipeline.sh --skip-gene
```

---

## 📊 Check Results

```bash
# View results
cat eval/vanilla_seed42/eval_*/results.json
cat eval/distilled_seed42/eval_*/results.json

# Or on WandB
# https://wandb.ai/matercomus/hoser-distill-optuna-6
```

---

## 📁 What Gets Created

After first run:
```
hoser-distill-optuna-6/
├── gene/
│   ├── vanilla_seed42/
│   │   └── 2025-10-07_12-30-45.csv    ← Generated trajectories
│   └── distilled_seed42/
│       └── 2025-10-07_13-45-20.csv
└── eval/
    ├── vanilla_seed42/
    │   └── eval_2025-10-07_14-00-30/
    │       └── results.json             ← Evaluation metrics
    └── distilled_seed42/
        └── eval_2025-10-07_15-15-45/
            └── results.json
```

All results are automatically uploaded to WandB!

---

## 🎯 For Your Workflow

### Now (Seed 42 Ready)
```bash
# Run in tmux session
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6
./run_gene_eval_pipeline.sh
```

### Later (When Seeds 43 & 44 Finish Training)

First, add the new models:
```bash
# Download or copy distilled models to:
models/distilled_25epoch_seed43.pth
models/distilled_25epoch_seed44.pth
```

Then run:
```bash
# Option 1: Run just the new seeds with distilled models
./run_gene_eval_pipeline.sh --seed 43 --models distilled
./run_gene_eval_pipeline.sh --seed 44 --models distilled

# Option 2: Batch run all new seeds
./run_all_seeds.sh --seeds "43 44" --models distilled

# Option 3: Run all seeds (will skip 42 due to caching)
./run_all_seeds.sh
```

The pipeline is smart enough to:
- ✅ Skip seed 42 (already done)
- ✅ Skip vanilla for 43 & 44 (vanilla only needs one seed)
- 🧬 Only generate & evaluate distilled for seeds 43 & 44

---

## 💡 Pro Tips

1. **Always run in tmux** so you can disconnect your laptop
2. **Default behavior is cached** - re-running is instant if already done
3. **Use `--force` sparingly** - only when you need to regenerate
4. **Check WandB** for live progress and comparisons
5. **Results are preserved** - old files stay, new ones added

---

## 🆘 If Something Goes Wrong

### Re-run just generation for one model:
```bash
./run_gene_eval_pipeline.sh --seed 42 --models vanilla --skip-eval --force
```

### Re-run just evaluation (fast, uses cached trajectories):
```bash
./run_gene_eval_pipeline.sh --seed 42 --models vanilla --skip-gene
```

### Force complete re-run:
```bash
./run_gene_eval_pipeline.sh --seed 42 --force
```

