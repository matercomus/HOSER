# Abnormal OD Pair Analysis Workflow

Complete pipeline for testing models on challenging abnormal trajectory scenarios using cross-dataset (BJUT Beijing) as an unseen test set.

## Quick Start

```bash
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6

# Phase 1-2: Detect abnormalities (real + generated) for both datasets
uv run python ../python_pipeline.py --eval-dir . --only abnormal 2>&1 | tee abnormal_baseline.log

# Phase 3: Extract abnormal OD pairs from BJUT
uv run python ../tools/extract_abnormal_od_pairs.py \
  --detection-results ../data/BJUT_Beijing/train/real_data/detection_results.json \
                     ../data/BJUT_Beijing/test/real_data/detection_results.json \
  --real-data ../data/BJUT_Beijing/train.csv \
              ../data/BJUT_Beijing/test.csv \
  --dataset BJUT_Beijing \
  --output abnormal_od_pairs_bjut.json

# Phase 4: Generate trajectories for abnormal OD pairs
uv run python ../tools/generate_abnormal_od.py \
  --od-pairs abnormal_od_pairs_bjut.json \
  --model-dir models \
  --output-dir gene_abnormal/Beijing/seed42 \
  --num-traj 50 \
  --max-pairs 20 \
  --seed 42

# Phase 5: Evaluate model performance on abnormal OD pairs
uv run python ../tools/evaluate_abnormal_od.py \
  --generated-dir gene_abnormal/Beijing/seed42 \
  --real-abnormal-file ../data/BJUT_Beijing/train.csv \
  --abnormal-od-pairs abnormal_od_pairs_bjut.json \
  --output-dir eval_abnormal/Beijing
```

## Workflow Overview

### Phase 1-2: Establish Baselines (✅ Implemented in pipeline)

**Purpose**: Understand abnormal patterns in real-world data

**Command**:
```bash
uv run python ../python_pipeline.py --eval-dir . --only abnormal
```

**Output**:
- `abnormal/Beijing/{train,test}/real_data/` - Beijing dataset abnormalities
- `abnormal/BJUT_Beijing/{train,test}/real_data/` - BJUT dataset abnormalities  
- `abnormal/{dataset}/{split}/generated/{model}/` - Generated trajectory abnormalities
- `abnormal/{dataset}/{split}/comparison_report.json` - Real vs generated comparison

**What to look for**:
- Abnormal rates in real data (~X%)
- If models hallucinate abnormalities (>5% difference from real)
- Which models maintain realistic distributions

---

### Phase 3: Extract Abnormal OD Pairs

**Purpose**: Identify specific challenging origin-destination pairs

**Script**: `tools/extract_abnormal_od_pairs.py`

**Command**:
```bash
uv run python ../tools/extract_abnormal_od_pairs.py \
  --detection-results abnormal/BJUT_Beijing/train/real_data/detection_results.json \
                     abnormal/BJUT_Beijing/test/real_data/detection_results.json \
  --real-data ../data/BJUT_Beijing/train.csv \
              ../data/BJUT_Beijing/test.csv \
  --dataset BJUT_Beijing \
  --output abnormal_od_pairs_bjut.json
```

**Output**: `abnormal_od_pairs_bjut.json`
```json
{
  "dataset": "BJUT_Beijing",
  "total_unique_od_pairs": 150,
  "od_pairs_by_category": {
    "speeding": [[o1, d1], [o2, d2], ...],
    "detour": [...],
    "suspicious_stops": [...],
    "circuitous": [...],
    "unusual_duration": [...]
  }
}
```

**What to look for**:
- How many unique abnormal OD pairs were found
- Which categories have the most abnormal patterns
- These OD pairs will become your targeted test set

---

### Phase 4: Generate for Abnormal OD Pairs

**Purpose**: Test if models can handle challenging edge cases

**Script**: `tools/generate_abnormal_od.py`

**Command**:
```bash
uv run python ../tools/generate_abnormal_od.py \
  --od-pairs abnormal_od_pairs_bjut.json \
  --model-dir models \
  --output-dir gene_abnormal/Beijing/seed42 \
  --num-traj 50 \
  --seed 42
```

**Options**:
- `--num-traj`: Trajectories per OD pair (default: 100)
- `--max-pairs`: Limit OD pairs per category (default: all)
- `--cuda`: CUDA device (default: 0)

**Output**:
- `gene_abnormal/Beijing/seed42/distilled_abnormal_od.csv`
- `gene_abnormal/Beijing/seed42/vanilla_abnormal_od.csv`
- `gene_abnormal/Beijing/seed42/{model}_abnormal_od.csv` (one per model)

**What happens**:
- Loads abnormal OD pairs from Phase 3
- For each model, generates N trajectories for each OD pair
- Uses Beijing road network (models trained on this)
- Total trajectories = #OD_pairs × N × #models

---

### Phase 5: Evaluate Abnormal OD Performance

**Purpose**: Measure model performance on targeted abnormal scenarios

**Script**: `tools/evaluate_abnormal_od.py`

**Command**:
```bash
uv run python ../tools/evaluate_abnormal_od.py \
  --generated-dir gene_abnormal/Beijing/seed42 \
  --real-abnormal-file ../data/BJUT_Beijing/train.csv \
  --abnormal-od-pairs abnormal_od_pairs_bjut.json \
  --output-dir eval_abnormal/Beijing
```

**Output**:
```
eval_abnormal/Beijing/
  distilled/
    detection/detection_results.json  # Abnormalities in generated
    metrics/evaluation_results.json   # EDR, DTW, Hausdorff
    abnormal_od_evaluation.json       # Combined results
  vanilla/
    [same structure]
  comparison_report.json              # Cross-model comparison
```

**Metrics computed**:
1. **Abnormality Reproduction**: Do generated trajectories maintain abnormal patterns?
2. **Similarity Metrics**: How close to real abnormal trajectories?
   - EDR (Edit Distance on Real sequence)
   - DTW (Dynamic Time Warping)
   - Hausdorff (Maximum distance)
3. **Category-specific**: Performance per abnormality type

**What to look for**:
- Which models reproduce abnormal patterns better
- Which models generate more realistic abnormal trajectories
- Which abnormality categories are hardest for models

---

## Expected Results Structure

```
hoser-distill-optuna-6/
├── abnormal/                          # Phase 1-2 output
│   ├── Beijing/
│   │   ├── train/
│   │   │   ├── real_data/
│   │   │   ├── generated/distilled/
│   │   │   ├── generated/vanilla/
│   │   │   └── comparison_report.json
│   │   └── test/...
│   └── BJUT_Beijing/...
├── abnormal_od_pairs_bjut.json        # Phase 3 output
├── gene_abnormal/                     # Phase 4 output
│   └── Beijing/seed42/
│       ├── distilled_abnormal_od.csv
│       └── vanilla_abnormal_od.csv
└── eval_abnormal/                     # Phase 5 output
    └── Beijing/
        ├── distilled/
        ├── vanilla/
        └── comparison_report.json
```

## Key Questions Answered

1. **Do models hallucinate abnormalities?** (Phase 1-2)
   - Compare abnormal rates: real vs generated
   
2. **What OD pairs are challenging?** (Phase 3)
   - Extract abnormal OD pairs from unseen dataset
   
3. **Can models handle edge cases?** (Phase 4-5)
   - Generate for abnormal OD pairs
   - Measure similarity and abnormality reproduction

## Tips

### Adjust Detection Sensitivity

If Phase 1-2 finds 0% abnormalities, thresholds may be too strict. Edit `config/abnormal_detection.yaml`:

```yaml
categories:
  speeding:
    speed_limit_kmh: 60  # Lower from 80
    percentile_threshold: 90  # Lower from 95
  
  detour:
    detour_ratio_threshold: 1.3  # Lower from 1.5
```

### Limit OD Pairs for Quick Testing

Use `--max-pairs` to limit evaluation:

```bash
uv run python tools/generate_abnormal_od.py \
  --od-pairs abnormal_od_pairs_bjut.json \
  --max-pairs 10 \  # Only 10 OD pairs per category
  --num-traj 20 \   # Fewer trajectories per pair
  ...
```

### Run Phases in tmux

Since generation (Phase 4) can take hours:

```bash
tmux new-session -s abnormal-od
cd /home/matt/Dev/HOSER/hoser-distill-optuna-6

# Run Phase 4
uv run python ../tools/generate_abnormal_od.py ...

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t abnormal-od
```

## Benefits of This Workflow

1. **Systematic**: Baselines → extract → generate → evaluate
2. **Generalizable**: Tests on unseen dataset (BJUT)
3. **Targeted**: Focuses on challenging edge cases
4. **Actionable**: Identifies specific OD pairs where models struggle
5. **Comprehensive**: Combines abnormality detection + similarity metrics

