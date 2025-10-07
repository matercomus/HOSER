#!/bin/bash
# Trajectory Generation & Evaluation Pipeline for HOSER Distillation Comparison
# 
# This script runs trajectory generation (gene.py) and evaluation (evaluation.py)
# for both vanilla and distilled 25-epoch models with proper WandB tracking.
#
# Directory structure:
#   hoser-distill-optuna-6/
#     ├── models/              # Model checkpoints
#     ├── gene/                # Generated trajectories
#     │   ├── vanilla/
#     │   └── distilled/
#     ├── eval/                # Evaluation results
#     │   ├── vanilla/
#     │   └── distilled/
#     └── data -> ../../../data  # Symlink to dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

WANDB_PROJECT="hoser-distill-optuna-6"
DATASET="Beijing"
CUDA_DEVICE=0
NUM_GENE=5000
SEED=42

echo "=========================================="
echo "HOSER Distillation: Gene & Eval Pipeline"
echo "=========================================="
echo "Project: $WANDB_PROJECT"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Trajectories: $NUM_GENE"
echo "=========================================="
echo ""

# =============================================================================
# PHASE 1: Generate trajectories for VANILLA model
# =============================================================================
echo "🧬 [1/4] Generating trajectories: VANILLA (25 epochs, seed $SEED)"
echo "-------------------------------------------------------------------"

VANILLA_GENE_DIR="$SCRIPT_DIR/gene/vanilla_seed$SEED"
mkdir -p "$VANILLA_GENE_DIR"

uv run python gene.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --cuda "$CUDA_DEVICE" \
  --num_gene "$NUM_GENE" \
  --model_path "$SCRIPT_DIR/models/vanilla_25epoch_seed$SEED.pth" \
  --nx_astar \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "gene_vanilla_seed${SEED}_nx_astar" \
  --wandb_tags vanilla baseline 25epochs seed$SEED generation nx_astar

# Move generated file to organized directory
LATEST_GENE_FILE=$(ls -t gene/$DATASET/seed$SEED/*.csv | head -1)
if [ -f "$LATEST_GENE_FILE" ]; then
    cp "$LATEST_GENE_FILE" "$VANILLA_GENE_DIR/"
    echo "✅ Generated file copied to: $VANILLA_GENE_DIR/"
else
    echo "❌ Error: No generated file found"
    exit 1
fi

echo ""

# =============================================================================
# PHASE 2: Evaluate VANILLA trajectories
# =============================================================================
echo "📊 [2/4] Evaluating trajectories: VANILLA"
echo "-------------------------------------------------------------------"

VANILLA_EVAL_DIR="$SCRIPT_DIR/eval/vanilla_seed$SEED"
mkdir -p "$VANILLA_EVAL_DIR"

# Create temporary directory with required structure for evaluation.py
VANILLA_TEMP_EVAL_DIR="$SCRIPT_DIR/eval/.temp_vanilla_seed$SEED"
mkdir -p "$VANILLA_TEMP_EVAL_DIR"
ln -sf "$PROJECT_ROOT/data/$DATASET" "$VANILLA_TEMP_EVAL_DIR/hoser_format"

# Copy generated file to temp eval dir
cp "$VANILLA_GENE_DIR"/*.csv "$VANILLA_TEMP_EVAL_DIR/hoser_format/"

uv run python evaluation.py \
  --run_dir "$VANILLA_TEMP_EVAL_DIR" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "eval_vanilla_seed${SEED}" \
  --wandb_tags vanilla baseline 25epochs seed$SEED evaluation

# Move evaluation results to organized directory
if [ -d "$VANILLA_TEMP_EVAL_DIR/eval_"* ]; then
    mv "$VANILLA_TEMP_EVAL_DIR/eval_"* "$VANILLA_EVAL_DIR/"
    echo "✅ Evaluation results saved to: $VANILLA_EVAL_DIR/"
fi

# Cleanup temp directory
rm -rf "$VANILLA_TEMP_EVAL_DIR"

echo ""

# =============================================================================
# PHASE 3: Generate trajectories for DISTILLED model
# =============================================================================
echo "🧬 [3/4] Generating trajectories: DISTILLED (25 epochs, seed $SEED)"
echo "-------------------------------------------------------------------"

DISTILLED_GENE_DIR="$SCRIPT_DIR/gene/distilled_seed$SEED"
mkdir -p "$DISTILLED_GENE_DIR"

uv run python gene.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --cuda "$CUDA_DEVICE" \
  --num_gene "$NUM_GENE" \
  --model_path "$SCRIPT_DIR/models/distilled_25epoch_seed$SEED.pth" \
  --nx_astar \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "gene_distilled_seed${SEED}_nx_astar" \
  --wandb_tags distilled final 25epochs seed$SEED generation nx_astar

# Move generated file to organized directory
LATEST_GENE_FILE=$(ls -t gene/$DATASET/seed$SEED/*.csv | head -1)
if [ -f "$LATEST_GENE_FILE" ]; then
    cp "$LATEST_GENE_FILE" "$DISTILLED_GENE_DIR/"
    echo "✅ Generated file copied to: $DISTILLED_GENE_DIR/"
else
    echo "❌ Error: No generated file found"
    exit 1
fi

echo ""

# =============================================================================
# PHASE 4: Evaluate DISTILLED trajectories
# =============================================================================
echo "📊 [4/4] Evaluating trajectories: DISTILLED"
echo "-------------------------------------------------------------------"

DISTILLED_EVAL_DIR="$SCRIPT_DIR/eval/distilled_seed$SEED"
mkdir -p "$DISTILLED_EVAL_DIR"

# Create temporary directory with required structure for evaluation.py
DISTILLED_TEMP_EVAL_DIR="$SCRIPT_DIR/eval/.temp_distilled_seed$SEED"
mkdir -p "$DISTILLED_TEMP_EVAL_DIR"
ln -sf "$PROJECT_ROOT/data/$DATASET" "$DISTILLED_TEMP_EVAL_DIR/hoser_format"

# Copy generated file to temp eval dir
cp "$DISTILLED_GENE_DIR"/*.csv "$DISTILLED_TEMP_EVAL_DIR/hoser_format/"

uv run python evaluation.py \
  --run_dir "$DISTILLED_TEMP_EVAL_DIR" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "eval_distilled_seed${SEED}" \
  --wandb_tags distilled final 25epochs seed$SEED evaluation

# Move evaluation results to organized directory
if [ -d "$DISTILLED_TEMP_EVAL_DIR/eval_"* ]; then
    mv "$DISTILLED_TEMP_EVAL_DIR/eval_"* "$DISTILLED_EVAL_DIR/"
    echo "✅ Evaluation results saved to: $DISTILLED_EVAL_DIR/"
fi

# Cleanup temp directory
rm -rf "$DISTILLED_TEMP_EVAL_DIR"

echo ""
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo "📂 Results Location:"
echo "   Models:     $SCRIPT_DIR/models/"
echo "   Generated:  $SCRIPT_DIR/gene/"
echo "   Evaluated:  $SCRIPT_DIR/eval/"
echo ""
echo "📊 WandB Project: https://wandb.ai/matercomus/$WANDB_PROJECT"
echo ""
echo "🔍 Quick Comparison:"
echo "   Vanilla results:   $VANILLA_EVAL_DIR/eval_*/results.json"
echo "   Distilled results: $DISTILLED_EVAL_DIR/eval_*/results.json"
echo ""

