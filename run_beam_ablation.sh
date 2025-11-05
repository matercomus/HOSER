#!/bin/bash
# Beam Ablation Study: Compare A* search vs Beam Search for Issue 1.8
#
# This script tests whether distillation benefits depend on the search method:
# - A* search (original HOSER paper method)
# - Beam search width=4 (current distillation default)
#
# Estimated runtime: 4-6 hours total
# Results saved to timestamped directories (non-destructive)

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
EVAL_DIR="./hoser-distill-optuna-6"
NUM_TRAJ=1000
SEED=42
BACKUP_DIR="${EVAL_DIR}.backup-$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "Beam Ablation Study: A* vs Beam Search"
echo "================================================================"
echo "Evaluation directory: $EVAL_DIR"
echo "Number of trajectories: $NUM_TRAJ"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo ""

# Backup existing evaluation directory
echo "=== Step 1: Backing up evaluation directory ==="
if [ -d "$EVAL_DIR" ]; then
    cp -r "$EVAL_DIR" "$BACKUP_DIR"
    echo "✅ Backup created: $BACKUP_DIR"
else
    echo "⚠️  Warning: $EVAL_DIR not found, skipping backup"
fi
echo ""

# Run A* search (original HOSER method)
echo "=== Step 2: Running A* search (original HOSER) ==="
echo "Method: Greedy A* without beam search"
echo "Started at: $(date)"
echo ""

uv run python python_pipeline.py \
    --eval-dir "$EVAL_DIR" \
    --only generation,base_eval \
    --use-astar \
    --num-gene "$NUM_TRAJ" \
    --seed "$SEED"

ASTAR_EXIT=$?
if [ $ASTAR_EXIT -eq 0 ]; then
    echo ""
    echo "✅ A* search complete at $(date)"
else
    echo ""
    echo "❌ A* search failed with exit code $ASTAR_EXIT"
    exit $ASTAR_EXIT
fi
echo ""

# Run Beam search (current method)
echo "=== Step 3: Running Beam search (width=4) ==="
echo "Method: Beam search with width 4"
echo "Started at: $(date)"
echo ""

uv run python python_pipeline.py \
    --eval-dir "$EVAL_DIR" \
    --only generation,base_eval \
    --num-gene "$NUM_TRAJ" \
    --seed "$SEED"

BEAM_EXIT=$?
if [ $BEAM_EXIT -eq 0 ]; then
    echo ""
    echo "✅ Beam search complete at $(date)"
else
    echo ""
    echo "❌ Beam search failed with exit code $BEAM_EXIT"
    exit $BEAM_EXIT
fi
echo ""

# Summary
echo "================================================================"
echo "Beam Ablation Study Complete!"
echo "================================================================"
echo "End time: $(date)"
echo ""
echo "📂 Backup: $BACKUP_DIR"
echo "📂 Generated trajectories: $EVAL_DIR/gene/Beijing/seed$SEED/"
echo "📂 Evaluation results: $EVAL_DIR/eval/"
echo ""
echo "Next steps:"
echo "  1. Compare evaluation metrics between A* and Beam search"
echo "  2. Check if distillation benefit differs between methods"
echo "  3. Update Issue 1.8 with findings"
echo ""
echo "To compare results:"
echo "  cd $EVAL_DIR/eval"
echo "  ls -lt | head -10  # View recent evaluation runs"
echo "================================================================"
