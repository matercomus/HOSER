#!/bin/bash
# Beam Ablation Study: Complete A* vs Beam Search with Normalized Metrics
#
# This script completes the beam ablation study with normalized metrics:
# 1. Re-evaluates existing A* trajectories with normalized metrics
# 2. Generates trajectories using beam search
# 3. Evaluates beam search with normalized metrics
#
# Estimated runtime: 5-7 hours total

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
EVAL_DIR="./hoser-distill-optuna-6"
NUM_TRAJ=1000
SEED=42

echo "================================================================"
echo "Beam Ablation Study: A* vs Beam Search (Normalized Metrics)"
echo "================================================================"
echo "Evaluation directory: $EVAL_DIR"
echo "Number of trajectories: $NUM_TRAJ"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo ""
echo "✨ Using NEW normalized metrics (Hausdorff_norm, DTW_norm)"
echo "   Added in Issue #14, commit 241b4cf"
echo ""

# Verify normalized metrics are in evaluation.py
if grep -q "Hausdorff_norm" evaluation.py; then
    echo "✅ Confirmed: evaluation.py has normalized metrics"
else
    echo "❌ ERROR: evaluation.py missing normalized metrics!"
    echo "   Please ensure you're on main branch with commit 241b4cf"
    exit 1
fi
echo ""

# Step 1: Re-evaluate A* trajectories with normalized metrics
echo "=== Step 1: Re-evaluating A* results with normalized metrics ==="
echo "Using existing A* trajectories from seed$SEED"
echo "Started at: $(date)"
echo ""

uv run python python_pipeline.py \
    --eval-dir "$EVAL_DIR" \
    --only base_eval \
    --seed "$SEED"

ASTAR_EVAL_EXIT=$?
if [ $ASTAR_EVAL_EXIT -eq 0 ]; then
    echo ""
    echo "✅ A* re-evaluation complete at $(date)"
else
    echo ""
    echo "❌ A* re-evaluation failed with exit code $ASTAR_EVAL_EXIT"
    exit $ASTAR_EVAL_EXIT
fi
echo ""

# Step 2: Run Beam search generation and evaluation
echo "=== Step 2: Running Beam search (width=4) ==="
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
echo "Beam Ablation Study Complete with Normalized Metrics!"
echo "================================================================"
echo "End time: $(date)"
echo ""
echo "📂 Generated trajectories: $EVAL_DIR/gene/Beijing/seed$SEED/"
echo "📂 Evaluation results: $EVAL_DIR/eval/"
echo ""
echo "📊 Both A* and Beam now evaluated with normalized metrics:"
echo "   - Hausdorff_norm (km/point) - trajectory-length independent ⭐"
echo "   - DTW_norm (km/point) - trajectory-length independent ⭐"
echo "   - Hausdorff_km (km) - raw distance (backward compatible)"
echo "   - DTW_km (km) - raw distance (backward compatible)"
echo ""
echo "Next steps:"
echo "  1. Compare A* vs Beam using normalized metrics"
echo "  2. Check if distillation benefit differs between methods"
echo "  3. Update Issue #8 with findings"
echo ""
echo "To compare results:"
echo "  cd $EVAL_DIR/eval"
echo "  ls -lt | head -20  # View recent evaluation runs"
echo ""
echo "Expected comparison:"
echo "  - A* (distilled): ~0.28 traj/s"
echo "  - A* (vanilla): ~0.05 traj/s (6x slower)"
echo "  - Beam (distilled): ? traj/s"
echo "  - Beam (vanilla): ? traj/s"
echo "================================================================"
