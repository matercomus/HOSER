#!/bin/bash
###############################################################################
# Beijing: Sequential Training (3 models) → A* Evaluation
#
# This script trains 3 new Beijing models to complete the 3×3 grid,
# then runs A* evaluation on all 6 models (3 vanilla + 3 distilled).
#
# Usage: ./train_then_eval_beijing.sh
#
# Models to train:
#   1. vanilla_25epoch_seed43.pth
#   2. vanilla_25epoch_seed44.pth
#   3. distilled_25epoch_seed43.pth
#
# Timeline: ~87 hours total (~3.6 days)
#   - Training: 75 hours (3 × 25 hours)
#   - Evaluation: 12 hours
###############################################################################

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
HOSER_ROOT="/home/matt/Dev/HOSER"
DATA_DIR="/home/matt/Dev/HOSER-dataset"
CONFIG_FILE="$HOSER_ROOT/config/Beijing.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Logging setup
mkdir -p "$HOSER_ROOT/logs"
MAIN_LOG="$HOSER_ROOT/logs/train_then_eval_beijing_${TIMESTAMP}.log"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

# Start banner
log_info "================================================================"
log_info "Beijing: Sequential Training + A* Evaluation"
log_info "================================================================"
log_info "Start time: $(date)"
log_info "Log file: $MAIN_LOG"
log_info ""
log_info "Phase 1: Training 3 models (75 hours)"
log_info "  1. vanilla_seed43 (~25 hours)"
log_info "  2. vanilla_seed44 (~25 hours)"
log_info "  3. distilled_seed43 (~25 hours)"
log_info ""
log_info "Phase 2: A* Evaluation (12 hours)"
log_info "  - All 6 models (3 vanilla + 3 distilled)"
log_info "  - 2 OD sources (train + test)"
log_info ""
log_info "Total estimated time: ~87 hours (~3.6 days)"
log_info "================================================================"
log_info ""

cd "$HOSER_ROOT"

# Verify prerequisites
log_info "Verifying prerequisites..."

if [ ! -d "$DATA_DIR" ]; then
    log_error "Dataset directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

log_success "Prerequisites verified"
log_info ""

# Track overall start time
OVERALL_START=$(date +%s)

###############################################################################
# PHASE 1: MODEL TRAINING (3 models)
###############################################################################

log_info "================================================================"
log_info "PHASE 1: TRAINING NEW MODELS"
log_info "================================================================"
log_info ""

# Model 1: vanilla_seed43
MODEL_1_START=$(date +%s)
log_info "🚀 Training Model 1/3: vanilla_25epoch_seed43"
log_info "   Seed: 43"
log_info "   Epochs: 25"
log_info "   Distillation: Disabled"
log_info ""

# Ensure distillation is disabled for vanilla
export TRAIN_VANILLA=1  # Flag to disable distillation if script supports it

uv run python train_with_distill.py \
    --dataset Beijing \
    --config "$CONFIG_FILE" \
    --seed 43 \
    --cuda 0 \
    --data_dir "$DATA_DIR" 2>&1 | tee -a "$MAIN_LOG"

MODEL_1_EXIT=${PIPESTATUS[0]}
MODEL_1_END=$(date +%s)
MODEL_1_DURATION=$((MODEL_1_END - MODEL_1_START))
MODEL_1_HOURS=$((MODEL_1_DURATION / 3600))
MODEL_1_MINUTES=$(((MODEL_1_DURATION % 3600) / 60))

if [ $MODEL_1_EXIT -eq 0 ]; then
    log_success "Model 1/3 complete: vanilla_seed43"
    log_info "   Duration: ${MODEL_1_HOURS}h ${MODEL_1_MINUTES}m"
else
    log_error "Model 1/3 FAILED with exit code: $MODEL_1_EXIT"
    exit $MODEL_1_EXIT
fi
log_info ""

# Model 2: vanilla_seed44
MODEL_2_START=$(date +%s)
log_info "🚀 Training Model 2/3: vanilla_25epoch_seed44"
log_info "   Seed: 44"
log_info "   Epochs: 25"
log_info "   Distillation: Disabled"
log_info ""

uv run python train_with_distill.py \
    --dataset Beijing \
    --config "$CONFIG_FILE" \
    --seed 44 \
    --cuda 0 \
    --data_dir "$DATA_DIR" 2>&1 | tee -a "$MAIN_LOG"

MODEL_2_EXIT=${PIPESTATUS[0]}
MODEL_2_END=$(date +%s)
MODEL_2_DURATION=$((MODEL_2_END - MODEL_2_START))
MODEL_2_HOURS=$((MODEL_2_DURATION / 3600))
MODEL_2_MINUTES=$(((MODEL_2_DURATION % 3600) / 60))

if [ $MODEL_2_EXIT -eq 0 ]; then
    log_success "Model 2/3 complete: vanilla_seed44"
    log_info "   Duration: ${MODEL_2_HOURS}h ${MODEL_2_MINUTES}m"
else
    log_error "Model 2/3 FAILED with exit code: $MODEL_2_EXIT"
    exit $MODEL_2_EXIT
fi
log_info ""

# Model 3: distilled_seed43
MODEL_3_START=$(date +%s)
log_info "🚀 Training Model 3/3: distilled_25epoch_seed43"
log_info "   Seed: 43"
log_info "   Epochs: 25"
log_info "   Distillation: Enabled"
log_info ""
log_warn "   NOTE: Ensure distillation is enabled in $CONFIG_FILE"
log_info ""

# Unset vanilla flag for distilled model
unset TRAIN_VANILLA

uv run python train_with_distill.py \
    --dataset Beijing \
    --config "$CONFIG_FILE" \
    --seed 43 \
    --cuda 0 \
    --data_dir "$DATA_DIR" 2>&1 | tee -a "$MAIN_LOG"

MODEL_3_EXIT=${PIPESTATUS[0]}
MODEL_3_END=$(date +%s)
MODEL_3_DURATION=$((MODEL_3_END - MODEL_3_START))
MODEL_3_HOURS=$((MODEL_3_DURATION / 3600))
MODEL_3_MINUTES=$(((MODEL_3_DURATION % 3600) / 60))

if [ $MODEL_3_EXIT -eq 0 ]; then
    log_success "Model 3/3 complete: distilled_seed43"
    log_info "   Duration: ${MODEL_3_HOURS}h ${MODEL_3_MINUTES}m"
else
    log_error "Model 3/3 FAILED with exit code: $MODEL_3_EXIT"
    exit $MODEL_3_EXIT
fi
log_info ""

# Phase 1 summary
PHASE1_END=$(date +%s)
PHASE1_DURATION=$((PHASE1_END - OVERALL_START))
PHASE1_HOURS=$((PHASE1_DURATION / 3600))
PHASE1_MINUTES=$(((PHASE1_DURATION % 3600) / 60))

log_info "================================================================"
log_success "PHASE 1 COMPLETE: All 3 models trained successfully"
log_info "================================================================"
log_info "Training Summary:"
log_info "  Model 1 (vanilla_seed43): ${MODEL_1_HOURS}h ${MODEL_1_MINUTES}m"
log_info "  Model 2 (vanilla_seed44): ${MODEL_2_HOURS}h ${MODEL_2_MINUTES}m"
log_info "  Model 3 (distilled_seed43): ${MODEL_3_HOURS}h ${MODEL_3_MINUTES}m"
log_info "  Total Phase 1 duration: ${PHASE1_HOURS}h ${PHASE1_MINUTES}m"
log_info "================================================================"
log_info ""

# Verify model files
log_info "Verifying model files..."
MODELS_DIR="$HOSER_ROOT/hoser-distill-optuna-6/models"

if [ -f "$MODELS_DIR/vanilla_25epoch_seed43.pth" ]; then
    SIZE=$(du -h "$MODELS_DIR/vanilla_25epoch_seed43.pth" | cut -f1)
    log_success "  vanilla_25epoch_seed43.pth ($SIZE)"
else
    log_error "  vanilla_25epoch_seed43.pth NOT FOUND"
fi

if [ -f "$MODELS_DIR/vanilla_25epoch_seed44.pth" ]; then
    SIZE=$(du -h "$MODELS_DIR/vanilla_25epoch_seed44.pth" | cut -f1)
    log_success "  vanilla_25epoch_seed44.pth ($SIZE)"
else
    log_error "  vanilla_25epoch_seed44.pth NOT FOUND"
fi

if [ -f "$MODELS_DIR/distilled_25epoch_seed43.pth" ]; then
    SIZE=$(du -h "$MODELS_DIR/distilled_25epoch_seed43.pth" | cut -f1)
    log_success "  distilled_25epoch_seed43.pth ($SIZE)"
else
    log_error "  distilled_25epoch_seed43.pth NOT FOUND"
fi

log_info ""

###############################################################################
# PHASE 2: A* EVALUATION
###############################################################################

log_info "================================================================"
log_info "PHASE 2: A* EVALUATION (All 6 Beijing Models)"
log_info "================================================================"
log_info ""
log_info "Evaluating models:"
log_info "  - vanilla_25epoch_seed42.pth (existing)"
log_info "  - vanilla_25epoch_seed43.pth (new)"
log_info "  - vanilla_25epoch_seed44.pth (new)"
log_info "  - distilled_25epoch_seed42.pth (existing)"
log_info "  - distilled_25epoch_seed43.pth (new)"
log_info "  - distilled_25epoch_seed44.pth (existing)"
log_info ""
log_info "This will:"
log_info "  1. Generate A* trajectories (5000 per model × 2 OD sources)"
log_info "  2. Evaluate with normalized metrics"
log_info "  3. Run paired statistical analysis"
log_info "  4. Generate trajectory-level metrics"
log_info ""

PHASE2_START=$(date +%s)

# Run A* evaluation script
./scripts/run_astar_evaluation.sh beijing 2>&1 | tee -a "$MAIN_LOG"

EVAL_EXIT=${PIPESTATUS[0]}
PHASE2_END=$(date +%s)
PHASE2_DURATION=$((PHASE2_END - PHASE2_START))
PHASE2_HOURS=$((PHASE2_DURATION / 3600))
PHASE2_MINUTES=$(((PHASE2_DURATION % 3600) / 60))

if [ $EVAL_EXIT -ne 0 ]; then
    log_error "A* evaluation FAILED with exit code: $EVAL_EXIT"
    exit $EVAL_EXIT
fi

log_success "PHASE 2 COMPLETE: A* evaluation finished"
log_info "  Duration: ${PHASE2_HOURS}h ${PHASE2_MINUTES}m"
log_info ""

###############################################################################
# FINAL SUMMARY
###############################################################################

OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_HOURS=$((OVERALL_DURATION / 3600))
OVERALL_MINUTES=$(((OVERALL_DURATION % 3600) / 60))

log_info "================================================================"
log_success "ALL OPERATIONS COMPLETE!"
log_info "================================================================"
log_info "Timeline:"
log_info "  Phase 1 (Training): ${PHASE1_HOURS}h ${PHASE1_MINUTES}m"
log_info "  Phase 2 (Evaluation): ${PHASE2_HOURS}h ${PHASE2_MINUTES}m"
log_info "  Total duration: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m"
log_info ""
log_info "Deliverables:"
log_info "  Models: 3 new models in hoser-distill-optuna-6/models/"
log_info "  Evaluations: A* results for all 6 models"
log_info "  Paired analysis: Statistical comparisons complete"
log_info ""
log_info "Next steps:"
log_info "  1. Verify cross-seed variance (should be <10%)"
log_info "  2. Check paired_analysis/ for model comparisons"
log_info "  3. Run cross-seed analysis tools"
log_info "  4. Update Issue #55 with results"
log_info ""
log_info "Log saved to: $MAIN_LOG"
log_info "End time: $(date)"
log_info "================================================================"

exit 0

