#!/bin/bash
###############################################################################
# A* Search Evaluation Pipeline Runner
# 
# Usage: ./run_astar_evaluation.sh <dataset>
#   dataset: beijing or porto
#
# This script:
#   1. Creates timestamped log files
#   2. Backs up existing eval directory
#   3. Runs full pipeline with A* search
#   4. Handles errors and provides status updates
###############################################################################

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "  dataset: beijing or porto"
    exit 1
fi

DATASET=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# Validate dataset
if [[ "$DATASET" != "beijing" && "$DATASET" != "porto" ]]; then
    echo "Error: Dataset must be 'beijing' or 'porto'"
    exit 1
fi

# Project root
PROJECT_ROOT="/home/matt/Dev/HOSER"

# Dataset-specific paths
if [ "$DATASET" == "beijing" ]; then
    EVAL_DIR="$PROJECT_ROOT/hoser-distill-optuna-6"
    DATASET_NAME="Beijing"
elif [ "$DATASET" == "porto" ]; then
    EVAL_DIR="$PROJECT_ROOT/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732"
    DATASET_NAME="Porto"
fi

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory if it doesn't exist
mkdir -p "$EVAL_DIR/logs"

# Main log file for this operation
MAIN_LOG="$EVAL_DIR/logs/astar_evaluation_${TIMESTAMP}.log"

# Start logging
log_info "================================================================"
log_info "A* Search Evaluation Pipeline - $DATASET_NAME Dataset"
log_info "================================================================"
log_info "Start time: $(date)"
log_info "Evaluation directory: $EVAL_DIR"
log_info "Log file: $MAIN_LOG"
log_info ""

# Check if eval directory exists
if [ ! -d "$EVAL_DIR" ]; then
    log_error "Evaluation directory not found: $EVAL_DIR"
    exit 1
fi

cd "$EVAL_DIR"

# Step 1: Backup existing eval directory
log_info "Step 1: Backing up existing evaluation directory..."

if [ -d "eval" ]; then
    BACKUP_DIR="eval.backup.astar_${TIMESTAMP}"
    log_info "Creating backup: $BACKUP_DIR"
    
    cp -r eval "$BACKUP_DIR"
    
    if [ $? -eq 0 ]; then
        BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
        log_success "Backup created successfully: $BACKUP_DIR (Size: $BACKUP_SIZE)"
    else
        log_error "Backup failed!"
        exit 1
    fi
else
    log_warn "No existing eval directory found, skipping backup"
fi

# Step 2: Backup existing paired_analysis directory if it exists
if [ -d "paired_analysis" ]; then
    PAIRED_BACKUP_DIR="paired_analysis.backup.astar_${TIMESTAMP}"
    log_info "Backing up paired_analysis: $PAIRED_BACKUP_DIR"
    cp -r paired_analysis "$PAIRED_BACKUP_DIR"
    log_success "Paired analysis backup created"
fi

log_info ""

# Step 3: Check for existing A* results
log_info "Step 2: Checking for existing generated trajectories..."

GENE_DIR=$(find . -type d -name "gene" -o -name "generated" | head -1)
if [ -n "$GENE_DIR" ]; then
    TOTAL_CSV=$(find "$GENE_DIR" -name "*.csv" | wc -l)
    ASTAR_CSV=$(find "$GENE_DIR" -name "*_astar.csv" | wc -l)
    BEAM_CSV=$((TOTAL_CSV - ASTAR_CSV))
    
    log_info "Found $TOTAL_CSV trajectory files:"
    log_info "  - Beam search: $BEAM_CSV files"
    log_info "  - A* search: $ASTAR_CSV files"
    
    if [ $ASTAR_CSV -gt 0 ]; then
        log_warn "A* search files already exist. Pipeline will regenerate or use existing."
    fi
fi

log_info ""

# Step 4: Run the pipeline
log_info "Step 3: Running full evaluation pipeline with A* search..."
log_info "This will:"
log_info "  1. Generate trajectories using A* search"
log_info "  2. Evaluate all trajectories"
log_info "  3. Run paired statistical analysis"
log_info "  4. Run cross-dataset evaluation"
log_info "  5. Run abnormality detection"
log_info "  6. Run scenario analysis"
log_info ""

PIPELINE_LOG="$EVAL_DIR/logs/pipeline_${TIMESTAMP}.log"
log_info "Pipeline output will be logged to: $PIPELINE_LOG"
log_info ""

START_TIME=$(date +%s)

# Run the pipeline
log_info "Starting pipeline execution..."
uv run python ../python_pipeline.py --use-astar 2>&1 | tee "$PIPELINE_LOG"

PIPELINE_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_info ""
log_info "================================================================"

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    log_success "Pipeline completed successfully!"
else
    log_error "Pipeline failed with exit code: $PIPELINE_EXIT_CODE"
fi

log_info "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_info "End time: $(date)"

# Step 5: Verify results
log_info ""
log_info "Step 4: Verifying results..."

# Check for trajectory_metrics.json files
TRAJ_METRICS=$(find eval -name "trajectory_metrics.json" 2>/dev/null | wc -l)
log_info "Trajectory metrics files generated: $TRAJ_METRICS"

# Check for paired analysis results
if [ -d "paired_analysis" ]; then
    PAIRED_JSON=$(find paired_analysis -name "paired_comparison.json" 2>/dev/null | wc -l)
    PAIRED_MD=$(find paired_analysis -name "paired_comparison.md" 2>/dev/null | wc -l)
    log_info "Paired analysis results:"
    log_info "  - JSON files: $PAIRED_JSON"
    log_info "  - Markdown files: $PAIRED_MD"
else
    log_warn "No paired_analysis directory found"
fi

# Check for A* generated files
if [ -n "$GENE_DIR" ]; then
    NEW_ASTAR_CSV=$(find "$GENE_DIR" -name "*_astar.csv" -newer "$BACKUP_DIR" 2>/dev/null | wc -l)
    if [ $NEW_ASTAR_CSV -gt 0 ]; then
        log_success "Generated $NEW_ASTAR_CSV new A* trajectory files"
    fi
fi

log_info ""
log_info "================================================================"
log_info "Summary for $DATASET_NAME Dataset"
log_info "================================================================"
log_info "Status: $([ $PIPELINE_EXIT_CODE -eq 0 ] && echo 'SUCCESS ✅' || echo 'FAILED ❌')"
log_info "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_info "Logs saved to:"
log_info "  - Main log: $MAIN_LOG"
log_info "  - Pipeline log: $PIPELINE_LOG"
log_info "Backup location: $BACKUP_DIR"
log_info "================================================================"

exit $PIPELINE_EXIT_CODE

