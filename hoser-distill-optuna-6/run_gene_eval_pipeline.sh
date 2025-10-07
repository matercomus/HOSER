#!/bin/bash
# Trajectory Generation & Evaluation Pipeline for HOSER Distillation Comparison
# 
# This script runs trajectory generation (gene.py) and evaluation (evaluation.py)
# for both vanilla and distilled 25-epoch models with proper WandB tracking.
#
# Usage:
#   ./run_gene_eval_pipeline.sh [OPTIONS]
#
# Options:
#   --seed SEED              Random seed (default: 42)
#   --models MODEL1,MODEL2   Models to run (default: vanilla,distilled)
#                           Options: vanilla, distilled
#   --skip-gene             Skip generation (use existing trajectories)
#   --skip-eval             Skip evaluation
#   --cuda DEVICE           CUDA device (default: 0)
#   --num-gene N            Number of trajectories (default: 5000)
#
# Examples:
#   # Run both models with seed 42 (default)
#   ./run_gene_eval_pipeline.sh
#
#   # Run only distilled model with seed 43
#   ./run_gene_eval_pipeline.sh --seed 43 --models distilled
#
#   # Run both models for seeds 42,43,44
#   for seed in 42 43 44; do
#     ./run_gene_eval_pipeline.sh --seed $seed
#   done
#
#   # Re-evaluate existing generated trajectories (skip generation)
#   ./run_gene_eval_pipeline.sh --seed 43 --skip-gene

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default configuration
WANDB_PROJECT="hoser-distill-optuna-6"
DATASET="Beijing"
CUDA_DEVICE=0
NUM_GENE=5000
SEED=42
MODELS="vanilla,distilled"
SKIP_GENE=false
SKIP_EVAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --skip-gene)
            SKIP_GENE=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --cuda)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --num-gene)
            NUM_GENE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seed SEED              Random seed (default: 42)"
            echo "  --models MODEL1,MODEL2   Models to run (default: vanilla,distilled)"
            echo "  --skip-gene             Skip generation"
            echo "  --skip-eval             Skip evaluation"
            echo "  --cuda DEVICE           CUDA device (default: 0)"
            echo "  --num-gene N            Number of trajectories (default: 5000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "HOSER Distillation: Gene & Eval Pipeline"
echo "=========================================="
echo "Project: $WANDB_PROJECT"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Models: $MODELS"
echo "Trajectories: $NUM_GENE"
echo "Skip Gene: $SKIP_GENE"
echo "Skip Eval: $SKIP_EVAL"
echo "=========================================="
echo ""

# Convert comma-separated models to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Function to process a single model
process_model() {
    local MODEL=$1
    local PHASE_NUM=$2
    local TOTAL_PHASES=$3
    
    echo "=========================================="
    echo "Processing: $MODEL (seed $SEED)"
    echo "=========================================="
    
    MODEL_PATH="$SCRIPT_DIR/models/${MODEL}_25epoch_seed${SEED}.pth"
    GENE_DIR="$SCRIPT_DIR/gene/${MODEL}_seed${SEED}"
    EVAL_DIR="$SCRIPT_DIR/eval/${MODEL}_seed${SEED}"
    
    # Check if model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠️  Model not found: $MODEL_PATH"
        echo "    Skipping $MODEL (seed $SEED)"
        echo ""
        return
    fi
    
    # =============================================================================
    # GENERATION PHASE
    # =============================================================================
    if [ "$SKIP_GENE" = false ]; then
        echo "🧬 [$PHASE_NUM/$TOTAL_PHASES] Generating trajectories: $MODEL (seed $SEED)"
        echo "-------------------------------------------------------------------"
        
        mkdir -p "$GENE_DIR"
        
        # Determine tag based on model type
        if [ "$MODEL" = "vanilla" ]; then
            TAGS="vanilla baseline 25epochs seed$SEED generation nx_astar"
        else
            TAGS="distilled final 25epochs seed$SEED generation nx_astar"
        fi
        
        uv run python gene.py \
          --dataset "$DATASET" \
          --seed "$SEED" \
          --cuda "$CUDA_DEVICE" \
          --num_gene "$NUM_GENE" \
          --model_path "$MODEL_PATH" \
          --nx_astar \
          --wandb \
          --wandb_project "$WANDB_PROJECT" \
          --wandb_run_name "gene_${MODEL}_seed${SEED}_nx_astar" \
          --wandb_tags $TAGS
        
        # Move generated file to organized directory
        LATEST_GENE_FILE=$(ls -t gene/$DATASET/seed$SEED/*.csv 2>/dev/null | head -1)
        if [ -f "$LATEST_GENE_FILE" ]; then
            cp "$LATEST_GENE_FILE" "$GENE_DIR/"
            echo "✅ Generated file copied to: $GENE_DIR/"
        else
            echo "❌ Error: No generated file found for $MODEL (seed $SEED)"
            return
        fi
        
        echo ""
    else
        echo "⏭️  Skipping generation for $MODEL (seed $SEED) - using existing trajectories"
        echo ""
    fi
    
    # =============================================================================
    # EVALUATION PHASE
    # =============================================================================
    if [ "$SKIP_EVAL" = false ]; then
        PHASE_NUM=$((PHASE_NUM + 1))
        echo "📊 [$PHASE_NUM/$TOTAL_PHASES] Evaluating trajectories: $MODEL (seed $SEED)"
        echo "-------------------------------------------------------------------"
        
        # Check if generated file exists
        GENE_FILE=$(ls -t "$GENE_DIR"/*.csv 2>/dev/null | head -1)
        if [ ! -f "$GENE_FILE" ]; then
            echo "❌ Error: No generated trajectories found in $GENE_DIR"
            echo "    Run without --skip-gene first, or check if generation completed"
            return
        fi
        
        mkdir -p "$EVAL_DIR"
        
        # Create temporary directory with required structure for evaluation.py
        TEMP_EVAL_DIR="$SCRIPT_DIR/eval/.temp_${MODEL}_seed${SEED}"
        mkdir -p "$TEMP_EVAL_DIR"
        ln -sf "$PROJECT_ROOT/data/$DATASET" "$TEMP_EVAL_DIR/hoser_format"
        
        # Copy generated file to temp eval dir
        cp "$GENE_FILE" "$TEMP_EVAL_DIR/hoser_format/"
        
        # Determine tags based on model type
        if [ "$MODEL" = "vanilla" ]; then
            TAGS="vanilla baseline 25epochs seed$SEED evaluation"
        else
            TAGS="distilled final 25epochs seed$SEED evaluation"
        fi
        
        uv run python evaluation.py \
          --run_dir "$TEMP_EVAL_DIR" \
          --wandb \
          --wandb_project "$WANDB_PROJECT" \
          --wandb_run_name "eval_${MODEL}_seed${SEED}" \
          --wandb_tags $TAGS
        
        # Move evaluation results to organized directory
        EVAL_RESULT_DIR=$(ls -td "$TEMP_EVAL_DIR/eval_"* 2>/dev/null | head -1)
        if [ -d "$EVAL_RESULT_DIR" ]; then
            mv "$EVAL_RESULT_DIR" "$EVAL_DIR/"
            echo "✅ Evaluation results saved to: $EVAL_DIR/"
        fi
        
        # Cleanup temp directory
        rm -rf "$TEMP_EVAL_DIR"
        
        echo ""
    else
        echo "⏭️  Skipping evaluation for $MODEL (seed $SEED)"
        echo ""
    fi
}

# Calculate total phases
TOTAL_PHASES=0
if [ "$SKIP_GENE" = false ]; then
    TOTAL_PHASES=$((TOTAL_PHASES + ${#MODEL_ARRAY[@]}))
fi
if [ "$SKIP_EVAL" = false ]; then
    TOTAL_PHASES=$((TOTAL_PHASES + ${#MODEL_ARRAY[@]}))
fi

# Process each model
CURRENT_PHASE=1
for MODEL in "${MODEL_ARRAY[@]}"; do
    process_model "$MODEL" "$CURRENT_PHASE" "$TOTAL_PHASES"
    if [ "$SKIP_GENE" = false ]; then
        CURRENT_PHASE=$((CURRENT_PHASE + 1))
    fi
    if [ "$SKIP_EVAL" = false ]; then
        CURRENT_PHASE=$((CURRENT_PHASE + 1))
    fi
done

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
echo "🔍 Results by model (seed $SEED):"
for MODEL in "${MODEL_ARRAY[@]}"; do
    EVAL_DIR="$SCRIPT_DIR/eval/${MODEL}_seed${SEED}"
    if [ -d "$EVAL_DIR" ]; then
        RESULT_FILE=$(ls -t "$EVAL_DIR/eval_"*/results.json 2>/dev/null | head -1)
        if [ -f "$RESULT_FILE" ]; then
            echo "   $MODEL: $RESULT_FILE"
        fi
    fi
done
echo ""

