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
#   --od-source SOURCE      OD pair source: train or test (default: train)
#                           train: Use training OD pairs (seen during training)
#                           test: Use test OD pairs (evaluates generalization)
#   --skip-gene             Skip generation (use existing trajectories)
#   --skip-eval             Skip evaluation
#   --force                 Force re-run even if results exist (overrides caching)
#   --cuda DEVICE           CUDA device (default: 0)
#   --num-gene N            Number of trajectories (default: 5000)
#   --search-mode MODE      Search algorithm (default: model_astar)
#                           Options: model_astar, nx_astar, beam_search
#
# Examples:
#   # Run both models on both train and test ODs (default, uses cache if available)
#   ./run_gene_eval_pipeline.sh
#
#   # Run only train OD evaluation (memorization test)
#   ./run_gene_eval_pipeline.sh --od-source train
#
#   # Run only test OD evaluation (generalization test)
#   ./run_gene_eval_pipeline.sh --od-source test
#
#   # Force re-run everything even if results exist
#   ./run_gene_eval_pipeline.sh --force
#
#   # Run only distilled model with seed 43 on both OD sources
#   ./run_gene_eval_pipeline.sh --seed 43 --models distilled
#
#   # Run both models for seeds 42,43,44 on both train and test ODs
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
OD_SOURCE="train,test"
SKIP_GENE=false
SKIP_EVAL=false
FORCE=false

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
        --od-source)
            OD_SOURCE="$2"
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
        --force)
            FORCE=true
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
            echo "  --od-source SOURCE      OD source: train or test (default: train)"
            echo "  --skip-gene             Skip generation"
            echo "  --skip-eval             Skip evaluation"
            echo "  --force                 Force re-run even if results exist"
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
echo "OD Sources: $OD_SOURCE (train=memorization, test=generalization)"
echo "Trajectories: $NUM_GENE"
echo "Skip Gene: $SKIP_GENE"
echo "Skip Eval: $SKIP_EVAL"
echo "Force Re-run: $FORCE"
echo "=========================================="
echo ""

# Convert comma-separated models to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Convert OD sources to array (supports "train", "test", or "train,test")
IFS=',' read -ra OD_SOURCE_ARRAY <<< "$OD_SOURCE"

# Function to process a single model with a specific OD source
process_model() {
    local MODEL=$1
    local OD_SRC=$2
    local PHASE_NUM=$3
    local TOTAL_PHASES=$4
    
    echo "=========================================="
    echo "Processing: $MODEL (seed $SEED, ${OD_SRC}_od)"
    echo "=========================================="
    
    MODEL_PATH="$SCRIPT_DIR/models/${MODEL}_25epoch_seed${SEED}.pth"
    GENE_DIR="$SCRIPT_DIR/gene/${MODEL}_seed${SEED}_${OD_SRC}od"
    EVAL_DIR="$SCRIPT_DIR/eval/${MODEL}_seed${SEED}_${OD_SRC}od"
    
    # Check if model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠️  Model not found: $MODEL_PATH"
        echo "    Skipping $MODEL (seed $SEED, ${OD_SRC}_od)"
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
        
        # Check if generation already exists (caching)
        EXISTING_GENE_FILE=$(ls -t "$GENE_DIR"/*.csv 2>/dev/null | head -1)
        if [ -f "$EXISTING_GENE_FILE" ] && [ "$FORCE" = false ]; then
            echo "✅ Found existing generated file: $(basename "$EXISTING_GENE_FILE")"
            echo "⏭️  Skipping generation (use --force to regenerate)"
            echo ""
        else
            if [ "$FORCE" = true ] && [ -f "$EXISTING_GENE_FILE" ]; then
                echo "🔄 Force re-run: regenerating trajectories (existing file will be kept)"
            fi
            
            # Determine tag based on model type and OD source
            if [ "$MODEL" = "vanilla" ]; then
                TAGS="vanilla baseline 25epochs seed$SEED ${OD_SRC}_od generation beam4"
            else
                TAGS="distilled final 25epochs seed$SEED ${OD_SRC}_od generation beam4"
            fi
            
            uv run python gene.py \
              --dataset "$DATASET" \
              --seed "$SEED" \
              --cuda "$CUDA_DEVICE" \
              --num_gene "$NUM_GENE" \
              --model_path "$MODEL_PATH" \
              --od_source "$OD_SRC" \
              --beam_search \
              --beam_width 4 \
              --wandb \
              --wandb_project "$WANDB_PROJECT" \
              --wandb_run_name "gene_${MODEL}_seed${SEED}_${OD_SRC}od_beam4" \
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
        fi
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
        
        # Check if evaluation already exists (caching)
        EXISTING_EVAL_RESULT=$(ls -td "$EVAL_DIR/eval_"*/results.json 2>/dev/null | head -1)
        if [ -f "$EXISTING_EVAL_RESULT" ] && [ "$FORCE" = false ]; then
            echo "✅ Found existing evaluation: $(dirname "$EXISTING_EVAL_RESULT" | xargs basename)"
            echo "⏭️  Skipping evaluation (use --force to re-evaluate)"
            echo ""
        else
            if [ "$FORCE" = true ] && [ -f "$EXISTING_EVAL_RESULT" ]; then
                echo "🔄 Force re-run: re-evaluating trajectories (existing results will be kept)"
            fi
            
            # Create temporary directory with required structure for evaluation.py
            TEMP_EVAL_DIR="$SCRIPT_DIR/eval/.temp_${MODEL}_seed${SEED}_${OD_SRC}od"
            mkdir -p "$TEMP_EVAL_DIR"
            ln -sf "$PROJECT_ROOT/data/$DATASET" "$TEMP_EVAL_DIR/hoser_format"
            
            # Copy generated file to temp eval dir
            cp "$GENE_FILE" "$TEMP_EVAL_DIR/hoser_format/"
            
            # Determine tags based on model type and OD source
            if [ "$MODEL" = "vanilla" ]; then
                TAGS="vanilla baseline 25epochs seed$SEED ${OD_SRC}_od evaluation"
            else
                TAGS="distilled final 25epochs seed$SEED ${OD_SRC}_od evaluation"
            fi
            
            uv run python evaluation.py \
              --run_dir "$TEMP_EVAL_DIR" \
              --wandb \
              --wandb_project "$WANDB_PROJECT" \
              --wandb_run_name "eval_${MODEL}_seed${SEED}_${OD_SRC}od" \
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
        fi
    else
        echo "⏭️  Skipping evaluation for $MODEL (seed $SEED)"
        echo ""
    fi
}

# Calculate total phases (models × OD sources × phases per model)
TOTAL_PHASES=0
if [ "$SKIP_GENE" = false ]; then
    TOTAL_PHASES=$((TOTAL_PHASES + ${#MODEL_ARRAY[@]} * ${#OD_SOURCE_ARRAY[@]}))
fi
if [ "$SKIP_EVAL" = false ]; then
    TOTAL_PHASES=$((TOTAL_PHASES + ${#MODEL_ARRAY[@]} * ${#OD_SOURCE_ARRAY[@]}))
fi

# Process each OD source and model combination
CURRENT_PHASE=1
for OD_SRC in "${OD_SOURCE_ARRAY[@]}"; do
    for MODEL in "${MODEL_ARRAY[@]}"; do
        process_model "$MODEL" "$OD_SRC" "$CURRENT_PHASE" "$TOTAL_PHASES"
        if [ "$SKIP_GENE" = false ]; then
            CURRENT_PHASE=$((CURRENT_PHASE + 1))
        fi
        if [ "$SKIP_EVAL" = false ]; then
            CURRENT_PHASE=$((CURRENT_PHASE + 1))
        fi
    done
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
echo "🔍 Results by OD source and model (seed $SEED):"
for OD_SRC in "${OD_SOURCE_ARRAY[@]}"; do
    echo ""
    echo "  📍 ${OD_SRC^^} OD Pairs ($([ "$OD_SRC" = "train" ] && echo "memorization" || echo "generalization")):"
    for MODEL in "${MODEL_ARRAY[@]}"; do
        EVAL_DIR="$SCRIPT_DIR/eval/${MODEL}_seed${SEED}_${OD_SRC}od"
        if [ -d "$EVAL_DIR" ]; then
            RESULT_FILE=$(ls -t "$EVAL_DIR/eval_"*/results.json 2>/dev/null | head -1)
            if [ -f "$RESULT_FILE" ]; then
                echo "     - $MODEL: $(basename "$(dirname "$RESULT_FILE")")"
            fi
        fi
    done
done
echo ""

