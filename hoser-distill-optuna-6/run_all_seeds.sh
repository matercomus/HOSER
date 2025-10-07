#!/bin/bash
# Batch runner for gene & eval pipeline across multiple seeds
# 
# Usage:
#   ./run_all_seeds.sh [OPTIONS]
#
# Options:
#   --seeds "42 43 44"      Seeds to run (default: "42 43 44")
#   --models MODEL1,MODEL2  Models to run (default: vanilla,distilled)
#   --skip-gene            Skip generation
#   --skip-eval            Skip evaluation
#   --force                Force re-run even if results exist
#   --cuda DEVICE          CUDA device (default: 0)
#   --num-gene N           Number of trajectories (default: 5000)
#
# Examples:
#   # Run all seeds with both models (default)
#   ./run_all_seeds.sh
#
#   # Run only seeds 43 and 44 with distilled models
#   ./run_all_seeds.sh --seeds "43 44" --models distilled
#
#   # Re-evaluate all existing generated trajectories
#   ./run_all_seeds.sh --skip-gene

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration
SEEDS="42 43 44"
MODELS="vanilla,distilled"
SKIP_GENE=false
SKIP_EVAL=false
FORCE=false
CUDA_DEVICE=0
NUM_GENE=5000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
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
            echo "  --seeds \"42 43 44\"      Seeds to run (default: \"42 43 44\")"
            echo "  --models MODEL1,MODEL2  Models to run (default: vanilla,distilled)"
            echo "  --skip-gene            Skip generation"
            echo "  --skip-eval            Skip evaluation"
            echo "  --force                Force re-run even if results exist"
            echo "  --cuda DEVICE          CUDA device (default: 0)"
            echo "  --num-gene N           Number of trajectories (default: 5000)"
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
echo "Batch Gene & Eval: Multiple Seeds"
echo "=========================================="
echo "Seeds: $SEEDS"
echo "Models: $MODELS"
echo "=========================================="
echo ""

# Convert seeds to array
SEED_ARRAY=($SEEDS)
TOTAL_SEEDS=${#SEED_ARRAY[@]}

# Run pipeline for each seed
for i in "${!SEED_ARRAY[@]}"; do
    SEED="${SEED_ARRAY[$i]}"
    SEED_NUM=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "SEED $SEED_NUM/$TOTAL_SEEDS: $SEED"
    echo "=========================================="
    echo ""
    
    # Build arguments for run_gene_eval_pipeline.sh
    ARGS="--seed $SEED --models $MODELS --cuda $CUDA_DEVICE --num-gene $NUM_GENE"
    
    if [ "$SKIP_GENE" = true ]; then
        ARGS="$ARGS --skip-gene"
    fi
    
    if [ "$SKIP_EVAL" = true ]; then
        ARGS="$ARGS --skip-eval"
    fi
    
    if [ "$FORCE" = true ]; then
        ARGS="$ARGS --force"
    fi
    
    # Run pipeline for this seed
    "$SCRIPT_DIR/run_gene_eval_pipeline.sh" $ARGS
    
    echo ""
    echo "✅ Completed seed $SEED ($SEED_NUM/$TOTAL_SEEDS)"
    echo ""
done

echo "=========================================="
echo "✅ All Seeds Complete!"
echo "=========================================="
echo ""
echo "📊 Summary:"
for SEED in $SEEDS; do
    echo "  Seed $SEED:"
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    for MODEL in "${MODEL_ARRAY[@]}"; do
        EVAL_DIR="$SCRIPT_DIR/eval/${MODEL}_seed${SEED}"
        if [ -d "$EVAL_DIR" ]; then
            RESULT_FILE=$(ls -t "$EVAL_DIR/eval_"*/results.json 2>/dev/null | head -1)
            if [ -f "$RESULT_FILE" ]; then
                echo "    - $MODEL: ✅ $(basename $(dirname "$RESULT_FILE"))"
            else
                echo "    - $MODEL: ⚠️  No results found"
            fi
        else
            echo "    - $MODEL: ⏭️  Skipped"
        fi
    done
done
echo ""
echo "📊 WandB Project: https://wandb.ai/matercomus/hoser-distill-optuna-6"
echo ""

