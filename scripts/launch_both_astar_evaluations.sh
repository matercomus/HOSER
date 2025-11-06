#!/bin/bash
###############################################################################
# Launch A* Evaluations for Both Beijing and Porto
# 
# This script starts both evaluations in separate tmux sessions
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTAR_SCRIPT="$SCRIPT_DIR/run_astar_evaluation.sh"

echo "================================================================"
echo "Launching A* Evaluations for Beijing and Porto"
echo "================================================================"
echo ""

# Check if script exists
if [ ! -f "$ASTAR_SCRIPT" ]; then
    echo "Error: run_astar_evaluation.sh not found at $ASTAR_SCRIPT"
    exit 1
fi

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first."
    exit 1
fi

# Launch Beijing evaluation in tmux
echo "🚀 Launching Beijing A* evaluation in tmux session 'beijing-astar'..."
tmux new-session -d -s beijing-astar "$ASTAR_SCRIPT beijing"

if [ $? -eq 0 ]; then
    echo "✅ Beijing evaluation started successfully"
else
    echo "❌ Failed to start Beijing evaluation"
    exit 1
fi

echo ""

# Launch Porto evaluation in tmux
echo "🚀 Launching Porto A* evaluation in tmux session 'porto-astar'..."
tmux new-session -d -s porto-astar "$ASTAR_SCRIPT porto"

if [ $? -eq 0 ]; then
    echo "✅ Porto evaluation started successfully"
else
    echo "❌ Failed to start Porto evaluation"
    # Kill Beijing session if Porto fails
    tmux kill-session -t beijing-astar 2>/dev/null
    exit 1
fi

echo ""
echo "================================================================"
echo "Both evaluations launched successfully!"
echo "================================================================"
echo ""
echo "📊 Monitor progress:"
echo "  tmux ls                          # List all sessions"
echo "  tmux attach -t beijing-astar     # Attach to Beijing"
echo "  tmux attach -t porto-astar       # Attach to Porto"
echo ""
echo "📋 View logs:"
echo "  tail -f ~/Dev/HOSER/hoser-distill-optuna-6/logs/astar_evaluation_*.log"
echo "  tail -f ~/Dev/HOSER/hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/logs/astar_evaluation_*.log"
echo ""
echo "⏱️  Expected completion: 12-18 hours"
echo ""
echo "🌙 You can safely disconnect your laptop now!"
echo "================================================================"

# Show current sessions
echo ""
echo "Active tmux sessions:"
tmux ls

