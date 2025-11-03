#!/bin/bash
# Wait for worktree setup to complete before allowing agent to proceed
# This script BLOCKS until setup is complete - DO NOT SKIP!

set -e

echo "========================================="
echo "⏳ Waiting for worktree setup to complete..."
echo "========================================="
echo ""
echo "This will take 1-4 minutes. Please be patient."
echo ""

# Check if we're in a worktree
if [ ! -d .cursor ]; then
    echo "❌ ERROR: Not in a worktree directory (no .cursor/ found)"
    echo "   Please cd to your worktree directory first"
    exit 1
fi

# Wait for setup log to appear (means setup started)
echo "⏳ Waiting for setup to start..."
WAIT_COUNT=0
while [ ! -f .cursor/worktree-setup.log ]; do
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $WAIT_COUNT -ge 300 ]; then
        echo ""
        echo "⚠️  Setup hasn't started after 5 minutes"
        echo "   Running setup manually..."
        bash .cursor/setup-worktree-unix.sh
        break
    fi
    echo -n "."
done
echo ""

# Wait for setup to complete (check for completion message)
if [ -f .cursor/worktree-setup.log ]; then
    echo "✅ Setup started, waiting for completion..."
    echo ""
    
    # Show setup progress
    tail -f .cursor/worktree-setup.log &
    TAIL_PID=$!
    
    # Wait for completion message
    while ! grep -q "Worktree setup complete" .cursor/worktree-setup.log 2>/dev/null; do
        sleep 2
    done
    
    # Kill tail
    kill $TAIL_PID 2>/dev/null || true
    wait $TAIL_PID 2>/dev/null || true
fi

echo ""
echo "========================================="
echo "✅ Setup complete! Verifying..."
echo "========================================="
echo ""

# Verify all requirements are met
VERIFICATION_FAILED=false

# Check 1: Symlink exists
if [ -L .cursor/plans ]; then
    echo "✅ Symlink: .cursor/plans exists"
else
    echo "❌ Symlink: .cursor/plans NOT found"
    VERIFICATION_FAILED=true
fi

# Check 2: Agent ID exists
if [ -f .cursor/worktree-agent-id ]; then
    AGENT_ID=$(cat .cursor/worktree-agent-id)
    echo "✅ Agent ID: $AGENT_ID"
else
    echo "❌ Agent ID: NOT found"
    VERIFICATION_FAILED=true
fi

# Check 3: Virtual environment exists
if [ -d .venv ]; then
    echo "✅ Virtual environment: .venv exists"
else
    echo "❌ Virtual environment: .venv NOT found"
    VERIFICATION_FAILED=true
fi

# Check 4: Python works
if uv run python -c "print('OK')" &>/dev/null; then
    echo "✅ Python: Working"
else
    echo "❌ Python: NOT working"
    VERIFICATION_FAILED=true
fi

echo ""

if [ "$VERIFICATION_FAILED" = true ]; then
    echo "❌ Setup verification FAILED"
    echo "   Please check .cursor/worktree-setup.log for errors"
    exit 1
fi

echo "========================================="
echo "🎉 Worktree is ready! You can start working."
echo "========================================="
echo ""
echo "Your agent ID: $AGENT_ID"
echo ""
echo "Next steps:"
echo "1. Read the plan: cat .cursor/plans/*.plan.md"
echo "2. Claim a task using your agent ID: $AGENT_ID"
echo "3. Start implementing!"
echo ""

