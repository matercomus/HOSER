#!/bin/bash
set -e  # Exit on any error

# Setup logging to file
LOG_FILE=".cursor/worktree-setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "🔧 Setting up worktree environment..."
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "ROOT_WORKTREE_PATH: ${ROOT_WORKTREE_PATH:-'(not set)'}"
echo "========================================="

# Step 1: Sync dependencies
echo "📦 Syncing dependencies with uv..."
uv sync

# Step 2: Setup .cursor/plans symlink
echo "🔗 Setting up .cursor/plans symlink..."

# Remove existing directory/symlink if present
rm -rf .cursor/plans

# Determine root worktree path with fallback
if [ -n "$ROOT_WORKTREE_PATH" ]; then
    ROOT_PATH="$ROOT_WORKTREE_PATH"
    echo "   Using ROOT_WORKTREE_PATH: $ROOT_PATH"
else
    # Get the common git directory (points to main repo's .git)
    # Then get the parent directory (the actual root repo path)
    GIT_COMMON_DIR=$(git rev-parse --path-format=absolute --git-common-dir)
    ROOT_PATH=$(dirname "$GIT_COMMON_DIR")
    echo "   ROOT_WORKTREE_PATH not set, using git: $ROOT_PATH"
fi

# Create symlink
ln -s "$ROOT_PATH/.cursor/plans" .cursor/plans

# Verify symlink was created successfully
if [ -L .cursor/plans ]; then
    echo "   ✅ Symlink created successfully"
    ls -la .cursor/plans/
else
    echo "   ❌ ERROR: Symlink creation failed"
    exit 1
fi

# Step 3: Verify plans are accessible
if ls .cursor/plans/*.plan.md &> /dev/null; then
    PLAN_COUNT=$(ls -1 .cursor/plans/*.plan.md | wc -l)
    echo "   📋 Found $PLAN_COUNT plan file(s)"
else
    echo "   ⚠️  WARNING: No plan files found (this might be okay)"
fi

echo "✅ Worktree setup complete!"
echo "========================================="
echo ""

