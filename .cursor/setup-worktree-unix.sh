#!/bin/bash
set -e  # Exit on any error
set -u  # Exit on undefined variables

# Ensure .cursor directory exists (required for logging)
mkdir -p .cursor

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
if command -v uv &> /dev/null; then
    uv sync
else
    echo "⚠️  WARNING: uv not found, skipping dependency sync"
fi

# Step 2: Setup .cursor/plans symlink
echo "🔗 Setting up .cursor/plans symlink..."

# Remove existing directory/symlink if present
rm -rf .cursor/plans

# Determine root worktree path with validation
if [ -n "${ROOT_WORKTREE_PATH:-}" ]; then
    ROOT_PATH="$ROOT_WORKTREE_PATH"
    echo "   Using ROOT_WORKTREE_PATH: $ROOT_PATH"
else
    # Get the common git directory (points to main repo's .git)
    # Then get the parent directory (the actual root repo path)
    GIT_COMMON_DIR=$(git rev-parse --path-format=absolute --git-common-dir)
    ROOT_PATH=$(dirname "$GIT_COMMON_DIR")
    echo "   ROOT_WORKTREE_PATH not set, using git: $ROOT_PATH"
fi

# Validate root path exists
if [ ! -d "$ROOT_PATH" ]; then
    echo "   ❌ ERROR: Root path does not exist: $ROOT_PATH"
    exit 1
fi

# Validate target plans directory exists
if [ ! -d "$ROOT_PATH/.cursor/plans" ]; then
    echo "   ⚠️  WARNING: Target plans directory doesn't exist: $ROOT_PATH/.cursor/plans"
    echo "   Creating it..."
    mkdir -p "$ROOT_PATH/.cursor/plans"
fi

# Create symlink
echo "   Creating symlink: .cursor/plans -> $ROOT_PATH/.cursor/plans"
ln -s "$ROOT_PATH/.cursor/plans" .cursor/plans

# Verify symlink was created successfully
if [ -L .cursor/plans ]; then
    echo "   ✅ Symlink created successfully"
    TARGET=$(readlink .cursor/plans)
    echo "   Symlink target: $TARGET"
    
    # Verify symlink is not broken
    if [ -e .cursor/plans ]; then
        ls -la .cursor/plans/ || echo "   (Plans directory is empty)"
    else
        echo "   ⚠️  WARNING: Symlink is broken (target doesn't exist)"
    fi
else
    echo "   ❌ ERROR: Symlink creation failed"
    exit 1
fi

# Step 3: Verify plans are accessible
if ls .cursor/plans/*.plan.md &> /dev/null; then
    PLAN_COUNT=$(ls -1 .cursor/plans/*.plan.md | wc -l)
    echo "   📋 Found $PLAN_COUNT plan file(s)"
else
    echo "   ℹ️  No plan files found (this is okay for new projects)"
fi

echo "✅ Worktree setup complete!"
echo "========================================="
echo ""

