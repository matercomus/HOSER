#!/bin/bash
# Fix existing worktrees by running the setup script in each one

set -e

echo "🔧 Fixing all existing worktrees..."
echo ""

# Get root repo path
ROOT_REPO=$(git rev-parse --path-format=absolute --git-common-dir | xargs dirname)
echo "Root repository: $ROOT_REPO"
echo ""

# Get list of all worktrees (skip header line and main worktree)
WORKTREES=$(git worktree list --porcelain | grep "^worktree " | cut -d' ' -f2 | grep -v "^$ROOT_REPO$")

if [ -z "$WORKTREES" ]; then
    echo "No worktrees found to fix."
    exit 0
fi

count=0
fixed=0
skipped=0

for worktree_path in $WORKTREES; do
    count=$((count + 1))
    worktree_name=$(basename "$worktree_path")
    
    echo "[$count] Processing: $worktree_name"
    
    # Check if .cursor directory exists
    if [ ! -d "$worktree_path/.cursor" ]; then
        echo "    ⚠️  No .cursor directory, skipping"
        skipped=$((skipped + 1))
        continue
    fi
    
    # Check if setup script exists in this worktree, if not copy it from root
    if [ ! -f "$worktree_path/.cursor/setup-worktree-unix.sh" ]; then
        if [ -f "$ROOT_REPO/.cursor/setup-worktree-unix.sh" ]; then
            echo "    📋 Copying setup script from root repo"
            cp "$ROOT_REPO/.cursor/setup-worktree-unix.sh" "$worktree_path/.cursor/setup-worktree-unix.sh"
            chmod +x "$worktree_path/.cursor/setup-worktree-unix.sh"
        else
            echo "    ⚠️  Setup script not found in root repo, skipping"
            skipped=$((skipped + 1))
            continue
        fi
    fi
    
    # Copy worktrees.json if it doesn't exist or is outdated
    if [ -f "$ROOT_REPO/.cursor/worktrees.json" ]; then
        if [ ! -f "$worktree_path/.cursor/worktrees.json" ] || \
           ! cmp -s "$ROOT_REPO/.cursor/worktrees.json" "$worktree_path/.cursor/worktrees.json"; then
            echo "    📋 Copying worktrees.json from root repo"
            cp "$ROOT_REPO/.cursor/worktrees.json" "$worktree_path/.cursor/worktrees.json"
        fi
    fi
    
    # Remove existing plans directory/symlink
    rm -rf "$worktree_path/.cursor/plans"
    
    # Create symlink
    ln -s "$ROOT_REPO/.cursor/plans" "$worktree_path/.cursor/plans"
    
    # Verify
    if [ -L "$worktree_path/.cursor/plans" ]; then
        echo "    ✅ Fixed symlink"
        fixed=$((fixed + 1))
    else
        echo "    ❌ Failed to create symlink"
    fi
    
    echo ""
done

echo "========================================="
echo "Summary:"
echo "  Total worktrees: $count"
echo "  Fixed: $fixed"
echo "  Skipped: $skipped"
echo "========================================="

