# Multi-Agent Worktree Quick Start

## 1. Create Worktrees (one-time setup)
```bash
# Create 2-4 worktrees for different phases
git worktree add ../PROJECT-phase1 -b feat/phase1
git worktree add ../PROJECT-phase2 -b feat/phase2
git worktree add ../PROJECT-testing -b feat/testing

# Cursor automatically runs (via worktrees.json):
# - uv sync
# - ln -s $ROOT/.cursor/plans .cursor/plans
```

## 2. Agents Claim Tasks
```bash
# Open plan file
cat .cursor/plans/*.plan.md

# Find available task (marked [ ])
# Check dependencies are met
# Check no file conflicts

# Update status
# [ ] → [IN PROGRESS - Agent1 - timestamp]
```

## 3. Work and Commit Regularly
```bash
# Make changes
# Test: uv tool run ruff check .

# Commit every 30-60 minutes
git add file.py
git commit -m "feat: implement method X"
git push origin feat/phase1

# Update plan with progress
```

## 4. Merge Phase by Phase
```bash
# After completing Phase 1:
git checkout main
git pull origin main
git merge feat/phase1 --no-ff
git push origin main

# Test integration
uv run pytest

# Phase 2 starts with stable Phase 1 in main
```

## 5. Cleanup After Completion
```bash
# Remove worktrees
git worktree remove --force /path/to/PROJECT-phase1

# Delete branches
git branch -d feat/phase1
```

## Decision Flowchart

Should I use multi-agent worktrees?
- Large feature (10+ tasks)? YES → Use it
- Clear module boundaries? YES → Use it
- Need parallel work? YES → Use it
- 1-3 simple tasks? NO → Use single branch
- Tightly coupled changes? NO → Sequential work better

## Common Scenarios

**Scenario:** Two agents want same task
- First to update plan wins
- Second agent refreshes plan and picks different task

**Scenario:** Task blocked waiting for dependency
- Mark as [BLOCKED - reason]
- Choose different available task
- Check back later when dependency done

**Scenario:** Found bug in another agent's work
- Add note to plan
- Optionally fix in your worktree
- Coordinate via plan notes

