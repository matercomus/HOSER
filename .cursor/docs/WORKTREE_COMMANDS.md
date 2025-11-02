# Git Worktree Commands Reference

## Setup

Create worktree:
```bash
git worktree add ../PROJECT-name -b branch-name
```

List all worktrees:
```bash
git worktree list
```

Verify symlink setup:
```bash
ls -la .cursor/plans/
# Should show: .cursor/plans -> /full/path/to/root/.cursor/plans
```

## Daily Workflow

Read plan:
```bash
cat .cursor/plans/*.plan.md
less .cursor/plans/*.plan.md  # For long plans
```

Check task status:
```bash
grep "Task X.Y" .cursor/plans/*.plan.md
```

Check file conflicts:
```bash
grep "filename.py" .cursor/plans/*.plan.md | grep "IN PROGRESS"
```

Commit changes:
```bash
git add file.py
git commit -m "feat: description"
git push origin branch-name
```

## Validation

Lint check:
```bash
uv tool run ruff check .
uv tool run ruff check file.py  # Single file
```

Run tests:
```bash
uv run pytest
uv run pytest tests/test_file.py  # Single test
```

Check git status:
```bash
git status
git diff  # See uncommitted changes
```

View commit history:
```bash
git log --oneline -10
git log --oneline main..HEAD  # Commits ahead of main
```

## Merging

Merge to main:
```bash
git checkout main
git pull origin main
git merge feat/branch --no-ff -m "Merge message"
git push origin main
```

Return to feature branch:
```bash
git checkout feat/branch
```

## Cleanup

Remove worktree:
```bash
git worktree remove /path/to/worktree
git worktree remove --force /path/to/worktree  # With untracked files
```

Delete branch:
```bash
git branch -d branch-name  # Safe (checks if merged)
git branch -D branch-name  # Force delete
```

List remaining worktrees:
```bash
git worktree list
```

## Troubleshooting

Recreate broken symlink:
```bash
rm -rf .cursor/plans
ln -s $(git rev-parse --show-toplevel)/.cursor/plans .cursor/plans
```

Sync dependencies:
```bash
uv sync
```

Find root worktree path:
```bash
git rev-parse --show-toplevel
```

Check which worktree you're in:
```bash
pwd
git worktree list | grep "$(pwd)"
```

