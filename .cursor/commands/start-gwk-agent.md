# Start Git Worktree Agent

**Required Rules:** @multi-agent-workflow @plan-file-best-practices

Initialize a git worktree agent and claim a task from a plan file following multi-agent workflow best practices.

## Workflow

### 1. Identify and Read Plan
User provides plan name. Read from `.cursor/plans/<name>*.plan.md`.

Example: `/start-gwk-agent add-skip` → read `.cursor/plans/add-skip-*.plan.md`

If plan not found, list available plans:
```bash
ls .cursor/plans/*.plan.md
```

### 2. Verify Environment
Check if already in a worktree:
```bash
ls -la .cursor/plans/  # Symlink = worktree, directory = root
```

### 3. Show Available Tasks
Parse the plan file and display tasks following @plan-file-best-practices format:

Show:
- Tasks with status `[ ]` (available)
- Dependencies (must be `[DONE]`)
- Files to be modified
- Any `[IN PROGRESS]` or `[BLOCKED]` tasks

Format output:
```
Available Tasks:
✅ Task 1.1: Add phase decorator
   Dependencies: None
   Files: python_pipeline.py (lines 73-92)
   Status: Available
   
✅ Task 1.2: Add config field
   Dependencies: None
   Files: python_pipeline.py (lines 89-132)
   Status: Available

❌ Task 2.1: Extract generation method
   Dependencies: Task 1.1 (NOT DONE)
   Files: python_pipeline.py (lines ~1350)
   Status: BLOCKED
```

### 4. Check for File Conflicts
Before task selection, check if other agents are working on same files per @multi-agent-workflow:

```bash
grep -n "filename.py" .cursor/plans/*.plan.md | grep "IN PROGRESS"
```

If conflicts found:
```
⚠️  WARNING: File conflicts detected
Agent2 is editing python_pipeline.py (Task 2.3)

Suggested alternatives:
- Task 1.2 (different file section)
- Wait for Task 2.3 to complete
```

### 5. Guide Task Selection
Ask user which task to claim. Validate:
- Dependencies are satisfied (all marked `[DONE]`)
- No file conflicts with other agents
- Task is available (status `[ ]`)

### 6. Determine Worktree and Branch Names
Extract semantic names from task details per @multi-agent-workflow:

Pattern: `PROJECT-phase-description`
Branch: `feat/branch-name` from task

Example for Task 1.1:
- Worktree: `HOSER-phase1-foundation`
- Branch: `feat/phase-decorator-foundation`

Suggest command:
```bash
git worktree add ../HOSER-phase1-foundation -b feat/phase-decorator-foundation
```

### 7. Setup Symlinks
After user creates worktree, guide symlink setup per @multi-agent-workflow:

```bash
cd ../HOSER-phase1-foundation
ln -s $(git rev-parse --show-toplevel)/.cursor/plans .cursor/plans
```

Verify:
```bash
ls -la .cursor/plans/  # Should show symlink
```

### 8. Claim Task in Plan
Update the plan file following @plan-file-best-practices status format:

Change from:
```markdown
### Task X.Y: Name [ ]
```

To:
```markdown
### Task X.Y: Name [IN PROGRESS - AgentN - YYYY-MM-DD HH:MM]
```

Where `N` is next available agent number (check existing plan for Agent1, Agent2, etc.)

### 9. Display Task Implementation Details
Show from plan:
1. Task description
2. Implementation code snippets
3. Validation commands
4. Commit message template

Format clearly with syntax highlighting.

### 10. Confirmation
```
✅ Worktree agent initialized!
📁 Worktree: ../HOSER-phase1-foundation
🌿 Branch: feat/phase-decorator-foundation
📋 Task: 1.1 - Add phase decorator infrastructure
📝 Plan updated: [IN PROGRESS - Agent1 - 2025-11-02 18:45]

Next steps:
1. Implement changes from task description
2. Run validation commands
3. Commit with template message
4. Update plan status to [DONE - Agent1 - timestamp]
```

## Error Handling

**Plan file not found:**
- List available plans in `.cursor/plans/`
- Ask user to specify correct plan name

**No available tasks:**
- Show all task statuses
- Identify blocking dependencies
- Suggest waiting or coordinating with other agents

**Worktree already exists:**
- Ask if continuing in existing worktree
- Or suggest new worktree name

**Not in git repository:**
- Explain git worktree workflow requires git repo
- Guide to initialize git if needed

**File conflicts detected:**
- Show which agent is editing conflicting files
- Suggest alternative tasks
- Option to coordinate with other agent

## Usage

```bash
/start-gwk-agent <plan-name>
```

Examples:
```bash
/start-gwk-agent add-skip-c4eaffdb
/start-gwk-agent phase-decorator-pattern
/start-gwk-agent elegant-phase
```

## Notes

Follow @multi-agent-workflow for:
- Worktree naming conventions
- Branch naming patterns
- Commit message format
- Merge strategies

Follow @plan-file-best-practices for:
- Status marker format
- Task numbering
- Validation commands
- Agent notes

