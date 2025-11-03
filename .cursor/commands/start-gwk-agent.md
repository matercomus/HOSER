# Start Git Worktree Agent

**Required Rules:** @multi-agent-workflow @plan-file-best-practices

Initialize a git worktree agent and claim a task from a plan file following multi-agent workflow best practices.

## 🚨 CRITICAL: You MUST wait 1-4 minutes after creating worktree for setup to complete! 🚨

**DO NOT skip this step!** See Step 7 below for waiting instructions.

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

### 7. 🚨 MANDATORY: Wait for Setup (Run This Command!) 🚨

**REQUIRED**: After creating worktree, run this ONE command:

```bash
cd ../HOSER-phase1-foundation
bash .cursor/wait-for-setup.sh
```

**This script will:**
- ⏳ **Block until setup completes** (you cannot skip)
- 📊 Show setup progress in real-time
- ✅ Verify everything is ready
- 🤖 Display your agent ID

**DO NOT:**
- ❌ Skip running `wait-for-setup.sh`
- ❌ Try to work before script completes
- ❌ Assume setup is instant
- ❌ Manually run other setup scripts

**The script handles everything - just run it and wait!**

Expected output:
```
=========================================
⏳ Waiting for worktree setup to complete...
=========================================

This will take 1-4 minutes. Please be patient.

⏳ Waiting for setup to start...
✅ Setup started, waiting for completion...

[... setup progress shown here ...]

=========================================
✅ Setup complete! Verifying...
=========================================

✅ Symlink: .cursor/plans exists
✅ Agent ID: PdPfi
✅ Virtual environment: .venv exists
✅ Python: Working

=========================================
🎉 Worktree is ready! You can start working.
=========================================

Your agent ID: PdPfi

Next steps:
1. Read the plan: cat .cursor/plans/*.plan.md
2. Claim a task using your agent ID: PdPfi
3. Start implementing!
```

### 8. Claim Task in Plan

**First, get your unique agent ID:**
```bash
AGENT_ID=$(cat .cursor/worktree-agent-id)
echo "My agent ID: $AGENT_ID"
# Example output: "PdPfi" or "H7WqM"
```

Update the plan file following @plan-file-best-practices status format:

Change from:
```markdown
### Task X.Y: Name [ ]
```

To:
```markdown
### Task X.Y: Name [IN PROGRESS - <AGENT_ID> - YYYY-MM-DD HH:MM]
```

Replace `<AGENT_ID>` with your ID from above (e.g., "PdPfi", "H7WqM", "LqAbJ")

**IMPORTANT**: After updating the plan:
1. **Save the file** (changes are instant via symlink)
2. **Verify**: `cat .cursor/plans/*.plan.md | grep "Task X.Y"`
3. **Close the file** in your editor
4. **Note**: Other agents must close/reopen or use "Revert File" (Ctrl+K R) to see changes

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
🔗 Symlink: Automatically created by setup script (waited 2m15s)
🐍 Environment: Dependencies synced with uv

⏱️  Setup completed successfully after automatic execution.

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

**Worktree setup failed:**
- Check "Output" → "Worktrees Setup" in Cursor
- Check `.cursor/worktree-setup.log` in worktree
- Verify `.cursor/worktrees.json` exists in root repo
- Manually run `bash .cursor/setup-worktree-unix.sh`

**Missing .cursor/plans symlink:**
- Indicates setup script didn't run or failed
- Check setup log for errors
- Run setup script manually
- Verify `$ROOT_WORKTREE_PATH` is set correctly

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

