# [Feature Name] - Multi-Agent Implementation Plan

## Worktree Setup Instructions

Recommended worktrees (2-4 for this project):
```bash
# Phase 1: [Description]
git worktree add ../PROJECT-phase1-name -b feat/phase1-name

# Phase 2: [Description]
git worktree add ../PROJECT-phase2-name -b feat/phase2-name

# Integration Testing
git worktree add ../PROJECT-integration -b feat/integration-testing
```

Cursor will automatically run (via .cursor/worktrees.json):
- `uv sync` (install dependencies)
- Create symlink: `.cursor/plans -> root/.cursor/plans`

## Agent Coordination Protocol

### Reading the Plan
1. Read entire plan before claiming any task
2. Check task dependencies are met
3. Verify no file conflicts with other agents
4. Update status when claiming: `[IN PROGRESS - AgentX - timestamp]`

### Task Status Markers
- `[ ]` - Available (can be claimed if dependencies met)
- `[IN PROGRESS - AgentX - YYYY-MM-DD HH:MM]` - Being worked on
- `[DONE - AgentX - YYYY-MM-DD HH:MM]` - Completed
- `[BLOCKED - Reason]` - Cannot proceed (waiting on dependency)

### Updating Progress
Update plan file frequently:
- When starting task (status to IN PROGRESS)
- After completing subtasks (check boxes)
- When done (status to DONE)
- When blocked (status to BLOCKED with reason)

Other agents see updates in real-time via symlinked plan.

---

## Phase 1: [Phase Name]

**Dependencies:** None (or list phase dependencies)

**Estimated Time:** X hours

**Optimal Agents:** 1-2

**Merge Strategy:** Complete all Phase 1 tasks → merge to main → test

### Task 1.1: [Task Name] [ ]

**Owner:** *Unassigned*

**Files:** `config/file1.yaml`, `tools/new_tool.py`

**Dependencies:** None (or Task IDs like "Task 1.2")

**Subtasks:**
- [ ] Create config file
- [ ] Implement core function
- [ ] Add validation
- [ ] Test changes

**Validation Commands:**
```bash
# Verify file exists and is valid
cat config/file1.yaml
uv tool run ruff check config/file1.yaml
```

---

[Repeat for all phases and tasks]

---

## Current Status Summary

**Last Updated:** *Never*

**Active Agents:** 0

**Completed Tasks:** 0/N

**Blocked Tasks:** 0

### Progress by Phase
- Phase 1: 0/X tasks complete
- Phase 2: 0/Y tasks complete
- Phase 3: 0/Z tasks complete

### Next Available Tasks
Agents can immediately start:
- Task 1.1 (no dependencies)
- Task 1.2 (no dependencies)

### Agent Notes
(Agents add notes here for coordination)

