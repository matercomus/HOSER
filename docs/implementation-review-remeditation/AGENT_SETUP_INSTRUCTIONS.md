# Setting Up Cursor for HOSER Peer Review Remediation

**Purpose**: Instructions for configuring Cursor to systematically fix peer review issues using GitHub MCP integration.

---

## File Structure

All agent configuration files are in place:

```
/home/matt/Dev/HOSER/
├── docs/implementation-review-remeditation/
│   ├── README.md                              # Overview and quick start
│   ├── AGENT_SYSTEM_PROMPT.md                 # 📘 Detailed agent instructions
│   ├── AGENT_QUICK_START.md                   # 🚀 Quick reference guide
│   ├── GITHUB_MCP_AGENT_GUIDE.md              # Complete GitHub MCP workflow
│   ├── IMPLEMENTATION_REMEDIATION_REPORT.md   # Detailed issue analysis
│   ├── IMPLEMENTATION_FIXES_CHECKLIST.md      # Quick task reference
│   ├── IMPLEMENTATION_ROADMAP.md              # Timeline and dependencies
│   └── COMPREHENSIVE_PEER_REVIEW.md           # Original peer review
└── .cursor/rules/
    └── hoser-peer-review-agent.mdc            # 🎯 Cursor-specific rules
```

---

## Step 1: Cursor Rules Setup

The Cursor rule file is already in place at:
```
/home/matt/Dev/HOSER/.cursor/rules/hoser-peer-review-agent.mdc
```

This file will be **automatically loaded** by Cursor when working in the HOSER repository.

### What the Rule Does
- Enforces systematic workflow
- Requires GitHub MCP usage
- Prevents common mistakes
- Enforces validation before closing issues
- Requires documentation updates with code changes
- Enforces atomic commits

---

## Step 2: Verify GitHub MCP is Available

Before starting an agent session, verify GitHub MCP is installed and accessible:

```bash
# Check if GitHub MCP server is running
# In Cursor, open Command Palette (Cmd/Ctrl+Shift+P)
# Search for "MCP: List Servers"
# Verify "github" is in the list
```

---

## Step 3: Starting an Agent Session

### Initial Prompt for First-Time Setup

```
I'm working on the HOSER Knowledge Distillation repository at /home/matt/Dev/HOSER.

Task: Set up GitHub MCP infrastructure for peer review remediation.

Please:
1. Read docs/implementation-review-remeditation/AGENT_SYSTEM_PROMPT.md
2. Read docs/implementation-review-remeditation/GITHUB_MCP_AGENT_GUIDE.md
3. Follow the "Initial Setup Workflow" from the guide:
   - Create milestones (Phase 1, 2, 3, Deferred)
   - Create all labels (P0-P3, feasibility, categories)
   - Create project board: "HOSER Peer Review Remediation"
   - Create all 35 GitHub issues from COMPREHENSIVE_PEER_REVIEW.md
   - Add all issues to project board

Context:
- GitHub MCP is already installed and available
- Documentation is in docs/implementation-review-remeditation/
- Repository has 35 issues to fix from peer review
```

### Prompt for Working on Issues

```
I'm working on the HOSER repository peer review remediation.

Task: Fix the next priority issue from Phase [1|2|3].

Please:
1. Read docs/implementation-review-remeditation/AGENT_SYSTEM_PROMPT.md if you haven't
2. Use GitHub MCP to find the next open issue in Phase [X]
3. Check if the issue has any blocking dependencies
4. Read the issue details from IMPLEMENTATION_REMEDIATION_REPORT.md
5. Follow the standard workflow:
   - Start work (update GitHub issue)
   - Implement fix as specified in documentation
   - Update all related documentation
   - Complete ALL validation steps
   - Commit with proper message format
   - Close issue with completion comment

Context:
- Follow docs/implementation-review-remeditation/AGENT_QUICK_START.md
- Use GitHub MCP for ALL tracking
- Do NOT skip validation steps
- Do NOT close issues without validation
```

---

## Step 4: Monitoring Agent Behavior

### Signs Agent is Working Correctly ✅

**GitHub Activity**:
- Issues are being updated with "in-progress" label
- Detailed progress comments appear on issues
- Issues close only after validation
- Project board columns update appropriately

**Git Activity**:
- Commits reference issue numbers: `[Issue X.X]`
- Commit messages are detailed and specific
- Changes are atomic (one logical change per commit)
- Documentation commits accompany code commits

**File Changes**:
- Code and docs change together
- Validation results documented
- No unrelated changes bundled together

### Signs Agent Needs Correction ❌

**Bad Behaviors**:
- Issues close without validation comments
- Commits don't reference issue numbers
- Multiple unrelated changes in one commit
- Documentation diverges from code
- Working on multiple issues simultaneously
- Skipping validation steps

**If You See These**: Stop the agent and provide correction:

```
STOP. You are not following the workflow correctly.

Issues I've observed:
- [Describe specific problematic behavior]

Required corrections:
1. Read docs/implementation-review-remeditation/AGENT_SYSTEM_PROMPT.md again
2. Follow the workflow protocol exactly
3. Use GitHub MCP for ALL tracking
4. Validate BEFORE closing issues
5. Work on ONE issue at a time

Please restart with the correct workflow.
```

---

## Step 5: Phase Completion Checklist

After each phase (1, 2, or 3), verify:

### Phase Completion Criteria
- [ ] All phase issues closed OR marked blocked with clear reason
- [ ] All closed issues have validation comments
- [ ] Documentation matches code exactly
- [ ] No regressions introduced
- [ ] Progress report generated and posted
- [ ] IMPLEMENTATION_FIXES_CHECKLIST.md updated (items checked off)
- [ ] Project board shows accurate status

### Command for Agent to Generate Report

```
Task: Generate Phase [X] completion report.

Please:
1. Use GitHub MCP to list all issues in Phase [X] milestone
2. Count: total, completed, blocked, in-progress
3. Calculate completion percentage
4. List any remaining issues with their status
5. Generate comprehensive report
6. Post report to tracking issue (or create if needed)

Use the progress report format from GITHUB_MCP_AGENT_GUIDE.md
```

---

## Step 6: Troubleshooting

### Issue: Agent Not Using GitHub MCP

**Symptom**: Agent is tracking manually or not updating GitHub issues

**Fix**:
```
You MUST use GitHub MCP for ALL issue tracking.

Before proceeding:
1. Verify GitHub MCP is available: list available MCP tools
2. Read: docs/implementation-review-remeditation/GITHUB_MCP_AGENT_GUIDE.md
3. For EVERY issue:
   - update_issue() to set labels
   - add_issue_comment() to document progress
   - close_issue() when complete

Do NOT track manually. Use GitHub MCP tools exclusively.
```

### Issue: Agent Skipping Validation

**Symptom**: Issues close without validation steps documented

**Fix**:
```
STOP. You are skipping validation steps.

Requirement: Before closing ANY issue, you MUST:
1. Complete ALL validation steps from IMPLEMENTATION_REMEDIATION_REPORT.md
2. Document results in GitHub issue comment
3. Verify all files in "Files to Modify" are updated
4. Ensure no regressions introduced

Example validation comment format:
## Validation Results

- [x] Test 1: PASSED - [details]
- [x] Test 2: PASSED - [details]
- [x] All files updated
- [x] No regressions

Restart Issue [X.X] with proper validation.
```

### Issue: Agent Making Unrelated Changes

**Symptom**: Commits contain multiple unrelated file changes

**Fix**:
```
Your commits are not atomic. Each commit should contain ONE logical change.

Rules:
- ONE issue per commit (or multiple commits for one issue)
- Related changes only (code + its documentation = OK)
- Unrelated changes = separate commits

Please:
1. Review git diff before committing
2. Stage only related files
3. Commit with proper message format
4. Separate unrelated changes into different commits
```

### Issue: Documentation Diverges from Code

**Symptom**: Code changes without corresponding doc updates

**Fix**:
```
Documentation MUST match code exactly.

Rule: Whenever you change code, you MUST update ALL related documentation IN THE SAME COMMIT.

For each code change, check:
1. README.md - Does it reference this code?
2. Docs in docs/ - Do any explain this functionality?
3. Comments in code - Do they match the new behavior?
4. Config files - Do they need updates?

Restart Issue [X.X] and update both code AND docs together.
```

---

## Step 7: Best Practices for Agent Supervision

### Start with Easy Issues

Begin with quick wins to verify agent behavior:
1. Issue 1.4: JSD binning docs (1 hour, docs-only)
2. Issue 3.5: Environment info (1 hour, docs-only)

These allow you to verify:
- Agent uses GitHub MCP correctly
- Agent follows commit message format
- Agent validates before closing

### Monitor Early, Intervene Early

- Check the **first 2-3 issues** closely
- Verify workflow adherence before letting agent continue
- Correct bad patterns immediately before they become habits

### Use Progress Checkpoints

After every **5 issues**, request a progress report:
```
Pause and generate a progress report.

Include:
- Issues completed in last session
- Current phase status
- Any issues encountered
- Next 5 issues planned
```

### Keep Sessions Focused

- Work on **one phase** at a time
- Complete Phase 1 before Phase 2
- Don't jump between phases
- Finish blocked issues from previous phases first

---

## Step 8: Example Agent Session Flow

### Session 1: Setup
```
Task: Set up GitHub MCP infrastructure
Expected: ~30 minutes
Outcome: Milestones, labels, project board, 35 issues created
```

### Session 2: Quick Wins (Phase 1)
```
Task: Complete Issue 1.4 (JSD binning docs)
Expected: ~1 hour
Outcome: Documentation fix, validated, committed, closed
Verification: Check GitHub issue, commit, project board
```

### Session 3: First Code Fix (Phase 1)
```
Task: Complete Issue 1.2 (Bonferroni correction)
Expected: ~4 hours
Outcome: Code fix, tests pass, docs updated, validated, committed
Verification: Check code, docs, tests, GitHub issue
```

### Session 4: Continue Phase 1
```
Task: Complete remaining Phase 1 issues
Expected: Multiple sessions
Outcome: All Phase 1 issues fixed/mitigated/documented
Verification: Phase 1 completion report
```

---

## Step 9: Success Metrics

### You're Ready to Proceed When

After **initial 3 issues**, verify:
- [x] GitHub issues updated correctly
- [x] Commits follow format
- [x] Validation documented before closing
- [x] Documentation matches code
- [x] Project board accurate
- [x] No regressions introduced

If all ✅, continue with confidence.
If any ❌, stop and correct before proceeding.

---

## Step 10: Emergency Stop Procedures

### When to Stop Immediately

🛑 Stop agent if you observe:
- Multiple issues closing without validation
- Code changes without doc updates
- Commits with unrelated changes
- Working on blocked issues
- Skipping validation steps
- Manual tracking instead of GitHub MCP

### How to Stop and Reset

1. **Stop the agent** (cancel current operation)
2. **Review recent changes** (`git log`, GitHub issues)
3. **Revert if necessary** (`git revert` bad commits)
4. **Provide correction** (detailed feedback on what went wrong)
5. **Request re-read** of system prompt and rules
6. **Test on simple issue** before resuming main work

---

## Summary

✅ **Files created**:
- `AGENT_SYSTEM_PROMPT.md` - Detailed instructions
- `AGENT_QUICK_START.md` - Quick reference
- `.cursor/rules/hoser-peer-review-agent.mdc` - Cursor rules

✅ **GitHub MCP integrated** throughout documentation

✅ **Workflow defined** with validation requirements

✅ **Error handling** and troubleshooting guides included

✅ **Quality controls** enforce best practices

**Next Step**: Start an agent session with the initial setup prompt from Step 3.

---

**Remember**: The goal is systematic, high-quality remediation. Monitor early, correct quickly, validate thoroughly. Quality over speed.

