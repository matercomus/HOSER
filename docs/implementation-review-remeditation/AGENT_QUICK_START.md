# Quick Start Guide

**For**: Working on HOSER peer review remediation  
**Goal**: Fix 35 repository issues systematically using GitHub MCP

---

## 📋 Pre-Flight Checklist

Before starting ANY work:

- [ ] Read `AGENT_SYSTEM_PROMPT.md` in full
- [ ] Read `GITHUB_MCP_AGENT_GUIDE.md` 
- [ ] Verify GitHub MCP is available
- [ ] Confirm project board exists
- [ ] Understand which phase you're working on

---

## 🚀 First-Time Setup (Run Once)

```bash
# 1. Verify you're in the right repository
pwd  # Should be: /home/matt/Dev/HOSER

# 2. Read the documentation
cd docs/implementation-review-remeditation/
ls -la  # Verify all files exist

# 3. Use GitHub MCP to create infrastructure
# See GITHUB_MCP_AGENT_GUIDE.md Section "Initial Setup Workflow"
# - Create milestones (Phase 1, 2, 3, Deferred)
# - Create labels (P0-P3, fix/mitigate/document/defer, categories)
# - Create project board
# - Create all 35 issues from COMPREHENSIVE_PEER_REVIEW.md
```

---

## 🔄 Standard Issue Resolution Loop

### 1. Query Next Issue
```python
# Get next open issue from current phase
issues = list_issues(milestone="Phase 1: Critical Fixes", state="open", sort="created")
next_issue = issues[0]  # First non-blocked issue
```

### 2. Check Dependencies
```python
# Before starting, verify no blockers
if has_blocking_dependencies(next_issue):
    update_issue(next_issue.number, labels=["blocked"])
    add_issue_comment(next_issue.number, "⛔ Blocked by #X")
    continue  # Skip to next issue
```

### 3. Read Documentation
```bash
# Open these files for the specific issue:
# - IMPLEMENTATION_REMEDIATION_REPORT.md (detailed fix)
# - IMPLEMENTATION_FIXES_CHECKLIST.md (quick reference)
# - IMPLEMENTATION_ROADMAP.md (context)
```

### 4. Start Work
```python
update_issue(next_issue.number, labels=["in-progress"])
add_issue_comment(next_issue.number, 
    "📝 Implementation started\n\nTimestamp: [UTC timestamp]")
```

### 5. Implement Fix
- Follow "Required Changes" from documentation EXACTLY
- Make ONE change at a time
- Test each change before proceeding

### 6. Update Documentation
- Update ALL docs that reference changed code
- Ensure docs match code exactly

### 7. Validate
```bash
# Complete ALL validation steps from documentation
# Examples:
python -m pytest tests/
python evaluate.py --dataset Beijing
grep -r "old_value" docs/  # Verify all updated
```

### 8. Comment Progress
```python
add_issue_comment(next_issue.number, """
## Validation Results

**Changes Made**:
- Modified `file.py` lines 100-150
- Updated `docs/README.md` lines 45-60

**Validation**:
- [x] Tests pass
- [x] Documentation matches code
- [x] No regressions

**Files Modified**:
- `path/to/file.py`
- `docs/README.md`
""")
```

### 9. Commit
```bash
git add path/to/file.py docs/README.md
git commit -m "[Issue X.X] fix: Brief description

- Detailed change 1
- Detailed change 2

Addresses: Issue X.X from peer review
Status: Fixed
Validation: All tests pass"
```

### 10. Complete
```python
add_issue_comment(next_issue.number, 
    "✅ Validated and complete\n\n[summary of what was fixed]")
close_issue(next_issue.number)
# Update IMPLEMENTATION_FIXES_CHECKLIST.md (check off item)
```

---

## 🎯 Priority Order

Work on issues in this order:

1. **P0-Critical** (8 issues) - Must fix before publication
2. **P1-Major** (11 issues) - High priority
3. **P2-Moderate** (8 issues) - Medium priority
4. **P3-Minor** (5 issues) - Nice to have

Within each priority: Lower issue number first (1.1, 1.2, 1.3...)

---

## ⚠️ Critical Rules

### ALWAYS Do
✅ Use GitHub MCP for ALL tracking
✅ Read documentation before implementing
✅ Validate ALL changes before closing
✅ Update docs with code changes
✅ Commit with issue references
✅ Work on ONE issue at a time
✅ Check dependencies first

### NEVER Do
❌ Skip validation steps
❌ Work on blocked issues
❌ Close without validation
❌ Bundle unrelated changes
❌ Assume fixes work
❌ Create issues manually
❌ Work on multiple issues simultaneously

---

## 📝 Commit Message Template

```
[Issue X.X] <type>: <brief description>

- Detailed change 1
- Detailed change 2
- Detailed change 3

Addresses: Issue X.X from peer review
Status: Fixed|Mitigated|Documented
Validation: [brief summary]
```

**Types**: `fix`, `feat`, `docs`, `test`, `refactor`, `stat`

---

## 🔍 Validation Checklist

Before closing ANY issue:

- [ ] ALL validation steps completed
- [ ] ALL files in "Files to Modify" updated
- [ ] Documentation matches code exactly
- [ ] Tests pass (if applicable)
- [ ] No regressions introduced
- [ ] GitHub issue has completion comment

---

## 🚨 Error Handling

### If Blocked by Dependency
```python
update_issue(issue_num, labels=["blocked"])
add_issue_comment(issue_num, "⛔ Blocked by #X - waiting for completion")
# Move to next issue
```

### If Validation Fails
```python
update_issue(issue_num, labels=["needs-revision"])
add_issue_comment(issue_num, "⚠️ Validation failed: [reason]\n\nRetrying...")
# Re-read documentation, try alternative approach
# Do NOT close issue
```

### If Unclear How to Proceed
```python
update_issue(issue_num, labels=["needs-help"])
add_issue_comment(issue_num, "❓ Question: [specific question]\n\nNeed clarification on...")
# Wait for response before proceeding
```

---

## 📊 Progress Tracking

### After Each Issue
```bash
# 1. Update checklist
# Edit: docs/implementation-review-remeditation/IMPLEMENTATION_FIXES_CHECKLIST.md
# Check off completed items

# 2. Verify GitHub updated
# - Issue closed
# - Project board moved to "Complete"
# - Labels updated
```

### After Each Phase
```python
# Generate progress report
report = generate_progress_report()
add_issue_comment(TRACKING_ISSUE_NUM, report)
```

---

## 💡 Pro Tips

1. **Read documentation FIRST** - Don't guess, follow the plan
2. **Validate THOROUGHLY** - Testing prevents rework
3. **Commit ATOMICALLY** - One logical change per commit
4. **Document EVERYTHING** - Future you will thank you
5. **Ask WHEN UNSURE** - Better to clarify than to fix mistakes
6. **Work SEQUENTIALLY** - Complete one before starting next

---

## 📚 Essential Files

**Read before starting**:
- `README.md` - Overview
- `AGENT_SYSTEM_PROMPT.md` - Detailed instructions
- `GITHUB_MCP_AGENT_GUIDE.md` - GitHub MCP workflow

**Reference while working**:
- `IMPLEMENTATION_REMEDIATION_REPORT.md` - Detailed fix instructions
- `IMPLEMENTATION_FIXES_CHECKLIST.md` - Quick reference
- `IMPLEMENTATION_ROADMAP.md` - Dependencies and timeline

**Original review**:
- `COMPREHENSIVE_PEER_REVIEW.md` - Full peer review details

---

## 🎓 Example Session

```python
# 1. Get next issue
issues = list_issues(milestone="Phase 1: Critical Fixes", state="open")
issue = issues[0]  # Issue 1.4: JSD Binning Documentation Mismatch

# 2. Start
update_issue(4, labels=["in-progress"])
add_issue_comment(4, "🤖 Started: 2026-01-15T10:30:00Z")

# 3. Read docs
# Read IMPLEMENTATION_REMEDIATION_REPORT.md Section 1.4
# Action: Replace "50 bins" with "100 bins" in docs/EVALUATION_ANALYSIS.md

# 4. Implement
grep -r "50 bins" docs/
# Found in: docs/EVALUATION_ANALYSIS.md line 143
sed -i 's/50 bins/100 bins/g' docs/EVALUATION_ANALYSIS.md

# 5. Validate
grep -r "50 bins" docs/  # Should return 0 results
grep -r "100 bins" docs/  # Verify replacement

# 6. Comment
add_issue_comment(4, "✏️ Updated docs/EVALUATION_ANALYSIS.md line 143\n\n✅ Validation: No instances of '50 bins' remain")

# 7. Commit
git add docs/EVALUATION_ANALYSIS.md
git commit -m "[Issue 1.4] docs: Fix JSD binning documentation mismatch

- Replaced '50 bins' with '100 bins' in EVALUATION_ANALYSIS.md line 143
- Verified all instances updated

Addresses: Issue 1.4 from peer review
Status: Fixed
Validation: grep confirms no '50 bins' remain in docs"

# 8. Complete
add_issue_comment(4, "✅ Complete: JSD binning documentation now matches code (100 bins)")
close_issue(4)

# 9. Update checklist
# Edit IMPLEMENTATION_FIXES_CHECKLIST.md and check off Issue 1.4
```

---

## 🎯 Success Metrics

**You're doing it right when**:
- Issues close quickly with full validation
- Commits are atomic and well-documented
- Documentation always matches code
- No regressions introduced
- GitHub issues are thoroughly documented
- Progress is visible and trackable

**You need to adjust when**:
- Issues reopen due to incomplete fixes
- Multiple unrelated changes in one commit
- Documentation diverges from code
- Validation steps skipped
- Issues closed without comments
- Progress is unclear

---

**Remember**: Quality over speed. Follow the plan. Validate thoroughly. Document everything.

**Need help?** Read `AGENT_SYSTEM_PROMPT.md` for detailed instructions.

