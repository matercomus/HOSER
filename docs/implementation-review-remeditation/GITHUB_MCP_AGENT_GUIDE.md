# Using GitHub MCP for Systematic Remediation

**Purpose**: Instructions for using GitHub MCP tools for systematic peer review remediation.

---

## Core Principles

### Zero Manual Setup Required
- All tracking uses GitHub MCP tools only
- Title-based categorization (no labels/milestones needed)
- Search via GitHub's native search patterns
- Fully automated workflow

### Title-Based Issue Categorization

**Format**: `[Priority][Phase][Feasibility] Issue X.X: Brief Title`

**Examples**:
- `[P0][Phase-1][fix] Issue 1.4: JSD Binning Documentation Mismatch`
- `[P0][Deferred][defer] Issue 1.1: Hyperparameter Optimization Confound`
- `[P1][Phase-2][document] Issue 2.1: No Teacher Baseline`
- `[P2][Phase-3][mitigate] Issue 3.1: Sample Size Not Justified`

**Priority Tags**: `[P0]` `[P1]` `[P2]` `[P3]`  
**Phase Tags**: `[Phase-1]` `[Phase-2]` `[Phase-3]` `[Deferred]`  
**Feasibility Tags**: `[fix]` `[mitigate]` `[document]` `[defer]`

---

## Automated Setup Workflow

### Step 1: Verify GitHub Issues Are Enabled

**Check via GitHub MCP**:
```python
# Test if issues are accessible
result = list_issues(
    owner="matercomus",
    repo="HOSER",
    state="open",
    perPage=1
)
# If this succeeds without error, issues are enabled
```

### Step 2: Create All 35 Issues Automatically

Read `IMPLEMENTATION_REMEDIATION_REPORT.md` and create issues using GitHub MCP:

```python
# For each issue in documentation:
issue_write(
    method="create",
    owner="matercomus",
    repo="HOSER",
    title="[P0][Phase-1][fix] Issue 1.4: JSD Binning Documentation Mismatch",
    body="""
## Metadata
- **Priority**: P0-Critical
- **Phase**: Phase 1 (Week 1)  
- **Feasibility**: fix
- **Category**: documentation
- **Effort**: 1 hour
- **Dependencies**: None

## Problem Statement
Documentation claims "50 bins" but code uses 100 bins for JSD calculation.

## Evidence
**File**: `docs/EVALUATION_COMPARISON.md`
- Line 234: "We use 50 spatial bins for JSD calculation"

**File**: `models/lmtad_distil.py`
- Line 156: `n_bins = 100  # JSD binning for location probability distribution`

## Required Changes
1. Update `docs/EVALUATION_COMPARISON.md` line 234: Change "50 bins" to "100 bins"
2. Add clarification about why 100 bins was chosen
3. Verify all other documentation references to JSD bins

## Validation Steps
- [ ] Search codebase for all "50 bin" references
- [ ] Verify code consistently uses 100 bins
- [ ] Documentation updated to match code
- [ ] No contradictory claims remain

## Files to Modify
- `docs/EVALUATION_COMPARISON.md`
- `docs/LMTAD-Distillation.md` (if applicable)

---
**Reference**: `docs/implementation-review-remeditation/IMPLEMENTATION_REMEDIATION_REPORT.md` - Issue 1.4
"""
)
```

### Step 3: Verify Setup Complete

Test search queries to ensure categorization works:

```python
# Query Phase 1 issues
phase1 = search_issues(
    query='is:open "[Phase-1]" in:title repo:matercomus/HOSER'
)
print(f"Phase 1 issues: {len(phase1)} (expected: 8)")

# Query P0 issues  
p0 = search_issues(
    query='is:open "[P0]" in:title repo:matercomus/HOSER'
)
print(f"P0 issues: {len(p0)} (expected: 8)")

# Query fix-type issues
fixable = search_issues(
    query='is:open "[fix]" in:title repo:matercomus/HOSER'
)
print(f"Fixable issues: {len(fixable)}")
```

---

## Working Through Issues

### Daily Workflow

**1. Find Next Issue to Work On**:
```python
# Get Phase 1 open issues
issues = search_issues(
    query='is:open "[Phase-1]" in:title repo:matercomus/HOSER',
    sort="created",
    order="asc"
)

# Filter for non-blocked issues (check dependencies in body)
next_issue = issues[0]  # First unblocked issue
```

**2. Start Work**:
```python
# Add status comment
add_issue_comment(
    owner="matercomus",
    repo="HOSER",
    issue_number=next_issue.number,
    body="🤖 **Status**: IN PROGRESS\n\n**Started**: 2026-01-05T10:30:00Z\n\nImplementing changes as specified."
)
```

**3. Implement Fix**:
```python
# Read detailed instructions from issue body
implementation_details = issue_read(
    method="get",
    owner="matercomus",
    repo="HOSER",
    issue_number=next_issue.number
)
# Implement the fix based on "Required Changes" section
```

**4. Update Progress**:
```python
add_issue_comment(
    owner="matercomus",
    repo="HOSER",
    issue_number=next_issue.number,
    body="""
## Progress Update

✏️ **Modified Files**:
- `docs/EVALUATION_COMPARISON.md` (line 234)
- `docs/LMTAD-Distillation.md` (line 89)

**Changes**:
- Updated "50 bins" to "100 bins" in all documentation
- Added clarification about bin size choice
"""
)
```

**5. Validate**:
```python
# Run validation steps from issue body
validation_result = run_validation_steps()

add_issue_comment(
    owner="matercomus",
    repo="HOSER",
    issue_number=next_issue.number,
    body="""
## Validation Results

- [x] Searched codebase for all "50 bin" references: PASSED
- [x] Verified code consistently uses 100 bins: PASSED
- [x] Documentation updated to match code: PASSED
- [x] No contradictory claims remain: PASSED

✅ All validation steps passed.
"""
)
```

**6. Complete**:
```python
# Close issue
issue_write(
    method="update",
    owner="matercomus",
    repo="HOSER",
    issue_number=next_issue.number,
    state="closed"
)

add_issue_comment(
    owner="matercomus",
    repo="HOSER",
    issue_number=next_issue.number,
    body="✅ **Status**: COMPLETE\n\nValidated and closed."
)
```

### Handling Dependencies

**Check if issue is blocked**:
```python
# Read issue body
issue = issue_read(method="get", owner="matercomus", repo="HOSER", issue_number=15)

# Parse dependencies section from body
if "Depends on Issue" in issue.body:
    # Check if blocking issue is closed
    blocking_issue_num = extract_dependency_number(issue.body)
    blocking_issue = issue_read(method="get", owner="matercomus", repo="HOSER", issue_number=blocking_issue_num)
    
    if blocking_issue.state == "open":
        # Add blocked comment
        add_issue_comment(
            owner="matercomus",
            repo="HOSER",
            issue_number=15,
            body=f"⛔ **Status**: BLOCKED\n\nWaiting on Issue #{blocking_issue_num} to be completed."
        )
        # Skip to next issue
```

---

## Search Query Patterns

### By Phase
```python
# Phase 1 (Week 1 - Critical)
search_issues(query='is:open "[Phase-1]" in:title repo:matercomus/HOSER')

# Phase 2 (Week 2 - Major)
search_issues(query='is:open "[Phase-2]" in:title repo:matercomus/HOSER')

# Phase 3 (Week 3 - Moderate/Minor)
search_issues(query='is:open "[Phase-3]" in:title repo:matercomus/HOSER')

# Deferred (Experimental work)
search_issues(query='is:open "[Deferred]" in:title repo:matercomus/HOSER')
```

### By Priority
```python
# P0 Critical (must fix before publication)
search_issues(query='is:open "[P0]" in:title repo:matercomus/HOSER')

# P1 Major (high priority)
search_issues(query='is:open "[P1]" in:title repo:matercomus/HOSER')

# P2 Moderate
search_issues(query='is:open "[P2]" in:title repo:matercomus/HOSER')

# P3 Minor
search_issues(query='is:open "[P3]" in:title repo:matercomus/HOSER')
```

### By Feasibility
```python
# Directly fixable
search_issues(query='is:open "[fix]" in:title repo:matercomus/HOSER')

# Can mitigate
search_issues(query='is:open "[mitigate]" in:title repo:matercomus/HOSER')

# Document limitation
search_issues(query='is:open "[document]" in:title repo:matercomus/HOSER')

# Requires experiments (deferred)
search_issues(query='is:open "[defer]" in:title repo:matercomus/HOSER')
```

### Combined Queries
```python
# Phase 1 P0 issues
search_issues(query='is:open "[P0][Phase-1]" in:title repo:matercomus/HOSER')

# Phase 2 fixable issues
search_issues(query='is:open "[Phase-2][fix]" in:title repo:matercomus/HOSER')

# All closed issues
search_issues(query='is:closed "Issue" in:title repo:matercomus/HOSER')
```

---

## Progress Tracking

### Generate Status Report
```python
# Count issues by phase
phase1_open = len(search_issues(query='is:open "[Phase-1]" in:title repo:matercomus/HOSER'))
phase1_closed = len(search_issues(query='is:closed "[Phase-1]" in:title repo:matercomus/HOSER'))
phase1_total = phase1_open + phase1_closed

# Repeat for Phase 2, 3, Deferred

print(f"""
## Progress Report

### Phase 1 (Critical)
- Complete: {phase1_closed}/{phase1_total} ({phase1_closed/phase1_total*100:.0f}%)
- Remaining: {phase1_open}

### Phase 2 (Major)
...
""")
```

### Update Checklist Document
After completing each issue, update `IMPLEMENTATION_FIXES_CHECKLIST.md`:

```python
# Mark issue as complete in checklist
update_checklist_file(issue_number, github_issue_number, status="complete")
```

---

## Issue Body Template

Standard structure for all issues:

```markdown
## Metadata
- **Priority**: P0-Critical | P1-Major | P2-Moderate | P3-Minor
- **Phase**: Phase 1 | Phase 2 | Phase 3 | Deferred
- **Feasibility**: fix | mitigate | document | defer
- **Category**: code-fix | documentation | statistical | experimental
- **Effort**: X hours
- **Dependencies**: Issue X.X | None

## Problem Statement
[Clear description of what is wrong and why it matters]

## Evidence
[File paths, line numbers, specific code/documentation that demonstrates the issue]

## Required Changes
[Specific, actionable steps to fix the issue]

## Validation Steps
- [ ] Validation step 1
- [ ] Validation step 2
- [ ] Validation step 3

## Files to Modify
- `path/to/file1.py`
- `path/to/file2.md`

---
**Reference**: `docs/implementation-review-remeditation/IMPLEMENTATION_REMEDIATION_REPORT.md` - Issue X.X
```

---

## Best Practices

### 1. Always Comment on Progress
```python
# At start
add_issue_comment(issue_number, "🤖 Status: IN PROGRESS")

# During work
add_issue_comment(issue_number, "✏️ Modified: file.py (lines 100-150)")

# After validation
add_issue_comment(issue_number, "✅ Validation: All checks passed")
```

### 2. Check Dependencies First
```python
# Parse issue body for "Dependencies" section
# Verify all blocking issues are closed before starting
```

### 3. Atomic Commits
```python
# One commit per issue
git commit -m "[Issue 1.4] fix: update JSD binning documentation to match code"
```

### 4. Validate Before Closing
```python
# Always complete all validation steps in issue body
# Document validation results in comment
# Only close after ALL validation passes
```

### 5. Keep Documentation Synchronized
```python
# When updating code, update corresponding docs in same commit
# When updating docs, verify code matches
```

---

## Troubleshooting

### Issue Not Found by Search
- Verify exact title format with brackets
- Check for typos in search query
- Use broader search: `"Issue 1.4" in:title repo:matercomus/HOSER`

### Can't Close Issue
- Verify you have write access to repository
- Check issue number is correct
- Ensure using `issue_write(method="update", state="closed")`

### Validation Fails
- Do not close issue
- Add comment documenting failure
- Fix the issue and re-validate
- Update progress comment with new validation results

---

## GitHub MCP Tool Reference

### Available Tools

**Issue Management**:
- `issue_write(method="create", ...)` - Create new issue
- `issue_write(method="update", ...)` - Update issue (close, reopen)
- `issue_read(method="get", ...)` - Get issue details
- `issue_read(method="get_comments", ...)` - Get issue comments
- `add_issue_comment(...)` - Add comment to issue

**Querying**:
- `list_issues(...)` - List repository issues
- `search_issues(...)` - Search issues across GitHub

**NOT Available**:
- Milestone creation/management
- Label creation/management  
- Project board management

---

## Summary

**Key Points**:
- ✅ Zero manual setup - fully automated
- ✅ Title-based categorization
- ✅ Search via GitHub native patterns
- ✅ Comment-based status tracking
- ✅ Works entirely with GitHub MCP tools

**Workflow**:
1. Create 35 issues with structured titles
2. Query by title patterns to find next issue
3. Comment progress throughout implementation
4. Validate before closing
5. Track progress via search queries

**Documentation synchronization is critical** - always update code and docs together.

