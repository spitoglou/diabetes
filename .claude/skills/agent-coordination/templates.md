# Agent Coordination Templates

Load this file when creating reports or handoffs.

---

## Category Quick Reference

| Category | Use For | Example Filename |
|----------|---------|------------------|
| `analysis/` | Research, EDA, data exploration | `analysis-user-behavior-20251216.md` |
| `arch/` | Architecture decisions, ADRs, system design | `arch-api-redesign-20251216.md` |
| `bugs/` | Bug reports, root cause analysis | `bugs-login-failure-20251216.md` |
| `commits/` | Commit summaries, changelog entries | `commits-release-v2-20251216.md` |
| `design/` | UI/UX reviews, design specs | `design-dashboard-review-20251216.md` |
| `exec/` | Execution logs, command outputs | `exec-migration-20251216.md` |
| `handoff/` | Agent coordination, context transfers | `handoff-arch-to-backend-20251216.md` |
| `implementation/` | Implementation plans, code specs | `implementation-auth-module-20251216.md` |
| `review/` | Code reviews, PR reviews | `review-pr-123-20251216.md` |
| `tests/` | Test plans, test results, coverage | `tests-auth-coverage-20251216.md` |
| `security/` | Security scans, threat models, compliance | `security-scan-api-20251216.md` |
| `sre/` | SLOs, postmortems, capacity plans | `sre-postmortem-outage-20251216.md` |
| `rfc/` | Design proposals, RFCs | `RFC-0001-auth-system.md` |
| `ci/` | CI pipeline results | `ci-main-20251216-1430.md` |

---

## Report Template

```markdown
# [Category]: [Topic]

**Agent:** [Agent Name]
**Date:** YYYY-MM-DD
**Status:** Draft | Active | Completed | Superseded

---

## Summary
[2-4 sentences: what was done, key findings, outcome]

---

## Key Decisions
- **[Decision 1]:** [What + Why]
- **[Decision 2]:** [What + Why]

---

## Details
[Main content - organized with headers]

---

## Action Items

### For [Agent/Role]:
- [ ] [Task] (Priority: High/Med/Low)

---

## Dependencies
**Depends on:** [report.md] - [Why]
**Blocks:** [What needs this first]

---

## References
- [report.md] - [How it relates]
```

---

## Handoff Template

Use for agent-to-agent coordination:

```markdown
# Handoff: [From Agent] → [To Agent]

**Topic:** [What's being handed off]
**Date:** YYYY-MM-DD

---

## Context
[Background the receiving agent needs]

## Completed Work
- [What was done]
- [Decisions made]

## Action Items for [To Agent]
- [ ] [Specific task with acceptance criteria]
- [ ] [Another task]

## Must Read
- This report: [Specific sections]
- Related: [other-report.md]

## Success Criteria
- [How to verify completion]
```

---

## Registry Entry Format

```markdown
### YYYY-MM-DD
- report-name | Status | Summary in one line
- another-report | Status | Another summary
```

**Status values:** Active | Completed | Superseded

---

## Task Prompt Template

For invoking agents with full context:

```
Task(agent-name, "
**Objective:** [Clear goal]

**Context from registry:**
- [Report 1]: [Key decision/finding]
- [Report 2]: [Relevant constraint]
- Current state: [What exists now]

**Requirements:**
1. [Specific deliverable]
2. [Another deliverable]

**Files to work with:**
- path/to/file1
- path/to/file2

**Expected output:**
- Report: .claude/reports/[cat]/[name]-YYYYMMDD.md
- Code: [path if applicable]
- Success: [How to verify]

**Mode:** [If agent has modes: specify which]
")
```

---

## OpenSpec Integration

### When Reports Relate to OpenSpec

Add OpenSpec reference to report header when applicable:

```markdown
# [Category]: [Topic]

**Agent:** [Agent Name]
**Date:** YYYY-MM-DD
**Status:** Completed
**OpenSpec Change:** [change-id](../../openspec/changes/[change-id]/)

---
```

### Report Category → OpenSpec Mapping

| Report Category | OpenSpec Relationship |
|-----------------|----------------------|
| `rfc/` | May become OpenSpec proposal if proposing changes |
| `arch/` | Can feed into OpenSpec `design.md` files |
| `review/` | Evidence for OpenSpec pre-archive quality checks |
| `security/` | May trigger OpenSpec security-related proposals |
| `tests/` | Verification for OpenSpec implementation |

### OpenSpec Verification Report Template

For pre-archive verification (link in OpenSpec `tasks.md`):

```markdown
# Verification: [OpenSpec Change ID]

**Agent:** [Agent Name]
**Date:** YYYY-MM-DD
**OpenSpec Change:** [change-id](../../openspec/changes/[change-id]/)
**Status:** Completed

---

## Scope
Files/areas reviewed for this OpenSpec change:
- [file1.py]
- [file2.py]

## Findings

### Critical (Must Fix Before Archive)
- None | [List issues]

### High (Should Fix)
- None | [List issues]

### Medium/Low (Track as Tech Debt)
- None | [List issues with TD-NNN references]

## Verdict
- [ ] **APPROVED** - Safe to archive
- [ ] **BLOCKED** - Issues must be resolved first

## Notes
[Any additional context for the OpenSpec change owner]
```

### Tech Debt Entry with OpenSpec Link

When creating tech debt from agent findings related to an OpenSpec change:

```markdown
- [ ] **TD-NNN**: [Description]
  - **Impact:** [Critical|High|Medium|Low]
  - **Source:** [report-name.md](category/report-name.md)
  - **OpenSpec:** [change-id](../../openspec/changes/[change-id]/)
  - **Created:** YYYY-MM-DD
```
