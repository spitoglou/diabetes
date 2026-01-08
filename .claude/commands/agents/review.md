---
name: Agents: Review
description: Invoke code-quality agent for code review of specified scope.
category: Agents
tags: [agents, review, code-quality]
---

**Purpose**
Run a code review using the code-quality agent on specified files or the entire codebase.

**Steps**
1. Check `.claude/reports/_registry.md` for recent reviews to avoid duplication.
2. Determine scope from user input (specific files, directories, or full codebase).
3. Invoke the code-quality agent in review mode:
   ```
   Task(code-quality, "
   **Objective:** Code review focusing on security, correctness, performance, maintainability.
   
   **Scope:** [specified files/directories or 'src/']
   
   **Context from prior work:**
   - [Reference any relevant recent reports from registry]
   
   **Output:**
   - Report: .claude/reports/review/review-[scope]-YYYYMMDD.md
   
   **Mode:** review
   ")
   ```
4. Verify report was created using `verify.py`.
5. Update `_registry.md` with the new report.
6. If critical issues found, consider creating OpenSpec proposal or tech debt entries.

**Arguments**
- `$ARGUMENTS` - Optional: specific files or directories to review (defaults to `src/`)

**Example Usage**
```
/agents:review src/services/
/agents:review src/routes/auth.py src/services/session_service.py
/agents:review  # Reviews entire src/ directory
```

**Integration**
- Links to OpenSpec: If review identifies issues requiring changes, reference in OpenSpec proposals
- Links to Tech Debt: Add findings to `_tech-debt.md` if not immediately addressed
