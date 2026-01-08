---
name: Agents: Coverage
description: Invoke test-engineer agent for test coverage analysis.
category: Agents
tags: [agents, tests, coverage, quality]
---

**Purpose**
Analyze test coverage and identify gaps using the test-engineer agent.

**Steps**
1. Check `.claude/reports/_registry.md` for recent coverage analyses.
2. Determine scope from user input (specific modules or full codebase).
3. Invoke the test-engineer agent:
   ```
   Task(test-engineer, "
   **Objective:** Analyze test coverage and identify critical gaps.
   
   **Scope:** [specified modules or full codebase]
   
   **Commands to run:**
   - pytest --cov=src --cov-report=term-missing -v
   
   **Focus Areas:**
   - Critical paths (auth, leave workflow, data integrity)
   - Services layer coverage
   - Repository layer coverage
   - Route/endpoint coverage
   
   **Context from prior work:**
   - [Reference any relevant recent reports from registry]
   
   **Output:**
   - Report: .claude/reports/tests/coverage-analysis-YYYYMMDD.md
   ")
   ```
4. Verify report was created using `verify.py`.
5. Update `_registry.md` with the new report.
6. For critical gaps, create tech debt entries or OpenSpec proposals.

**Arguments**
- `$ARGUMENTS` - Optional: specific modules to analyze (defaults to full codebase)

**Example Usage**
```
/agents:coverage
/agents:coverage src/services/
/agents:coverage --focus auth  # Focus on authentication-related coverage
```

**Integration**
- Use during OpenSpec implementation to verify test coverage
- Link coverage gaps to tech debt registry
- Consider OpenSpec proposals for significant test infrastructure needs
