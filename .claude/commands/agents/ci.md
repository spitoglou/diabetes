---
name: Agents: CI
description: Run local CI pipeline (lint, type-check, test, security scan).
category: Agents
tags: [agents, ci, pipeline, quality]
---

**Purpose**
Run a comprehensive local CI pipeline to verify code quality before commits or OpenSpec archiving.

**Steps**
1. Run quality checks sequentially:

   ```bash
   # Step 1: Linting
   echo "=== Step 1/4: Linting ==="
   uv run ruff check src tests
   
   # Step 2: Type checking
   echo "=== Step 2/4: Type Checking ==="
   uv run mypy src --ignore-missing-imports
   
   # Step 3: Tests
   echo "=== Step 3/4: Running Tests ==="
   uv run pytest --tb=short -q
   
   # Step 4: (Optional) Security scan on changed files
   echo "=== Step 4/4: Security Overview ==="
   # Quick security check using grep patterns
   ```

2. Report results summary:
   - Lint: Pass/Fail with error count
   - Types: Pass/Fail with error count (baseline ~152)
   - Tests: Pass/Fail with test count
   - Security: Any obvious issues

3. If all pass, output success message.
4. If failures, provide specific remediation steps.

**Quality Gate Criteria**
- [ ] Ruff: Zero errors
- [ ] Mypy: No increase from baseline (~152 errors)
- [ ] Tests: All pass (361+ expected)

**Arguments**
- `$ARGUMENTS` - Optional: `--fix` to auto-fix linting issues, `--quick` to skip tests

**Example Usage**
```
/agents:ci           # Full CI pipeline
/agents:ci --fix     # Fix linting issues first, then run full pipeline
/agents:ci --quick   # Skip tests (lint + type-check only)
```

**Integration**
- Use before `openspec archive` to ensure quality gate passes
- Use before committing significant changes
- Results can be saved to `.claude/reports/ci/ci-YYYYMMDD-HHMM.md` if needed
