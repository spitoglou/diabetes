---
name: agent-coordination
description: Coordination protocol for main Claude Code agent. Explicit user invocation required ("mobilize agents", "coordinate", "check registry"). Provides agent orchestration, registry management, and handoff protocols. Subagents never access this - main agent provides context in task prompts.
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, TodoWrite
---

# Agent Coordination Protocol

## Core Principle

**Build on existing work. Never recreate.**

---

## Two Registries

The system maintains two distinct registries:

| Registry | Purpose | Updated When |
|----------|---------|--------------|
| `_registry.md` | Index of completed work (reports) | After each agent produces a report |
| `_tech-debt.md` | Tracked improvements to address later | When issues are deferred, incidents occur |

**Decision rule:**
- **Registry** → "What work has been done?" (past/present state)
- **Tech Debt** → "What do we need to fix later?" (future work)

---

## 4-Step Workflow

### Step 1: Check Prior Work

Before invoking any agent, check what already exists:

```bash
# Check registry for recent reports
cat .claude/reports/_registry.md | head -50

# Check if archiving needed (>50 entries)
ENTRIES=$(grep -c "^- .*|.*|" .claude/reports/_registry.md 2>/dev/null || echo 0)
[ "$ENTRIES" -gt 50 ] && echo "⚠️ Registry has $ENTRIES entries - suggest /archive"

# Check relevant tech debt (if working on that area)
grep -i "[area-keyword]" .claude/reports/_tech-debt.md
```

**Read relevant reports** before proceeding to understand current state.

### Step 2: Context Injection

Subagents NEVER read registry or reports directly. Main agent provides ALL context:

```
Task(agent-name, "
[Objective]

Context from prior work:
- [Report X]: [key decisions/findings]
- [Tech debt TD-NNN]: [relevant constraint]
- Current state: [what exists now]

Requirements:
- [Specific deliverables]

Output location:
- Report: .claude/reports/[category]/[name]-YYYYMMDD.md
")
```

### Step 3: Sequencing

**Rule:** Will Agent B need Agent A's output?
- YES → Sequential (verify between each)
- NO → Parallel

### Step 4: Verify and Update

After each agent completes:

```bash
# Verify deliverables exist
.claude/skills/agent-coordination/scripts/verify.sh "[category]" "[name]" "[date]"
```

**Then update registries:**

1. **Always:** Add report to `_registry.md`
   ```
   - [report-name] | [Status] | [1-line summary]
   ```

2. **If issues deferred:** Add to `_tech-debt.md`
   ```
   - [ ] **TD-NNN**: [Description]
     - **Impact:** [Critical|High|Medium|Low]
     - **Source:** [report-name or postmortem-name]
   ```

---

## Report Categories

All reports go to `.claude/reports/[category]/`:

| Category | Folder | Use For | Typical Agents |
|----------|--------|---------|----------------|
| analysis | `analysis/` | Research, EDA, data exploration | data-engineer, data-viz |
| arch | `arch/` | Architecture decisions, ADRs, system design | architect, rfc |
| bugs | `bugs/` | Bug reports, root cause analysis | code-quality (debug) |
| commits | `commits/` | Commit summaries, changelog entries | devops (git) |
| design | `design/` | UI/UX reviews, design specs | ux-designer |
| exec | `exec/` | Execution logs, command outputs | devops |
| handoff | `handoff/` | Agent coordination, context transfers | (main agent) |
| implementation | `implementation/` | Implementation plans, code specs | backend, frontend |
| review | `review/` | Code reviews, PR reviews | code-quality (review) |
| tests | `tests/` | Test plans, test results, coverage | test-engineer, qa |
| security | `security/` | Security scans, threat models, compliance | security-engineer |
| sre | `sre/` | SLOs, postmortems, capacity plans | sre |
| rfc | `rfc/` | Design proposals, RFCs | rfc |
| ci | `ci/` | CI pipeline results | (bash/devops) |
| archive | `archive/` | Old reports (moved, not deleted) | (archive script) |

**Naming convention:** `[category]-[topic]-YYYYMMDD.md`

---

## Agent Reference

| Agent | Modes | Primary Output Category |
|-------|-------|------------------------|
| code-quality | review, debug, qa-strategy | review/, bugs/, tests/ |
| test-engineer | - | tests/ |
| architect | system, pipeline | arch/ |
| security-engineer | scan, threat-model, compliance | security/ |
| sre | reliability-review, incident, capacity | sre/ |
| rfc | author, review, decision | rfc/ |
| data-engineer | collect, analyze, preprocess | analysis/ |
| data-viz-specialist | - | analysis/, design/ |
| devops | infra, git | exec/, commits/, ci/ |
| docs | general, webdev | (documentation files) |
| frontend | - | implementation/ |
| backend | - | implementation/ |
| ml-engineer | train, evaluate, deploy | analysis/, implementation/ |
| lrl-nlp-expert | - | analysis/ |
| ux-designer | design, copy | design/ |

---

## When to Update Tech Debt

Tech debt entries are created when:

| Situation | Action |
|-----------|--------|
| Code review finds issue, won't fix now | Add as Medium/Low priority |
| Postmortem identifies prevention action | Add as Critical/High priority |
| RFC defers a requirement | Add as Medium priority |
| Security scan finds non-blocking issue | Add as High priority |
| Manual identification | Use `/debt add` |

**Format in `_tech-debt.md`:**
```markdown
- [ ] **TD-NNN**: Brief description
  - **Impact:** Critical | High | Medium | Low
  - **Source:** [link to originating report]
  - **Created:** YYYY-MM-DD
```

---

## OpenSpec Integration

The agent coordination system works alongside OpenSpec for spec-driven development.

### When Agent Findings Become OpenSpec Proposals

Agent findings that require code changes SHOULD become OpenSpec proposals:

| Agent Finding | Action |
|---------------|--------|
| Critical security vulnerability | Create OpenSpec change proposal |
| Architecture recommendation requiring refactor | Feed into OpenSpec `design.md` |
| Test gap in critical path (significant) | Create OpenSpec change proposal |
| Performance issue requiring refactor | Create OpenSpec change proposal |
| Minor issues, style, small fixes | Track as tech debt only |

### Linking Reports to OpenSpec

When agent work relates to an existing OpenSpec change:

1. **Reference in report header:**
   ```markdown
   **OpenSpec Change:** [change-id](../../openspec/changes/[change-id]/)
   ```

2. **Update tech debt with OpenSpec link:**
   ```markdown
   - [ ] **TD-NNN**: Description
     - **OpenSpec:** [change-id](../../openspec/changes/[change-id]/)
   ```

3. **Cross-reference in OpenSpec tasks.md:**
   ```markdown
   ## 3. Verification
   - [x] 3.1 Security scan - see `.claude/reports/security/[report].md`
   ```

### Report Categories → OpenSpec Mapping

| Report Category | OpenSpec Relationship |
|-----------------|----------------------|
| `rfc/` | May become OpenSpec proposal if proposing changes |
| `arch/` | Can feed into OpenSpec `design.md` files |
| `review/` | Evidence for OpenSpec pre-archive quality checks |
| `security/` | May trigger OpenSpec security-related proposals |
| `tests/` | Verification for OpenSpec implementation |

### OpenSpec Pre-Archive Verification

Before archiving an OpenSpec change, consider running verification agents:

```bash
# For security-sensitive changes (auth, data handling, input validation)
Task(security-engineer, "Scan [affected files] for OWASP vulnerabilities")

# For all changes with significant new code
Task(code-quality, "Review [affected files] for correctness and maintainability")

# For changes affecting critical user paths
Task(test-engineer, "Verify test coverage for [affected functionality]")
```

**Link verification reports** in the archived change's `tasks.md`.

---

## Commands

| Command | Purpose |
|---------|---------|
| /agents:review | Code review with code-quality agent |
| /agents:security | Security scan with security-engineer agent |
| /agents:ci | Local CI pipeline (lint → test → security) |
| /review-full | Multi-level review (L1: peer → L2: arch → L3: security → L4: reliability) |
| /rfc | Create/review design documents |
| /slo | Define service level objectives |
| /postmortem | Incident analysis and learning |
| /debt | View and manage tech debt |
| /archive | Move old registry entries to archive |

---

## Skill Resources

```
skills/agent-coordination/
├── SKILL.md              # This file
├── templates.md          # Report templates
├── reference.md          # Verification details, retry logic
└── scripts/
    ├── verify.sh         # Deliverable verification
    └── archive_reports.py # Registry archiving (dated snapshots)
```

---

**Version:** 5.2.0
