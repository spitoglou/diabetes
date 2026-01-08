# Tech Debt Registry

**Last Updated:** 2026-01-08

> **Purpose:** Track deferred improvements. Review weekly.

---

## Summary

| Priority | Count | Oldest |
|----------|-------|--------|
| Critical | 0 | - |
| High | 0 | - |
| Medium | 0 | - |
| Low | 0 | - |

---

## Critical (Immediate Attention)

<!-- Items that block work or pose security/reliability risks -->

---

## High Priority

<!-- Items that should be addressed soon -->

---

## Medium Priority

<!-- Items to address when convenient -->

---

## Low Priority

<!-- Nice-to-have improvements -->

---

## Resolved

<!-- Completed items - keep for reference -->
<!-- Format: - [x] **TD-NNN**: Description (Resolved: YYYY-MM-DD) -->

---

## Notes

**Creating entries:**
```markdown
- [ ] **TD-NNN**: Brief description
  - **Impact:** Critical | High | Medium | Low
  - **Source:** [report-name.md](category/report-name.md)
  - **Created:** YYYY-MM-DD
```

**With OpenSpec link:**
```markdown
- [ ] **TD-NNN**: Brief description
  - **Impact:** Critical | High | Medium | Low
  - **Source:** [report-name.md](category/report-name.md)
  - **OpenSpec:** [change-id](../../openspec/changes/[change-id]/)
  - **Created:** YYYY-MM-DD
```

**Debt is created from:**
- `/postmortem` action items (P0/P1 → Critical/High)
- `/review-full` findings marked "won't fix now"
- `/agents:security` non-blocking vulnerabilities
- `/rfc` deferred requirements
- Manual identification via `/debt add`
