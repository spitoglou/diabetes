---
name: Context
description: Familiarize with codebase, OpenSpec context, and active changes.
category: Session
tags: [context, session, initialization]
---

**Purpose**
Initialize session context by reviewing the codebase structure, OpenSpec specifications, and active changes.

**Steps**
1. Read `openspec/project.md` to understand:
   - Project purpose and tech stack
   - Code style and architecture patterns
   - Testing strategy and git workflow
   - Domain context
   - Important constraints and implementation notes

2. Consult `openspec/AGENTS.md` for OpenSpec workflow conventions

3. Remember that OpenSpec is a collaborative specification system. `openspec` is the command-line interface for managing OpenSpec specifications. It is NOT a python module, it is a command-line tool. Familiarize yourself with the `openspec` command-line tool by running `openspec help` to see all available commands and options.
    
4. Run `openspec list` to see active change proposals and their status.

5. Run `openspec list --specs` to enumerate existing capability specifications.

6. Parse the Codebase:
   - Understand the directory structure and file organization
   - Identify key components and their roles
   - Review important configuration files and settings
   - Find documentation and resources related to the project. Main documentation resides in `docs` directory.
   - Consult `reports\handoffs` directory, `reports\_registry.md`, and `reports\_tech-debt.md` for current implementation context.

7. Summarize for the user:
   - Brief project overview
   - Active changes in progress (if any)
   - Available capabilities/specs
   - Any pending tasks or blockers noted in proposals

**Reference**
- When performing file system operations, first try methods for windows systems and assume powershell commands are present.
- Use `openspec show <id>` for details on specific changes or specs
-
