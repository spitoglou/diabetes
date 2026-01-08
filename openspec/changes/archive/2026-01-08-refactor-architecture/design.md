# Design: Refactor Project Architecture

## Context

The diabetes prediction system has grown organically, resulting in:
- 13 root-level scripts with unclear purposes
- Dead code from abandoned features (AIDA provider, SQLite backend)
- Duplicate code (create.py = dataset_generator.py, two mongo.py files)
- Hardcoded credentials in version-controlled files
- No unified entry point for users

This refactoring addresses technical debt while maintaining all working functionality.

## Goals / Non-Goals

### Goals
- Remove all dead and duplicate code
- Centralize configuration with environment variable support
- Organize scripts into logical directories
- Add type safety for maintainability
- Create a unified CLI entry point

### Non-Goals
- Change ML pipeline behavior or algorithms
- Modify database schema
- Add new features (this is purely structural)
- Change the core data flow

## Decisions

### D1: Pydantic Settings for Configuration

**Decision:** Use `pydantic-settings` for unified configuration.

**Rationale:**
- Type-safe configuration with validation
- Automatic `.env` file loading
- Environment variable support (12-factor app)
- IDE autocompletion for config values
- Already using Pydantic in FastAPI

**Alternatives Considered:**
- `python-dotenv` only: Simpler but no validation or type safety
- `dynaconf`: More features but overkill for this project
- Keep current approach: Already proven problematic (duplication, no secrets protection)

### D2: Typer for CLI

**Decision:** Use `typer` for unified command-line interface.

**Rationale:**
- Built on Click, battle-tested
- Automatic help generation
- Type hints for argument validation
- Subcommand support matches our use case
- Good developer experience

**Alternatives Considered:**
- `argparse`: Standard library but verbose, no subcommand discoverability
- `click`: Good but Typer adds type hint support
- Keep separate scripts: Current state is confusing

### D3: Script Organization Structure

**Decision:** Organize into `scripts/{training,serving,utils}/`.

```
scripts/
├── training/
│   ├── train_simple.py      # Quick model training
│   ├── train_full.py        # Full experiment framework
│   ├── generate_datasets.py # Dataset creation
│   └── run_experiment.py    # Multi-patient experiments
├── serving/
│   ├── server.py            # FastAPI server
│   ├── client.py            # CGM data simulator
│   └── predictor.py         # MongoDB watcher + predictions
└── utils/
    ├── check_datasets.py    # Dataset diagnostics
    └── export_part_of_day.py # Excel export
```

**Rationale:**
- Matches the two main workflows (training vs real-time)
- Utils for supporting scripts
- Clear mental model for new developers

**Alternatives Considered:**
- Flat `scripts/` directory: Gets cluttered as project grows
- Keep in root: Current state is confusing
- `src/cli/` location: Mixes library code with executable scripts

### D4: Gradual Type Hint Addition

**Decision:** Add types to `src/` modules only, not scripts.

**Rationale:**
- Core library code benefits most from types
- Scripts are entry points, less reused
- Reduces initial effort
- mypy can be configured to be strict on `src/` only

### D5: Phase Ordering

**Decision:** Execute phases in order: Cleanup → Config → Organization → Types

**Rationale:**
- Phase 1 (cleanup) has zero risk, provides immediate wins
- Phase 2 (config) fixes security issues early
- Phase 3 (organization) can build on clean foundation
- Phase 4 (types) is enhancement, can be done incrementally

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing workflows | Medium | High | Each phase tested independently; git rollback available |
| Import path changes confuse users | Medium | Medium | Update all documentation; provide migration guide |
| Config migration misses a file | Low | Medium | Grep for old config imports after Phase 2 |
| Type hints introduce runtime errors | Low | Low | Types are hints only; extensive testing |

## Migration Plan

### Phase 1: Dead Code Removal
1. Delete 7 identified files
2. Update mobile/main.py import
3. Verify with tests
4. **Rollback:** `git checkout -- <files>`

### Phase 2: Configuration
1. Create new config structure
2. Migrate consumers one file at a time
3. Delete old config files
4. **Rollback:** Restore old config files from git

### Phase 3: Script Organization
1. Create directory structure
2. Move files (preserving git history with `git mv`)
3. Update imports
4. Create CLI
5. Update documentation
6. **Rollback:** Move files back, delete CLI

### Phase 4: Type Safety
1. Add types file by file
2. Run mypy after each file
3. Fix errors incrementally
4. **Rollback:** Remove type hints (though rarely needed)

## Open Questions

1. **Should we rename patient-specific scripts?**
   - Current: `simple_train_559.py` has hardcoded patient ID
   - Proposed: `train_simple.py` with `--patient` argument
   - Decision: Yes, parameterize in Phase 3

2. **Should mobile/ be reorganized too?**
   - Current: Separate mobile/ directory with its own structure
   - Proposed: Keep separate for now, address in future change
   - Decision: Out of scope for this refactoring

3. **Should we add pre-commit hooks?**
   - Would enforce mypy, formatting
   - Decision: Out of scope, can be separate proposal
