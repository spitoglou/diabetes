# Change: Refactor Project Architecture

## Why

The diabetes prediction codebase has accumulated significant architectural debt:
- **~300 lines of dead/duplicate code** causing maintenance confusion
- **13 unorganized root scripts** with no clear entry points
- **Hardcoded credentials** in version control (security risk)
- **Configuration sprawl** across 3 files with duplicated settings
- **No type safety** making refactoring risky

This refactoring will clean up the codebase, improve security, and establish patterns for sustainable growth.

## What Changes

### Phase 1: Dead Code Removal (Critical)
- Delete `create.py` (100% duplicate of `dataset_generator.py`)
- Delete `main.py` (empty stub)
- Delete `sandbox.py` (dev scratch file)
- Delete `live_test.py` (broken imports)
- Delete `src/bgc_providers/aida_bgc_provider.py` (references non-existent data)
- Delete `src/helpers/diabetes/madex_old.py` (superseded by `madex.py`)
- Delete `mobile/src/mongo.py` (duplicate of `src/mongo.py`)

### Phase 2: Configuration Consolidation (High)
- Move credentials to `.env` file (gitignored)
- Create `.env.example` template
- Consolidate 3 config files into single Pydantic Settings class
- Remove duplicate `OHIO_ID` definitions
- **BREAKING**: Config import paths will change

### Phase 3: Script Organization (Medium)
- Create `scripts/` directory with `training/`, `serving/`, `utils/` subdirectories
- Move and rename root scripts to appropriate locations
- Create unified CLI using Typer (`cli.py`)
- Update CLAUDE.md with new command structure
- **BREAKING**: Script locations and names will change

### Phase 4: Type Safety & Quality (Medium)
- Add type hints to all `src/` modules
- Configure mypy for static type checking
- Add `/health` endpoint to FastAPI server
- Standardize logging (loguru everywhere)
- Expand test coverage
- Create comprehensive documentation (README, API docs, architecture guide)

## Impact

- **Affected specs**: None yet (first capability spec will be created)
- **Affected code**: 
  - 7 files deleted (Phase 1)
  - 3 config files consolidated (Phase 2)
  - 13 root scripts reorganized (Phase 3)
  - ~10 src/ modules updated with types (Phase 4)
- **Breaking changes**: Config imports (Phase 2), script paths (Phase 3)
- **Risk level**: Low for Phase 1, Medium for Phases 2-4
- **Rollback**: Each phase can be reverted independently via git
