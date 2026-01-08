# Architecture Assessment: Diabetes Prediction System

**Date:** 2026-01-08
**Status:** Completed
**Author:** architect agent

---

## Executive Summary

The diabetes blood glucose prediction system has solid ML foundations using PyCaret and tsfresh, but suffers from significant architectural debt. Key issues include 100% code duplication between scripts, hardcoded credentials, 13 unorganized root scripts, dead code, and configuration sprawl across 3 files. Immediate cleanup can remove ~300 lines of dead/duplicate code with zero risk.

---

## Current Architecture

### Package Management

This project uses **[UV](https://docs.astral.sh/uv/)** for dependency management:

```bash
# Environment setup
uv venv                  # Create virtual environment
uv sync                  # Install from uv.lock

# Dependency management
uv add <package>         # Add dependency
uv remove <package>      # Remove dependency

# Running scripts
uv run python <script>   # Run with correct environment
```

**Key Files:**
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions

### Directory Structure

```
diabetes/
├── pyproject.toml             # UV project config & dependencies
├── uv.lock                    # Locked dependency versions
├── config/                    # Configuration (3 files, some duplication)
│   ├── mongo_config.py        # MongoDB connection settings
│   ├── server_config.py       # Server + patient config (OHIO_ID, etc.)
│   └── simulation_config.py   # Simulation settings (also has OHIO_ID)
├── data/ohio/                 # Raw Ohio T1DM XML data
├── dataframes/                # Processed pickle datasets
├── models/                    # Trained model pickles
├── mobile/                    # Mobile app (Flask)
│   ├── main.py
│   └── src/mongo.py           # DUPLICATE of src/mongo.py
├── src/                       # Core modules
│   ├── bgc_providers/
│   │   ├── aida_bgc_provider.py   # DEAD CODE (references missing files)
│   │   └── ohio_bgc_provider.py   # Active - Ohio dataset parser
│   ├── featurizers/
│   │   └── tsfresh.py         # Time-series feature extraction
│   ├── helpers/
│   │   ├── dataframe.py       # DataFrame utilities
│   │   ├── experiment.py      # PyCaret experiment orchestration
│   │   ├── fhir.py            # FHIR message creation
│   │   ├── misc.py            # Utility functions
│   │   └── diabetes/
│   │       ├── cega.py        # Clarke Error Grid Analysis
│   │       ├── madex.py       # MADEX metric
│   │       └── madex_old.py   # DEAD CODE (old version)
│   ├── interfaces/
│   │   └── bgc_provider_interface.py
│   └── mongo.py               # MongoDB wrapper
├── tests/                     # Test files
└── [13 root scripts]          # Training, serving, utilities
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                        ROOT SCRIPTS                              │
├──────────────────┬──────────────────┬───────────────────────────┤
│  TRAINING        │  REAL-TIME       │  UTILITIES                │
│                  │                  │                           │
│  simple_train    │  server.py       │  check_datasets.py        │
│  train_best      │  client.py       │  create.py (DUPLICATE)    │
│  final_run.py    │  load_model...   │  dataset_generator.py     │
│                  │                  │  create_part_of_day.py    │
│                  │                  │  main.py (DEAD)           │
│                  │                  │  sandbox.py (DEAD)        │
│                  │                  │  live_test.py (DEAD)      │
└────────┬─────────┴────────┬─────────┴───────────────────────────┘
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│  src/helpers/   │  │  src/mongo.py   │
│  experiment.py  │  │  (+ duplicate   │
│                 │  │   in mobile/)   │
└────────┬────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         src/featurizers/            │
│         tsfresh.py                  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│       src/bgc_providers/            │
│       ohio_bgc_provider.py          │
│       (aida_bgc_provider.py DEAD)   │
└─────────────────────────────────────┘
```

### Data Flow

**Training Pipeline:**
```
Ohio XML Data → OhioBgcProvider → TsfreshFeaturizer → DataFrame
     ↓
create.py / dataset_generator.py (DUPLICATES)
     ↓
dataframes/*.pkl
     ↓
simple_train_559.py OR train_best_model_559.py
     ↓
PyCaret Experiment → models/*.pkl
```

**Real-Time Pipeline:**
```
CGM Device (simulated)
     ↓
client.py (streams FHIR)
     ↓
server.py (FastAPI :8000)
     ↓
MongoDB (measurements_{patient_id})
     ↓
load_model_and_predict.py (change stream watcher)
     ↓
MongoDB (predictions_{patient_id})
```

---

## Issues Identified

### Critical

| ID | Issue | Impact | Files |
|----|-------|--------|-------|
| C1 | **100% Code Duplication** - `create.py` and `dataset_generator.py` are byte-for-byte identical | Maintenance nightmare, confusion | `create.py`, `dataset_generator.py` |
| C2 | **Duplicate MongoDB Wrapper** - Same class exists in two locations | Changes must be made twice | `src/mongo.py`, `mobile/src/mongo.py` |
| C3 | **Dead Code in Production** - 3 root scripts don't work (`main.py` is stub, `live_test.py` imports non-existent `db`, `sandbox.py` is dev code) | Confusion, false documentation | `main.py`, `sandbox.py`, `live_test.py` |

### High Priority

| ID | Issue | Impact | Files |
|----|-------|--------|-------|
| H1 | **Hardcoded Credentials** - MongoDB URIs and Neptune API tokens in version control | Security vulnerability | `config/mongo_config.py`, `src/helpers/experiment.py` |
| H2 | **Configuration Sprawl** - `OHIO_ID` defined in both `server_config.py` and `simulation_config.py` | Inconsistent behavior if mismatched | `config/*.py` |
| H3 | **13 Root Scripts** - No clear entry point, unclear which to use | Onboarding difficulty | Root directory |
| H4 | **Two Training Scripts** - `simple_train_559.py` vs `train_best_model_559.py` with no guidance | User confusion | Both training scripts |
| H5 | **Dead Provider** - `aida_bgc_provider.py` imports from non-existent `data/aida/` | Import errors if used | `src/bgc_providers/aida_bgc_provider.py` |

### Medium Priority

| ID | Issue | Impact | Files |
|----|-------|--------|-------|
| M1 | **No Type Hints** - Core modules lack type annotations | IDE support, maintainability | `src/**/*.py` |
| M2 | **Mixed Database Backends** - `live_test.py` uses SQLite, everything else uses MongoDB | Confusion | `live_test.py` |
| M3 | **Old MADEX Version** - `madex_old.py` kept alongside `madex.py` | Dead code | `src/helpers/diabetes/madex_old.py` |
| M4 | **Hardcoded Patient ID** - Training scripts hardcode patient 559 | Limited flexibility | `simple_train_559.py`, `train_best_model_559.py` |
| M5 | **No Dependency Injection** - MongoDB/config hardwired into modules | Testing difficulty | Multiple files |

### Low Priority

| ID | Issue | Impact | Files |
|----|-------|--------|-------|
| L1 | **No Unified CLI** - Each script is standalone | UX fragmentation | Root scripts |
| L2 | **Inconsistent Logging** - Mix of loguru and print statements | Log management | Multiple files |
| L3 | **No Health Checks** - Server lacks `/health` endpoint | Operations | `server.py` |

---

## Recommended Refactorings

| ID | Description | Priority | Effort | Files Affected | Rationale |
|----|-------------|----------|--------|----------------|-----------|
| R1 | Delete `create.py` (keep `dataset_generator.py`) | Critical | Small | 1 file | 100% duplicate, zero risk |
| R2 | Delete `main.py`, `sandbox.py`, `live_test.py` | Critical | Small | 3 files | Dead code, no dependencies |
| R3 | Delete `mobile/src/mongo.py`, import from `src/` | Critical | Small | 2 files | Duplicate class |
| R4 | Delete `aida_bgc_provider.py` | High | Small | 1 file | References non-existent data |
| R5 | Delete `madex_old.py` | High | Small | 1 file | Superseded by `madex.py` |
| R6 | Move credentials to `.env` + python-dotenv | High | Medium | 3 config files | Security fix |
| R7 | Consolidate config into single Pydantic Settings class | High | Medium | 3 → 1 config file | Single source of truth |
| R8 | Rename `dataset_generator.py` → `generate_datasets.py` | Medium | Small | 1 file + imports | Clearer naming |
| R9 | Add CLI wrapper using Typer | Medium | Medium | New `cli.py` | Unified entry point |
| R10 | Parameterize patient ID in training scripts | Medium | Small | 2 files | Flexibility |
| R11 | Add type hints to `src/` modules | Medium | Large | ~10 files | Maintainability |
| R12 | Add `/health` endpoint to server | Low | Small | `server.py` | Operations |
| R13 | Standardize logging (loguru everywhere) | Low | Medium | ~8 files | Consistency |
| R14 | Create ADR for training script choice | Low | Small | New doc | Documentation |
| R15 | Add `__init__.py` exports for clean imports | Low | Small | `src/` packages | Developer experience |
| R16 | Move root scripts to `scripts/` directory | Low | Medium | 13 files | Organization |

---

## Proposed Target Architecture

### Phase 1: Immediate Cleanup (Week 1)

```bash
# Safe deletions - no dependencies
git rm create.py                           # Duplicate of dataset_generator.py
git rm main.py                             # Empty stub
git rm sandbox.py                          # Dev scratch file
git rm live_test.py                        # Uses non-existent imports
git rm src/bgc_providers/aida_bgc_provider.py  # Dead provider
git rm src/helpers/diabetes/madex_old.py   # Old version
git rm mobile/src/mongo.py                 # Duplicate wrapper
```

**Result:** -298 lines of dead/duplicate code

### Phase 2: Configuration Consolidation (Week 2)

**Before:**
```
config/
├── mongo_config.py      # MONGO_URI, MONGO_DB
├── server_config.py     # DATABASE, OHIO_ID, WINDOW_STEPS, ...
└── simulation_config.py # INTERVAL, OHIO_ID (duplicate!)
```

**After:**
```
config/
├── .env                 # Secrets (gitignored)
├── .env.example         # Template for secrets
└── settings.py          # Pydantic Settings class
```

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    database: str = "bgc"
    
    # Patient Configuration
    ohio_id: int = 559
    window_steps: int = 12
    prediction_horizon: int = 6
    
    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    # Simulation
    interval: float = 0.5
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Phase 3: Script Organization (Week 3)

**Before:** 13 scripts in root

**After:**
```
diabetes/
├── cli.py                    # Unified CLI entry point
├── scripts/
│   ├── training/
│   │   ├── train_simple.py   # Renamed from simple_train_559.py
│   │   ├── train_full.py     # Renamed from train_best_model_559.py
│   │   └── generate_datasets.py
│   ├── serving/
│   │   ├── server.py
│   │   ├── client.py
│   │   └── predictor.py      # Renamed from load_model_and_predict.py
│   └── utils/
│       ├── check_datasets.py
│       └── export_part_of_day.py
```

**CLI Usage (with UV):**
```bash
uv run python cli.py train --patient 559 --mode simple
uv run python cli.py train --patient 559 --mode full
uv run python cli.py serve
uv run python cli.py predict --watch
uv run python cli.py datasets check
uv run python cli.py datasets generate --patient 559
```

### Phase 4: Type Safety & Testing (Week 4-5)

1. Add type hints to all `src/` modules
2. Add mypy configuration
3. Expand test coverage (currently minimal)
4. Add integration tests for real-time pipeline

---

## Migration Path

### Week 1: Safe Cleanup
- [ ] Delete 7 dead/duplicate files (R1-R5)
- [ ] Verify all imports still work
- [ ] Update CLAUDE.md if any script names mentioned

### Week 2: Security & Config
- [ ] Create `.env` file with secrets
- [ ] Add `.env` to `.gitignore`
- [ ] Create Pydantic Settings class
- [ ] Update all imports to use new settings
- [ ] Remove old config files

### Week 3: Organization
- [ ] Create `scripts/` directory structure
- [ ] Move and rename scripts
- [ ] Create `cli.py` with Typer
- [ ] Update CLAUDE.md with new commands

### Week 4-5: Quality
- [ ] Add type hints progressively
- [ ] Configure mypy
- [ ] Add missing tests
- [ ] Document training script decision tree

---

## Appendix

### A. Root Script Inventory

| Script | Lines | Status | Recommendation |
|--------|-------|--------|----------------|
| check_datasets.py | 127 | Active | Keep |
| client.py | 44 | Active | Keep |
| create.py | 26 | **DUPLICATE** | **DELETE** |
| create_part_of_day_files.py | 24 | Active | Keep |
| dataset_generator.py | 26 | Active | Keep (rename) |
| final_run.py | 24 | Active | Keep |
| live_test.py | 58 | **BROKEN** | **DELETE** |
| load_model_and_predict.py | 149 | Active | Keep |
| main.py | 6 | **STUB** | **DELETE** |
| sandbox.py | 22 | **DEV** | **DELETE** |
| server.py | 55 | Active | Keep |
| simple_train_559.py | 285 | Active | Keep |
| train_best_model_559.py | 160 | Active | Keep |

### B. Training Script Decision Tree

```
Which training script should I use?

├─ Quick experimentation or first-time training?
│  └─ Use simple_train_559.py
│     - Direct PyCaret usage
│     - Faster execution
│     - Outputs comparison table + best model
│
└─ Full experiment with holdout evaluation?
   └─ Use train_best_model_559.py
      - Uses Experiment framework
      - Holdout + unseen data evaluation
      - More comprehensive metrics
      - Optional Neptune logging
```

### C. Immediate Action Commands

```bash
# Phase 1: Delete dead code (copy-paste ready)
git rm create.py
git rm main.py
git rm sandbox.py
git rm live_test.py
git rm src/bgc_providers/aida_bgc_provider.py
git rm src/helpers/diabetes/madex_old.py
git rm mobile/src/mongo.py

# Verify nothing breaks (using UV)
uv run python -c "from src.helpers.experiment import Experiment; print('OK')"
uv run python -c "from src.mongo import MongoDB; print('OK')"
uv run python check_datasets.py
```
