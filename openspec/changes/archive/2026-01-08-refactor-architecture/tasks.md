# Tasks: Refactor Project Architecture

## Phase 1: Dead Code Removal ✅ COMPLETE

### 1.1 Delete Duplicate/Dead Files
- [x] 1.1.1 Delete `create.py` (duplicate of `dataset_generator.py`)
- [x] 1.1.2 Delete `main.py` (empty stub - just prints "Hello from diabetes!")
- [x] 1.1.3 Delete `sandbox.py` (dev scratch file for MongoDB testing)
- [x] 1.1.4 Delete `live_test.py` (imports non-existent `db` module)
- [x] 1.1.5 Delete `src/bgc_providers/aida_bgc_provider.py` (references non-existent `data/aida/`)
- [x] 1.1.6 Delete `src/helpers/diabetes/madex_old.py` (superseded by `madex.py`)
- [x] 1.1.7 Delete `mobile/src/mongo.py` (duplicate of `src/mongo.py`)

### 1.2 Verify Phase 1
- [x] 1.2.1 Verified imports work correctly
- [x] 1.2.2 Verified MongoDB module imports
- [x] 1.2.3 Verified check_datasets.py runs
- [x] 1.2.4 All tests passing
- [x] 1.2.5 Updated mobile/main.py to use settings

---

## Phase 2: Configuration Consolidation ✅ COMPLETE

### 2.1 Create Environment-Based Config
- [x] 2.1.1 Created `.env.example` with all required variables (no secrets)
- [x] 2.1.2 Created `.env` with actual values
- [x] 2.1.3 Added `.env` to `.gitignore`

### 2.2 Create Pydantic Settings
- [x] 2.2.1 Added `pydantic-settings` dependency
- [x] 2.2.2 Created `config/settings.py` with unified Settings class
- [x] 2.2.3 Included all settings: MongoDB, patient config, server, simulation, Neptune

### 2.3 Migrate Config Consumers
- [x] 2.3.1 Updated `server.py` to use new settings
- [x] 2.3.2 Updated `client.py` to use new settings
- [x] 2.3.3 Updated `load_model_and_predict.py` to use new settings
- [x] 2.3.4 Updated `src/mongo.py` to use new settings
- [x] 2.3.5 Updated `src/helpers/experiment.py` to use new settings (Neptune from env)
- [x] 2.3.6 Updated `mobile/main.py` to use new settings

### 2.4 Remove Old Config Files
- [x] 2.4.1 Deleted `config/mongo_config.py`
- [x] 2.4.2 Deleted `config/server_config.py`
- [x] 2.4.3 Deleted `config/simulation_config.py`
- [x] 2.4.4 Deleted template files (*.py.template)

### 2.5 Verify Phase 2
- [x] 2.5.1 Verified server starts correctly
- [x] 2.5.2 All tests passing
- [x] 2.5.3 No hardcoded credentials in git-tracked files (moved to .env)

---

## Phase 3: Script Organization ✅ COMPLETE

### 3.1 Create Unified CLI
- [x] 3.1.1 Added `typer` dependency
- [x] 3.1.2 Created `cli.py` with unified command interface
- [x] 3.1.3 CLI provides single entry point for all operations

### 3.2 CLI Commands Implemented
- [x] 3.2.1 `cli.py info` - Show current configuration
- [x] 3.2.2 `cli.py train` - Run model training
- [x] 3.2.3 `cli.py serve` - Start FastAPI server
- [x] 3.2.4 `cli.py stream` - Start CGM streaming client
- [x] 3.2.5 `cli.py predict` - Start prediction watcher
- [x] 3.2.6 `cli.py check-datasets` - Check dataset availability

### 3.3 Organize Scripts into Directories
- [x] 3.3.1 Created `scripts/training/` directory
- [x] 3.3.2 Moved `simple_train_559.py`, `train_best_model_559.py`, `dataset_generator.py`, `final_run.py`
- [x] 3.3.3 Created `scripts/serving/` directory
- [x] 3.3.4 Moved `server.py`, `client.py`, `load_model_and_predict.py`
- [x] 3.3.5 Created `scripts/utils/` directory
- [x] 3.3.6 Moved `check_datasets.py`, `create_part_of_day_files.py`
- [x] 3.3.7 Added `__init__.py` files to all script directories
- [x] 3.3.8 Updated CLI imports to reference new script locations

### 3.4 Verify Phase 3
- [x] 3.4.1 Verified `uv run python cli.py --help` works
- [x] 3.4.2 Verified `uv run python cli.py info` shows config
- [x] 3.4.3 All tests passing (updated test imports)

---

## Phase 4: Type Safety, Quality & Documentation ✅ COMPLETE

### 4.1 Add Type Hints to Core Modules
- [x] 4.1.1 Added types to `src/mongo.py`
- [x] 4.1.2 Added types to `src/interfaces/bgc_provider_interface.py`
- [x] 4.1.3 Added types to `src/helpers/dataframe.py`
- [x] 4.1.4 Added types to `src/helpers/fhir.py`
- [x] 4.1.5 Added types to `src/helpers/misc.py`
- [x] 4.1.6 Added types to `src/helpers/diabetes/madex.py`

### 4.2 Configure Type Checking
- [x] 4.2.1 Added `mypy` dependency
- [x] 4.2.2 Added `[tool.mypy]` to `pyproject.toml`
- [x] 4.2.3 mypy passes on typed modules

### 4.3 Server Improvements
- [x] 4.3.1 Added `/health` endpoint with Pydantic response model
- [x] 4.3.2 Added proper Pydantic models for FHIR data
- [x] 4.3.3 Added global exception handler

### 4.4 Logging Standardization
- [x] 4.4.1 Created `src/helpers/logging.py` with centralized Loguru configuration
- [x] 4.4.2 Supports JSON format, file output, and configurable levels
- [x] 4.4.3 Added LOG_LEVEL, LOG_JSON, LOG_FILE to settings

### 4.5 Expand Test Coverage
- [x] 4.5.1 Expanded from 4 to 36 tests
- [x] 4.5.2 Added `tests/test_config.py` - Settings tests
- [x] 4.5.3 Added `tests/test_mongo.py` - MongoDB wrapper tests with mocking
- [x] 4.5.4 Added `tests/test_server.py` - FastAPI endpoint tests
- [x] 4.5.5 Added `tests/test_helpers.py` - Helper function tests (misc, fhir, dataframe, madex)
- [x] 4.5.6 Added `httpx` dev dependency for FastAPI TestClient

### 4.6 Comprehensive Documentation
- [x] 4.6.1 Created `docs/architecture.md` - System design and components
- [x] 4.6.2 Created `docs/api.md` - REST API reference with examples
- [x] 4.6.3 Created `docs/training.md` - Model training guide
- [x] 4.6.4 Created `docs/development.md` - Development setup and contributing
- [x] 4.6.5 Updated `CLAUDE.md` with new CLI commands and configuration
- [x] 4.6.6 Updated `README.md` with CLI usage and new project structure

### 4.7 Verify Phase 4
- [x] 4.7.1 mypy passes on core modules
- [x] 4.7.2 All 36 tests passing

---

## Final Verification ✅ COMPLETE

- [x] All 4 phases complete
- [x] All 36 tests passing
- [x] No hardcoded credentials in repository (moved to .env)
- [x] CLAUDE.md updated with new CLI commands
- [x] README.md updated with new structure
- [x] Comprehensive documentation in docs/

---

## Summary of Changes

### Files Deleted (Phase 1)
- `create.py`, `main.py`, `sandbox.py`, `live_test.py`
- `src/bgc_providers/aida_bgc_provider.py`
- `src/helpers/diabetes/madex_old.py`
- `mobile/src/mongo.py`
- `config/mongo_config.py`, `config/server_config.py`, `config/simulation_config.py`
- Config template files

### Files Created
- `.env.example` - Environment variable template
- `.env` - Actual environment values (gitignored)
- `config/settings.py` - Unified Pydantic Settings
- `cli.py` - Unified CLI with Typer
- `src/helpers/logging.py` - Centralized logging configuration
- `scripts/training/__init__.py` - Package init
- `scripts/serving/__init__.py` - Package init
- `scripts/utils/__init__.py` - Package init
- `docs/architecture.md` - System architecture documentation
- `docs/api.md` - API reference documentation
- `docs/training.md` - Training guide
- `docs/development.md` - Development guide
- `tests/test_config.py` - Config tests
- `tests/test_mongo.py` - MongoDB tests
- `tests/test_server.py` - Server tests
- `tests/test_helpers.py` - Helper tests

### Files Moved (Phase 3)
- `server.py` → `scripts/serving/server.py`
- `client.py` → `scripts/serving/client.py`
- `load_model_and_predict.py` → `scripts/serving/load_model_and_predict.py`
- `simple_train_559.py` → `scripts/training/simple_train_559.py`
- `train_best_model_559.py` → `scripts/training/train_best_model_559.py`
- `dataset_generator.py` → `scripts/training/dataset_generator.py`
- `final_run.py` → `scripts/training/final_run.py`
- `check_datasets.py` → `scripts/utils/check_datasets.py`
- `create_part_of_day_files.py` → `scripts/utils/create_part_of_day_files.py`

### Files Modified
- `cli.py` - Updated imports for new script locations
- `src/mongo.py` - Uses settings + type hints
- `src/helpers/experiment.py` - Neptune from settings
- `mobile/main.py` - Uses settings
- `src/helpers/misc.py` - Type hints
- `src/helpers/fhir.py` - Type hints
- `src/helpers/dataframe.py` - Type hints
- `src/interfaces/bgc_provider_interface.py` - Type hints
- `src/helpers/diabetes/madex.py` - Type hints
- `.gitignore` - Added .env
- `pyproject.toml` - Added dependencies and mypy config
- `CLAUDE.md` - Updated with CLI commands and config
- `README.md` - Updated with new structure

### Dependencies Added
- `pydantic-settings` - Environment-based configuration
- `typer` - CLI framework
- `mypy` (dev) - Type checking
- `httpx` (dev) - FastAPI test client
