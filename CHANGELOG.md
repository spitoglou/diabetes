# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-08

### Added
- **Unified CLI** (`cli.py`) with Typer framework for all operations
  - `info` - Show current configuration
  - `train` - Train prediction models
  - `serve` - Start FastAPI server
  - `stream` - Stream CGM data
  - `predict` - Run prediction watcher
  - `check-datasets` - Verify dataset availability
- **Pydantic Settings** (`config/settings.py`) for type-safe configuration
- **Environment-based configuration** via `.env` file support
- **Health endpoint** (`/health`) for API monitoring
- **Centralized logging** (`src/helpers/logging.py`) with Loguru
- **Type hints** for core modules (mongo, helpers, interfaces)
- **Expanded test suite** from 4 to 36 tests
- **Comprehensive documentation** in `docs/`:
  - `architecture.md` - System design overview
  - `api.md` - REST API reference
  - `training.md` - Model training guide
  - `development.md` - Development setup guide
- **OpenSpec integration** for change management

### Changed
- **Scripts reorganized** into `scripts/` subdirectories:
  - `scripts/training/` - Model training scripts
  - `scripts/serving/` - API server and streaming
  - `scripts/utils/` - Utility scripts
- **Configuration consolidated** from multiple files to single Settings class
- **README.md** updated with CLI commands and new structure
- **CLAUDE.md** updated with new configuration and commands

### Removed
- **Dead code files**: `create.py`, `main.py`, `sandbox.py`, `live_test.py`
- **Duplicate files**: `mobile/src/mongo.py`
- **Obsolete files**: `src/bgc_providers/aida_bgc_provider.py`, `src/helpers/diabetes/madex_old.py`
- **Old config files**: `mongo_config.py`, `server_config.py`, `simulation_config.py` templates
- **requirements.txt** (replaced by `uv.lock`)

## [0.1.0] - Initial Release

### Added
- Blood glucose prediction using PyCaret and tsfresh
- Ohio T1DM dataset support
- FastAPI server for CGM data ingestion
- MongoDB storage for measurements and predictions
- Real-time prediction pipeline
- Clarke Error Grid Analysis (CEGA) metrics
- MADEX diabetes-specific error metric
