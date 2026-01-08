# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-01-08

### Added
- **Historical data generation** (`data generate-historical`) CLI command
  - Generate simglucose CGM data for any date range
  - Supports all virtual patients (adult, adolescent, child)
  - Configurable insulin mode and random seed
  - Dry-run mode for testing without server

### Fixed
- **Realistic CGM intervals** for streaming clients
  - Ohio clients: 300s (5-minute Guardian sampling)
  - Simglucose client: 180s (3-minute Dexcom sampling)
  - Added `SIMGLUCOSE_INTERVAL` setting separate from `INTERVAL`

### Changed
- **Settings documentation** improved with CGM sensor sampling explanations

## [0.4.0] - 2026-01-08

### Added
- **Simglucose client command** (`serve simglucose-client`) for synthetic CGM simulation
  - Uses FDA-approved UVA/Padova T1D simulator with 30 virtual patients
  - Supports time-synced streaming (fast-forward to current time of day)
  - Three insulin modes: `none` (open-loop), `basal`, `basal-bolus`
  - Configurable meal schedules via `--meals` option
  - Reproducible simulations via `--seed` option
- **ConfigurableController** for customizable insulin delivery parameters
  - `--basal-rate, -b`: Basal insulin rate in U/hr (0.0-5.0)
  - `--target-glucose, -t`: Target glucose for corrections in mg/dL (70-200)
  - `--carb-ratio, -c`: Carbohydrate ratio in g/U (1-50)
  - `--correction-factor, -f`: Correction factor in mg/dL/U (5-200)
  - `--pre-bolus-minutes, -B`: Pre-meal bolus timing in minutes (0-45)
- **SimglucoseProvider** class in `src/bgc_providers/` for simglucose integration
- **Comprehensive documentation** for insulin parameters (`docs/simglucose-insulin-parameters.md`)
  - Clinical significance and algorithms for each parameter
  - Validation scenarios with 24-hour timetables
- **Test suite** for ConfigurableController (25 tests)
  - BBController equivalence validation
  - Parameter range validation
  - Custom behavior verification
- **OpenSpec specifications** for insulin-controller capability

### Changed
- **Settings** extended with simglucose configuration options
- **CLI** updated with new insulin parameter options for `serve simglucose-client`

## [0.3.0] - 2026-01-08

### Added
- **Synced client command** (`serve synced-client`) for real-time timestamp streaming
  - Starts from dataset reading closest to current time of day
  - Uses current system timestamps instead of historical ones
  - Supports `--patient` and `--verbose` options
- **CLI parameter options** for patient and model configuration:
  - `--patient, -p` option for `serve client` command
  - `--patient, -p` option for `serve predict` command
  - `--window, -w` option for `serve predict` command
  - `--horizon, -H` option for `serve predict` command
- **Simglucose results** for adult and adolescent patient simulations

### Fixed
- **Prediction watcher** now correctly passes window and horizon parameters to model loader

### Changed
- **Documentation** updated with new CLI parameter options

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
