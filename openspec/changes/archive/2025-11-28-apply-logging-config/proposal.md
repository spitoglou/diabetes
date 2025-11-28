# Change: Apply DEBUG and LOG_LEVEL configuration consistently

## Why
The `DEBUG` and `LOG_LEVEL` environment variables are defined in configuration but inconsistently applied across the codebase:
- `LOG_LEVEL` is only used in 2 of 6+ entry points (`final_run.py`, `load_model_and_predict.py`)
- `DEBUG` is only checked in one place (`server.py:149`)
- Multiple files use `print()` statements instead of logger calls
- Entry points like `server.py`, `client.py`, and `mobile/main.py` don't call `setup_logging()`

This leads to inconsistent logging behavior and no way to control verbosity across the application.

## What Changes
- All entry points SHALL call `setup_logging()` at startup
- All `print()` statements in `src/` SHALL be replaced with appropriate logger calls
- `DEBUG` mode SHALL enable additional diagnostic logging where beneficial
- `debug_print()` helper SHALL use logger instead of print

## Impact
- Affected specs: `core` (Standardized Logging requirement)
- Affected code:
  - Entry points: `server.py`, `client.py`, `mobile/main.py`
  - Helper modules: `src/helpers/experiment.py`, `src/helpers/misc.py`
  - Metrics: `src/helpers/diabetes/cega.py`, `src/helpers/diabetes/madex.py`
