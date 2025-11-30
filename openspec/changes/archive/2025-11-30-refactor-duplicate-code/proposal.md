# Change: Refactor duplicate code across repositories, clients, and helpers

## Why
Pylint reports 6 duplicate code instances (R0801) in active code, violating DRY principles and increasing maintenance burden:

1. **Repositories** (3 instances): `MeasurementRepository` and `PredictionRepository` share identical `get_recent()`, `count()`, `delete_all()`, and `_get_collection()` implementations.

2. **Client scripts** (1 instance): `client.py` and `client_twin.py` duplicate the POST request logic for sending readings.

3. **DataFrame column fixing** (2 instances): Same column renaming/sanitization logic exists in `helpers/dataframe.py`, `pipeline/data_pipeline.py`, and `services/prediction_service.py`.

## What Changes
- Create `BaseRepository` class with shared repository methods
- Have `MeasurementRepository` and `PredictionRepository` inherit from it
- Extract `send_reading()` helper function for client scripts
- Consolidate DataFrame column sanitization to single function in `helpers/dataframe.py`

## Impact
- Affected specs: `core` (Repository Pattern requirement)
- Affected code:
  - `src/repositories/base_repository.py` (new)
  - `src/repositories/measurement_repository.py`
  - `src/repositories/prediction_repository.py`
  - `src/helpers/client_helpers.py` (new)
  - `src/helpers/dataframe.py`
  - `src/pipeline/data_pipeline.py`
  - `src/services/prediction_service.py`
  - `client.py`
  - `client_twin.py`
