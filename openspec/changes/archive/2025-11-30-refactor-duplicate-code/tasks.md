## 1. Repository Refactoring
- [x] 1.1 Create `src/repositories/base_repository.py` with shared methods
- [x] 1.2 Refactor `MeasurementRepository` to inherit from `BaseRepository`
- [x] 1.3 Refactor `PredictionRepository` to inherit from `BaseRepository`
- [x] 1.4 Update repository tests if needed

## 2. Client Helper Extraction
- [x] 2.1 Create `src/helpers/client_helpers.py` with `send_reading()` function
- [x] 2.2 Update `client.py` to use `send_reading()`
- [x] 2.3 Update `client_twin.py` to use `send_reading()`

## 3. DataFrame Helper Consolidation
- [x] 3.1 Ensure `fix_column_names()` in `helpers/dataframe.py` handles all use cases
- [x] 3.2 Update `pipeline/data_pipeline.py` to use shared helper
- [x] 3.3 Update `services/prediction_service.py` to use shared helper

## 4. Verification
- [x] 4.1 Run tests to verify no regressions
- [x] 4.2 Run pylint to verify duplicate code warnings resolved
