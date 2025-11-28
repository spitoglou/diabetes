# Implementation Tasks

## 1. Configuration Management
- [x] 1.1 Create `src/config.py` with `Config` dataclass for all settings
- [x] 1.2 Extract Neptune credentials to environment variables
- [x] 1.3 Extract MongoDB connection string to environment variables
- [x] 1.4 Create `.env.example` with all required variables documented
- [x] 1.5 Add `python-dotenv` to requirements and load in entry points
- [x] 1.6 Replace hardcoded patient IDs, paths, and thresholds with config

## 2. Repository Pattern
- [x] 2.1 Create `src/repositories/` directory structure
- [x] 2.2 Implement `MeasurementRepository` with CRUD operations
- [x] 2.3 Implement `PredictionRepository` with CRUD operations
- [x] 2.4 Add collection name validation (whitelist patient IDs)
- [x] 2.5 Add query projections for efficient data retrieval
- [x] 2.6 Remove duplicate `mobile/src/mongo.py`, import from shared

## 3. Dependency Injection
- [x] 3.1 Refactor `Experiment.__init__` to accept provider as parameter
- [x] 3.2 Create `PredictionService` class with constructor injection
- [x] 3.3 Update `server.py` to use FastAPI dependency injection
- [x] 3.4 Update `load_model_and_predict.py` to use injected services
- [x] 3.5 Create factory functions for production and test configurations

## 4. Separation of Concerns
- [x] 4.1 Create `src/pipeline/` directory structure
- [x] 4.2 Extract `DataPipeline` class (gap removal, null handling)
- [x] 4.3 Extract `FeatureEngineer` class (tsfresh coordination)
- [x] 4.4 Extract `ModelTrainer` class (PyCaret setup, comparison, selection)
- [x] 4.5 Extract `MetricsCalculator` class (CEGA, MADEX, RMSE)
- [x] 4.6 Create `ExperimentOrchestrator` that composes pipeline components
- [x] 4.7 Extract shared `fix_column_names()` to `src/helpers/dataframe.py`

## 5. Logging Standardization
- [x] 5.1 Create `src/logging_config.py` with standard format
- [x] 5.2 Replace all `print()` calls in `experiment.py` with logger
- [x] 5.3 Replace all `print()` calls in `mongo.py` with logger
- [x] 5.4 Replace all `print()` calls in `server.py` with logger
- [x] 5.5 Add contextual logging (patient, window, horizon) to pipeline

## 6. Update Entry Points
- [x] 6.1 Update `final_run.py` to use new architecture
- [x] 6.2 Update `load_model_and_predict.py` to use new services
- [x] 6.3 Update `server.py` to use repository injection
- [x] 6.4 Update `mobile/main.py` to use shared repository

## 7. Testing Infrastructure
- [x] 7.1 Create `tests/fixtures.py` with mock data and repositories
- [x] 7.2 Add unit tests for `MeasurementRepository`
- [x] 7.3 Add unit tests for `PredictionRepository`
- [x] 7.4 Add unit tests for `DataPipeline`
- [x] 7.5 Add integration test for prediction workflow

## 8. Documentation
- [x] 8.1 Update `openspec/project.md` with new architecture
- [x] 8.2 Document configuration setup in README
