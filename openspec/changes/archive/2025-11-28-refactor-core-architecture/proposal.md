# Change: Refactor Core Architecture

## Why

The current codebase has critical architectural issues that block maintainability, testability, and security:

1. **Security Risk**: Hardcoded API credentials (Neptune token) in source code
2. **Tight Coupling**: Components directly instantiate dependencies, making testing impossible
3. **Mixed Concerns**: The `Experiment` class handles 7+ responsibilities (data loading, feature engineering, training, evaluation, logging, persistence)
4. **Code Duplication**: MongoDB wrapper and column-fixing logic duplicated across modules
5. **No Repository Pattern**: Direct database access in API endpoints creates injection risks
6. **Inconsistent Logging**: Mix of `print()`, `logger`, and no logging

These issues make the codebase difficult to extend, test, or deploy safely.

## What Changes

### Configuration Management
- Extract all credentials to environment variables
- Create centralized `Config` class with validation
- Add `.env.example` template to repository
- Remove all hardcoded paths and magic numbers

### Dependency Injection
- Refactor `Experiment` class to accept providers via constructor
- Create `PredictionService` class with injected dependencies
- Enable component substitution for testing

### Repository Pattern
- Create `MeasurementRepository` for database operations
- Create `PredictionRepository` for prediction storage
- Add input validation and collection name whitelisting
- Abstract MongoDB access behind repository interface

### Separation of Concerns
- Split `Experiment` class into focused components:
  - `DataPipeline`: Data loading and transformation
  - `FeatureEngineer`: Feature extraction coordination
  - `ModelTrainer`: PyCaret model training
  - `MetricsCalculator`: Evaluation metrics (CEGA, MADEX, RMSE)
- Extract duplicate `fix_names()` logic to shared utility

### Logging Standardization
- Replace all `print()` calls with structured logging
- Add context (patient, window, horizon) to log messages
- Configure consistent log format across modules

## Impact

- **Affected code:**
  - `src/helpers/experiment.py` - Major refactor (split into 4 modules)
  - `src/mongo.py` - Replace with repository pattern
  - `mobile/src/mongo.py` - Remove duplicate, import from shared
  - `server.py` - Use repository injection
  - `load_model_and_predict.py` - Use dependency injection
  - `final_run.py` - Update to use new architecture
  - `config/` - New configuration module

- **Breaking changes:**
  - `Experiment` class API changes (constructor signature)
  - MongoDB wrapper replaced with repositories
  - Configuration now requires `.env` file

- **New files:**
  - `src/config.py` - Centralized configuration
  - `src/repositories/measurement_repository.py`
  - `src/repositories/prediction_repository.py`
  - `src/pipeline/data_pipeline.py`
  - `src/pipeline/feature_engineer.py`
  - `src/pipeline/model_trainer.py`
  - `src/pipeline/metrics_calculator.py`
  - `.env.example` - Configuration template

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | Maintain backward-compatible `Experiment` facade initially |
| Model compatibility | Keep pickle format unchanged, only refactor orchestration |
| Database migrations | No schema changes, only access patterns |
| Learning curve | Document new architecture in `ARCHITECTURE.md` |
