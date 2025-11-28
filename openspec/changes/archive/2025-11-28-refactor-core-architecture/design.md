# Design: Core Architecture Refactoring

## Context

The diabetes prediction system has grown organically during PhD research, prioritizing rapid experimentation over architectural cleanliness. Now that the core ML approach is validated, the codebase needs refactoring to support:
- Safe deployment (no hardcoded credentials)
- Maintainability (clear separation of concerns)
- Testability (dependency injection enables mocking)
- Extensibility (add new data sources, models, or metrics)

## Goals / Non-Goals

**Goals:**
- Remove all hardcoded credentials from source code
- Enable unit testing of individual components
- Establish clear module boundaries
- Reduce code duplication
- Standardize logging across modules

**Non-Goals:**
- Changing the ML model architecture or algorithms
- Migrating to a different database
- Adding new features or capabilities
- Changing the prediction pipeline logic
- Full HIPAA compliance (requires broader effort)

## Decisions

### Decision 1: Layered Architecture

Adopt a layered architecture with clear boundaries:

```
┌─────────────────────────────────────────────────┐
│                 Entry Points                     │
│   (final_run.py, server.py, mobile/main.py)     │
├─────────────────────────────────────────────────┤
│              Service Layer                       │
│   (ExperimentOrchestrator, PredictionService)   │
├─────────────────────────────────────────────────┤
│              Pipeline Layer                      │
│   (DataPipeline, FeatureEngineer, ModelTrainer) │
├─────────────────────────────────────────────────┤
│             Repository Layer                     │
│   (MeasurementRepository, PredictionRepository) │
├─────────────────────────────────────────────────┤
│              Infrastructure                      │
│   (MongoDB, Config, Logging)                    │
└─────────────────────────────────────────────────┘
```

**Rationale:** Clear layers enable testing at each level and make dependencies explicit.

**Alternatives considered:**
- Hexagonal architecture: Overkill for this project size
- No refactoring: Blocks testing and safe deployment

### Decision 2: Constructor Injection

Use constructor injection for all dependencies rather than a DI framework:

```python
class ExperimentOrchestrator:
    def __init__(
        self,
        provider: BgcProviderInterface,
        feature_engineer: FeatureEngineer,
        model_trainer: ModelTrainer,
        metrics_calculator: MetricsCalculator,
        config: Config
    ):
        self.provider = provider
        # ...
```

**Rationale:** 
- Python's simplicity doesn't warrant a DI container
- Constructor injection is explicit and type-checkable
- Easy to understand for contributors

**Alternatives considered:**
- `dependency-injector` library: Adds complexity without significant benefit
- Service locator pattern: Hides dependencies, harder to test

### Decision 3: Environment-Based Configuration

Use `.env` files with `python-dotenv` for configuration:

```python
# src/config.py
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Config:
    mongo_uri: str = os.getenv("MONGO_URI", "")
    neptune_token: str = os.getenv("NEPTUNE_API_TOKEN", "")
    neptune_project: str = os.getenv("NEPTUNE_PROJECT", "")
    database_name: str = os.getenv("DATABASE_NAME", "diabetes_db")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Domain constants
    glucose_low: int = 70
    glucose_high: int = 180
    valid_patient_ids: tuple = (559, 563, 570, 575, 588, 591)
```

**Rationale:**
- Industry standard approach
- Works in development and production
- Easy to configure in Docker/cloud environments

**Alternatives considered:**
- YAML config files: Requires additional library, less standard for secrets
- Command-line arguments: Verbose for many settings

### Decision 4: Repository Pattern for Data Access

Implement repository classes that encapsulate MongoDB operations:

```python
# src/repositories/measurement_repository.py
class MeasurementRepository:
    def __init__(self, db_client, database_name: str, config: Config):
        self.db = db_client[database_name]
        self.config = config
    
    def save(self, patient_id: str, measurement: dict) -> str:
        self._validate_patient_id(patient_id)
        collection = self.db[f"measurements_{patient_id}"]
        result = collection.insert_one(measurement)
        return str(result.inserted_id)
    
    def get_recent(self, patient_id: str, limit: int = 100) -> List[dict]:
        self._validate_patient_id(patient_id)
        collection = self.db[f"measurements_{patient_id}"]
        return list(collection.find().sort("_id", -1).limit(limit))
    
    def _validate_patient_id(self, patient_id: str) -> None:
        if int(patient_id) not in self.config.valid_patient_ids:
            raise ValueError(f"Invalid patient ID: {patient_id}")
```

**Rationale:**
- Centralizes validation logic
- Enables mocking for tests
- Prevents collection name injection

### Decision 5: Pipeline Component Extraction

Split `Experiment` class into focused pipeline components:

| Component | Responsibility |
|-----------|----------------|
| `DataPipeline` | Load data, remove gaps, handle missing values |
| `FeatureEngineer` | Coordinate tsfresh feature extraction |
| `ModelTrainer` | PyCaret setup, model comparison, selection |
| `MetricsCalculator` | CEGA, MADEX, RMSE calculation |
| `ExperimentOrchestrator` | Compose components, manage workflow |

**Rationale:**
- Single Responsibility Principle
- Each component testable in isolation
- Clear interfaces between stages

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing scripts | High | Create backward-compatible facade initially |
| Extended refactoring time | Medium | Implement incrementally, test after each phase |
| Team unfamiliarity | Low | Document architecture, create examples |
| Over-engineering | Medium | Keep implementations simple, no premature abstractions |

## Migration Plan

**Phase 1: Foundation (non-breaking)**
1. Add `src/config.py` with environment loading
2. Add `.env.example` template
3. Create repository classes alongside existing code
4. Add logging configuration

**Phase 2: Gradual Migration**
1. Update entry points to use Config
2. Replace direct MongoDB access with repositories
3. Update imports to use shared modules

**Phase 3: Experiment Refactoring**
1. Extract pipeline components
2. Create ExperimentOrchestrator
3. Deprecate old Experiment class
4. Update final_run.py to use new orchestrator

**Rollback:** Each phase is independently deployable. If issues arise, revert the specific phase without affecting others.

## New Directory Structure

```
src/
├── config.py                    # Centralized configuration
├── logging_config.py            # Logging setup
├── repositories/
│   ├── __init__.py
│   ├── measurement_repository.py
│   └── prediction_repository.py
├── pipeline/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   └── metrics_calculator.py
├── services/
│   ├── __init__.py
│   ├── experiment_orchestrator.py
│   └── prediction_service.py
├── helpers/
│   ├── dataframe.py             # Shared utilities (fix_column_names)
│   ├── diabetes/
│   │   ├── cega.py
│   │   └── madex.py
│   └── ...
├── bgc_providers/               # Unchanged
├── featurizers/                 # Unchanged
└── interfaces/                  # Unchanged
```

## Open Questions

1. **Backward compatibility period**: How long should the old `Experiment` class be maintained as a facade?
   - Recommendation: 1-2 release cycles

2. **Neptune integration**: Should Neptune logging be optional or required?
   - Recommendation: Keep optional, default to disabled

3. **Test database**: Should we use a test MongoDB instance or in-memory mocks?
   - Recommendation: Use mocks for unit tests, real MongoDB for integration tests
