# core Specification

## Purpose
TBD - created by archiving change refactor-core-architecture. Update Purpose after archive.
## Requirements
### Requirement: Centralized Configuration
The system SHALL load all configuration from environment variables via a centralized `Config` class.

The system SHALL NOT contain hardcoded credentials, API tokens, or connection strings in source code.

The system SHALL provide a `.env.example` file documenting all required environment variables.

#### Scenario: Application startup with valid configuration
- **WHEN** the application starts with all required environment variables set
- **THEN** the Config class loads successfully with validated values

#### Scenario: Application startup with missing credentials
- **WHEN** the application starts without required credentials (MONGO_URI, NEPTUNE_API_TOKEN)
- **THEN** the system raises a clear configuration error before attempting connections

#### Scenario: Configuration validation
- **WHEN** an invalid patient ID is referenced
- **THEN** the system rejects the operation with a validation error

---

### Requirement: Repository Pattern for Data Access
The system SHALL access MongoDB through repository classes that encapsulate all database operations.

The system SHALL validate patient IDs against a whitelist before constructing collection names.

#### Scenario: Save measurement via repository
- **WHEN** a valid FHIR measurement is submitted for a known patient ID
- **THEN** the MeasurementRepository stores it and returns the record ID

#### Scenario: Reject invalid patient ID
- **WHEN** a measurement is submitted for an unknown patient ID
- **THEN** the repository raises a ValueError before database access

#### Scenario: Retrieve recent measurements
- **WHEN** recent measurements are requested for a patient
- **THEN** the repository returns them sorted by timestamp descending with configurable limit

---

### Requirement: Dependency Injection
The system SHALL use constructor injection for all service dependencies.

Components SHALL NOT instantiate their own dependencies internally.

#### Scenario: Experiment orchestration with injected provider
- **WHEN** an ExperimentOrchestrator is created
- **THEN** it accepts a BgcProviderInterface implementation via constructor

#### Scenario: Testing with mock dependencies
- **WHEN** a component is instantiated with mock dependencies
- **THEN** it operates correctly using the mocked implementations

---

### Requirement: Separated Pipeline Components
The ML pipeline SHALL be composed of focused, single-responsibility components:
- DataPipeline: Data loading and cleaning
- FeatureEngineer: Feature extraction coordination
- ModelTrainer: Model training and selection
- MetricsCalculator: Evaluation metrics

#### Scenario: Data pipeline execution
- **WHEN** raw glucose data is processed through DataPipeline
- **THEN** it returns cleaned data with gaps removed and nulls handled

#### Scenario: Feature engineering execution
- **WHEN** cleaned data is processed through FeatureEngineer
- **THEN** it returns a feature DataFrame suitable for model training

#### Scenario: Model training execution
- **WHEN** feature data is processed through ModelTrainer
- **THEN** it returns trained models with comparison metrics

#### Scenario: Metrics calculation
- **WHEN** predictions are evaluated through MetricsCalculator
- **THEN** it returns CEGA zones, MADEX, and RMSE values

---

### Requirement: Standardized Logging
The system SHALL use loguru for all logging with consistent format.

The system SHALL NOT use print() for operational output in production code.

Log messages SHALL include contextual information (patient ID, window, horizon) where applicable.

All entry points SHALL call `setup_logging()` before performing operations to ensure LOG_LEVEL is respected.

When DEBUG mode is enabled, the system SHALL log additional diagnostic information including:
- Full request/response payloads in API endpoints
- Intermediate data shapes during pipeline execution
- Verbose feature extraction details

#### Scenario: Structured log output
- **WHEN** an experiment runs
- **THEN** log messages include patient ID, window size, and horizon in structured format

#### Scenario: Error logging with context
- **WHEN** an error occurs during prediction
- **THEN** the error is logged with full context and stack trace

#### Scenario: LOG_LEVEL respected across all entry points
- **WHEN** LOG_LEVEL is set to WARNING
- **THEN** INFO and DEBUG messages are suppressed in all entry points (server, client, mobile, final_run, load_model_and_predict)

#### Scenario: DEBUG mode enables verbose logging
- **WHEN** DEBUG=true is set in environment
- **THEN** additional diagnostic information is logged including payload details and intermediate values

### Requirement: Experiment Execution
The Experiment workflow SHALL be coordinated by ExperimentOrchestrator which composes pipeline components.

The system SHALL maintain backward compatibility through an Experiment facade during migration.

#### Scenario: Run experiment with new architecture
- **WHEN** final_run.py executes an experiment
- **THEN** it uses ExperimentOrchestrator with injected dependencies

#### Scenario: Backward compatible experiment execution
- **WHEN** legacy code instantiates Experiment class directly
- **THEN** it continues to work using the facade pattern

---

### Requirement: API Data Ingestion
The FastAPI server SHALL use dependency injection for repository access.

The server SHALL validate all input data before database operations.

#### Scenario: POST measurement with repository injection
- **WHEN** a FHIR observation is POSTed to /bg/reading
- **THEN** the endpoint uses an injected MeasurementRepository

#### Scenario: Invalid measurement rejected
- **WHEN** a malformed FHIR observation is POSTed
- **THEN** the endpoint returns 400 with validation error details

