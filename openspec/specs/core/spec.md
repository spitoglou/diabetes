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

### Requirement: CLI Entry Points
The system SHALL provide named CLI commands for core executable scripts via `pyproject.toml` entry points.

The following commands SHALL be available after package installation:
- `diabetes-server`: Start the FastAPI server for receiving CGM measurements
- `diabetes-client`: Start the glucose data streaming client
- `diabetes-predict`: Start the real-time prediction watcher service

Each CLI command SHALL invoke the appropriate main function from its module without requiring `python <script>.py` invocation.

#### Scenario: Start server via CLI
- **WHEN** a user runs `diabetes-server` from the command line
- **THEN** the FastAPI server starts on port 8000 and accepts CGM measurements

#### Scenario: Start client via CLI
- **WHEN** a user runs `diabetes-client` from the command line
- **THEN** the glucose streaming client starts and sends data to the server

#### Scenario: Start prediction service via CLI
- **WHEN** a user runs `diabetes-predict` from the command line
- **THEN** the prediction watcher service starts monitoring for new measurements

#### Scenario: CLI commands available after install
- **WHEN** the package is installed via `uv sync` or `pip install -e .`
- **THEN** all three CLI commands are available in the environment's PATH

### Requirement: Database Change Monitor
The system SHALL provide a database monitoring script that watches all collections in the configured MongoDB database for changes.

The monitor SHALL log a message when an insert or update operation is detected on any collection.

The monitor SHALL use MongoDB change streams to receive real-time notifications.

The monitor SHALL be available via the `diabetes-db-monitor` CLI command.

#### Scenario: Monitor detects insert operation
- **WHEN** a new document is inserted into any collection in the default database
- **THEN** the monitor logs a message indicating the collection name and operation type

#### Scenario: Monitor detects update operation
- **WHEN** a document is updated in any collection in the default database
- **THEN** the monitor logs a message indicating the collection name and operation type

#### Scenario: Start monitor via CLI
- **WHEN** a user runs `diabetes-db-monitor` from the command line
- **THEN** the monitor starts watching all collections and logs connection status

#### Scenario: Monitor handles connection errors gracefully
- **WHEN** the MongoDB connection is unavailable or interrupted
- **THEN** the monitor logs an error message and exits gracefully

### Requirement: Time-Synchronized Digital Twin Client
The system SHALL provide a time-synchronized streaming client that simulates a digital twin of a patient by broadcasting glucose data relevant to the current time of day.

The client SHALL search the BG provider's dataset to find a starting point where the recorded time of day (hour:minute) is closest to the current runtime time.

The client SHALL replace the original date in the dataset with the current date while preserving the time component, creating timestamps that appear as if the patient is being monitored in real-time.

The client SHALL be available via the `diabetes-client-twin` CLI command.

#### Scenario: Start streaming at current time of day
- **WHEN** a user runs `diabetes-client-twin` at 14:30
- **THEN** the client finds the first glucose reading in the dataset near 14:30 and starts streaming from that point

#### Scenario: Timestamps reflect current date
- **WHEN** the client streams a glucose reading originally recorded on 2020-04-15 at 14:35
- **THEN** the transmitted FHIR observation has a timestamp of today's date at 14:35

#### Scenario: Continuous streaming after start
- **WHEN** the client starts streaming from a time-matched position
- **THEN** it continues streaming subsequent readings in sequence with the standard 5-second interval

#### Scenario: Wrap-around at dataset end
- **WHEN** the client reaches the end of the dataset
- **THEN** it logs a completion message and stops gracefully

