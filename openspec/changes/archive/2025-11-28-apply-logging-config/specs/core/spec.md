## MODIFIED Requirements

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
