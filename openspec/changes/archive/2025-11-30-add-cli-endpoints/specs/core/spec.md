## ADDED Requirements

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
