# Project Structure Specification

## ADDED Requirements

### Requirement: Unified Configuration
The system SHALL use a single Pydantic Settings class for all configuration, loading values from environment variables with `.env` file support.

#### Scenario: Configuration loads from environment
- **WHEN** the application starts
- **THEN** configuration values are loaded from environment variables
- **AND** missing values fall back to defaults defined in Settings class

#### Scenario: Secrets not in version control
- **WHEN** configuration contains secrets (MongoDB URI, API tokens)
- **THEN** secrets are loaded from `.env` file
- **AND** `.env` file is listed in `.gitignore`
- **AND** `.env.example` template is provided in version control

### Requirement: Organized Script Structure
The system SHALL organize executable scripts into categorized subdirectories under `scripts/`.

#### Scenario: Training scripts location
- **WHEN** a user wants to train models
- **THEN** training scripts are found in `scripts/training/`
- **AND** includes `train_simple.py`, `train_full.py`, `generate_datasets.py`

#### Scenario: Serving scripts location
- **WHEN** a user wants to run the real-time prediction system
- **THEN** serving scripts are found in `scripts/serving/`
- **AND** includes `server.py`, `client.py`, `predictor.py`

#### Scenario: Utility scripts location
- **WHEN** a user wants to run utility operations
- **THEN** utility scripts are found in `scripts/utils/`
- **AND** includes `check_datasets.py`, `export_part_of_day.py`

### Requirement: Unified CLI Entry Point
The system SHALL provide a single CLI entry point (`cli.py`) with subcommands for all major operations.

#### Scenario: CLI help displays available commands
- **WHEN** user runs `uv run python cli.py --help`
- **THEN** help text displays available subcommands: `train`, `serve`, `predict`, `datasets`

#### Scenario: Train command supports patient selection
- **WHEN** user runs `uv run python cli.py train --patient 559 --mode simple`
- **THEN** training executes for patient 559 using simple training mode

#### Scenario: Datasets command checks availability
- **WHEN** user runs `uv run python cli.py datasets check`
- **THEN** system displays dataset availability status for configured patient

### Requirement: No Dead Code
The system SHALL NOT contain unused, duplicate, or broken code files in the repository.

#### Scenario: No duplicate scripts
- **WHEN** reviewing root directory
- **THEN** no two scripts have identical functionality

#### Scenario: No broken imports
- **WHEN** any Python file is imported
- **THEN** all its imports resolve successfully

### Requirement: Type-Safe Core Modules
The system SHALL provide type hints for all public functions in `src/` modules.

#### Scenario: Type hints present
- **WHEN** reviewing any public function in `src/`
- **THEN** function has type hints for parameters and return value

#### Scenario: Type checking passes
- **WHEN** running `uv run mypy src/`
- **THEN** no type errors are reported

### Requirement: Health Check Endpoint
The system SHALL provide a `/health` endpoint on the FastAPI server for operational monitoring.

#### Scenario: Health check returns status
- **WHEN** GET request sent to `/health`
- **THEN** response status is 200
- **AND** response body contains `{"status": "healthy"}`

### Requirement: Comprehensive Documentation
The system SHALL provide comprehensive documentation covering installation, usage, architecture, and development workflows.

#### Scenario: README provides quickstart
- **WHEN** a new user clones the repository
- **THEN** `README.md` exists with project overview, installation instructions, and quickstart guide
- **AND** user can run the system following only README instructions

#### Scenario: Architecture documentation exists
- **WHEN** a developer needs to understand system design
- **THEN** `docs/architecture.md` exists with component diagrams and data flow descriptions
- **AND** describes both training pipeline and real-time prediction pipeline

#### Scenario: API documentation exists
- **WHEN** a developer needs to integrate with the REST API
- **THEN** `docs/api.md` exists with endpoint documentation
- **AND** includes request/response examples for each endpoint

#### Scenario: Training guide exists
- **WHEN** a user wants to train models
- **THEN** `docs/training.md` exists with model training instructions
- **AND** explains parameters, dataset requirements, and evaluation metrics

#### Scenario: Deployment guide exists
- **WHEN** an operator needs to deploy to production
- **THEN** `docs/deployment.md` exists with deployment instructions
- **AND** covers environment setup, configuration, and monitoring

#### Scenario: Development guide exists
- **WHEN** a contributor wants to develop new features
- **THEN** `docs/development.md` exists with setup instructions and coding guidelines
- **AND** explains testing, linting, and contribution workflow

#### Scenario: Code has docstrings
- **WHEN** reviewing any public function in `src/`
- **THEN** function has a docstring explaining purpose, parameters, and return value
