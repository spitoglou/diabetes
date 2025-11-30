## MODIFIED Requirements
### Requirement: Centralized Configuration
The system SHALL load all configuration from environment variables via a centralized `Config` class.

The system SHALL NOT contain hardcoded credentials, API tokens, connection strings, or service URLs in source code.

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

#### Scenario: Client uses configured server URL
- **WHEN** a client script sends data to the server
- **THEN** it uses the SERVER_URL from configuration (default: http://localhost:8000)
