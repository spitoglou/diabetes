## MODIFIED Requirements
### Requirement: Repository Pattern for Data Access
The system SHALL access MongoDB through repository classes that encapsulate all database operations.

The system SHALL provide a `BaseRepository` class containing shared repository functionality (collection access, validation, common CRUD operations).

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

#### Scenario: Repository inheritance
- **WHEN** a new repository type is needed
- **THEN** it can inherit from BaseRepository to reuse common functionality
