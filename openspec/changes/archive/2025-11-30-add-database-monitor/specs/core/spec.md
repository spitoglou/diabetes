## ADDED Requirements

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
