## ADDED Requirements

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
