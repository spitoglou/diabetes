# cli-commands Specification

## Purpose
TBD - created by archiving change add-synced-client. Update Purpose after archive.
## Requirements
### Requirement: Synced Client Command
The CLI SHALL provide a `serve synced-client` command that streams CGM data with current system timestamps.

#### Scenario: Synced client starts with current time match
- **GIVEN** the Ohio dataset contains glucose readings throughout the day
- **WHEN** user runs `uv run python cli.py serve synced-client`
- **THEN** streaming starts from the dataset reading closest to the current time of day
- **AND** each reading is sent with the current system timestamp

#### Scenario: Synced client uses current date
- **GIVEN** the synced client is running
- **WHEN** a glucose reading is streamed
- **THEN** the `effectiveDateTime` in the FHIR payload uses today's date
- **AND** the time increments by INTERVAL seconds from the start time

#### Scenario: Synced client supports patient selection
- **GIVEN** multiple patient datasets exist
- **WHEN** user runs `uv run python cli.py serve synced-client --patient 563`
- **THEN** data is streamed from patient 563's dataset
- **AND** readings use current system timestamps

#### Scenario: Synced client supports verbose mode
- **WHEN** user runs `uv run python cli.py serve synced-client --verbose`
- **THEN** detailed logging shows each reading's original and mapped timestamps
- **AND** server response status is logged

#### Scenario: Time matching handles midnight wrap-around
- **GIVEN** current time is 23:55
- **AND** dataset has readings at 23:50 and 00:05
- **WHEN** finding the closest time match
- **THEN** 23:50 is selected as the closest match (5 minutes vs 10 minutes circular distance)

