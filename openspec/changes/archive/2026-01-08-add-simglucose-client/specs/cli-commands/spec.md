# cli-commands Specification

## Purpose
Define CLI commands for the diabetes blood glucose prediction system, including serving commands for streaming CGM data.

## ADDED Requirements

### Requirement: Simglucose Client Command
The CLI SHALL provide a `serve simglucose-client` command that streams live-simulated CGM data using the simglucose library.

#### Scenario: Simglucose client streams synthetic glucose values
- **GIVEN** simglucose library is installed
- **WHEN** user runs `uv run python cli.py serve simglucose-client`
- **THEN** the simulation starts with the default virtual patient
- **AND** CGM values are streamed to the server with current system timestamps

#### Scenario: Simglucose client syncs to current time of day
- **GIVEN** current wall-clock time is 14:30
- **WHEN** user runs `uv run python cli.py serve simglucose-client`
- **THEN** the simulation fast-forwards to 14:30 simulated time
- **AND** any meals scheduled before 14:30 have already affected glucose levels
- **AND** streaming continues from that point in real-time

#### Scenario: Simglucose client supports patient selection
- **GIVEN** simglucose provides 30 virtual patients
- **WHEN** user runs `uv run python cli.py serve simglucose-client --patient adolescent#001`
- **THEN** the simulation uses the adolescent#001 patient model

#### Scenario: Simglucose client supports reproducible runs
- **WHEN** user runs `uv run python cli.py serve simglucose-client --seed 42`
- **THEN** the simulation produces deterministic results
- **AND** running with the same seed produces identical glucose sequences

#### Scenario: Simglucose client supports no-sync mode
- **WHEN** user runs `uv run python cli.py serve simglucose-client --no-sync`
- **THEN** the simulation starts from midnight (00:00)
- **AND** does not fast-forward to current time

#### Scenario: Simglucose client supports verbose mode
- **WHEN** user runs `uv run python cli.py serve simglucose-client --verbose`
- **THEN** detailed logging shows simulation time, CGM value, and meal events
- **AND** server response status is logged

#### Scenario: Default meal schedule affects glucose
- **GIVEN** default meal schedule includes breakfast at 07:00 (45g carbs)
- **AND** simulation has fast-forwarded past 07:00
- **WHEN** CGM values are streamed
- **THEN** glucose levels reflect the post-breakfast rise and fall pattern
