# Change: Add Time-Synchronized Digital Twin Client

## Why
The current `client.py` streams glucose data sequentially from the beginning of the dataset, which doesn't reflect realistic time-of-day patterns. Since glucose concentration varies significantly by time of day (morning, afternoon, evening, night), a "digital twin" simulation should start streaming from a point in the dataset that matches the current time of day. This provides more realistic simulation for testing and demonstration purposes.

## What Changes
- Add new `client_twin.py` script that:
  - Finds the first glucose reading in the dataset matching the current time of day (hour:minute)
  - Replaces the original timestamp with the current date while preserving the time
  - Streams data from that point forward, simulating a real patient's readings at the current moment
- Add CLI entry point `diabetes-client-twin` for easy invocation

## Impact
- Affected specs: `core` (adding time-synced streaming capability)
- Affected code: New `client_twin.py` script, `pyproject.toml` (new CLI entry point)
- No breaking changes to existing functionality
- Existing `client.py` remains unchanged for sequential streaming use cases
