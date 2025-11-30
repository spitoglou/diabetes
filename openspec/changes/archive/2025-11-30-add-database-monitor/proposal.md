# Change: Add Database Monitor Script

## Why
Developers and operators need visibility into database activity for debugging and monitoring purposes. A dedicated script that watches all collections in the default MongoDB database and logs inserts/updates provides real-time observability without modifying existing services.

## What Changes
- Add new `db_monitor.py` script that uses MongoDB change streams to watch all collections
- Add CLI entry point `diabetes-db-monitor` for easy invocation
- Leverage existing `MongoDB` wrapper and `Config` for connection/configuration

## Impact
- Affected specs: `core` (adding database monitoring capability)
- Affected code: New `db_monitor.py` script, `pyproject.toml` (new CLI entry point)
- No breaking changes to existing functionality
