# Change: Add CLI Entry Points

## Why
The project has three key executable scripts (`client.py`, `server.py`, `load_model_and_predict.py`) that currently require running with `python <script>.py`. Adding CLI entry points via `pyproject.toml` enables cleaner invocation through named commands (e.g., `diabetes-server`, `diabetes-client`, `diabetes-predict`) after package installation.

## What Changes
- Add `[project.scripts]` section to `pyproject.toml` with three CLI entry points
- Each entry point maps to the main function in its respective module
- Refactor scripts to expose callable entry point functions if needed

## Impact
- Affected specs: `core` (adding CLI capability requirement)
- Affected code: `pyproject.toml`, `client.py`, `server.py`, `load_model_and_predict.py`
- No breaking changes to existing functionality
