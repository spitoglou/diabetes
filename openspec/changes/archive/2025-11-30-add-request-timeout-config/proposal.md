# Change: Add configurable HTTP request timeout for client scripts

## Why
The client scripts (`client.py` and `client_twin.py`) make HTTP requests without a timeout parameter. This can cause the program to hang indefinitely if the server is unresponsive. Pylint flags this as W3101 (missing-timeout).

## What Changes
- Add `REQUEST_TIMEOUT` environment variable to centralized Config (default: 30 seconds)
- Update `client.py` to use `config.request_timeout` in requests
- Update `client_twin.py` to use `config.request_timeout` in requests
- Update `.env.example` with the new variable

## Impact
- Affected specs: `core` (Centralized Configuration requirement)
- Affected code: `client.py`, `client_twin.py`, `src/config.py`, `.env.example`
