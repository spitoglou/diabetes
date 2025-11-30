# Change: Add configurable server URL for client scripts

## Why
The client scripts (`client.py` and `client_twin.py`) have the server endpoint hardcoded as `http://localhost:8000`. This violates the Centralized Configuration requirement and prevents deployment flexibility (e.g., connecting to remote servers, different ports, or HTTPS endpoints).

## What Changes
- Add `SERVER_URL` environment variable to centralized Config
- Update `client.py` to use config for server endpoint
- Update `client_twin.py` to use config for server endpoint
- Update `.env.example` with the new variable

## Impact
- Affected specs: `core` (Centralized Configuration requirement)
- Affected code: `client.py`, `client_twin.py`, `src/config.py`, `.env.example`
