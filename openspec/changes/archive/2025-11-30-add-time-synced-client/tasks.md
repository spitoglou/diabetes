## 1. Implementation

- [x] 1.1 Create `client_twin.py` script in project root
- [x] 1.2 Add function to find dataset index closest to current time of day
- [x] 1.3 Add function to replace original date with current date in readings
- [x] 1.4 Implement time-synced streaming loop
- [x] 1.5 Add `main()` entry point function

## 2. CLI Integration

- [x] 2.1 Add `diabetes-client-twin` entry point to `pyproject.toml`
- [x] 2.2 Add `client_twin.py` to hatch build targets

## 3. Verification

- [x] 3.1 Run `uv sync` to install new entry point
- [x] 3.2 Verify `diabetes-client-twin` starts and finds time-matched position
- [x] 3.3 Verify timestamps use current date with original time

## 4. Documentation

- [x] 4.1 Update README.md with new CLI command
