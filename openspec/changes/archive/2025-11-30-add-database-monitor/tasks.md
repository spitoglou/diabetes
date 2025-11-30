## 1. Implementation

- [x] 1.1 Create `db_monitor.py` script in project root
- [x] 1.2 Implement database change stream watcher for all collections
- [x] 1.3 Add logging for insert and update operations
- [x] 1.4 Add `main()` entry point function

## 2. CLI Integration

- [x] 2.1 Add `diabetes-db-monitor` entry point to `pyproject.toml`
- [x] 2.2 Add `db_monitor.py` to hatch build targets

## 3. Verification

- [x] 3.1 Run `uv sync` to install new entry point
- [x] 3.2 Verify `diabetes-db-monitor` starts and connects to MongoDB
- [x] 3.3 Test that inserts are detected and logged
- [x] 3.4 Test that updates are detected and logged

## 4. Documentation

- [x] 4.1 Update README.md with new CLI command
