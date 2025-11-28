## 1. Entry Point Logging Setup
- [x] 1.1 Add `setup_logging()` call to `server.py` startup
- [x] 1.2 Add `setup_logging()` call to `client.py` 
- [x] 1.3 Add `setup_logging()` call to `mobile/main.py`

## 2. Replace print() with Logger in Core Modules
- [x] 2.1 Replace `print()` calls in `src/helpers/experiment.py` with `logger.debug()` or `logger.info()`
- [x] 2.2 Update `debug_print()` in `src/helpers/misc.py` to use logger
- [x] 2.3 Replace `print()` warnings in `src/helpers/diabetes/cega.py` with `logger.warning()`
- [x] 2.4 Replace `print()` in `src/helpers/diabetes/madex.py` with `logger.debug()` (verbose mode)

## 3. Expand DEBUG Flag Usage
- [x] 3.1 Add DEBUG-conditional logging in `client.py` for payload details
- [x] 3.2 Add DEBUG-conditional logging in prediction service for feature details
- [x] 3.3 Document DEBUG behavior in `.env.example`

## 4. Validation
- [x] 4.1 Verify all entry points respect LOG_LEVEL setting
- [x] 4.2 Verify DEBUG=true enables additional output
- [x] 4.3 Run existing tests to ensure no regressions (33 passed)
