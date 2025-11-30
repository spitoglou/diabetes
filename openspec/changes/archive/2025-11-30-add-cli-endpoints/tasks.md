## 1. Refactor Entry Points

- [x] 1.1 Add `main()` function to `server.py` wrapping the uvicorn.run call
- [x] 1.2 Add `main()` function to `client.py` wrapping `stream_data()` call
- [x] 1.3 Add `main()` wrapper in `load_model_and_predict.py` calling `run_watcher()`

## 2. Configure CLI Entry Points

- [x] 2.1 Add `[project.scripts]` section to `pyproject.toml`
- [x] 2.2 Define `diabetes-server` entry point → `server:main`
- [x] 2.3 Define `diabetes-client` entry point → `client:main`
- [x] 2.4 Define `diabetes-predict` entry point → `load_model_and_predict:main`

## 3. Verification

- [x] 3.1 Run `uv sync` to install entry points
- [x] 3.2 Verify `diabetes-server` invocation works
- [x] 3.3 Verify `diabetes-client` command is installed
- [x] 3.4 Verify `diabetes-predict` command is installed

## 4. Documentation

- [x] 4.1 Update README.md with new CLI commands in the usage section
