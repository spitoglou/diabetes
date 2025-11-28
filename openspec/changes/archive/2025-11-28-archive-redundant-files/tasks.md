## 1. Preparation

- [x] 1.1 Verify no active code imports the files to be archived
- [x] 1.2 Create `archive/deprecated-scripts/` directory
- [x] 1.3 Create `archive/deprecated-config/` directory

## 2. Archive Root Scripts

- [x] 2.1 Move `main.py` to `archive/deprecated-scripts/`
- [x] 2.2 Move `create.py` to `archive/deprecated-scripts/`
- [x] 2.3 Move `dataset_generator.py` to `archive/deprecated-scripts/`
- [x] 2.4 Move `sandbox.py` to `archive/deprecated-scripts/`
- [x] 2.5 Move `create_part_of_day_files.py` to `archive/deprecated-scripts/`
- [x] 2.6 Move `legacy_live_plot.py` to `archive/deprecated-scripts/`

## 3. Archive Config Folder

- [x] 3.1 Move `config/` folder to `archive/deprecated-config/`
- [x] 3.2 Remove empty `config/` directory if any remnants remain

## 4. Cleanup

- [x] 4.1 Verify project still runs (`uv run python -c "from src.config import get_config; print(get_config())"`)
- [x] 4.2 Verify no broken imports in active code
