# Tasks for add-simglucose-client

## Implementation Tasks

1. [x] Add `simglucose` dependency to `pyproject.toml`
   - Run `uv add simglucose`
   - Verify installation with `uv run python -c "import simglucose"`

2. [x] Add simglucose settings to `config/settings.py`
   - `SIMGLUCOSE_PATIENT`: Default virtual patient (e.g., "adult#001")
   - `SIMGLUCOSE_MEALS`: Default meal schedule as string "7:45,12:70,16:15,18:80,23:10"
   - `SIMGLUCOSE_SEED`: Optional random seed for reproducibility

3. [x] Create `SimglucoseProvider` class in `src/bgc_providers/simglucose_provider.py`
   - Initialize simglucose environment with patient and scenario
   - Method `fast_forward_to_time(target_time)` to advance simulation to current time
   - Generator `simulate_glucose_stream()` yielding CGM values with current timestamps

4. [x] Add `stream_simglucose_data()` function to `scripts/serving/client.py`
   - Uses SimglucoseProvider
   - Fast-forward to current time on startup (configurable)
   - Stream with INTERVAL-second delays

5. [x] Add `serve simglucose-client` CLI command to `cli.py`
   - Options: `--patient`, `--verbose`, `--seed`, `--no-sync`
   - `--no-sync` skips fast-forward (start from midnight)

6. [x] Update documentation
   - Add command to README.md and CLAUDE.md
   - Document available patients and examples

## Insulin Mode Enhancement

12. [x] Add insulin mode support to `SimglucoseProvider`
    - `none`: No insulin delivery (open-loop)
    - `basal`: Constant basal insulin based on patient parameters (default)
    - `basal-bolus`: Basal + meal boluses using BBController

13. [x] Add `--insulin-mode` / `-i` CLI option (default: basal)

14. [x] Add `SIMGLUCOSE_INSULIN_MODE` setting to `config/settings.py`

15. [x] Update documentation with insulin mode examples

## Validation Tasks

16. [x] Verify CLI command registered: `uv run python cli.py serve --help` shows `simglucose-client`
17. [x] Verify no-sync mode streams from midnight with stable glucose (~142 mg/dL)
18. [x] Verify sync mode fast-forwards to current time (391 steps for 19:35)
19. [x] Verify different patients work (adult#001, adolescent#001)
20. [x] Verify seed produces consistent results
21. [x] Verify `--insulin-mode basal` shows insulin delivery in verbose output
22. [x] Verify `--insulin-mode none` shows no insulin in verbose output
