# Tasks for add-configurable-insulin

## Implementation Tasks

### Phase 1: ConfigurableController Class

1. [x] Create `ConfigurableController` class in `src/bgc_providers/simglucose_provider.py`
   - Extends simglucose `Controller` base class
   - Constructor accepts: `patient`, `meal_schedule`, `basal_rate`, `target_glucose`, `carb_ratio`, `correction_factor`, `pre_bolus_minutes`
   - Loads patient defaults from simglucose's quest CSV when parameters are None
   - Implements `policy()` method returning `Action(basal, bolus)`

2. [x] Implement basal rate calculation
   - Default: `u2ss × BW / 6000` (U/min) from patient parameters
   - Override: Convert user-provided U/hr to U/min
   - Validation: 0.0-5.0 U/hr range

3. [x] Implement meal bolus calculation
   - Formula: `meal_carbs / carb_ratio`
   - Default CR from quest CSV lookup
   - Override: Use user-provided CR

4. [x] Implement correction bolus calculation
   - Formula: `(current_glucose - target_glucose) / correction_factor` when glucose > 150
   - Default CF from quest CSV lookup
   - Default target: 140 mg/dL
   - Override: Use user-provided values
   - Note: Correction only applied at meal time (BBController behavior)

5. [x] Implement pre-bolus timing logic
   - Track meal schedule with delivery status
   - Deliver bolus `pre_bolus_minutes` before scheduled meal time
   - Skip meal bolus at actual meal time (already pre-bolused)
   - Still allow correction bolus at meal time
   - Default: 0 minutes (bolus at meal time, matching BBController)

6. [x] Add parameter validation with clear error messages
   - Basal rate: 0.0-5.0 U/hr
   - Target glucose: 70-200 mg/dL
   - Carb ratio: 1-50 g/U
   - Correction factor: 5-200 mg/dL/U
   - Pre-bolus minutes: 0-45 min

### Phase 2: Settings and CLI Integration

7. [x] Add settings to `config/settings.py`
   ```python
   SIMGLUCOSE_BASAL_RATE: float | None = None
   SIMGLUCOSE_TARGET_GLUCOSE: float = 140.0
   SIMGLUCOSE_CARB_RATIO: float | None = None
   SIMGLUCOSE_CORRECTION_FACTOR: float | None = None
   SIMGLUCOSE_PRE_BOLUS_MINUTES: int = 0
   ```

8. [x] Add CLI options to `serve simglucose-client` command in `cli.py`
   - `--basal-rate` / `-b`: Basal rate override (U/hr)
   - `--target-glucose` / `-t`: Target glucose (mg/dL)
   - `--carb-ratio` / `-c`: Carbohydrate ratio (g/U)
   - `--correction-factor` / `-f`: Correction factor (mg/dL/U)
   - `--pre-bolus-minutes` / `-B`: Pre-bolus time (minutes)

9. [x] Update `SimglucoseProvider` to use `ConfigurableController`
   - Replace `BasalOnlyController` usage
   - Replace `BBController` usage
   - Pass new parameters through from CLI/settings

10. [x] Update `stream_simglucose_data()` in `scripts/serving/client.py`
    - Accept new parameters
    - Pass through to SimglucoseProvider

### Phase 3: Validation Tests

11. [x] Create `tests/test_configurable_controller.py`
    - Test parameter validation (valid and invalid ranges)
    - Test default value loading from quest CSV

12. [x] Add BBController equivalence tests
    - Compare basal rates for adult#001
    - Compare meal bolus calculations (multiple carb amounts)
    - Compare correction bolus calculations (at meal time)
    - Note: BBController only applies correction at meal time

13. [x] Add custom parameter tests
    - Verify custom basal rate overrides patient default
    - Verify custom carb ratio changes meal bolus size
    - Verify custom target glucose changes correction bolus

14. [x] Add validation tests
    - Verify out-of-range parameters raise ValueError

### Phase 4: Documentation

15. [x] Update README.md
    - Add examples for new CLI options
    - Document parameter ranges and defaults

16. [x] Update CLAUDE.md
    - Add examples for new CLI options

17. [x] Add inline documentation to ConfigurableController
    - Full docstrings with algorithmic formulas
    - Clinical context in comments
    - Parameter descriptions with units and ranges

## Validation Checklist

- [x] `uv run python cli.py serve simglucose-client --help` shows all new options
- [x] Default parameters produce identical results to BBController
- [x] Custom basal rate overrides patient default
- [x] Custom target glucose changes correction calculations
- [x] Custom carb ratio changes meal bolus size
- [x] Custom correction factor changes correction bolus size
- [x] Pre-bolus delivers insulin before meal time
- [x] Validation errors shown for out-of-range parameters
- [x] Verbose mode shows parameter values and calculation breakdown
- [x] All tests pass: `uv run pytest tests/test_configurable_controller.py`

## Dependencies

- Tasks 1-6 can be done in parallel (controller implementation)
- Tasks 7-10 depend on task 1 (controller must exist)
- Tasks 11-14 depend on tasks 1-10 (need full implementation to test)
- Tasks 15-17 can start after task 8 (once CLI is defined)
