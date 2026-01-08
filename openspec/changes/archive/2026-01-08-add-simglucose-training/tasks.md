# Tasks for add-simglucose-training

## Implementation Tasks

### Phase 1: SimglucoseProvider Enhancement

1. [x] Implement `SimglucoseProvider.tsfresh_dataframe()` method
   - Run simulation for configurable number of days
   - Collect CGM readings into DataFrame matching Ohio format
   - Support `truncate` parameter for limiting rows
   - Add `simulation_days` parameter (default: 14)

2. [x] Add `SimglucoseProvider.get_glycose_levels()` method
   - Generate list of glucose events from simulation
   - Match OhioBgcProvider return format for compatibility

3. [x] Add simulation parameters for training data generation
   - `train_seed`: Seed for reproducible training data
   - `test_seed`: Different seed for test data
   - `simulation_days`: Duration of simulation

### Phase 2: Provider Factory

4. [x] Create `create_provider()` factory function in `src/bgc_providers/factory.py`
   - Accept `data_source` parameter ("ohio" or "simglucose")
   - Return appropriate provider instance
   - Handle patient ID format differences

5. [x] Update `timeseries_dataframe()` function to use factory
   - Add `data_source` parameter
   - Replace hardcoded OhioBgcProvider with factory call

6. [x] Update `create_ds_name()` for new naming convention
   - Include data source prefix in filename (sim_ for simglucose)
   - Maintain backward compatibility for existing Ohio datasets

### Phase 3: Experiment Class Updates

7. [x] Add `data_source` parameter to `Experiment.__init__()`
   - Default to "ohio" for backward compatibility
   - Store as instance attribute for use in training

8. [x] Update `Experiment.train_parameters` and `unseen_data_parameters`
   - Include `data_source` in parameter dictionaries
   - Pass through to provider factory

9. [x] Auto-set interval based on data source
   - Ohio: 5 min (Guardian sensor)
   - Simglucose: 3 min (Dexcom sensor)
   - Uses `get_sample_interval()` from factory

### Phase 4: CLI Updates

10. [x] Add `--data-source / -d` option to `train simple` command
    - Choices: "ohio", "simglucose"
    - Default: "ohio"
    - Pass to Experiment constructor

11. [x] Add `--data-source / -d` option to `train full` command
    - Same implementation as train simple

12. [x] Update CLI help text and examples
    - Document data source options
    - Show examples for both Ohio and simglucose training

### Phase 5: Model Naming

13. [x] Update model filename convention to include data source
    - Ohio: `{patient}_{window}_{horizon}_{rank}_{Model}_{uuid}.pkl`
    - Simglucose: `sim_{patient}_{window}_{horizon}_{rank}_{Model}_{uuid}.pkl`
    - Updated `Experiment.log_best_models()` method

14. [x] Update `load_trained_model()` to handle new naming
    - Search for models with data source prefix
    - Auto-detect simglucose from patient ID (contains `#`)
    - Fall back to old naming for backward compatibility

### Phase 6: Documentation

15. [x] Update CLAUDE.md with simglucose training examples
16. [x] Update README.md with data source documentation
17. [x] Update docs/training.md with simglucose training guide
18. [x] Update docs/architecture.md with provider factory diagram

### Phase 7: Testing

19. [x] Add unit tests for `SimglucoseProvider.tsfresh_dataframe()`
    - Verify DataFrame schema matches Ohio format
    - Test truncate parameter
    - Test simulation_days parameter

20. [x] Add unit tests for provider factory
    - Test Ohio provider creation
    - Test Simglucose provider creation
    - Test invalid data source error

21. [x] Add integration test for simglucose training pipeline
    - Test timeseries_dataframe() with simglucose data source
    - Test Experiment train parameters configuration
    - Test load_trained_model() auto-detection of simglucose

22. [x] Add backward compatibility tests
    - Verify existing Ohio training parameters unchanged
    - Verify existing model loading interface unchanged

## Validation Checklist

- [x] `uv run python cli.py train simple -p adult#001 -d simglucose --help` shows options
- [x] Provider factory creates SimglucoseProvider correctly
- [x] Dataset naming includes `sim_` prefix for simglucose
- [x] Model naming includes `sim_` prefix for simglucose
- [x] Ohio training unchanged: default data_source is "ohio"
- [x] All existing tests pass (98 tests)
- [x] New tests pass (26 tests for simglucose training)

## Files Changed

- `src/bgc_providers/simglucose_provider.py` - Added `tsfresh_dataframe()`, `get_glycose_levels()`, `_generate_training_data()`, `meal_in_grams` parameter
- `src/bgc_providers/factory.py` - New file with `create_provider()` and `get_sample_interval()`
- `src/helpers/experiment.py` - Added `data_source` parameter, updated naming conventions
- `cli.py` - Added `--data-source` option to train commands
- `scripts/serving/load_model_and_predict.py` - Updated `load_trained_model()` with data source auto-detection
- `CLAUDE.md` - Updated documentation with simglucose training examples
- `README.md` - Added simglucose training documentation
- `docs/training.md` - Added simglucose training guide
- `docs/architecture.md` - Added provider factory documentation
- `tests/test_simglucose_training.py` - New test file with 26 tests (including 3 integration tests)
