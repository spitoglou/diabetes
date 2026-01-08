# Add Simglucose Training Data Support

## Why

The training pipeline currently only supports Ohio T1DM dataset (5-minute intervals). With the addition of `SimglucoseProvider` for streaming synthetic CGM data at 3-minute intervals (Dexcom sensor), there's a mismatch:

1. **No training data for simglucose**: The `Experiment` class only uses `OhioBgcProvider`
2. **Interval mismatch**: Models trained on Ohio (5-min) cannot be used with simglucose data (3-min)
3. **CLI inconsistency**: The `--interval` option exists but training always uses Ohio data

Users need the ability to train models on simglucose-generated data to make predictions on simglucose streams.

## What Changes

1. **Implement `tsfresh_dataframe()` in SimglucoseProvider** - Generate training-compatible DataFrames from simulation
2. **Add `--data-source` CLI option** to train commands - Choose between `ohio` and `simglucose`
3. **Update Experiment class** to accept a provider parameter instead of hardcoding `OhioBgcProvider`
4. **Generate and cache simglucose training datasets** - Similar to Ohio dataset caching in `dataframes/`

## Scope

**In scope:**
- SimglucoseProvider.tsfresh_dataframe() implementation
- CLI `--data-source ohio|simglucose` option for train commands
- Provider abstraction in Experiment class
- Training dataset generation for simglucose virtual patients
- Documentation updates

**Out of scope:**
- Mixed data source training (combining Ohio and simglucose)
- Real-time model switching based on data source
- Automatic interval detection from data

## Risks

- **Medium**: Simglucose data characteristics may differ from real patient data (Ohio), affecting model generalization
- **Low**: Training on synthetic data is faster (no XML parsing) but produces different feature distributions
- **Low**: Generated datasets may be large for long simulation periods

## Dependencies

- Existing `SimglucoseProvider` class
- Existing `BgcProviderInterface` with `tsfresh_dataframe()` method
- Existing `Experiment` class and training pipeline
