# training-data-source Specification

## Purpose

Define requirements for multi-source training data support, enabling ML model training on both Ohio T1DM dataset (5-minute intervals) and simglucose synthetic data (3-minute intervals).

## ADDED Requirements

### Requirement: Data Source CLI Option

The CLI train commands SHALL provide a `--data-source` option to select the training data source.

#### Scenario: Train with Ohio data (default)
- **GIVEN** the Ohio T1DM dataset is available in `data/ohio/`
- **WHEN** user runs `uv run python cli.py train simple -p 559`
- **THEN** training uses OhioBgcProvider with 5-minute intervals
- **AND** model is saved with Ohio-compatible naming

#### Scenario: Train with simglucose data
- **GIVEN** simglucose library is installed
- **WHEN** user runs `uv run python cli.py train simple -p adult#001 --data-source simglucose`
- **THEN** training uses SimglucoseProvider with 3-minute intervals
- **AND** model is saved with simglucose-specific naming

#### Scenario: Invalid data source rejected
- **WHEN** user runs `uv run python cli.py train simple --data-source invalid`
- **THEN** CLI displays error message listing valid options
- **AND** exits with non-zero status

### Requirement: SimglucoseProvider Training Data Generation

The SimglucoseProvider SHALL implement `tsfresh_dataframe()` to generate training-compatible DataFrames.

#### Scenario: Generate training DataFrame
- **GIVEN** SimglucoseProvider is initialized with patient "adult#001"
- **WHEN** `tsfresh_dataframe()` is called
- **THEN** simulation runs for the configured number of days
- **AND** returns DataFrame with columns: date_time, mock_date, time_of_day, part_of_day, time, bg_value, id
- **AND** readings are at 3-minute intervals

#### Scenario: DataFrame schema matches Ohio format
- **GIVEN** OhioBgcProvider and SimglucoseProvider instances
- **WHEN** `tsfresh_dataframe()` is called on both
- **THEN** returned DataFrames have identical column names and types
- **AND** both can be processed by TsfreshFeaturizer

#### Scenario: Truncate parameter limits rows
- **GIVEN** SimglucoseProvider instance
- **WHEN** `tsfresh_dataframe(truncate=1000)` is called
- **THEN** returned DataFrame has at most 1000 rows

### Requirement: Provider Factory

The training pipeline SHALL use a factory function to create the appropriate data provider.

#### Scenario: Factory creates Ohio provider
- **GIVEN** data_source="ohio" and patient="559"
- **WHEN** `get_provider()` is called
- **THEN** returns OhioBgcProvider instance
- **AND** provider is configured for the specified patient

#### Scenario: Factory creates Simglucose provider
- **GIVEN** data_source="simglucose" and patient="adult#001"
- **WHEN** `get_provider()` is called
- **THEN** returns SimglucoseProvider instance
- **AND** provider is configured with appropriate insulin mode and meals

### Requirement: Interval Auto-Configuration

The training pipeline SHALL automatically set the correct sample interval based on data source.

#### Scenario: Ohio data uses 5-minute interval
- **GIVEN** user specifies `--data-source ohio`
- **WHEN** training starts
- **THEN** sample interval is set to 5 minutes
- **AND** any conflicting `--interval` value is ignored with warning

#### Scenario: Simglucose data uses 3-minute interval
- **GIVEN** user specifies `--data-source simglucose`
- **WHEN** training starts
- **THEN** sample interval is set to 3 minutes
- **AND** any conflicting `--interval` value is ignored with warning

### Requirement: Dataset Naming Convention

Simglucose training datasets SHALL include data source prefix, while Ohio datasets retain existing naming for backward compatibility.

#### Scenario: Ohio dataset naming (backward compatible)
- **GIVEN** training with Ohio data for patient 559
- **WHEN** dataset is saved
- **THEN** filename follows existing pattern `{patient}_{scope}_{size}_{window}_{horizon}.pkl`
- **AND** no source prefix is added
- **AND** existing cached datasets continue to work

#### Scenario: Simglucose dataset naming
- **GIVEN** training with simglucose data for patient adult#001
- **WHEN** dataset is saved
- **THEN** filename follows pattern `simglucose_{patient}_{scope}_{size}_{window}_{horizon}.pkl`
- **AND** source prefix distinguishes from Ohio datasets

### Requirement: Model Naming Convention

Trained models SHALL include data source in the filename for compatibility tracking.

#### Scenario: Simglucose model naming
- **GIVEN** training completes with simglucose data
- **WHEN** model is saved
- **THEN** filename follows pattern `simglucose_{patient}_{window}_{horizon}_best_{Model}_{uuid}.pkl`

#### Scenario: Model loading with data source
- **GIVEN** model file `simglucose_adult#001_12_6_best_ExtraTreesRegressor_abc123.pkl` exists
- **WHEN** prediction watcher starts for patient adult#001 with simglucose source
- **THEN** correct model is loaded based on data source prefix
