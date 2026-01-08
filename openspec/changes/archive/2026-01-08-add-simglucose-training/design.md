# Design: Simglucose Training Data Support

## Overview

Enable training ML models on simglucose-generated synthetic CGM data, allowing predictions on simglucose streams with matching 3-minute intervals.

## Architecture

### Current State

```
Training Pipeline (Ohio only):
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│  OhioBgcProvider │───>│ tsfresh_dataframe()│───>│   Experiment    │
│  (hardcoded)    │    │                   │    │   (training)    │
└─────────────────┘    └───────────────────┘    └─────────────────┘
```

### Proposed State

```
Training Pipeline (pluggable):
┌─────────────────┐
│  --data-source  │
│  ohio|simglucose│
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│ Provider Factory │───>│ tsfresh_dataframe()│───>│   Experiment    │
│ (OhioBgcProvider │    │                   │    │   (training)    │
│  or Simglucose) │    └───────────────────┘    └─────────────────┘
└─────────────────┘
```

## Key Design Decisions

### 1. SimglucoseProvider.tsfresh_dataframe() Implementation

**Approach**: Run simulation for a configurable period (e.g., 7-30 days) and collect glucose readings into a DataFrame matching Ohio format.

**DataFrame Schema** (must match OhioBgcProvider output):
```python
columns = [
    "date_time",    # datetime - Full timestamp
    "mock_date",    # date - Date component
    "time_of_day",  # time - Time component
    "part_of_day",  # str - "morning", "afternoon", etc.
    "time",         # float - Hours since start
    "bg_value",     # int - Glucose value in mg/dL
    "id",           # str - Patient identifier
]
```

**Parameters**:
- `simulation_days`: Number of days to simulate (default: 14)
- `truncate`: Limit number of rows (0 = no limit)

**Calculation**: 
- 3-minute intervals × 24 hours × days = readings
- 14 days = 14 × 24 × 20 = 6,720 readings

### 2. Provider Factory Pattern

**Location**: New function in `src/helpers/experiment.py`

```python
def get_provider(data_source: str, patient: str, scope: str) -> BgcProviderInterface:
    """Factory function to create the appropriate provider.
    
    Args:
        data_source: "ohio" or "simglucose"
        patient: Patient ID (559 for Ohio, "adult#001" for simglucose)
        scope: "train" or "test"
    
    Returns:
        Provider instance implementing BgcProviderInterface
    """
    if data_source == "ohio":
        return OhioBgcProvider(scope=scope, ohio_no=patient)
    elif data_source == "simglucose":
        return SimglucoseProvider(patient_name=patient, ...)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
```

### 3. Dataset Naming Convention

**Current (Ohio)**:
```
dataframes/{patient}_{scope}_{size}_{window}_{horizon}.pkl
# Example: dataframes/559_train_0_12_6.pkl
```

**Proposed (with data source)**:
```
# Simglucose (new): includes source prefix
dataframes/simglucose_{patient}_{scope}_{size}_{window}_{horizon}.pkl
# Example: dataframes/simglucose_adult#001_train_0_12_6.pkl

# Ohio: KEEP existing format for backward compatibility
dataframes/{patient}_{scope}_{size}_{window}_{horizon}.pkl
# Example: dataframes/559_train_0_12_6.pkl (unchanged)
```

**Backward Compatibility**: Existing Ohio datasets continue to work with old naming. Only simglucose datasets use the new prefixed format.

### 4. CLI Options

**Train commands**:
```bash
# Ohio (default, backward compatible)
uv run python cli.py train simple -p 559 -w 12 -h 6

# Simglucose (new)
uv run python cli.py train simple -p adult#001 -w 12 -h 6 --data-source simglucose -i 3
```

**Option definition**:
```python
data_source: str = typer.Option(
    "ohio",
    "--data-source",
    "-d",
    help="Data source: ohio (5-min intervals) or simglucose (3-min intervals)",
)
```

### 5. Simglucose Training Data Generation

**Decisions**:
- **Simulation duration**: 14 days for training (~6,720 readings at 3-min intervals)
- **Train/Test split**: Time-period based (train: days 1-14, test: days 15-21)
- **Insulin mode**: `basal-bolus` for realistic glucose patterns
- **Patient selection**: Single patient per model (matching Ohio approach)

**Rationale**: Time-period split better mimics real-world scenario where models are trained on historical data and evaluated on future data. Basal-bolus insulin provides realistic glucose control patterns.

### 6. Interval Validation

**Add validation** to ensure interval matches data source:
```python
if data_source == "ohio" and interval != 5:
    logger.warning("Ohio data uses 5-minute intervals, ignoring --interval")
    interval = 5
elif data_source == "simglucose" and interval != 3:
    logger.warning("Simglucose uses 3-minute intervals, ignoring --interval")
    interval = 3
```

## Trade-offs

| Decision | Pros | Cons |
|----------|------|------|
| Factory pattern | Clean abstraction, extensible | Slight complexity increase |
| Time-period split | Realistic train/test separation | Requires longer simulation |
| New dataset naming | Clear source identification | Breaks existing cache |
| Auto-interval from source | Prevents user error | Less flexible |

## Model Compatibility

**Important**: Models are NOT interchangeable between data sources:

| Model trained on | Can predict on | Cannot predict on |
|------------------|----------------|-------------------|
| Ohio (5-min) | Ohio streams | Simglucose streams |
| Simglucose (3-min) | Simglucose streams | Ohio streams |

The model filename should encode the data source for clarity:
```
models/{source}_{patient}_{window}_{horizon}_best_{Model}_{uuid}.pkl
# Example: models/simglucose_adult#001_12_6_best_ExtraTreesRegressor_abc123.pkl
```

## Testing Strategy

1. **Unit tests**: SimglucoseProvider.tsfresh_dataframe() output schema
2. **Integration tests**: Full training pipeline with simglucose data
3. **Validation tests**: Model predictions on simglucose streams
4. **Compatibility tests**: Verify Ohio training still works unchanged
