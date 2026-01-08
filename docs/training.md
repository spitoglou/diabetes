# Model Training Guide

This guide covers training blood glucose prediction models using the Diabetes BGC Prediction System.

## Data Sources

The system supports two data sources for training:

| Data Source | Type | Sensor | Sample Interval | Patients |
|-------------|------|--------|-----------------|----------|
| **Ohio T1DM** | Real patient data | Medtronic Guardian | 5 minutes | 559, 563, 570, 575, 588, 591 |
| **Simglucose** | Synthetic data | Dexcom G6 | 3 minutes | adult#001-010, adolescent#001-010, child#001-010 |

**Prediction time formula:** `horizon_steps × sample_interval = minutes ahead`

Examples:
- Ohio (6 steps × 5 min) = 30 min prediction
- Simglucose (6 steps × 3 min) = 18 min prediction

## Prerequisites

1. **Ohio T1DM Dataset** (for Ohio training): Place XML files in `data/ohio/`
2. **Simglucose** (for synthetic training): Installed automatically with dependencies
3. **MongoDB**: Running instance (optional, for real-time pipeline)
4. **Dependencies**: Install with `uv sync`

## Quick Start

### Check Dataset Availability

```bash
uv run python cli.py data check
```

This verifies the Ohio dataset files are accessible and lists available patient IDs.

### Train a Model

**Simple training (recommended for first run):**
```bash
uv run python cli.py train simple --patient 559
```

**With custom window and horizon:**
```bash
uv run python cli.py train simple --patient 559 --window 12 --horizon 6
uv run python cli.py train simple -p 559 -w 12 -h 6  # short form
```

**Full experiment with all options:**
```bash
uv run python cli.py train full -p 559 -w 12 -h 6 --no-neptune --speed 2
```

Options for `train full`:
- `--patient/-p`: Patient ID (default: 559)
- `--window/-w`: Window size in steps (default: 12)
- `--horizon/-h`: Prediction horizon in steps (default: 6)
- `--neptune/--no-neptune`: Enable/disable Neptune.ai logging (default: enabled)
- `--speed/-s`: Speed setting 1=full, 2=medium, 3=fast (default: 1)

**Direct script execution:**
```bash
uv run python scripts/training/train_best_model_559.py
```

### Train with Simglucose Synthetic Data

Train models using synthetic CGM data from the simglucose library:

```bash
# Basic simglucose training
uv run python cli.py train simple -p adult#001 -d simglucose

# Full experiment without Neptune
uv run python cli.py train full -p adult#001 -d simglucose --no-neptune

# Custom simulation duration (default: 14 days training, 7 days test)
uv run python cli.py train simple -p adult#001 -d simglucose --simulation-days 21
```

Available virtual patients:
- `adult#001` through `adult#010`
- `adolescent#001` through `adolescent#010`
- `child#001` through `child#010`

Simglucose uses basal-bolus insulin control by default for realistic glucose patterns.

## Training Pipeline

### 1. Data Loading

Use the provider factory to load data from either source:

```python
from src.bgc_providers.factory import create_provider

# Ohio data (real patients)
provider = create_provider("ohio", "559")
df = provider.tsfresh_dataframe()

# Simglucose data (synthetic)
provider = create_provider("simglucose", "adult#001")
df = provider.tsfresh_dataframe(simulation_days=14)
```

Or use providers directly:

```python
# Ohio provider
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
provider = OhioBgcProvider(ohio_no="559")

# Simglucose provider
from src.bgc_providers.simglucose_provider import SimglucoseProvider
provider = SimglucoseProvider(patient_name="adult#001", insulin_mode="basal-bolus")
```

Output DataFrame format (same for both providers):
| Column | Type | Description |
|--------|------|-------------|
| `date_time` | datetime | Measurement timestamp |
| `bg_value` | int | Blood glucose (mg/dL) |
| `id` | str | Patient identifier |
| `time` | float | Hours since start |
| `part_of_day` | str | morning/afternoon/evening/night |

### 2. Feature Extraction

The `TsfreshFeaturizer` extracts ~800 time-series features:

```python
from src.featurizers.tsfresh import TsfreshFeaturizer

featurizer = TsfreshFeaturizer(
    window_steps=6,           # 6 readings (30 min history)
    prediction_horizon=6,     # Predict 6 steps ahead (30 min)
    minimal_features=False    # Full feature set
)

labeled_df = featurizer.create_labeled_dataframe(provider)
```

**Feature categories:**
- Statistical: mean, std, min, max, median, quantiles
- Temporal: autocorrelation, partial autocorrelation
- Frequency: FFT coefficients, spectral density
- Complexity: entropy, sample entropy
- Trend: linear regression coefficients

### 3. Model Training with PyCaret

The `Experiment` class orchestrates PyCaret-based training:

```python
from src.helpers.experiment import Experiment

exp = Experiment(patient_id="559")
exp.setup_regressor(labeled_df)
best_model = exp.compare_models()
exp.save_best_model(best_model)
```

**Algorithms compared:**
- ExtraTreesRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- XGBoostRegressor
- LightGBM
- CatBoost
- Ridge, Lasso, ElasticNet
- SVR, KNN
- And more...

### 4. Model Evaluation

**Standard metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

**Diabetes-specific metrics:**

**MADEX (Mean Adjusted Exponent Error):**
```python
from src.helpers.diabetes.madex import madex

score = madex(y_true, y_pred, center=125)
```

Penalizes errors more heavily around critical glucose levels.

**Clarke Error Grid Analysis:**
```python
from src.helpers.diabetes.cega import calculate_cega_zones

zones = calculate_cega_zones(y_true, y_pred)
# Returns percentage in zones A, B, C, D, E
```

## Configuration Options

### Environment Variables

Set in `.env` or export:

```bash
# Patient and model settings
OHIO_ID=559
WINDOW_STEPS=6
PREDICTION_HORIZON=6

# Neptune.ai experiment tracking (optional)
NEPTUNE_PROJECT=your-workspace/diabetes-bgc
NEPTUNE_API_TOKEN=your-api-token
```

### Training Parameters

In `simple_train_559.py`:

```python
# Window size (number of CGM readings as input)
WINDOW_STEPS = 6  # 6 × 5 min = 30 min history

# Prediction horizon (steps ahead to predict)
PREDICTION_HORIZON = 6  # 30 min ahead

# Feature extraction
MINIMAL_FEATURES = False  # Use full ~800 features
```

## Model Naming Convention

Models are saved with this naming pattern:

**Ohio models:**
```
{patient}_{window}_{horizon}_{rank}_{ModelName}_{uuid}.pkl
```
Example: `559_12_6_1_ExtraTreesRegressor_be523b44.pkl`

**Simglucose models:**
```
sim_{patient}_{window}_{horizon}_{rank}_{ModelName}_{uuid}.pkl
```
Example: `sim_adult#001_6_6_1_ExtraTreesRegressor_a1b2c3d4.pkl`

Components:
- `sim_`: Prefix for simglucose models (distinguishes from Ohio)
- `559` or `adult#001`: Patient ID
- `12`: Window size (12 steps)
- `6`: Prediction horizon (6 steps)
- `1`: Model rank (1 = best)
- `ExtraTreesRegressor`: Model algorithm
- `be523b44`: Unique identifier

## Experiment Tracking with Neptune

Enable Neptune.ai tracking:

1. Set environment variables:
   ```bash
   export NEPTUNE_PROJECT="workspace/project"
   export NEPTUNE_API_TOKEN="your-token"
   ```

2. Training will automatically log:
   - Hyperparameters
   - Metrics (MAE, RMSE, R², MADEX)
   - Feature importance
   - Model artifacts

## Advanced Training

### Custom Feature Sets

```python
# Minimal features (faster, ~60 features)
featurizer = TsfreshFeaturizer(minimal_features=True)

# Full features (~800 features)
featurizer = TsfreshFeaturizer(minimal_features=False)
```

### Cross-Validation

PyCaret uses 10-fold cross-validation by default. Modify in experiment setup:

```python
exp.setup_regressor(
    data=labeled_df,
    fold=5,  # 5-fold CV
)
```

### Hyperparameter Tuning

After selecting the best model:

```python
tuned_model = exp.tune_model(best_model, n_iter=100)
```

## Output Files

| Location | Contents |
|----------|----------|
| `models/` | Trained model files (`.pkl`) |
| `dataframes/` | Cached feature DataFrames |
| `logs/` | Training logs (if configured) |

## Troubleshooting

### Memory Issues

Large feature sets may cause memory issues. Solutions:

1. Use minimal features: `MINIMAL_FEATURES = True`
2. Reduce window size: `WINDOW_STEPS = 6`
3. Process in batches

### Slow Training

1. Start with minimal features for prototyping
2. Use GPU-enabled algorithms (XGBoost, LightGBM with GPU)
3. Reduce cross-validation folds

### Missing Data

Ohio dataset may have gaps. The featurizer handles this by:
- Dropping windows with missing values
- Forward-filling small gaps (configurable)

## Example: Complete Training Script

```python
#!/usr/bin/env python
"""Train a blood glucose prediction model."""

from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.helpers.experiment import Experiment
from config.settings import settings

# Configuration
PATIENT_ID = settings.OHIO_ID
WINDOW_STEPS = settings.WINDOW_STEPS
PREDICTION_HORIZON = settings.PREDICTION_HORIZON

# Load data
print(f"Loading data for patient {PATIENT_ID}...")
provider = OhioBgcProvider(ohio_id=PATIENT_ID)

# Extract features
print("Extracting features...")
featurizer = TsfreshFeaturizer(
    window_steps=WINDOW_STEPS,
    prediction_horizon=PREDICTION_HORIZON,
)
labeled_df = featurizer.create_labeled_dataframe(provider)

print(f"Dataset shape: {labeled_df.shape}")

# Train model
print("Training models...")
exp = Experiment(patient_id=PATIENT_ID)
exp.setup_regressor(labeled_df)
best_model = exp.compare_models()

# Evaluate
print("Evaluating best model...")
exp.evaluate_model(best_model)

# Save
model_path = exp.save_best_model(best_model)
print(f"Model saved to: {model_path}")
```

Run with:
```bash
uv run python my_training_script.py
```
