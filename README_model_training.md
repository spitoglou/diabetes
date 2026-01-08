# Diabetes Prediction Model Training Scripts

This directory contains scripts to train the best PyCaret model for diabetes prediction using subject 559 data with 12 prediction steps and 30-minute horizon (6 steps).

## Scripts Overview

### 1. `check_datasets.py` - Dataset Availability Checker

**Purpose**: Check if required datasets exist and show information about available data.

**Usage**:

```bash
uv run python check_datasets.py
```

**What it does**:

- Checks if the required datasets for subject 559 (12 window, 6 horizon) exist
- Shows dataset shapes and basic statistics
- Lists all available datasets and trained models
- Provides guidance on next steps

### 2. `simple_train_559.py` - Simple PyCaret Training Script ⭐ RECOMMENDED

**Purpose**: Direct PyCaret approach to find the best model with minimal overhead.

**Usage**:

```bash
uv run python simple_train_559.py
```

**What it does**:

- Loads or creates the required featurized datasets
- Sets up PyCaret regression environment
- Compares 15+ machine learning models
- Selects and saves the top 3 best performing models
- Evaluates performance with diabetes-specific metrics
- Provides comprehensive results summary

**Output**:

- Best model saved in `models/` directory
- Performance metrics (RMSE, MAE, R², MADEX, RMADEX, Clarke Error Grid)
- Model comparison table

### 3. `train_best_model_559.py` - Full Experiment Framework

**Purpose**: Uses the complete Experiment class with Neptune tracking capabilities.

**Usage**:

```bash
uv run python train_best_model_559.py
```

**What it does**:

- Uses the full experimental framework from the project
- Includes gap correction and data cleaning
- Provides holdout and unseen data evaluation
- More comprehensive but takes longer to run

## Configuration Details

**Patient ID**: 559
**Window Steps**: 12 (representing 60 minutes of historical glucose data)
**Prediction Horizon**: 6 steps (representing 30 minutes ahead prediction)
**Measurement Frequency**: Every 5 minutes

## Requirements

This project uses [UV](https://docs.astral.sh/uv/) for package management.

```bash
# Initial setup
uv venv                  # Create virtual environment
uv sync                  # Install all dependencies from uv.lock
```

- Python 3.10-3.12 (as specified in pyproject.toml)
- All dependencies managed via `pyproject.toml` and `uv.lock`

## Recommended Workflow

1. **First, check data availability**:

   ```bash
   uv run python check_datasets.py
   ```

2. **Train the model** (recommended simple approach):

   ```bash
   uv run python simple_train_559.py
   ```

3. **Review results**: The script will output performance metrics and save the best model.

## Expected Output

After successful training, you should see:

- Model comparison table showing performance of different algorithms
- Best model saved as `.pkl` file in the `models/` directory
- Performance metrics including:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - R² (R-squared)
  - MADEX (Mean Absolute Deviation Error eXponent)
  - RMADEX (Root Mean Absolute Deviation Error eXponent)
  - Clarke Error Grid Analysis (diabetes-specific evaluation)

## File Locations

- **Datasets**: `dataframes/559_train_0_12_6.pkl` and `dataframes/559_test_0_12_6.pkl`
- **Models**: `models/559_12_6_best_[ModelName]_[ID].pkl`
- **Logs**: Console output with detailed progress information

## Troubleshooting

- If datasets don't exist, the scripts will automatically create them using the existing data pipeline
- If you encounter memory issues, try running with fewer models or restart your Python environment
- For PyCaret compatibility issues, ensure you're using Python 3.9-3.11 as specified in the project documentation

## Model Usage

Once trained, you can load and use the best model with:

```python
from pycaret.regression import load_model
model = load_model('models/559_12_6_best_[ModelName]_[ID]')
```

This model can then be integrated with the existing prediction pipeline in `load_model_and_predict.py`.
