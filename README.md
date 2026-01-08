# Diabetes Blood Glucose Prediction

Machine learning system for predicting blood glucose levels 30 minutes ahead using continuous glucose monitoring (CGM) data from Type 1 diabetic patients.

## Purpose

Helps prevent dangerous hypoglycemic (<70 mg/dL) and hyperglycemic (>180 mg/dL) episodes by providing advance warnings based on historical glucose patterns.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Setup
uv venv
uv sync
cp .env.example .env    # Configure environment

# Verify setup
uv run python cli.py info

# Check dataset availability
uv run python cli.py data check

# Train model
uv run python cli.py train simple --patient 559
```

## CLI Commands

The project provides a unified command-line interface:

```bash
uv run python cli.py --help          # Show all commands
uv run python cli.py info            # Show current configuration
uv run python cli.py data check      # Verify Ohio dataset
uv run python cli.py train simple    # Train model (simple)
uv run python cli.py train full      # Train model (full experiment)
uv run python cli.py serve start              # Start API server
uv run python cli.py serve client             # Stream CGM data (historical timestamps)
uv run python cli.py serve synced-client      # Stream CGM data (real-time timestamps)
uv run python cli.py serve simglucose-client  # Stream synthetic CGM data (simglucose)
uv run python cli.py serve predict            # Run predictions
```

### Train Commands with Options

```bash
# Simple training with custom parameters
uv run python cli.py train simple --patient 559 --window 12 --horizon 6
uv run python cli.py train simple -p 559 -w 12 -h 6  # short form

# Full experiment with all options
uv run python cli.py train full -p 559 -w 12 -h 6 --no-neptune --speed 2
```

### Serve Commands with Options

The `serve client`, `serve synced-client`, and `serve predict` commands support patient-specific options:

```bash
# Stream data for a specific patient (historical timestamps from dataset)
uv run python cli.py serve client --patient 570
uv run python cli.py serve client -p 570 -v  # verbose mode

# Stream data with real-time timestamps (starts from closest time of day)
uv run python cli.py serve synced-client --patient 570
uv run python cli.py serve synced-client -p 570 -v  # verbose mode

# Run predictions with custom parameters
uv run python cli.py serve predict --patient 570 --window 12 --horizon 6
uv run python cli.py serve predict -p 570 -w 12 -H 6  # short form

# Stream synthetic data from simglucose simulation
uv run python cli.py serve simglucose-client                    # Default patient, sync to current time
uv run python cli.py serve simglucose-client -p adolescent#001  # Different virtual patient
uv run python cli.py serve simglucose-client --seed 42          # Reproducible simulation
uv run python cli.py serve simglucose-client --no-sync -v       # Start from midnight, verbose
uv run python cli.py serve simglucose-client -i none            # No insulin (open-loop)
uv run python cli.py serve simglucose-client -i basal           # Basal insulin only (default)
uv run python cli.py serve simglucose-client -i basal-bolus     # Basal + meal boluses

# Configurable insulin parameters (override patient defaults)
uv run python cli.py serve simglucose-client -i basal-bolus --basal-rate 1.2      # Custom basal (U/hr)
uv run python cli.py serve simglucose-client -i basal-bolus --target-glucose 120  # Custom target (mg/dL)
uv run python cli.py serve simglucose-client -i basal-bolus --carb-ratio 12       # Custom CR (g/U)
uv run python cli.py serve simglucose-client -i basal-bolus --correction-factor 50 # Custom CF (mg/dL/U)
uv run python cli.py serve simglucose-client -i basal-bolus --pre-bolus-minutes 15 # Pre-bolus 15min before meals
```

## Real-Time Prediction System

Run in 3 separate terminals:

```bash
uv run python cli.py serve start                      # FastAPI server on port 8000
uv run python cli.py serve client -p 559              # Stream test data for patient 559
uv run python cli.py serve predict -p 559 -w 12 -H 6  # Monitor and predict
```

Or using direct scripts:

```bash
uv run python scripts/serving/server.py
uv run python scripts/serving/client.py
uv run python scripts/serving/load_model_and_predict.py
```

## Mobile Dashboard

```bash
cd mobile && uv run streamsync run main.py
```

## Project Structure

```
diabetes/
├── cli.py                  # Unified CLI entry point
├── config/
│   └── settings.py         # Pydantic configuration
├── scripts/
│   ├── training/           # Model training scripts
│   ├── serving/            # API server and streaming
│   └── utils/              # Utility scripts
├── src/
│   ├── bgc_providers/      # Data source providers (Ohio, AIDA)
│   ├── featurizers/        # TSFresh feature extraction
│   ├── helpers/            # Experiment, metrics, logging
│   ├── interfaces/         # Abstract base classes
│   └── mongo.py            # MongoDB wrapper
├── data/ohio/              # Ohio T1DM dataset (XML)
├── dataframes/             # Processed datasets (.pkl)
├── models/                 # Trained models (.pkl)
├── mobile/                 # Streamsync dashboard
├── docs/                   # Documentation
└── tests/                  # Pytest test suite
```

## Key Technologies

- **PyCaret** - Automated ML model training
- **TSFresh** - Time-series feature extraction
- **FastAPI** - REST API server
- **MongoDB** - Data storage
- **Streamsync** - Mobile dashboard

## Configuration

All settings are managed via environment variables. Copy `.env.example` to `.env` and customize:

```bash
OHIO_ID=559                          # Patient ID
MONGO_URI=mongodb://localhost:27017  # MongoDB connection
WINDOW_STEPS=6                       # History window (6 × 5min = 30min)
PREDICTION_HORIZON=6                 # Steps ahead (30min)
```

## Documentation

- [Architecture Overview](docs/architecture.md) - System design and components
- [API Reference](docs/api.md) - REST API endpoints
- [Training Guide](docs/training.md) - Model training instructions
- [Development Guide](docs/development.md) - Setup and contributing
- [CLAUDE.md](CLAUDE.md) - AI assistant instructions

## Testing

```bash
uv run pytest tests/                 # Run all tests
uv run pytest tests/ -v              # Verbose output
uv run pytest tests/ --cov=src       # With coverage
```

## Type Checking

```bash
uv run mypy src/
```
