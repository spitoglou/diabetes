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
uv run python cli.py check-datasets

# Train model
uv run python cli.py train --patient 559
```

## CLI Commands

The project provides a unified command-line interface:

```bash
uv run python cli.py --help         # Show all commands
uv run python cli.py info           # Show current configuration
uv run python cli.py check-datasets # Verify Ohio dataset
uv run python cli.py train          # Train model
uv run python cli.py serve          # Start API server
uv run python cli.py stream         # Stream CGM data
uv run python cli.py predict        # Run predictions
```

## Real-Time Prediction System

Run in 3 separate terminals:

```bash
uv run python cli.py serve          # FastAPI server on port 8000
uv run python cli.py stream         # Stream test data
uv run python cli.py predict        # Monitor and predict
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
