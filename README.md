# Diabetes Blood Glucose Prediction

[![CodeQL Advanced](https://github.com/spitoglou/diabetes/actions/workflows/codeql.yml/badge.svg)](https://github.com/spitoglou/diabetes/actions/workflows/codeql.yml)
[![Pylint](https://github.com/spitoglou/diabetes/actions/workflows/pylint.yml/badge.svg)](https://github.com/spitoglou/diabetes/actions/workflows/pylint.yml)

A machine learning system for predicting continuous glucose monitoring (CGM) readings, developed for a PhD thesis by Stavros Pitoglou.

## Overview

This project uses machine learning to predict glucose levels from historical data, supporting both batch training and real-time streaming prediction.

**Key Features:**
- Predictive models for glucose levels using the Ohio T1DM dataset
- Clinical accuracy evaluation (Clarke Error Grid, MADEX, RMSE)
- Real-time streaming prediction via MongoDB change streams
- Web-based dashboard for monitoring and visualization

## Quick Start

### Prerequisites
- Python 3.10-3.12
- [uv](https://docs.astral.sh/uv/) package manager
- MongoDB (local or Atlas)

### Installation

```bash
# Clone and enter directory
git clone <repository-url>
cd diabetes

# Install dependencies
uv sync

# Copy environment template and configure
cp .env.example .env
# Edit .env with your MongoDB URI and other settings
```

### Running

**CLI Commands:**
```bash
# Start the API server
uv run diabetes-server

# Start the glucose streaming client
uv run diabetes-client

# Start the digital twin client (time-synchronized)
uv run diabetes-client-twin

# Start the prediction service
uv run diabetes-predict

# Monitor database for changes (inserts/updates)
uv run diabetes-db-monitor
```

**Scripts:**
```bash
# Run an experiment
uv run python final_run.py

# Launch the web dashboard
cd mobile && uv run streamsync run .

# Run Jupyter notebooks
uv run jupyter notebook
```

## Project Structure

```
diabetes/
├── src/
│   ├── config.py                # Centralized configuration
│   ├── bgc_providers/           # Data source adapters (Ohio, AIDA)
│   ├── repositories/            # Data access layer
│   ├── pipeline/                # ML pipeline components
│   ├── services/                # Business logic
│   └── helpers/                 # Utilities and metrics
├── mobile/                      # StreamSync web dashboard
├── tests/                       # Unit tests
├── data/                        # Datasets (Ohio T1DM, AIDA)
├── models/                      # Trained model artifacts
├── dataframes/                  # Cached feature dataframes
├── final_run.py                 # Main experiment script
├── server.py                    # FastAPI backend
└── load_model_and_predict.py    # Real-time prediction service
```

## Configuration

Configuration is managed via environment variables (`.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | Required |
| `DATABASE_NAME` | MongoDB database name | `test_database_1` |
| `ENABLE_NEPTUNE` | ML experiment tracking | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEFAULT_PATIENT_ID` | Default patient for predictions | `559` |

See `.env.example` for all available options.

## Testing

```bash
uv run pytest
```

## Tech Stack

- **ML Framework:** PyCaret, scikit-learn, tsfresh
- **Database:** MongoDB
- **API:** FastAPI, uvicorn
- **Dashboard:** StreamSync, Plotly
- **Package Manager:** uv

## License

This project is part of PhD research. Contact the author for licensing information.
