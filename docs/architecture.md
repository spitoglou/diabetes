# Architecture Overview

This document describes the system architecture of the Diabetes Blood Glucose Prediction System.

## System Overview

The system predicts blood glucose levels 30 minutes ahead (6 steps × 5-minute intervals) using continuous glucose monitoring (CGM) data from the Ohio T1DM dataset.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Diabetes BGC Prediction System                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Ohio T1DM   │───>│  TSFresh     │───>│  PyCaret ML Pipeline     │  │
│  │  Dataset     │    │  Featurizer  │    │  (Model Training)        │  │
│  │  (XML)       │    │  (~800 feat) │    │                          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                    │                     │
│                                                    ▼                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  CGM Device  │───>│  FastAPI     │───>│  MongoDB                 │  │
│  │  (FHIR)      │    │  Server      │    │  (measurements_*)        │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                    │                     │
│                                                    ▼                     │
│                                          ┌──────────────────────────┐  │
│                                          │  Prediction Watcher      │  │
│                                          │  (Real-time inference)   │  │
│                                          └──────────────────────────┘  │
│                                                    │                     │
│                                                    ▼                     │
│                                          ┌──────────────────────────┐  │
│                                          │  MongoDB                 │  │
│                                          │  (predictions_*)         │  │
│                                          └──────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Providers

**OhioBgcProvider** (`src/bgc_providers/ohio_bgc_provider.py`)
- Parses Ohio T1DM XML dataset files
- Provides streaming simulation of CGM readings
- Converts data to tsfresh-compatible DataFrames

### 2. Feature Extraction

**TsfreshFeaturizer** (`src/featurizers/tsfresh.py`)
- Extracts ~800 time-series features using tsfresh library
- Creates sliding window feature datasets
- Supports both minimal and comprehensive feature sets

### 3. Model Training

**Experiment** (`src/helpers/experiment.py`)
- Orchestrates PyCaret-based model comparison
- Supports multiple regression algorithms
- Integrates with Neptune.ai for experiment tracking
- Custom metrics: MADEX (Mean Adjusted Exponent Error)

### 4. Real-Time Pipeline

**Server** (`server.py`)
- FastAPI REST API for receiving CGM readings
- Accepts FHIR-formatted observations
- Stores measurements in MongoDB

**Client** (`client.py`)
- Streams CGM data from dataset to server
- Simulates real-time data flow

**Predictor** (`load_model_and_predict.py`)
- Watches MongoDB for new measurements
- Runs inference using trained model
- Stores predictions in MongoDB

### 5. Configuration

**Settings** (`config/settings.py`)
- Pydantic-based configuration management
- Environment variable support via `.env`
- Type-safe settings access

## Data Flow

### Training Pipeline

```
1. Ohio XML Data → OhioBgcProvider.tsfresh_dataframe()
2. Raw DataFrame → TsfreshFeaturizer.create_labeled_dataframe()
3. Feature DataFrame → Experiment.setup_regressor()
4. PyCaret → Model comparison → Best model saved (.pkl)
```

### Inference Pipeline

```
1. CGM Reading (FHIR JSON) → POST /bg/reading
2. Server → MongoDB (measurements_559)
3. MongoDB Change Stream → Prediction Watcher
4. Load window of readings → TsfreshFeaturizer
5. Feature extraction → Model inference
6. Prediction → MongoDB (predictions_559)
```

## Database Schema

### measurements_{patient_id}
```json
{
  "_id": ObjectId,
  "status": "final",
  "category": [...],
  "code": {"text": "Blood Glucose Concentration"},
  "subject": {"identifier": "559"},
  "effectiveDateTime": "2024-01-08T12:00:00Z",
  "valueQuantity": {"value": 120.0, "unit": "mg/dL"},
  "device": {...}
}
```

### predictions_{patient_id}
```json
{
  "_id": ObjectId,
  "prediction_origin_time": ISODate,
  "prediction_time": ISODate,
  "prediction_value": 125.5
}
```

## Directory Structure

```
diabetes/
├── cli.py                    # Unified CLI entry point
├── server.py                 # FastAPI server
├── client.py                 # CGM data streaming client
├── load_model_and_predict.py # Real-time prediction watcher
├── config/
│   └── settings.py           # Pydantic settings
├── src/
│   ├── bgc_providers/        # Data providers
│   ├── featurizers/          # Feature extraction
│   ├── helpers/              # Utilities
│   │   ├── diabetes/         # CEGA, MADEX metrics
│   │   ├── experiment.py     # Training orchestration
│   │   └── ...
│   ├── interfaces/           # Abstract interfaces
│   └── mongo.py              # MongoDB wrapper
├── data/
│   └── ohio/                 # Ohio T1DM dataset (XML)
├── dataframes/               # Cached feature datasets
├── models/                   # Trained models (.pkl)
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Package Manager | UV |
| ML Framework | PyCaret, scikit-learn |
| Feature Extraction | tsfresh |
| API Framework | FastAPI |
| Database | MongoDB |
| Experiment Tracking | Neptune.ai |
| Configuration | Pydantic Settings |
| CLI | Typer |
| Logging | Loguru |
| Testing | pytest |
| Type Checking | mypy |

## Key Metrics

### Clarke Error Grid Analysis (CEGA)
Clinical accuracy metric with zones A-E:
- **Zone A**: Clinically accurate (±20% or hypoglycemic range)
- **Zone B**: Benign errors
- **Zone C**: Overcorrecting
- **Zone D**: Failure to detect
- **Zone E**: Erroneous treatment

### MADEX (Mean Adjusted Exponent Error)
Custom diabetes-specific metric that penalizes errors more heavily in critical glucose ranges (around 125 mg/dL).

```python
MADEX = (1/n) * Σ |y_pred - y|^exp
where exp = 2 - tanh((y - center) / range) * ((y_pred - y) / slope)
```
