# Project Context

## Purpose

Diabetes blood glucose (BG) prediction system developed for a PhD thesis by Stavros Pitoglou. The project uses machine learning to predict continuous glucose monitoring (CGM) readings from the Ohio T1DM dataset.

**Key Goals:**
- Build predictive models for glucose levels using historical data
- Evaluate prediction accuracy using clinical metrics (Clarke Error Grid, MADEX, RMSE)
- Support both batch training and real-time streaming prediction
- Provide a web-based interface for monitoring and visualization

## Tech Stack

**Core ML & Data Processing:**
- PyCaret (>=3.0.4) - Automated machine learning framework
- tsfresh (>=0.20.1) - Time series feature extraction
- scikit-learn - ML algorithms (via PyCaret)
- pandas, numpy - Data manipulation
- matplotlib - Visualization

**Database & Backend:**
- MongoDB (via pymongo >=4.4.1) - Data persistence
- FastAPI - REST API framework
- uvicorn - ASGI web server

**Monitoring & Logging:**
- loguru (>=0.7.0) - Logging
- Neptune (>=1.6.0) - ML experiment tracking

**Frontend:**
- StreamSync (>=0.2.8) - Web dashboard framework
- plotly - Interactive visualization

**Utilities:**
- typer (>=0.9.0) - CLI framework
- lxml (>=4.9.3) - XML parsing for FHIR data
- uv - Package management

**Runtime:** Python 3.10 (see `.python-version`)

## Project Conventions

### Code Style
- Standard Python conventions (PEP 8)
- Type hints required in all files (function signatures, class attributes, and variables where type is not obvious)
- loguru for logging instead of standard logging module
- Configuration via module-level constants and config files

### Architecture Patterns

**Layered Architecture:**
```
┌─────────────────────────────────────────────────┐
│                 Entry Points                     │
│   (final_run.py, server.py, mobile/main.py)     │
├─────────────────────────────────────────────────┤
│              Service Layer                       │
│   (ExperimentOrchestrator, PredictionService)   │
├─────────────────────────────────────────────────┤
│              Pipeline Layer                      │
│   (DataPipeline, FeatureEngineer, ModelTrainer) │
├─────────────────────────────────────────────────┤
│             Repository Layer                     │
│   (MeasurementRepository, PredictionRepository) │
├─────────────────────────────────────────────────┤
│              Infrastructure                      │
│   (MongoDB, Config, Logging)                    │
└─────────────────────────────────────────────────┘
```

**Provider Pattern:**
- Abstract `BgcProviderInterface` with implementations:
  - `OhioBgcProvider` - Ohio T1DM XML dataset
  - `AidaBgcProvider` - AIDA .dat files
- Enables swappable data sources via dependency injection

**Repository Pattern:**
- `MeasurementRepository` - CRUD for glucose measurements
- `PredictionRepository` - CRUD for predictions
- Patient ID validation (whitelist)
- Centralized database access

**Pipeline Components:**
- `DataPipeline` - Data loading, gap removal, cleaning
- `FeatureEngineer` - tsfresh feature extraction with caching
- `ModelTrainer` - PyCaret AutoML integration
- `MetricsCalculator` - CEGA, MADEX, RMSE computation

**Services:**
- `ExperimentOrchestrator` - Coordinates ML experiment workflow
- `PredictionService` - Real-time prediction from MongoDB

**Configuration:**
- Environment-based configuration via `.env` files
- Centralized `Config` dataclass in `src/config.py`
- No hardcoded credentials in source code

### Project Structure

```
diabetes/
├── src/
│   ├── config.py                # Centralized configuration
│   ├── logging_config.py        # Logging setup
│   ├── mongo.py                 # MongoDB connection wrapper
│   ├── bgc_providers/           # Data source adapters
│   │   ├── ohio_bgc_provider.py
│   │   └── aida_bgc_provider.py
│   ├── featurizers/
│   │   └── tsfresh.py           # Time series feature extraction
│   ├── repositories/            # Data access layer
│   │   ├── measurement_repository.py
│   │   └── prediction_repository.py
│   ├── pipeline/                # ML pipeline components
│   │   ├── data_pipeline.py
│   │   ├── feature_engineer.py
│   │   ├── model_trainer.py
│   │   └── metrics_calculator.py
│   ├── services/                # Business logic
│   │   ├── experiment_orchestrator.py
│   │   └── prediction_service.py
│   ├── helpers/
│   │   ├── experiment.py        # Legacy (backward compatibility)
│   │   ├── diabetes/
│   │   │   ├── cega.py          # Clarke Error Grid Analysis
│   │   │   └── madex.py         # Mean Adjusted Exponent Error
│   │   ├── fhir.py              # FHIR JSON conversion
│   │   ├── misc.py              # Helper functions
│   │   └── dataframe.py         # DataFrame utilities
│   └── interfaces/
│       └── bgc_provider_interface.py
├── config/                      # Configuration templates
├── data/                        # Datasets
├── models/                      # Trained model artifacts
├── dataframes/                  # Cached feature dataframes
├── mobile/                      # StreamSync web dashboard
├── tests/                       # Unit tests
├── .env.example                 # Environment variable template
├── server.py                    # FastAPI backend
├── final_run.py                 # Main experiment script
└── load_model_and_predict.py    # Real-time prediction service
```

### Testing Strategy
- Test framework: pytest (default configuration)
- Test location: `tests/`
- Current tests: Provider functionality, CEGA metric validation
- Gap: Limited integration and end-to-end tests

### Git Workflow
- Main branch: `main`
- Development branches: `catchup`, `thesis`, `intermediate`
- Commit convention: Conventional commits (partially followed)
  - Example: `chore(requirements): relax package version constraints`
- Tagged releases: `PhD_c`

## Domain Context

**Glucose Measurement:**
- Units: mg/dL (milligrams per deciliter)
- Normal range: 70-180 mg/dL
- Hypoglycemia: <70 mg/dL
- Hyperglycemia: >180 mg/dL
- CGM provides continuous readings every ~5 minutes

**Datasets:**
- **Ohio T1DM:** Type 1 diabetes patient data (XML format)
  - Patients: 559, 563, 570, 575, 588, 591
- **AIDA:** Alternative diabetes dataset (.dat format)

**Clinical Evaluation Metrics:**
- **Clarke Error Grid (CEGA):** Zones A-E classifying prediction accuracy
  - Zone A: Clinically accurate (within 20% or <70 mg/dL)
  - Zone B: Clinically acceptable
  - Zones C-E: Clinical errors
- **MADEX:** Mean Adjusted Exponent Error - weights hypoglycemia severity
- **RMSE:** Root Mean Squared Error

**Time Patterns:**
- `part_of_day`: morning (7-11), afternoon (12-16), evening (17-20), night (21-23), late_night (0-6)

**Prediction Task:**
- Input: 12-hour window of glucose readings (~72 data points)
- Output: Glucose value 30 min to 1 hour ahead
- Window/horizon configured in 5-minute intervals

## Important Constraints

- **Data Privacy:** Patient data from Ohio T1DM is research data; treat with appropriate care
- **Clinical Accuracy:** Zone A+B percentage is the primary success metric (>95% target)
- **Real-time Latency:** Streaming predictions must complete within measurement interval (~5 min)
- **Model Portability:** Models stored as pickle files; ensure scikit-learn version compatibility

## External Dependencies

**MongoDB Atlas:**
- Remote cloud database for measurement persistence
- Connection configured via `config/mongo_config.py`
- Supports change streams for event-driven prediction

**Neptune.ai:**
- Experiment tracking platform
- Project: `spitoglou/intermediate`
- Toggle via `neptune=True/False` in experiment runs

**Datasets:**
- Ohio T1DM Dataset (local, in `data/ohio/`)
- AIDA Dataset (local, in `data/aida/`)

## Key Entry Points

| Script | Purpose |
|--------|---------|
| `final_run.py` | Main experiment execution |
| `server.py` | FastAPI backend (port 8000) |
| `load_model_and_predict.py` | Real-time prediction service |
| `mobile/main.py` | StreamSync web dashboard |
| `client.py` | HTTP client for testing |
| `stream_data.ipynb` | Streaming simulation notebook |

## Configuration

**Environment Variables** (see `.env.example`):
```bash
# Required
MONGO_URI=mongodb+srv://...      # MongoDB connection string

# Optional
DATABASE_NAME=test_database_1    # MongoDB database name
ENABLE_NEPTUNE=false             # ML tracking toggle
NEPTUNE_API_TOKEN=...            # Neptune API token (if enabled)
NEPTUNE_PROJECT=user/project     # Neptune project name
DEBUG=false                      # Debug mode
LOG_LEVEL=INFO                   # Logging level
DEFAULT_PATIENT_ID=559           # Default patient for predictions
WINDOW_STEPS=12                  # Feature window (5-min intervals)
PREDICTION_HORIZON=6             # Prediction horizon (5-min intervals)
```

**Configuration Class** (`src/config.py`):
```python
from src.config import get_config
config = get_config()

# Access settings
config.mongo_uri
config.glucose_low  # 70
config.glucose_high  # 180
config.valid_patient_ids  # (559, 563, 570, 575, 588, 591)
config.is_valid_patient_id("559")  # True
```

**Legacy Configuration Templates** (in `config/`):
- `mongo_config.py.template` - MongoDB connection (deprecated)
- `server_config.py.template` - FastAPI settings (deprecated)

**Package Management:**
- Tool: `uv` (modern Python package manager) - use for all package operations
- Lock file: `uv.lock`
- Metadata: `pyproject.toml`

**Common uv commands:**
```bash
uv add <package>        # Add a dependency
uv remove <package>     # Remove a dependency
uv sync                 # Sync environment with lock file
uv run <script>         # Run a script in the virtual environment
```
