# Project Context

## Purpose
Machine learning system for predicting blood glucose levels 30 minutes ahead using continuous glucose monitoring (CGM) data from Type 1 diabetic patients. The system aims to help prevent dangerous hypoglycemic (low blood sugar <70 mg/dL) and hyperglycemic (high >180 mg/dL) episodes by providing advance warnings.

**Key Goals:**
- Predict glucose concentrations 6 steps (30 minutes) into the future
- Use historical glucose data (12 steps/60 minutes window) for predictions
- Evaluate predictions using clinical accuracy metrics (Clarke Error Grid, MADEX)
- Provide real-time prediction capability via a REST API and mobile dashboard
- Support multi-patient analysis (currently focusing on patient 559 from Ohio T1DM dataset)

## Tech Stack
- **Python 3.10-3.12** - Primary language (specified in pyproject.toml)
- **UV** - Python package manager (fast, modern replacement for pip/pip-tools)
- **PyCaret >=3.0.4** - Automated ML framework for regression model training and comparison
- **TSFresh >=0.20.1** - Time-series feature extraction (~800 features per window)
- **FastAPI** - Real-time prediction REST API server
- **Streamsync >=0.2.8** - Web/mobile dashboard UI framework
- **MongoDB >=4.4.1** - Storage for CGM measurements and predictions
- **Neptune >=1.6.0** - ML experiment tracking and logging
- **Loguru >=0.7.0** - Structured logging
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Plotly** - Visualization (Clarke Error Grid plots, interactive graphs)

### Package Management with UV
This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python package management.

**Setup:**
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create virtual environment and install dependencies
uv venv
uv sync
```

**Common Commands:**
```bash
uv sync                    # Install/update all dependencies from uv.lock
uv add <package>           # Add a new dependency
uv remove <package>        # Remove a dependency
uv lock                    # Update uv.lock without installing
uv run <command>           # Run a command in the virtual environment
uv pip list                # List installed packages
```

**Key Files:**
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions (commit this to git)
- `.python-version` - Specifies Python 3.10

## Project Conventions

### Code Style
- PEP 8 compliant formatting
- Type hints used selectively (not comprehensive)
- Google-style docstrings with detailed parameter descriptions
- Function naming: snake_case for functions
- Comments explaining clinical concepts and parameter meanings
- Loguru for comprehensive logging across all modules

### Naming Conventions
- **Models:** `{patient}_{window}_{horizon}_best_{ModelName}_{uuid}.pkl`
- **Datasets:** `{patient}_{scope}_{size}_{window}_{horizon}.pkl`
- **MongoDB Collections:** `measurements_{patient_id}`, `predictions_{patient_id}`
- **Config files:** Environment-specific with `.template` backups

### Architecture Patterns
**ML Pipeline:**
```
Ohio Dataset (XML) → OhioBgcProvider → Time-Series DataFrame
    → TsfreshFeaturizer → ~800 Feature Extraction
    → Experiment class → PyCaret Model Selection
    → Trained Model (.pkl)
```

**Real-Time Pipeline:**
```
CGM Device Data → client.py (FHIR format) → FastAPI Server (port 8000)
    → MongoDB Storage → load_model_and_predict.py
    → Model Prediction → Mobile/Dashboard Output
```

**Key Design Patterns:**
- Interface-based design: `BgcProviderInterface` for pluggable data sources
- Factory pattern: Multiple BGC providers (Ohio, AIDA) implementing common interface
- Class-based experiment orchestration: Encapsulated ML pipeline in `Experiment` class
- Configuration management: Separate config files per environment

### Testing Strategy
- **Framework:** Pytest
- **Test files:**
  - `tests/cega_test.py` - Clarke Error Grid validation (unit tests for zone classification)
  - `tests/ohio_test.py` - Data provider validation (integration tests for XML parsing)
  - `tests/check_test.py` - Dataset availability checks
- **Approach:** Simple assertion-based testing, direct execution against sample data
- **Run tests:** `uv run pytest tests/`

### Git Workflow
- **Main branch:** `main` (stable/production)
- **Development:** Feature branches (e.g., `catchup`)
- **Commit style:** Conventional commits (e.g., `chore(requirements): relax package version constraints`)

## Domain Context
**Type 1 Diabetes Management:**
- Target users: Diabetic patients, healthcare providers
- Data source: Ohio T1DM research dataset with XML glucose readings
- Measurement interval: 5 minutes (6 steps = 30 minutes)
- Window size: 12 steps (60 minutes historical data)
- Prediction horizon: 6 steps (30 minutes ahead)

**Clinical Thresholds:**
- **Hypoglycemia (Low):** <70 mg/dL (dangerous, needs immediate treatment)
- **Euglycemia (In-Range):** 70-180 mg/dL (safe range)
- **Hyperglycemia (High):** >180 mg/dL (elevated, preventative action)

**Clinical Metrics Implemented:**
1. **Clarke Error Grid Analysis (CEG)** - Divides predictions into 5 zones (A-E) based on clinical accuracy
2. **MADEX (Mean Adjusted Exponent Error)** - Diabetes-specific metric weighting errors by clinical severity
3. **RMADEX** - Root Mean Adjusted Exponent Error for normalized comparison

## Important Constraints
- Requires Ohio T1DM dataset in `data/ohio/` directory (XML format)
- MongoDB connection required for real-time prediction system
- Neptune.ai account required for experiment tracking
- Model training uses PyCaret which excludes expensive models (CatBoost, XGBoost) at speed=2
- FHIR JSON format required for CGM data interchange

## External Dependencies
- **Ohio T1DM Research Dataset** - XML files in `data/ohio/` directory with real patient glucose readings
- **MongoDB Atlas** - Cloud database for measurements and predictions storage
- **Neptune.ai** - ML experiment tracking (API token in `src/helpers/experiment.py`)
- **FHIR Standard** - Fast Healthcare Interoperability Resources format for healthcare data exchange

## Key Workflow Commands

**Initial Setup:**
```bash
uv venv                               # Create virtual environment
uv sync                               # Install all dependencies
```

**Training:**
```bash
uv run python check_datasets.py       # Check dataset availability
uv run python simple_train_559.py     # Train model (RECOMMENDED)
uv run python train_best_model_559.py # Full experiment framework
```

**Real-Time System (3 terminals):**
```bash
uv run python server.py               # FastAPI server on :8000
uv run python client.py               # Stream test data
uv run python load_model_and_predict.py  # Monitor & predict
```

**Mobile Dashboard:**
```bash
cd mobile && uv run streamsync run main.py   # Launch Streamsync UI
```

**Testing:**
```bash
uv run pytest tests/                  # Run all tests
```

**Dependency Management:**
```bash
uv add <package>                      # Add a new dependency
uv remove <package>                   # Remove a dependency
uv sync                               # Sync dependencies after pulling changes
```
