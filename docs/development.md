# Development Guide

This guide covers setting up a development environment and contributing to the Diabetes BGC Prediction System.

## Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) package manager
- MongoDB (for real-time pipeline)
- Git

## Initial Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd diabetes
```

### 2. Create Virtual Environment

```bash
uv venv
```

This creates a `.venv` directory with a Python virtual environment.

### 3. Install Dependencies

```bash
uv sync
```

This installs all dependencies from `uv.lock`, ensuring reproducible builds.

### 4. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Required
OHIO_ID=559
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=test_database_1

# Optional
DEBUG=false
LOG_LEVEL=INFO
NEPTUNE_PROJECT=
NEPTUNE_API_TOKEN=
```

### 5. Verify Setup

```bash
# Check CLI works
uv run python cli.py --help

# Run tests
uv run pytest tests/
```

## Project Structure

```
diabetes/
├── cli.py                    # Unified CLI entry point
├── config/
│   └── settings.py           # Pydantic settings configuration
├── scripts/
│   ├── training/             # Model training scripts
│   │   ├── simple_train_559.py
│   │   ├── train_best_model_559.py
│   │   ├── dataset_generator.py
│   │   └── final_run.py
│   ├── serving/              # Real-time pipeline
│   │   ├── server.py         # FastAPI server
│   │   ├── client.py         # CGM data streaming
│   │   └── load_model_and_predict.py
│   └── utils/                # Utility scripts
│       ├── check_datasets.py
│       └── create_part_of_day_files.py
├── src/
│   ├── bgc_providers/        # Data source adapters
│   ├── featurizers/          # Feature extraction
│   ├── helpers/              # Utility modules
│   │   ├── diabetes/         # Domain-specific metrics
│   │   ├── experiment.py     # Training orchestration
│   │   ├── fhir.py          # FHIR data helpers
│   │   ├── logging.py       # Logging configuration
│   │   └── misc.py          # General utilities
│   ├── interfaces/           # Abstract base classes
│   └── mongo.py             # MongoDB wrapper
├── tests/                    # Test suite
├── data/                     # Raw datasets
├── dataframes/               # Cached DataFrames
├── models/                   # Trained models
└── docs/                     # Documentation
```

## Development Workflow

### Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add a development dependency
uv add --dev <package>

# Sync after changes
uv sync
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_server.py

# Run specific test
uv run pytest tests/test_server.py::test_health_endpoint -v
```

### Type Checking

```bash
# Run mypy
uv run mypy src/
```

### Code Formatting

```bash
# Format with black (if installed)
uv run black .

# Sort imports with isort (if installed)
uv run isort .
```

## Writing Tests

### Test Location

Place tests in `tests/` directory, mirroring the source structure:

```
tests/
├── __init__.py
├── test_config.py       # Tests for config/settings.py
├── test_mongo.py        # Tests for src/mongo.py
├── test_server.py       # Tests for scripts/serving/server.py
└── test_helpers.py      # Tests for src/helpers/
```

### Test Patterns

**Unit test with mocking:**

```python
from unittest.mock import MagicMock, patch

def test_mongodb_insert():
    with patch("src.mongo.MongoClient") as mock_client:
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        
        from src.mongo import MongoDB
        mongo = MongoDB()
        # Test operations...
```

**FastAPI endpoint test:**

```python
from fastapi.testclient import TestClient
from scripts.serving.server import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
```

**Pytest fixtures:**

```python
import pytest

@pytest.fixture
def sample_dataframe():
    import pandas as pd
    return pd.DataFrame({
        "id": [1, 1, 1],
        "time": pd.date_range("2024-01-01", periods=3, freq="5min"),
        "bg_value": [100.0, 105.0, 110.0]
    })

def test_feature_extraction(sample_dataframe):
    # Use fixture in test
    assert len(sample_dataframe) == 3
```

## Adding New Features

### 1. Create Interface (if needed)

Add abstract base class in `src/interfaces/`:

```python
from abc import ABC, abstractmethod

class NewProviderInterface(ABC):
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass
```

### 2. Implement Feature

Create implementation in appropriate module:

```python
from src.interfaces.new_provider_interface import NewProviderInterface

class MyProvider(NewProviderInterface):
    def get_data(self) -> pd.DataFrame:
        # Implementation
        pass
```

### 3. Add Type Hints

Use type hints for all public functions:

```python
from typing import Optional
import pandas as pd

def process_data(
    df: pd.DataFrame,
    threshold: float = 0.5,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Process DataFrame with given parameters.
    
    Args:
        df: Input DataFrame
        threshold: Filtering threshold
        max_rows: Maximum rows to return
        
    Returns:
        Processed DataFrame
    """
    pass
```

### 4. Write Tests

Add tests for new functionality:

```python
def test_my_provider_returns_dataframe():
    provider = MyProvider()
    result = provider.get_data()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
```

### 5. Update Documentation

Update relevant docs in `docs/` if the feature affects user-facing functionality.

## Configuration

### Settings Class

Add new settings to `config/settings.py`:

```python
class Settings(BaseSettings):
    # Existing settings...
    
    # New setting with default
    MY_NEW_SETTING: str = "default_value"
    
    # New setting required
    REQUIRED_SETTING: str
```

### Environment Variables

Add to `.env.example`:

```bash
MY_NEW_SETTING=value
REQUIRED_SETTING=value
```

## Debugging

### Enable Debug Mode

```bash
export DEBUG=true
uv run python cli.py serve start
```

### Debug Logging

```python
from loguru import logger

logger.debug("Debug message with {variable}", variable=value)
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use IPython
from IPython import embed; embed()
```

## Common Tasks

### Add New API Endpoint

1. Add Pydantic models in `scripts/serving/server.py`:
   ```python
   class NewRequest(BaseModel):
       field: str
   
   class NewResponse(BaseModel):
       result: str
   ```

2. Add endpoint:
   ```python
   @app.post("/new-endpoint", response_model=NewResponse)
   async def new_endpoint(request: NewRequest) -> NewResponse:
       return NewResponse(result="success")
   ```

3. Add tests in `tests/test_server.py`

4. Update `docs/api.md`

### Add New CLI Command

1. Add command in `cli.py`:
   ```python
   @app.command()
   def new_command(
       option: str = typer.Option("default", help="Option description"),
   ) -> None:
       """Command description."""
       # Implementation
   ```

2. Test: `uv run python cli.py new-command --help`

### Add New Data Provider

1. Implement interface in `src/bgc_providers/`
2. Add to `src/bgc_providers/__init__.py`
3. Write tests
4. Update training scripts if needed

## Troubleshooting

### Import Errors

Ensure you're running from project root:
```bash
cd /path/to/diabetes
uv run python ...
```

### MongoDB Connection Issues

Check MongoDB is running:
```bash
mongosh --eval "db.adminCommand('ping')"
```

### Test Failures

Run with verbose output:
```bash
uv run pytest tests/ -v --tb=long
```

### Type Errors

Run mypy for details:
```bash
uv run mypy src/ --show-error-codes
```
