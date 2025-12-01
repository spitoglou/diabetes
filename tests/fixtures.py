"""
Test fixtures and mock data for testing.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        mongo_uri="mongodb://localhost:27017",
        database_name="test_database",
        neptune_api_token="",
        neptune_project="test/project",
        enable_neptune=False,
        debug=True,
        log_level="DEBUG",
        default_patient_id="559",
        window_steps=12,
        prediction_horizon=6,
    )


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()
    return db


@pytest.fixture
def mock_collection():
    """Create a mock MongoDB collection."""
    collection = MagicMock()
    collection.find.return_value.sort.return_value.limit.return_value = []
    collection.insert_one.return_value.inserted_id = "test_id_123"
    collection.count_documents.return_value = 0
    return collection


@pytest.fixture
def sample_measurements() -> List[Dict[str, Any]]:
    """Generate sample FHIR measurement data."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    measurements = []

    for i in range(20):
        measurement = {
            "_id": f"meas_{i}",
            "status": "final",
            "category": [{"coding": [{"code": "vital-signs"}]}],
            "code": {"text": "Glucose"},
            "subject": {"identifier": "559"},
            "effectiveDateTime": (base_time + timedelta(minutes=5 * i)).isoformat(),
            "valueQuantity": {
                "value": 120 + (i % 10) * 5,  # Values between 120-165
                "unit": "mg/dL",
            },
            "device": {"display": "CGM Device"},
        }
        measurements.append(measurement)

    return measurements


@pytest.fixture
def sample_predictions() -> List[Dict[str, Any]]:
    """Generate sample prediction data."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    predictions = []

    for i in range(10):
        prediction = {
            "_id": f"pred_{i}",
            "prediction_origin_time": base_time + timedelta(minutes=5 * i),
            "prediction_time": base_time + timedelta(minutes=5 * i + 30),
            "prediction_value": 125 + (i % 8) * 5,
        }
        predictions.append(prediction)

    return predictions


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generate sample time series DataFrame for feature extraction."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = []

    for i in range(100):
        records.append(
            {
                "bg_value": 120 + (i % 20) * 3,
                "date_time": base_time + timedelta(minutes=5 * i),
                "time_of_day": (base_time + timedelta(minutes=5 * i)).time(),
                "part_of_day": "morning" if i < 50 else "afternoon",
                "time": i + 1,
                "id": "a",
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Generate sample feature DataFrame."""
    # Simple features for testing
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
            "feature_3": [7.0, 8.0, 9.0],
            "label": [120.0, 130.0, 125.0],
            "start": [0, 1, 2],
            "end": [10, 11, 12],
            "start_time": [0, 5, 10],
            "end_time": [50, 55, 60],
        }
    )


def create_mock_measurement_repo(measurements: List[Dict] | None = None):
    """Create a mock MeasurementRepository."""
    repo = MagicMock()
    repo.get_recent.return_value = measurements or []
    repo.save.return_value = "inserted_id_123"
    repo.count.return_value = len(measurements) if measurements else 0
    return repo


def create_mock_prediction_repo(predictions: List[Dict] | None = None):
    """Create a mock PredictionRepository."""
    repo = MagicMock()
    repo.get_recent.return_value = predictions or []
    repo.save.return_value = "inserted_pred_id_123"
    repo.count.return_value = len(predictions) if predictions else 0
    return repo
