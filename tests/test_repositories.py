"""
Tests for repository classes.
"""

from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.repositories.measurement_repository import MeasurementRepository
from src.repositories.prediction_repository import PredictionRepository


@pytest.fixture
def config():
    """Test configuration."""
    return Config(
        mongo_uri="mongodb://localhost:27017",
        database_name="test_db",
        valid_patient_ids=(559, 563, 570),
    )


@pytest.fixture
def mock_db():
    """Mock MongoDB database."""
    db = MagicMock()
    return db


class TestMeasurementRepository:
    """Tests for MeasurementRepository."""

    def test_init(self, mock_db, config):
        """Test repository initialization."""
        repo = MeasurementRepository(mock_db, config)
        assert repo.db == mock_db
        assert repo.config == config

    def test_get_collection_valid_patient(self, mock_db, config):
        """Test getting collection for valid patient ID."""
        repo = MeasurementRepository(mock_db, config)
        _collection = repo._get_collection("559")
        mock_db.__getitem__.assert_called_with("measurements_559")

    def test_get_collection_invalid_patient(self, mock_db, config):
        """Test getting collection for invalid patient ID raises error."""
        repo = MeasurementRepository(mock_db, config)
        with pytest.raises(ValueError, match="Invalid patient ID"):
            repo._get_collection("999")

    def test_save_measurement(self, mock_db, config):
        """Test saving a measurement."""
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value.inserted_id = "test_id"
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        measurement = {"valueQuantity": {"value": 120}}

        result = repo.save("559", measurement)

        assert result == "test_id"
        mock_collection.insert_one.assert_called_once_with(measurement)

    def test_save_measurement_invalid_patient(self, mock_db, config):
        """Test saving measurement for invalid patient raises error."""
        repo = MeasurementRepository(mock_db, config)
        with pytest.raises(ValueError, match="Invalid patient ID"):
            repo.save("999", {"value": 120})

    def test_get_recent(self, mock_db, config):
        """Test getting recent measurements."""
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = [
            {"_id": "1", "value": 120},
            {"_id": "2", "value": 125},
        ]
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        results = repo.get_recent("559", limit=10)

        assert len(results) == 2
        mock_collection.find.assert_called_once()
        mock_cursor.sort.assert_called_with([("_id", -1)])
        mock_cursor.sort.return_value.limit.assert_called_with(10)

    def test_count(self, mock_db, config):
        """Test counting measurements."""
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 42
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        count = repo.count("559")

        assert count == 42
        mock_collection.count_documents.assert_called_once_with({})


class TestPredictionRepository:
    """Tests for PredictionRepository."""

    def test_init(self, mock_db, config):
        """Test repository initialization."""
        repo = PredictionRepository(mock_db, config)
        assert repo.db == mock_db
        assert repo.config == config

    def test_get_collection_valid_patient(self, mock_db, config):
        """Test getting collection for valid patient ID."""
        repo = PredictionRepository(mock_db, config)
        _collection = repo._get_collection("559")
        mock_db.__getitem__.assert_called_with("predictions_559")

    def test_save_prediction(self, mock_db, config):
        """Test saving a prediction."""
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value.inserted_id = "pred_id"
        mock_db.__getitem__.return_value = mock_collection

        repo = PredictionRepository(mock_db, config)
        prediction = {"prediction_value": 130}

        result = repo.save("559", prediction)

        assert result == "pred_id"
        mock_collection.insert_one.assert_called_once_with(prediction)

    def test_get_recent(self, mock_db, config):
        """Test getting recent predictions."""
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = [
            {"_id": "1", "prediction_value": 120},
        ]
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection

        repo = PredictionRepository(mock_db, config)
        results = repo.get_recent("559", limit=5)

        assert len(results) == 1
        mock_cursor.sort.return_value.limit.assert_called_with(5)
