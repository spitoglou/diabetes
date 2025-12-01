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

    def test_delete_all(self, mock_db, config):
        """Test deleting all measurements for a patient."""
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.deleted_count = 15
        mock_collection.delete_many.return_value = mock_result
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        deleted = repo.delete_all("559")

        assert deleted == 15
        mock_collection.delete_many.assert_called_once_with({})

    def test_delete_all_invalid_patient(self, mock_db, config):
        """Test delete_all with invalid patient raises error."""
        repo = MeasurementRepository(mock_db, config)
        with pytest.raises(ValueError, match="Invalid patient ID"):
            repo.delete_all("999")


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

    def test_get_by_prediction_time_range(self, mock_db, config):
        """Test getting predictions by time range."""
        from datetime import datetime

        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = [
            {"_id": "1", "prediction_time": datetime(2024, 1, 1, 10, 0)},
            {"_id": "2", "prediction_time": datetime(2024, 1, 1, 11, 0)},
        ]
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection

        repo = PredictionRepository(mock_db, config)
        start = datetime(2024, 1, 1, 9, 0)
        end = datetime(2024, 1, 1, 12, 0)
        results = repo.get_by_prediction_time_range("559", start, end)

        assert len(results) == 2
        mock_collection.find.assert_called_once()
        # Verify the query contains the time range filter
        call_args = mock_collection.find.call_args[0][0]
        assert "prediction_time" in call_args
        assert "$gte" in call_args["prediction_time"]
        assert "$lte" in call_args["prediction_time"]


class TestMeasurementRepositoryDateRange:
    """Tests for MeasurementRepository date range methods."""

    def test_get_by_date_range(self, mock_db, config):
        """Test getting measurements by date range."""
        from datetime import datetime

        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = [
            {"_id": "1", "effectiveDateTime": "2024-01-01T10:00:00"},
            {"_id": "2", "effectiveDateTime": "2024-01-01T11:00:00"},
        ]
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        start = datetime(2024, 1, 1, 9, 0)
        end = datetime(2024, 1, 1, 12, 0)
        results = repo.get_by_date_range("559", start, end)

        assert len(results) == 2
        mock_collection.find.assert_called_once()
        # Verify the query contains the date range filter
        call_args = mock_collection.find.call_args[0][0]
        assert "effectiveDateTime" in call_args
        assert "$gte" in call_args["effectiveDateTime"]
        assert "$lte" in call_args["effectiveDateTime"]

    def test_get_by_date_range_with_limit(self, mock_db, config):
        """Test getting measurements by date range with custom limit."""
        from datetime import datetime

        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = []
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        start = datetime(2024, 1, 1, 9, 0)
        end = datetime(2024, 1, 1, 12, 0)
        repo.get_by_date_range("559", start, end, limit=50)

        mock_cursor.sort.return_value.limit.assert_called_with(50)

    def test_watch(self, mock_db, config):
        """Test watching for measurement changes."""
        mock_collection = MagicMock()
        mock_change_stream = MagicMock()
        mock_collection.watch.return_value = mock_change_stream
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        result = repo.watch("559")

        assert result == mock_change_stream
        mock_collection.watch.assert_called_once()
        # Verify default pipeline
        call_args = mock_collection.watch.call_args[0][0]
        assert call_args == [{"$match": {"operationType": "insert"}}]

    def test_watch_with_custom_pipeline(self, mock_db, config):
        """Test watching with custom pipeline."""
        mock_collection = MagicMock()
        mock_change_stream = MagicMock()
        mock_collection.watch.return_value = mock_change_stream
        mock_db.__getitem__.return_value = mock_collection

        repo = MeasurementRepository(mock_db, config)
        custom_pipeline = [{"$match": {"operationType": "update"}}]
        result = repo.watch("559", pipeline=custom_pipeline)

        assert result == mock_change_stream
        mock_collection.watch.assert_called_once_with(custom_pipeline)
