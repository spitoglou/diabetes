"""Tests for prediction service."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import Config
from src.services.prediction_service import PredictionService


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        mongo_uri="mongodb://localhost:27017",
        database_name="test_database",
        models_path="test_models",
        window_steps=12,
        prediction_horizon=6,
        debug=False,
    )


@pytest.fixture
def mock_measurement_repo():
    """Create a mock measurement repository."""
    return MagicMock()


@pytest.fixture
def mock_prediction_repo():
    """Create a mock prediction repository."""
    return MagicMock()


@pytest.fixture
def mock_feature_engineer():
    """Create a mock feature engineer."""
    return MagicMock()


@pytest.fixture
def sample_measurements():
    """Create sample measurement data."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    measurements = []

    for i in range(15):
        measurements.append(
            {
                "effectiveDateTime": (base_time + timedelta(minutes=5 * i)).isoformat(),
                "valueQuantity": {"value": 120 + i * 2},
            }
        )

    return measurements


class TestPredictionServiceInit:
    """Tests for PredictionService initialization."""

    def test_init_with_all_params(
        self,
        test_config,
        mock_measurement_repo,
        mock_prediction_repo,
        mock_feature_engineer,
    ):
        """Test initialization with all parameters."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
            feature_engineer=mock_feature_engineer,
        )

        assert service.measurement_repo == mock_measurement_repo
        assert service.prediction_repo == mock_prediction_repo
        assert service.config == test_config
        assert service.feature_engineer == mock_feature_engineer
        assert service._model is None

    def test_init_creates_feature_engineer(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test that feature engineer is created if not provided."""
        with patch("src.services.prediction_service.FeatureEngineer") as MockEngineer:
            MockEngineer.return_value = MagicMock()

            PredictionService(
                measurement_repo=mock_measurement_repo,
                prediction_repo=mock_prediction_repo,
                config=test_config,
            )

            MockEngineer.assert_called_once_with(test_config)


class TestPredictionServiceLoadModel:
    """Tests for load_model method."""

    def test_load_model_success(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test successful model loading."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
        )

        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["feature_1", "feature_2"]

        with patch("src.services.prediction_service.glob.glob") as mock_glob:
            with patch("src.services.prediction_service.load_model") as mock_load:
                mock_glob.return_value = ["test_models/559_12_6_1_model.pkl"]
                mock_load.return_value = mock_model

                result = service.load_model(patient_id=559, window=12, horizon=6)

                assert result == mock_model
                assert service._model == mock_model
                assert service._model_features == ["feature_1", "feature_2"]

    def test_load_model_not_found(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test model loading when file not found."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
        )

        with patch("src.services.prediction_service.glob.glob") as mock_glob:
            mock_glob.return_value = []

            with pytest.raises(FileNotFoundError, match="No model found"):
                service.load_model(patient_id=999)

    def test_load_model_uses_config_defaults(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test that load_model uses config defaults."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
        )

        mock_model = MagicMock()
        mock_model.feature_names_in_ = []

        with patch("src.services.prediction_service.glob.glob") as mock_glob:
            with patch("src.services.prediction_service.load_model") as mock_load:
                mock_glob.return_value = ["model.pkl"]
                mock_load.return_value = mock_model

                service.load_model(patient_id=559)

                # Should use window_steps=12 and prediction_horizon=6 from config
                pattern = mock_glob.call_args[0][0]
                assert "559_12_6_1" in pattern


class TestPredictionServicePredictFromMeasurements:
    """Tests for predict_from_measurements method."""

    def test_predict_requires_model(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test that prediction requires model to be loaded."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
        )

        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.predict_from_measurements(patient_id="559")

    def test_predict_insufficient_measurements(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test error when insufficient measurements."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
        )
        service._model = MagicMock()
        mock_measurement_repo.get_recent.return_value = [{"data": 1}]  # Only 1

        with pytest.raises(ValueError, match="Insufficient measurements"):
            service.predict_from_measurements(patient_id="559", window=12)

    def test_predict_success(
        self,
        test_config,
        mock_measurement_repo,
        mock_prediction_repo,
        mock_feature_engineer,
        sample_measurements,
    ):
        """Test successful prediction."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
            feature_engineer=mock_feature_engineer,
        )

        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["feature_1"]
        service._model = mock_model
        service._model_features = ["feature_1"]

        mock_measurement_repo.get_recent.return_value = sample_measurements

        mock_features = pd.DataFrame({"feature_1": [1.0]})
        mock_feature_engineer.create_stream_features.return_value = mock_features

        with patch("src.services.prediction_service.predict_model") as mock_predict:
            mock_predict.return_value = pd.DataFrame({"prediction_label": [130.0]})

            result = service.predict_from_measurements(patient_id="559")

            assert "prediction_value" in result
            assert result["prediction_value"] == 130.0
            assert "prediction_origin_time" in result
            assert "prediction_time" in result


class TestPredictionServicePredictAndStore:
    """Tests for predict_and_store method."""

    def test_predict_and_store(
        self,
        test_config,
        mock_measurement_repo,
        mock_prediction_repo,
        mock_feature_engineer,
        sample_measurements,
    ):
        """Test prediction and storage."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
            feature_engineer=mock_feature_engineer,
        )

        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["feature_1"]
        service._model = mock_model
        service._model_features = ["feature_1"]

        mock_measurement_repo.get_recent.return_value = sample_measurements
        mock_prediction_repo.save.return_value = "prediction_id_123"

        mock_features = pd.DataFrame({"feature_1": [1.0]})
        mock_feature_engineer.create_stream_features.return_value = mock_features

        with patch("src.services.prediction_service.predict_model") as mock_predict:
            mock_predict.return_value = pd.DataFrame({"prediction_label": [130.0]})

            result = service.predict_and_store(patient_id="559")

            mock_prediction_repo.save.assert_called_once()
            assert result == "prediction_id_123"


class TestPredictionServiceHelperMethods:
    """Tests for helper methods."""

    def test_create_stream_dataframe(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test stream DataFrame creation."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
        )

        df = pd.DataFrame(
            {
                "bg_value": [100, 110, 120, 130, 140],
                "time": [1, 2, 3, 4, 5],
            }
        )

        result = service._create_stream_dataframe(df, window=3)

        assert len(result) == 3
        assert result.iloc[0]["bg_value"] == 120  # Last 3 rows

    def test_correct_feature_names_adds_missing(
        self, test_config, mock_measurement_repo, mock_prediction_repo
    ):
        """Test that missing features are added."""
        service = PredictionService(
            measurement_repo=mock_measurement_repo,
            prediction_repo=mock_prediction_repo,
            config=test_config,
        )

        df = pd.DataFrame({"feature_1": [1.0]})
        model_features = ["feature_1", "feature_2", "feature_3"]

        result = service._correct_feature_names(df, model_features)

        assert "feature_2" in result.columns
        assert "feature_3" in result.columns
