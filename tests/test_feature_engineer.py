"""Tests for feature engineering pipeline."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import Config
from src.pipeline.feature_engineer import FeatureEngineer


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        mongo_uri="mongodb://localhost:27017",
        database_name="test_database",
        dataframes_path="test_dataframes",
    )


@pytest.fixture
def sample_timeseries_df():
    """Create a sample time series DataFrame."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = []

    for i in range(50):
        records.append(
            {
                "bg_value": 120 + (i % 10) * 3,
                "date_time": base_time + timedelta(minutes=5 * i),
                "time_of_day": (base_time + timedelta(minutes=5 * i)).time(),
                "part_of_day": "morning" if i < 25 else "afternoon",
                "time": i + 1,
                "id": "a",
            }
        )

    return pd.DataFrame(records)


class TestFeatureEngineerInit:
    """Tests for FeatureEngineer initialization."""

    def test_init_with_config(self, test_config):
        """Test initialization with provided config."""
        engineer = FeatureEngineer(test_config)

        assert engineer.config == test_config

    def test_init_without_config(self):
        """Test initialization uses global config."""
        with patch("src.pipeline.feature_engineer.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            engineer = FeatureEngineer()

            assert engineer.config == mock_config


class TestFeatureEngineerGetCachePath:
    """Tests for _get_cache_path method."""

    def test_get_cache_path(self, test_config):
        """Test cache path generation."""
        engineer = FeatureEngineer(test_config)

        cache_path = engineer._get_cache_path(
            patient_id=559,
            scope="train",
            truncate=0,
            window=12,
            horizon=6,
        )

        assert cache_path == "test_dataframes/559_train_0_12_6.pkl"

    def test_get_cache_path_different_params(self, test_config):
        """Test cache path with different parameters."""
        engineer = FeatureEngineer(test_config)

        cache_path = engineer._get_cache_path(
            patient_id=570,
            scope="test",
            truncate=100,
            window=6,
            horizon=1,
        )

        assert cache_path == "test_dataframes/570_test_100_6_1.pkl"


class TestFeatureEngineerCreateFeatures:
    """Tests for create_features method."""

    def test_create_features(self, test_config, sample_timeseries_df):
        """Test feature creation."""
        engineer = FeatureEngineer(test_config)

        with patch("src.pipeline.feature_engineer.TsfreshFeaturizer") as MockFeaturizer:
            mock_featurizer = MagicMock()
            mock_featurizer.labeled_dataframe = pd.DataFrame(
                {"feature_1": [1, 2], "label": [100, 110]}
            )
            MockFeaturizer.return_value = mock_featurizer

            result = engineer.create_features(
                timeseries_df=sample_timeseries_df,
                window=12,
                horizon=6,
            )

            MockFeaturizer.assert_called_once()
            mock_featurizer.create_labeled_dataframe.assert_called_once()
            assert isinstance(result, pd.DataFrame)

    def test_create_features_minimal(self, test_config, sample_timeseries_df):
        """Test feature creation with minimal features."""
        engineer = FeatureEngineer(test_config)

        with patch("src.pipeline.feature_engineer.TsfreshFeaturizer") as MockFeaturizer:
            mock_featurizer = MagicMock()
            mock_featurizer.labeled_dataframe = pd.DataFrame(
                {"feature_1": [1], "label": [100]}
            )
            MockFeaturizer.return_value = mock_featurizer

            engineer.create_features(
                timeseries_df=sample_timeseries_df,
                window=12,
                horizon=6,
                minimal_features=True,
            )

            call_kwargs = MockFeaturizer.call_args[1]
            assert call_kwargs["minimal_features"] is True


class TestFeatureEngineerCreateFeaturesCached:
    """Tests for create_features_cached method."""

    def test_loads_from_cache_if_exists(self, test_config, sample_timeseries_df):
        """Test that cached features are loaded if available."""
        engineer = FeatureEngineer(test_config)
        cached_df = pd.DataFrame({"feature_1": [1, 2], "label": [100, 110]})

        with patch("src.pipeline.feature_engineer.path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("src.pipeline.feature_engineer.read_df") as mock_read:
                mock_read.return_value = cached_df

                result = engineer.create_features_cached(
                    timeseries_df=sample_timeseries_df,
                    patient_id=559,
                    scope="train",
                    truncate=0,
                    window=12,
                    horizon=6,
                )

                mock_read.assert_called_once()
                assert result.equals(cached_df)

    def test_creates_features_if_not_cached(self, test_config, sample_timeseries_df):
        """Test that features are created if not cached."""
        engineer = FeatureEngineer(test_config)

        with patch("src.pipeline.feature_engineer.path.exists") as mock_exists:
            mock_exists.return_value = False
            with patch.object(engineer, "create_features") as mock_create:
                mock_create.return_value = pd.DataFrame(
                    {"feature_1": [1], "label": [100]}
                )
                with patch("src.pipeline.feature_engineer.save_df"):
                    result = engineer.create_features_cached(
                        timeseries_df=sample_timeseries_df,
                        patient_id=559,
                        scope="train",
                        truncate=0,
                        window=12,
                        horizon=6,
                    )

                    mock_create.assert_called_once()

    def test_force_recreate_ignores_cache(self, test_config, sample_timeseries_df):
        """Test that force_recreate bypasses cache."""
        engineer = FeatureEngineer(test_config)

        with patch("src.pipeline.feature_engineer.path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch.object(engineer, "create_features") as mock_create:
                mock_create.return_value = pd.DataFrame(
                    {"feature_1": [1], "label": [100]}
                )
                with patch("src.pipeline.feature_engineer.save_df"):
                    engineer.create_features_cached(
                        timeseries_df=sample_timeseries_df,
                        patient_id=559,
                        scope="train",
                        truncate=0,
                        window=12,
                        horizon=6,
                        force_recreate=True,
                    )

                    mock_create.assert_called_once()

    def test_saves_to_cache_after_creation(self, test_config, sample_timeseries_df):
        """Test that new features are saved to cache."""
        engineer = FeatureEngineer(test_config)
        new_df = pd.DataFrame({"feature_1": [1], "label": [100]})

        with patch("src.pipeline.feature_engineer.path.exists") as mock_exists:
            mock_exists.return_value = False
            with patch.object(engineer, "create_features") as mock_create:
                mock_create.return_value = new_df
                with patch("src.pipeline.feature_engineer.save_df") as mock_save:
                    engineer.create_features_cached(
                        timeseries_df=sample_timeseries_df,
                        patient_id=559,
                        scope="train",
                        truncate=0,
                        window=12,
                        horizon=6,
                    )

                    mock_save.assert_called_once()


class TestFeatureEngineerCreateStreamFeatures:
    """Tests for create_stream_features method."""

    def test_create_stream_features(self, test_config, sample_timeseries_df):
        """Test stream feature creation."""
        engineer = FeatureEngineer(test_config)

        with patch("src.pipeline.feature_engineer.TsfreshFeaturizer") as MockFeaturizer:
            mock_featurizer = MagicMock()
            mock_featurizer.feature_dataframe = pd.DataFrame({"feature_1": [1]})
            MockFeaturizer.return_value = mock_featurizer

            result = engineer.create_stream_features(
                stream_df=sample_timeseries_df,
                window=12,
                horizon=6,
            )

            mock_featurizer.create_feature_dataframe.assert_called_once()
            assert mock_featurizer.chunks == 1
            assert isinstance(result, pd.DataFrame)
