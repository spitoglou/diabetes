"""Tests for model trainer pipeline."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import Config
from src.pipeline.model_trainer import ModelTrainer


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        mongo_uri="mongodb://localhost:27017",
        database_name="test_database",
        models_path="test_models",
    )


@pytest.fixture
def sample_train_df():
    """Create sample training DataFrame."""
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "label": [100.0, 110.0, 120.0, 130.0, 140.0],
            "start": [0, 1, 2, 3, 4],
            "end": [10, 11, 12, 13, 14],
        }
    )


class TestModelTrainerInit:
    """Tests for ModelTrainer initialization."""

    def test_init_with_config(self, test_config):
        """Test initialization with provided config."""
        trainer = ModelTrainer(test_config)

        assert trainer.config == test_config
        assert trainer._regressor is None
        assert trainer._best_models == []
        assert trainer._comparison_df is None

    def test_init_without_config(self):
        """Test initialization uses global config."""
        with patch("src.pipeline.model_trainer.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            trainer = ModelTrainer()

            # Note: pycaret's get_config is imported, not src's
            assert trainer._regressor is None


class TestModelTrainerSpeedExclusions:
    """Tests for speed exclusion settings."""

    def test_speed_exclusions_defined(self):
        """Test that speed exclusions are properly defined."""
        assert 1 in ModelTrainer.SPEED_EXCLUSIONS
        assert 2 in ModelTrainer.SPEED_EXCLUSIONS
        assert 3 in ModelTrainer.SPEED_EXCLUSIONS

    def test_speed_1_includes_all(self):
        """Test speed 1 excludes nothing."""
        assert not ModelTrainer.SPEED_EXCLUSIONS[1]

    def test_speed_3_excludes_most(self):
        """Test speed 3 excludes the most models."""
        assert len(ModelTrainer.SPEED_EXCLUSIONS[3]) > len(
            ModelTrainer.SPEED_EXCLUSIONS[2]
        )


class TestModelTrainerSetupRegressor:
    """Tests for setup_regressor method."""

    def test_setup_regressor(self, test_config, sample_train_df):
        """Test regressor setup."""
        trainer = ModelTrainer(test_config)

        with patch("src.pipeline.model_trainer.setup") as mock_setup:
            with patch("src.pipeline.model_trainer.add_metric"):
                mock_setup.return_value = MagicMock()

                trainer.setup_regressor(sample_train_df)

                mock_setup.assert_called_once()
                assert trainer._regressor is not None

    def test_setup_regressor_adds_custom_metrics(self, test_config, sample_train_df):
        """Test that custom metrics are added."""
        trainer = ModelTrainer(test_config)

        with patch("src.pipeline.model_trainer.setup") as mock_setup:
            with patch("src.pipeline.model_trainer.add_metric") as mock_add_metric:
                mock_setup.return_value = MagicMock()

                trainer.setup_regressor(sample_train_df)

                # Should add madex and rmadex metrics
                assert mock_add_metric.call_count == 2

    def test_setup_regressor_ignores_metadata_features(
        self, test_config, sample_train_df
    ):
        """Test that metadata features are ignored."""
        trainer = ModelTrainer(test_config)

        with patch("src.pipeline.model_trainer.setup") as mock_setup:
            with patch("src.pipeline.model_trainer.add_metric"):
                mock_setup.return_value = MagicMock()

                trainer.setup_regressor(sample_train_df)

                call_kwargs = mock_setup.call_args[1]
                assert "start" in call_kwargs["ignore_features"]
                assert "end" in call_kwargs["ignore_features"]


class TestModelTrainerCompareAndSelect:
    """Tests for compare_and_select method."""

    def test_compare_and_select_requires_setup(self, test_config):
        """Test that compare_and_select requires setup first."""
        trainer = ModelTrainer(test_config)

        with pytest.raises(RuntimeError, match="Must call setup_regressor"):
            trainer.compare_and_select()

    def test_compare_and_select(self, test_config):
        """Test model comparison and selection."""
        trainer = ModelTrainer(test_config)
        trainer._regressor = MagicMock()

        mock_models = [MagicMock(), MagicMock(), MagicMock()]

        with patch("src.pipeline.model_trainer.compare_models") as mock_compare:
            with patch("src.pipeline.model_trainer.pull") as mock_pull:
                mock_compare.return_value = mock_models
                mock_pull.return_value = pd.DataFrame({"Model": ["a", "b", "c"]})

                result = trainer.compare_and_select(n_select=3)

                assert len(result) == 3
                assert trainer._best_models == mock_models

    def test_compare_and_select_single_model_wrapped_in_list(self, test_config):
        """Test that single model is wrapped in list."""
        trainer = ModelTrainer(test_config)
        trainer._regressor = MagicMock()

        single_model = MagicMock()

        with patch("src.pipeline.model_trainer.compare_models") as mock_compare:
            with patch("src.pipeline.model_trainer.pull") as mock_pull:
                mock_compare.return_value = single_model  # Not a list
                mock_pull.return_value = pd.DataFrame()

                result = trainer.compare_and_select(n_select=1)

                assert isinstance(result, list)
                assert len(result) == 1


class TestModelTrainerPredict:
    """Tests for predict method."""

    def test_predict_requires_models(self, test_config):
        """Test that predict requires models to be available."""
        trainer = ModelTrainer(test_config)

        with pytest.raises(RuntimeError, match="No models available"):
            trainer.predict()

    def test_predict_uses_best_model_by_default(self, test_config):
        """Test that predict uses best model by default."""
        trainer = ModelTrainer(test_config)
        mock_model = MagicMock()
        trainer._best_models = [mock_model, MagicMock()]

        with patch("src.pipeline.model_trainer.predict_model") as mock_predict:
            mock_predict.return_value = pd.DataFrame()

            trainer.predict()

            mock_predict.assert_called_once_with(mock_model, data=None)

    def test_predict_with_custom_model(self, test_config):
        """Test predict with a specific model."""
        trainer = ModelTrainer(test_config)
        trainer._best_models = [MagicMock()]
        custom_model = MagicMock()

        with patch("src.pipeline.model_trainer.predict_model") as mock_predict:
            mock_predict.return_value = pd.DataFrame()

            trainer.predict(model=custom_model)

            mock_predict.assert_called_once_with(custom_model, data=None)

    def test_predict_with_data(self, test_config):
        """Test predict with custom data."""
        trainer = ModelTrainer(test_config)
        mock_model = MagicMock()
        trainer._best_models = [mock_model]
        test_data = pd.DataFrame({"feature_1": [1, 2]})

        with patch("src.pipeline.model_trainer.predict_model") as mock_predict:
            mock_predict.return_value = pd.DataFrame()

            trainer.predict(data=test_data)

            mock_predict.assert_called_once_with(mock_model, data=test_data)


class TestModelTrainerSaveModels:
    """Tests for save_models method."""

    def test_save_models(self, test_config):
        """Test saving models."""
        trainer = ModelTrainer(test_config)
        mock_model = MagicMock()
        mock_model.__str__ = MagicMock(return_value="LinearRegression()")
        trainer._best_models = [mock_model]

        with patch("src.pipeline.model_trainer.save_model") as mock_save:
            with patch("src.pipeline.model_trainer.uuid.uuid4") as mock_uuid:
                mock_uuid.return_value = "test-uuid"

                paths = trainer.save_models(patient_id=559, window=12, horizon=6)

                mock_save.assert_called_once()
                assert len(paths) == 1
                assert "559_12_6_1_LinearRegression" in paths[0]

    def test_save_models_multiple(self, test_config):
        """Test saving multiple models."""
        trainer = ModelTrainer(test_config)
        mock_model1 = MagicMock()
        mock_model1.__str__ = MagicMock(return_value="Model1()")
        mock_model2 = MagicMock()
        mock_model2.__str__ = MagicMock(return_value="Model2()")
        trainer._best_models = [mock_model1, mock_model2]

        with patch("src.pipeline.model_trainer.save_model"):
            with patch("src.pipeline.model_trainer.uuid.uuid4") as mock_uuid:
                mock_uuid.return_value = "test-uuid"

                paths = trainer.save_models(patient_id=559, window=12, horizon=6)

                assert len(paths) == 2


class TestModelTrainerProperties:
    """Tests for ModelTrainer properties."""

    def test_best_models_property(self, test_config):
        """Test best_models property."""
        trainer = ModelTrainer(test_config)
        trainer._best_models = ["model1", "model2"]

        assert trainer.best_models == ["model1", "model2"]

    def test_comparison_df_property(self, test_config):
        """Test comparison_df property."""
        trainer = ModelTrainer(test_config)
        df = pd.DataFrame({"Model": ["a", "b"]})
        trainer._comparison_df = df

        assert trainer.comparison_df is not None
        assert trainer.comparison_df.equals(df)

    def test_get_model_name(self):
        """Test _get_model_name static method."""
        result = ModelTrainer._get_model_name("LinearRegression(normalize=True)")

        assert result == "LinearRegression"
