"""Tests for experiment orchestrator service."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import Config
from src.pipeline.metrics_calculator import PredictionMetrics
from src.services.experiment_orchestrator import (
    ExperimentOrchestrator,
    ExperimentResults,
)


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        mongo_uri="mongodb://localhost:27017",
        database_name="test_database",
        models_path="test_models",
        enable_neptune=False,
        neptune_api_token="",
        neptune_project="test/project",
    )


@pytest.fixture
def mock_provider():
    """Create a mock BGC provider."""
    return MagicMock()


@pytest.fixture
def mock_data_pipeline():
    """Create a mock data pipeline."""
    pipeline = MagicMock()
    pipeline.load_timeseries.return_value = pd.DataFrame(
        {
            "bg_value": [100, 110, 120],
            "time": [1, 2, 3],
        }
    )
    pipeline.remove_gaps.return_value = pd.DataFrame()
    pipeline.remove_missing_and_inf.return_value = pd.DataFrame()
    pipeline.fix_column_names.return_value = pd.DataFrame()
    pipeline.align_columns.return_value = (pd.DataFrame(), pd.DataFrame())
    return pipeline


@pytest.fixture
def mock_feature_engineer():
    """Create a mock feature engineer."""
    engineer = MagicMock()
    engineer.create_features_cached.return_value = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "label": [100, 110, 120],
        }
    )
    return engineer


@pytest.fixture
def mock_model_trainer():
    """Create a mock model trainer."""
    trainer = MagicMock()
    trainer.compare_and_select.return_value = [MagicMock()]
    trainer.comparison_df = pd.DataFrame()
    trainer.predict.return_value = pd.DataFrame(
        {
            "label": [100, 110],
            "prediction_label": [105, 115],
        }
    )
    return trainer


@pytest.fixture
def mock_metrics_calculator():
    """Create a mock metrics calculator."""
    calculator = MagicMock()
    calculator.calculate_from_predictions.return_value = PredictionMetrics(
        cega_zones={"A": 90, "B": 8, "C": 1, "D": 1, "E": 0},
        rmse=10.0,
        rmadex=5.0,
    )
    return calculator


class TestExperimentResults:
    """Tests for ExperimentResults dataclass."""

    def test_default_values(self):
        """Test ExperimentResults default values."""
        results = ExperimentResults(
            patient_id=559,
            window=12,
            horizon=6,
        )

        assert results.patient_id == 559
        assert results.window == 12
        assert results.horizon == 6
        assert results.holdout_metrics is None
        assert results.unseen_metrics is None
        assert not results.best_models
        assert results.execution_time == 0.0

    def test_with_all_values(self):
        """Test ExperimentResults with all values."""
        metrics = PredictionMetrics(
            cega_zones={"A": 90, "B": 10, "C": 0, "D": 0, "E": 0},
            rmse=10.0,
            rmadex=5.0,
        )

        results = ExperimentResults(
            patient_id=559,
            window=12,
            horizon=6,
            holdout_metrics=metrics,
            unseen_metrics=metrics,
            best_models=["model1"],
            execution_time=60.0,
        )

        assert results.holdout_metrics == metrics
        assert results.execution_time == 60.0


class TestExperimentOrchestratorInit:
    """Tests for ExperimentOrchestrator initialization."""

    def test_init_with_all_components(
        self,
        test_config,
        mock_provider,
        mock_data_pipeline,
        mock_feature_engineer,
        mock_model_trainer,
        mock_metrics_calculator,
    ):
        """Test initialization with all components."""
        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
            data_pipeline=mock_data_pipeline,
            feature_engineer=mock_feature_engineer,
            model_trainer=mock_model_trainer,
            metrics_calculator=mock_metrics_calculator,
        )

        assert orchestrator.config == test_config
        assert orchestrator.provider == mock_provider
        assert orchestrator.data_pipeline == mock_data_pipeline
        assert orchestrator.feature_engineer == mock_feature_engineer
        assert orchestrator.model_trainer == mock_model_trainer
        assert orchestrator.metrics_calculator == mock_metrics_calculator

    def test_init_creates_default_components(self, test_config, mock_provider):
        """Test that default components are created if not provided."""
        with patch("src.services.experiment_orchestrator.DataPipeline") as MockPipeline:
            with patch(
                "src.services.experiment_orchestrator.FeatureEngineer"
            ) as MockEngineer:
                with patch(
                    "src.services.experiment_orchestrator.ModelTrainer"
                ) as MockTrainer:
                    with patch(
                        "src.services.experiment_orchestrator.MetricsCalculator"
                    ) as MockCalc:
                        MockPipeline.return_value = MagicMock()
                        MockEngineer.return_value = MagicMock()
                        MockTrainer.return_value = MagicMock()
                        MockCalc.return_value = MagicMock()

                        ExperimentOrchestrator(
                            provider=mock_provider,
                            config=test_config,
                        )

                        MockPipeline.assert_called_once()
                        MockEngineer.assert_called_once()
                        MockTrainer.assert_called_once()
                        MockCalc.assert_called_once()


class TestExperimentOrchestratorMakeParams:
    """Tests for _make_params method."""

    def test_make_params(self, test_config, mock_provider):
        """Test parameter dict creation."""
        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
        )

        params = orchestrator._make_params(
            patient_id=559,
            scope="train",
            window=12,
            horizon=6,
            minimal_features=True,
        )

        assert params["ohio_no"] == 559
        assert params["scope"] == "train"
        assert params["window_size"] == 12
        assert params["prediction_horizon"] == 6
        assert params["minimal_features"] is True


class TestExperimentOrchestratorNeptune:
    """Tests for Neptune integration."""

    def test_init_neptune_without_token(self, test_config, mock_provider):
        """Test Neptune init without API token."""
        test_config.neptune_api_token = ""

        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
        )

        orchestrator._init_neptune(
            patient_id=559,
            window=12,
            horizon=6,
            speed=3,
            perform_gap_corrections=True,
        )

        assert orchestrator._neptune_run is None

    def test_init_neptune_with_token(self, test_config, mock_provider):
        """Test Neptune init with API token."""
        test_config.neptune_api_token = "test-token"

        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
        )

        with patch(
            "src.services.experiment_orchestrator.neptune.init_run"
        ) as mock_init:
            mock_run = MagicMock()
            mock_init.return_value = mock_run

            orchestrator._init_neptune(
                patient_id=559,
                window=12,
                horizon=6,
                speed=3,
                perform_gap_corrections=True,
            )

            mock_init.assert_called_once()
            assert orchestrator._neptune_run == mock_run

    def test_log_neptune_metrics_no_run(self, test_config, mock_provider):
        """Test logging metrics when Neptune not initialized."""
        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
        )

        metrics = PredictionMetrics(
            cega_zones={"A": 90, "B": 10, "C": 0, "D": 0, "E": 0},
            rmse=10.0,
            rmadex=5.0,
        )

        # Should not raise
        orchestrator._log_neptune_metrics(metrics, metrics)

    def test_log_neptune_metrics_with_run(self, test_config, mock_provider):
        """Test logging metrics when Neptune is initialized."""
        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
        )

        # Create a mock Neptune run
        mock_run = MagicMock()
        orchestrator._neptune_run = mock_run

        holdout = PredictionMetrics(
            cega_zones={"A": 90, "B": 8, "C": 1, "D": 1, "E": 0},
            rmse=10.0,
            rmadex=5.0,
        )
        unseen = PredictionMetrics(
            cega_zones={"A": 85, "B": 10, "C": 2, "D": 2, "E": 1},
            rmse=12.0,
            rmadex=6.0,
        )

        orchestrator._log_neptune_metrics(holdout, unseen)

        # Verify metrics were logged
        mock_run.__setitem__.assert_any_call("holdout/cega", holdout.cega_zones)
        mock_run.__setitem__.assert_any_call("holdout/RMSE", holdout.rmse)
        mock_run.__setitem__.assert_any_call("holdout/RMADEX", holdout.rmadex)
        mock_run.__setitem__.assert_any_call("unseen/cega", unseen.cega_zones)
        mock_run.__setitem__.assert_any_call("unseen/RMSE", unseen.rmse)
        mock_run.__setitem__.assert_any_call("unseen/RMADEX", unseen.rmadex)

    def test_log_neptune_metrics_with_cega_figure(self, test_config, mock_provider):
        """Test logging metrics with CEGA figure."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
        )

        # Create a mock Neptune run
        mock_run = MagicMock()
        orchestrator._neptune_run = mock_run

        # Create a figure
        fig = Figure()
        fig_base, _ax = plt.subplots()

        holdout = PredictionMetrics(
            cega_zones={"A": 90, "B": 10, "C": 0, "D": 0, "E": 0},
            rmse=10.0,
            rmadex=5.0,
            cega_figure=fig,
        )
        unseen = PredictionMetrics(
            cega_zones={"A": 85, "B": 15, "C": 0, "D": 0, "E": 0},
            rmse=12.0,
            rmadex=6.0,
        )

        orchestrator._log_neptune_metrics(holdout, unseen)

        # Verify cega figure was logged
        mock_run.__getitem__.assert_called_with("cega")
        plt.close(fig_base)
        plt.close(fig)


class TestExperimentOrchestratorRun:
    """Tests for run method."""

    def test_run_returns_results(
        self,
        test_config,
        mock_provider,
        mock_data_pipeline,
        mock_feature_engineer,
        mock_model_trainer,
        mock_metrics_calculator,
    ):
        """Test that run returns ExperimentResults."""
        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
            data_pipeline=mock_data_pipeline,
            feature_engineer=mock_feature_engineer,
            model_trainer=mock_model_trainer,
            metrics_calculator=mock_metrics_calculator,
        )

        with patch("src.bgc_providers.ohio_bgc_provider.OhioBgcProvider"):
            with patch(
                "src.services.experiment_orchestrator.DataPipeline"
            ) as MockPipeline:
                mock_test_pipeline = MagicMock()
                mock_test_pipeline.load_timeseries.return_value = pd.DataFrame()
                mock_test_pipeline.remove_gaps.return_value = pd.DataFrame()
                mock_test_pipeline.remove_missing_and_inf.return_value = pd.DataFrame()
                mock_test_pipeline.fix_column_names.return_value = pd.DataFrame()
                MockPipeline.return_value = mock_test_pipeline

                result = orchestrator.run(
                    patient_id=559,
                    window=12,
                    horizon=6,
                    enable_neptune=False,
                )

                assert isinstance(result, ExperimentResults)
                assert result.patient_id == 559
                assert result.window == 12
                assert result.horizon == 6
                assert result.execution_time >= 0

    def test_run_calls_pipeline_components(
        self,
        test_config,
        mock_provider,
        mock_data_pipeline,
        mock_feature_engineer,
        mock_model_trainer,
        mock_metrics_calculator,
    ):
        """Test that run calls all pipeline components."""
        orchestrator = ExperimentOrchestrator(
            provider=mock_provider,
            config=test_config,
            data_pipeline=mock_data_pipeline,
            feature_engineer=mock_feature_engineer,
            model_trainer=mock_model_trainer,
            metrics_calculator=mock_metrics_calculator,
        )

        with patch("src.bgc_providers.ohio_bgc_provider.OhioBgcProvider"):
            with patch(
                "src.services.experiment_orchestrator.DataPipeline"
            ) as MockPipeline:
                mock_test_pipeline = MagicMock()
                mock_test_pipeline.load_timeseries.return_value = pd.DataFrame()
                mock_test_pipeline.remove_gaps.return_value = pd.DataFrame()
                mock_test_pipeline.remove_missing_and_inf.return_value = pd.DataFrame()
                mock_test_pipeline.fix_column_names.return_value = pd.DataFrame()
                MockPipeline.return_value = mock_test_pipeline

                orchestrator.run(
                    patient_id=559,
                    window=12,
                    horizon=6,
                    enable_neptune=False,
                )

                mock_data_pipeline.load_timeseries.assert_called_once()
                mock_feature_engineer.create_features_cached.assert_called()
                mock_model_trainer.setup_regressor.assert_called_once()
                mock_model_trainer.compare_and_select.assert_called_once()
                mock_model_trainer.save_models.assert_called_once()
                mock_metrics_calculator.calculate_from_predictions.assert_called()
