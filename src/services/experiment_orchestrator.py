"""Experiment orchestrator service coordinating the ML pipeline."""
# pylint: disable=wrong-import-position

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import neptune
from loguru import logger

from src.config import Config, get_config
from src.interfaces.bgc_provider_interface import BgcProviderInterface
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.feature_engineer import FeatureEngineer
from src.pipeline.metrics_calculator import MetricsCalculator, PredictionMetrics
from src.pipeline.model_trainer import ModelTrainer


@dataclass
class ExperimentResults:
    """Container for experiment results."""

    patient_id: int
    window: int
    horizon: int
    holdout_metrics: Optional[PredictionMetrics] = None
    unseen_metrics: Optional[PredictionMetrics] = None
    best_models: list = field(default_factory=list)
    comparison_df: Any = None
    execution_time: float = 0.0


class ExperimentOrchestrator:
    """
    Orchestrator for ML experiment workflows.

    Coordinates:
    - Data loading and preprocessing
    - Feature engineering
    - Model training and selection
    - Evaluation on holdout and unseen data
    - Neptune tracking (optional)
    """

    def __init__(
        self,
        provider: BgcProviderInterface,
        config: Config | None = None,
        data_pipeline: DataPipeline | None = None,
        feature_engineer: FeatureEngineer | None = None,
        model_trainer: ModelTrainer | None = None,
        metrics_calculator: MetricsCalculator | None = None,
    ):
        """
        Initialize the orchestrator with injected dependencies.

        Args:
            provider: Data provider for loading glucose data.
            config: Application configuration.
            data_pipeline: Data pipeline component.
            feature_engineer: Feature engineering component.
            model_trainer: Model training component.
            metrics_calculator: Metrics calculation component.
        """
        self.config = config or get_config()
        self.provider = provider

        # Use injected components or create defaults
        self.data_pipeline = data_pipeline or DataPipeline(provider, self.config)
        self.feature_engineer = feature_engineer or FeatureEngineer(self.config)
        self.model_trainer = model_trainer or ModelTrainer(self.config)
        self.metrics_calculator = metrics_calculator or MetricsCalculator()

        self._neptune_run = None

    def run(
        self,
        patient_id: int,
        window: int,
        horizon: int,
        min_per_measure: int = 5,
        best_models_no: int = 3,
        speed: int = 3,
        perform_gap_corrections: bool = True,
        minimal_features: bool = False,
        enable_neptune: bool | None = None,
    ) -> ExperimentResults:
        """
        Run a complete experiment.

        Args:
            patient_id: Patient identifier.
            window: Window size in measurement intervals.
            horizon: Prediction horizon in measurement intervals.
            min_per_measure: Minutes per measurement (default 5).
            best_models_no: Number of top models to select.
            speed: Training speed (1=slow, 3=fast).
            perform_gap_corrections: Remove gap-affected data.
            minimal_features: Use minimal feature set.
            enable_neptune: Override Neptune setting.

        Returns:
            ExperimentResults with all metrics and models.
        """
        start_time = time.time()

        # Determine Neptune setting
        use_neptune = (
            enable_neptune if enable_neptune is not None else self.config.enable_neptune
        )

        win_min = window * min_per_measure
        hor_min = horizon * min_per_measure

        logger.info(
            f"Starting experiment: Patient={patient_id}, "
            f"Window={window} ({win_min}min), Horizon={horizon} ({hor_min}min)"
        )

        # Initialize Neptune tracking
        if use_neptune:
            self._init_neptune(
                patient_id, window, horizon, speed, perform_gap_corrections
            )

        try:
            # Prepare parameters
            _train_params = self._make_params(
                patient_id, "train", window, horizon, minimal_features
            )
            _test_params = self._make_params(
                patient_id, "test", window, horizon, minimal_features
            )

            # Load and process training data
            logger.info("Processing training data")
            train_ts = self.data_pipeline.load_timeseries()
            train_df = self.feature_engineer.create_features_cached(
                train_ts,
                patient_id=patient_id,
                scope="train",
                truncate=0,
                window=window,
                horizon=horizon,
                minimal_features=minimal_features,
            )
            train_df = self.data_pipeline.remove_gaps(
                train_df, window, horizon, perform_gap_corrections
            )
            train_df = self.data_pipeline.remove_missing_and_inf(train_df)
            train_df = self.data_pipeline.fix_column_names(train_df)

            # Load and process test data
            logger.info("Processing test data")
            # Create a new provider for test data
            # pylint: disable-next=import-outside-toplevel
            from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider

            test_provider = OhioBgcProvider(scope="test", ohio_no=patient_id)
            test_pipeline = DataPipeline(test_provider, self.config)

            test_ts = test_pipeline.load_timeseries()
            test_df = self.feature_engineer.create_features_cached(
                test_ts,
                patient_id=patient_id,
                scope="test",
                truncate=0,
                window=window,
                horizon=horizon,
                minimal_features=minimal_features,
            )
            test_df = test_pipeline.remove_gaps(
                test_df, window, horizon, perform_gap_corrections
            )
            test_df = test_pipeline.remove_missing_and_inf(test_df)
            test_df = test_pipeline.fix_column_names(test_df)

            # Align columns
            train_df, test_df = self.data_pipeline.align_columns(train_df, test_df)

            # Setup and train models
            logger.info("Training models")
            self.model_trainer.setup_regressor(train_df, verbose=False)
            best_models = self.model_trainer.compare_and_select(
                n_select=best_models_no,
                speed=speed,
                verbose=True,
            )

            # Log to Neptune
            if use_neptune and self._neptune_run:
                self._neptune_run["model/comparison"].upload(
                    neptune.types.File.as_html(self.model_trainer.comparison_df)  # pyright: ignore[reportAttributeAccessIssue]
                )

            # Save models
            self.model_trainer.save_models(patient_id, window, horizon)

            # Evaluate on holdout
            logger.info("Evaluating on holdout data")
            holdout_legend = f"[{patient_id}]Holdout_W{win_min}_H{hor_min}"
            holdout_pred = self.model_trainer.predict()
            holdout_metrics = self.metrics_calculator.calculate_from_predictions(
                holdout_pred, legend=holdout_legend
            )
            plt.close("all")

            # Evaluate on unseen data
            logger.info("Evaluating on unseen data")
            unseen_legend = f"[{patient_id}]Unseen_W{win_min}_H{hor_min}"
            unseen_pred = self.model_trainer.predict(data=test_df)
            unseen_metrics = self.metrics_calculator.calculate_from_predictions(
                unseen_pred, legend=unseen_legend
            )
            plt.close("all")

            # Log metrics to Neptune
            if use_neptune and self._neptune_run:
                self._log_neptune_metrics(holdout_metrics, unseen_metrics)

            execution_time = time.time() - start_time
            logger.info(f"Experiment completed in {execution_time:.1f}s")

            # Log results
            logger.info(
                f"Holdout: {MetricsCalculator.format_metrics_summary(holdout_metrics)}"
            )
            logger.info(
                f"Unseen:  {MetricsCalculator.format_metrics_summary(unseen_metrics)}"
            )

            return ExperimentResults(
                patient_id=patient_id,
                window=window,
                horizon=horizon,
                holdout_metrics=holdout_metrics,
                unseen_metrics=unseen_metrics,
                best_models=best_models,
                comparison_df=self.model_trainer.comparison_df,
                execution_time=execution_time,
            )

        finally:
            if use_neptune and self._neptune_run:
                self._neptune_run["Execution Time"] = time.time() - start_time
                self._neptune_run.stop()
                self._neptune_run = None

    def _make_params(
        self,
        patient_id: int,
        scope: str,
        window: int,
        horizon: int,
        minimal_features: bool,
    ) -> Dict[str, Any]:
        """Create parameter dict for data loading."""
        return {
            "ohio_no": patient_id,
            "scope": scope,
            "train_ds_size": 0,
            "window_size": window,
            "prediction_horizon": horizon,
            "minimal_features": minimal_features,
        }

    def _init_neptune(
        self,
        patient_id: int,
        window: int,
        horizon: int,
        speed: int,
        perform_gap_corrections: bool,
    ) -> None:
        """Initialize Neptune tracking run."""
        if not self.config.neptune_api_token:
            logger.warning("Neptune API token not set, skipping Neptune tracking")
            return

        self._neptune_run = neptune.init_run(
            project=self.config.neptune_project,
            api_token=self.config.neptune_api_token,
        )

        self._neptune_run["parameters"] = {
            "patient": patient_id,
            "window": window,
            "horizon": horizon,
            "gap_corrections": perform_gap_corrections,
            "speed": speed,
        }

    def _log_neptune_metrics(
        self,
        holdout: PredictionMetrics,
        unseen: PredictionMetrics,
    ) -> None:
        """Log metrics to Neptune."""
        if not self._neptune_run:
            return

        self._neptune_run["holdout/cega"] = holdout.cega_zones
        self._neptune_run["holdout/RMSE"] = holdout.rmse
        self._neptune_run["holdout/RMADEX"] = holdout.rmadex

        self._neptune_run["unseen/cega"] = unseen.cega_zones
        self._neptune_run["unseen/RMSE"] = unseen.rmse
        self._neptune_run["unseen/RMADEX"] = unseen.rmadex

        if holdout.cega_figure:
            self._neptune_run["cega"].log(plt.gcf())
