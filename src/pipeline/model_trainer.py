"""Model training pipeline using PyCaret."""

import uuid
from typing import Any, List, Optional

import pandas as pd
from loguru import logger
from pycaret.regression import (
    add_metric,
    compare_models,
    get_config,
    predict_model,
    pull,
    save_model,
    setup,
)

from src.config import Config, get_config
from src.helpers.diabetes.madex import madex, rmadex


class ModelTrainer:
    """
    Model training component using PyCaret AutoML.

    Handles:
    - PyCaret regression setup
    - Model comparison and selection
    - Model persistence
    """

    # Models to exclude at different speed settings
    SPEED_EXCLUSIONS = {
        1: [],  # Include all models
        2: ["catboost", "xgboost"],
        3: ["catboost", "xgboost", "et", "rf", "ada", "gbr"],
    }

    IGNORE_FEATURES = [
        "start",
        "end",
        "start_time",
        "end_time",
        "start_time_of_day",
        "end_time_of_day",
    ]

    def __init__(self, config: Config | None = None):
        """
        Initialize the model trainer.

        Args:
            config: Application configuration.
        """
        self.config = config or get_config()
        self._regressor = None
        self._best_models: List[Any] = []
        self._comparison_df: Optional[pd.DataFrame] = None

    def setup_regressor(
        self,
        train_df: pd.DataFrame,
        target: str = "label",
        feature_selection: bool = True,
        normalize: bool = True,
        session_id: int = 1974,
        verbose: bool = False,
    ) -> Any:
        """
        Setup PyCaret regression environment.

        Args:
            train_df: Training DataFrame.
            target: Target column name.
            feature_selection: Enable feature selection.
            normalize: Enable normalization.
            session_id: Random seed for reproducibility.
            verbose: Enable verbose output.

        Returns:
            PyCaret setup object.
        """
        logger.info(f"Setting up regressor with {len(train_df)} samples")

        # Add custom metrics
        add_metric("madex", "MADEX", madex, greater_is_better=False)
        add_metric("rmadex", "RMADEX", rmadex, greater_is_better=False)

        self._regressor = setup(
            train_df,
            target=target,
            feature_selection=feature_selection,
            normalize=normalize,
            ignore_features=self.IGNORE_FEATURES,
            html=False,
            verbose=verbose,
            session_id=session_id,
        )

        logger.info("Regressor setup complete")
        return self._regressor

    def compare_and_select(
        self,
        n_select: int = 3,
        sort_metric: str = "RMADEX",
        speed: int = 3,
        verbose: bool = True,
    ) -> List[Any]:
        """
        Compare models and select the best ones.

        Args:
            n_select: Number of top models to select.
            sort_metric: Metric to sort by.
            speed: Speed setting (1=slow/all models, 3=fast/fewer models).
            verbose: Enable verbose output.

        Returns:
            List of best models.
        """
        if self._regressor is None:
            raise RuntimeError("Must call setup_regressor() first")

        exclude = self.SPEED_EXCLUSIONS.get(speed, [])
        logger.info(f"Comparing models (excluding: {exclude}, sort by: {sort_metric})")

        self._best_models = compare_models(
            exclude=exclude,
            sort=sort_metric,
            n_select=n_select,
            verbose=verbose,
        )

        self._comparison_df = pull()

        # Ensure it's a list
        if not isinstance(self._best_models, list):
            self._best_models = [self._best_models]

        logger.info(f"Selected {len(self._best_models)} best models")
        return self._best_models

    def predict(
        self,
        model: Any = None,
        data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Make predictions using a model.

        Args:
            model: Model to use (defaults to best model).
            data: Data to predict on (defaults to holdout set).

        Returns:
            DataFrame with predictions.
        """
        if model is None:
            if not self._best_models:
                raise RuntimeError(
                    "No models available. Run compare_and_select() first"
                )
            model = self._best_models[0]

        return predict_model(model, data=data)

    def save_models(
        self,
        patient_id: int,
        window: int,
        horizon: int,
        models: Optional[List[Any]] = None,
    ) -> List[str]:
        """
        Save models to disk.

        Args:
            patient_id: Patient identifier for naming.
            window: Window size for naming.
            horizon: Prediction horizon for naming.
            models: Models to save (defaults to best models).

        Returns:
            List of saved model paths.
        """
        if models is None:
            models = self._best_models

        saved_paths = []
        for index, model in enumerate(models):
            model_name = self._get_model_name(model)
            path = (
                f"{self.config.models_path}/{patient_id}_{window}_{horizon}"
                f"_{index + 1}_{model_name}_{uuid.uuid4()}"
            )
            save_model(model, path)
            saved_paths.append(path)
            logger.info(f"Saved model {index + 1}: {model_name} -> {path}")

        return saved_paths

    def get_config_param(self, param: str) -> Any:
        """
        Get a PyCaret configuration parameter.

        Args:
            param: Parameter name.

        Returns:
            Parameter value.
        """
        return get_config(param)

    @property
    def best_models(self) -> List[Any]:
        """Get the list of best models."""
        return self._best_models

    @property
    def comparison_df(self) -> Optional[pd.DataFrame]:
        """Get the model comparison DataFrame."""
        return self._comparison_df

    @staticmethod
    def _get_model_name(model: Any) -> str:
        """Extract model name from model object."""
        return str(model).split("(")[0]
