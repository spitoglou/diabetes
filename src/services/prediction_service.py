"""Service for real-time glucose predictions."""

import glob
import os
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from pycaret.regression import load_model, predict_model

from src.config import Config, get_config
from src.pipeline.feature_engineer import FeatureEngineer
from src.repositories.measurement_repository import MeasurementRepository
from src.repositories.prediction_repository import PredictionRepository


class PredictionService:
    """
    Service for real-time glucose prediction.

    Handles:
    - Loading trained models
    - Retrieving recent measurements
    - Creating features for prediction
    - Making and storing predictions
    """

    def __init__(
        self,
        measurement_repo: MeasurementRepository,
        prediction_repo: PredictionRepository,
        config: Config | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ):
        """
        Initialize the prediction service.

        Args:
            measurement_repo: Repository for measurement data.
            prediction_repo: Repository for prediction data.
            config: Application configuration.
            feature_engineer: Feature engineering component.
        """
        self.measurement_repo = measurement_repo
        self.prediction_repo = prediction_repo
        self.config = config or get_config()
        self.feature_engineer = feature_engineer or FeatureEngineer(self.config)

        self._model: Optional[Any] = None
        self._model_features: Optional[List[str]] = None

    def load_model(
        self,
        patient_id: str | int,
        window: int | None = None,
        horizon: int | None = None,
        model_rank: int = 1,
    ) -> Any:
        """
        Load a trained model from disk.

        Args:
            patient_id: Patient identifier.
            window: Window size (uses config default if not specified).
            horizon: Prediction horizon (uses config default if not specified).
            model_rank: Model rank (1 = best model).

        Returns:
            Loaded model object.

        Raises:
            FileNotFoundError: If no matching model file found.
        """
        window = window or self.config.window_steps
        horizon = horizon or self.config.prediction_horizon

        pattern = f"{self.config.models_path}/{patient_id}_{window}_{horizon}_{model_rank}*.pkl"
        model_files = glob.glob(pattern)

        if not model_files:
            raise FileNotFoundError(f"No model found matching pattern: {pattern}")

        model_path = os.path.splitext(model_files[0])[0]
        logger.info(f"Loading model from: {model_path}")

        self._model = load_model(model_path)
        self._model_features = self._model.feature_names_in_  # pyright: ignore[reportOptionalMemberAccess]

        return self._model

    def predict_from_measurements(
        self,
        patient_id: str,
        window: int | None = None,
        horizon: int | None = None,
    ) -> Dict[str, Any]:
        """
        Make a prediction from the most recent measurements.

        Args:
            patient_id: Patient identifier.
            window: Window size in measurement intervals.
            horizon: Prediction horizon in measurement intervals.

        Returns:
            Prediction dict with origin_time, prediction_time, and value.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        window = window or self.config.window_steps
        horizon = horizon or self.config.prediction_horizon

        # Retrieve recent measurements
        raw_measurements = self.measurement_repo.get_recent(
            patient_id, limit=window + 5
        )

        if len(raw_measurements) < window:
            raise ValueError(
                f"Insufficient measurements: {len(raw_measurements)} < {window} required"
            )

        # Convert to DataFrame and prepare
        measurements_df = self._prepare_measurements(raw_measurements)
        measurements_df = measurements_df.sort_values("date_time").reset_index(
            drop=True
        )

        # Create features
        stream_df = self._create_stream_dataframe(measurements_df, window)
        features = self.feature_engineer.create_stream_features(
            stream_df, window, horizon
        )

        # Correct feature names for model compatibility
        features = self._correct_feature_names(features, self._model_features)  # pyright: ignore[reportArgumentType]

        # Make prediction
        prediction_result = predict_model(self._model, features)
        predicted_value = prediction_result.prediction_label.iloc[0]

        # Get last measurement time
        last_measurement = measurements_df.iloc[-1]
        origin_time = last_measurement["date_time"]
        prediction_time = origin_time + timedelta(minutes=horizon * 5)

        prediction = {
            "prediction_origin_time": origin_time,
            "prediction_time": prediction_time,
            "prediction_value": float(predicted_value),
        }

        logger.info(
            f"Prediction for patient {patient_id}: "
            f"{predicted_value:.1f} mg/dL at {prediction_time}"
        )

        return prediction

    def predict_and_store(
        self,
        patient_id: str,
        window: int | None = None,
        horizon: int | None = None,
    ) -> str:
        """
        Make a prediction and store it in the database.

        Args:
            patient_id: Patient identifier.
            window: Window size.
            horizon: Prediction horizon.

        Returns:
            Inserted prediction record ID.
        """
        prediction = self.predict_from_measurements(patient_id, window, horizon)
        record_id = self.prediction_repo.save(patient_id, prediction)
        logger.info(f"Stored prediction with ID: {record_id}")
        return record_id

    def _prepare_measurements(self, raw_measurements: List[Dict]) -> pd.DataFrame:
        """Convert raw measurement dicts to DataFrame."""
        from src.helpers.misc import get_part_of_day

        records = []
        for idx, raw in enumerate(
            reversed(raw_measurements)
        ):  # Reverse to chronological order
            date_time = pd.to_datetime(raw["effectiveDateTime"])
            records.append(
                {
                    "bg_value": raw["valueQuantity"]["value"],
                    "date_time": date_time,
                    "time_of_day": date_time.time(),
                    "part_of_day": get_part_of_day(date_time.hour),
                    "time": idx + 1,
                    "id": "a",
                }
            )

        return pd.DataFrame(records)

    def _create_stream_dataframe(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Create stream DataFrame for feature extraction."""
        return df.tail(window).reset_index(drop=True)

    def _correct_feature_names(
        self,
        df: pd.DataFrame,
        model_features: List[str],
    ) -> pd.DataFrame:
        """
        Correct DataFrame column names for model compatibility.

        LightGBM doesn't support special characters in feature names.
        Also adds any missing features required by the model.
        """
        # First, sanitize column names
        new_names = {
            col: re.sub(r"[^A-Za-z0-9_]+", "", col.replace(" ", "_"))
            for col in df.columns
        }
        new_n_list = list(new_names.values())
        new_names = {
            col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }
        df = df.rename(columns=new_names)

        # Add missing features required by model
        for feat in model_features:
            if feat not in df.columns and feat != "label":
                logger.debug(f"Adding missing feature: {feat}")
                df[feat] = None

        return df
