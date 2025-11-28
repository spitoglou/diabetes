"""Metrics calculation for glucose prediction evaluation."""

from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error

from src.helpers.diabetes.cega import clarke_error_grid
from src.helpers.diabetes.madex import mean_adjusted_exponent_error


@dataclass
class PredictionMetrics:
    """Container for prediction evaluation metrics."""

    cega_zones: Dict[str, int]  # Clarke Error Grid zone counts
    rmse: float  # Root Mean Squared Error
    rmadex: float  # Root Mean Adjusted Exponent Error
    cega_figure: Optional[plt.Figure] = None

    @property
    def zone_a_b_percent(self) -> float:
        """Calculate percentage of predictions in clinically acceptable zones A+B."""
        total = sum(self.cega_zones.values())
        if total == 0:
            return 0.0
        return (self.cega_zones["A"] + self.cega_zones["B"]) / total * 100


class MetricsCalculator:
    """
    Calculator for glucose prediction evaluation metrics.

    Computes:
    - Clarke Error Grid Analysis (CEGA)
    - Root Mean Squared Error (RMSE)
    - Root Mean Adjusted Exponent Error (RMADEX)
    """

    ZONE_LABELS = ["A", "B", "C", "D", "E"]

    def calculate(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        legend: str = "",
        generate_plot: bool = True,
    ) -> PredictionMetrics:
        """
        Calculate all prediction metrics.

        Args:
            actual: Actual glucose values.
            predicted: Predicted glucose values.
            legend: Legend text for plots.
            generate_plot: Whether to generate CEGA plot.

        Returns:
            PredictionMetrics with all calculated values.
        """
        # Reset index to avoid alignment issues
        actual = actual.reset_index(drop=True)
        predicted = predicted.reset_index(drop=True)

        # Clarke Error Grid
        fig, zone_counts = clarke_error_grid(actual, predicted, legend)
        cega_zones = dict(zip(self.ZONE_LABELS, zone_counts))

        # RMSE
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # RMADEX
        rmadex = np.sqrt(mean_adjusted_exponent_error(actual, predicted))

        logger.info(
            f"Metrics [{legend}]: RMSE={rmse:.2f}, RMADEX={rmadex:.4f}, "
            f"Zone A+B={cega_zones['A'] + cega_zones['B']}/{sum(zone_counts)}"
        )

        return PredictionMetrics(
            cega_zones=cega_zones,
            rmse=rmse,
            rmadex=rmadex,
            cega_figure=fig if generate_plot and isinstance(fig, plt.Figure) else None,
        )

    def calculate_from_predictions(
        self,
        predictions_df: pd.DataFrame,
        actual_col: str = "label",
        predicted_col: str = "prediction_label",
        legend: str = "",
        generate_plot: bool = True,
    ) -> PredictionMetrics:
        """
        Calculate metrics from a predictions DataFrame.

        Args:
            predictions_df: DataFrame with actual and predicted values.
            actual_col: Column name for actual values.
            predicted_col: Column name for predicted values.
            legend: Legend text for plots.
            generate_plot: Whether to generate CEGA plot.

        Returns:
            PredictionMetrics with all calculated values.
        """
        actual_series = predictions_df[actual_col]
        predicted_series = predictions_df[predicted_col]

        # Ensure we have Series objects
        if isinstance(actual_series, pd.DataFrame):
            actual_series = actual_series.squeeze()
        if isinstance(predicted_series, pd.DataFrame):
            predicted_series = predicted_series.squeeze()

        # Ensure they are Series type for type checker
        if not isinstance(actual_series, pd.Series):
            actual_series = pd.Series(actual_series)
        if not isinstance(predicted_series, pd.Series):
            predicted_series = pd.Series(predicted_series)

        return self.calculate(
            actual_series,
            predicted_series,
            legend=legend,
            generate_plot=generate_plot,
        )

    @staticmethod
    def format_metrics_summary(metrics: PredictionMetrics) -> str:
        """
        Format metrics as a human-readable summary string.

        Args:
            metrics: PredictionMetrics to format.

        Returns:
            Formatted string.
        """
        zones_str = ", ".join(f"{k}={v}" for k, v in metrics.cega_zones.items())
        return (
            f"RMSE: {metrics.rmse:.2f} | "
            f"RMADEX: {metrics.rmadex:.4f} | "
            f"Zone A+B: {metrics.zone_a_b_percent:.1f}% | "
            f"Zones: [{zones_str}]"
        )
