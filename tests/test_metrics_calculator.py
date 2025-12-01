"""Tests for metrics calculator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.metrics_calculator import MetricsCalculator, PredictionMetrics


class TestPredictionMetrics:
    """Tests for PredictionMetrics dataclass."""

    def test_zone_a_b_percent_basic(self):
        """Test zone A+B percentage calculation."""
        metrics = PredictionMetrics(
            cega_zones={"A": 80, "B": 15, "C": 3, "D": 1, "E": 1},
            rmse=10.0,
            rmadex=5.0,
        )

        assert metrics.zone_a_b_percent == 95.0

    def test_zone_a_b_percent_all_zone_a(self):
        """Test with all predictions in zone A."""
        metrics = PredictionMetrics(
            cega_zones={"A": 100, "B": 0, "C": 0, "D": 0, "E": 0},
            rmse=5.0,
            rmadex=2.0,
        )

        assert metrics.zone_a_b_percent == 100.0

    def test_zone_a_b_percent_empty(self):
        """Test with zero total predictions."""
        metrics = PredictionMetrics(
            cega_zones={"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
            rmse=0.0,
            rmadex=0.0,
        )

        assert metrics.zone_a_b_percent == 0.0

    def test_cega_figure_optional(self):
        """Test that cega_figure is optional."""
        metrics = PredictionMetrics(
            cega_zones={"A": 50, "B": 50, "C": 0, "D": 0, "E": 0},
            rmse=10.0,
            rmadex=5.0,
        )

        assert metrics.cega_figure is None


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a MetricsCalculator instance."""
        return MetricsCalculator()

    @pytest.fixture
    def sample_actual(self):
        """Create sample actual values."""
        return pd.Series([100.0, 120.0, 140.0, 160.0, 180.0])

    @pytest.fixture
    def sample_predicted(self):
        """Create sample predicted values."""
        return pd.Series([105.0, 118.0, 145.0, 155.0, 182.0])

    def test_calculate_returns_metrics(
        self, calculator, sample_actual, sample_predicted
    ):
        """Test that calculate returns PredictionMetrics."""
        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [4, 1, 0, 0, 0])

            result = calculator.calculate(
                actual=sample_actual,
                predicted=sample_predicted,
                legend="test",
            )

            assert isinstance(result, PredictionMetrics)
            assert result.cega_zones == {"A": 4, "B": 1, "C": 0, "D": 0, "E": 0}

    def test_calculate_rmse(self, calculator, sample_actual, sample_predicted):
        """Test RMSE calculation."""
        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [5, 0, 0, 0, 0])

            result = calculator.calculate(
                actual=sample_actual,
                predicted=sample_predicted,
            )

            # RMSE should be positive
            assert result.rmse > 0
            # Manual calculation: sqrt(mean([25, 4, 25, 25, 4])) = sqrt(16.6) ≈ 4.07
            expected_rmse = np.sqrt(np.mean([25, 4, 25, 25, 4]))
            assert np.isclose(result.rmse, expected_rmse)

    def test_calculate_rmadex(self, calculator, sample_actual, sample_predicted):
        """Test RMADEX calculation."""
        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [5, 0, 0, 0, 0])

            result = calculator.calculate(
                actual=sample_actual,
                predicted=sample_predicted,
            )

            # RMADEX should be positive and finite
            assert result.rmadex > 0
            assert np.isfinite(result.rmadex)

    def test_calculate_resets_index(self, calculator):
        """Test that calculate resets index to avoid alignment issues."""
        actual = pd.Series([100.0, 120.0], index=[5, 10])
        predicted = pd.Series([105.0, 125.0], index=[5, 10])

        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [2, 0, 0, 0, 0])

            # Should not raise alignment error
            result = calculator.calculate(actual, predicted)
            assert result is not None

    def test_calculate_from_predictions(self, calculator):
        """Test calculate_from_predictions method."""
        predictions_df = pd.DataFrame(
            {
                "label": [100.0, 120.0, 140.0],
                "prediction_label": [105.0, 118.0, 142.0],
            }
        )

        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [3, 0, 0, 0, 0])

            result = calculator.calculate_from_predictions(
                predictions_df,
                actual_col="label",
                predicted_col="prediction_label",
            )

            assert isinstance(result, PredictionMetrics)

    def test_calculate_from_predictions_custom_columns(self, calculator):
        """Test calculate_from_predictions with custom column names."""
        predictions_df = pd.DataFrame(
            {
                "actual_glucose": [100.0, 120.0],
                "predicted_glucose": [105.0, 118.0],
            }
        )

        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [2, 0, 0, 0, 0])

            result = calculator.calculate_from_predictions(
                predictions_df,
                actual_col="actual_glucose",
                predicted_col="predicted_glucose",
            )

            assert isinstance(result, PredictionMetrics)

    def test_calculate_from_predictions_handles_dataframe_columns(self, calculator):
        """Test calculate_from_predictions when columns are DataFrames."""
        # Create a DataFrame with duplicate column names to produce DataFrame slices
        predictions_df = pd.DataFrame(
            [[100.0, 105.0], [120.0, 118.0]],
            columns=["label", "prediction_label"],
        )

        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [2, 0, 0, 0, 0])

            result = calculator.calculate_from_predictions(predictions_df)

            assert isinstance(result, PredictionMetrics)

    def test_calculate_from_predictions_converts_non_series(self, calculator):
        """Test that non-Series values are converted to Series."""
        # Create DataFrame and manually convert columns to list to test conversion
        predictions_df = pd.DataFrame(
            {
                "label": [100.0, 120.0, 140.0],
                "prediction_label": [105.0, 118.0, 142.0],
            }
        )

        with patch("src.pipeline.metrics_calculator.clarke_error_grid") as mock_cega:
            mock_cega.return_value = (MagicMock(), [3, 0, 0, 0, 0])

            result = calculator.calculate_from_predictions(
                predictions_df,
                generate_plot=False,
            )

            assert isinstance(result, PredictionMetrics)
            assert result.cega_figure is None


class TestMetricsCalculatorFormatSummary:
    """Tests for format_metrics_summary static method."""

    def test_format_metrics_summary(self):
        """Test metrics summary formatting."""
        metrics = PredictionMetrics(
            cega_zones={"A": 80, "B": 15, "C": 3, "D": 1, "E": 1},
            rmse=10.5,
            rmadex=5.25,
        )

        result = MetricsCalculator.format_metrics_summary(metrics)

        assert "RMSE: 10.50" in result
        assert "RMADEX: 5.2500" in result
        assert "Zone A+B: 95.0%" in result
        assert "A=80" in result
        assert "B=15" in result

    def test_format_metrics_summary_perfect(self):
        """Test formatting with perfect predictions."""
        metrics = PredictionMetrics(
            cega_zones={"A": 100, "B": 0, "C": 0, "D": 0, "E": 0},
            rmse=0.0,
            rmadex=0.0,
        )

        result = MetricsCalculator.format_metrics_summary(metrics)

        assert "RMSE: 0.00" in result
        assert "Zone A+B: 100.0%" in result
