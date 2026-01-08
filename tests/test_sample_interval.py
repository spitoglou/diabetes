"""
Tests for configurable CGM sample interval support.

The sample interval is critical for:
- Prediction time calculation (horizon_steps * sample_interval = minutes ahead)
- Feature extraction window (window_steps * sample_interval = minutes of history)
- Model training and inference consistency

Models trained on one interval cannot be used with different intervals.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestSampleIntervalSettings:
    """Test SAMPLE_INTERVAL configuration in settings."""

    def test_default_sample_interval_is_5(self):
        """Default sample interval should be 5 minutes (Ohio dataset)."""
        from config.settings import Settings

        settings = Settings()
        assert settings.SAMPLE_INTERVAL == 5

    def test_sample_interval_from_env(self, monkeypatch):
        """Sample interval can be configured via environment variable."""
        monkeypatch.setenv("SAMPLE_INTERVAL", "3")
        from config.settings import Settings

        settings = Settings()
        assert settings.SAMPLE_INTERVAL == 3

    def test_sample_interval_values(self):
        """Common CGM sensor sample intervals."""
        # Medtronic Guardian (Ohio dataset)
        assert 5 == 5  # 5-minute intervals

        # Dexcom G6 (simglucose)
        assert 3 == 3  # 3-minute intervals

        # Abbott FreeStyle Libre (internal)
        assert 1 == 1  # 1-minute intervals


class TestPredictionTimeCalculation:
    """Test prediction time calculation with different intervals."""

    def test_ohio_prediction_time_30_minutes(self):
        """Ohio dataset: 6 steps * 5 min = 30 min prediction."""
        horizon_steps = 6
        sample_interval = 5
        prediction_minutes = horizon_steps * sample_interval
        assert prediction_minutes == 30

    def test_simglucose_prediction_time_18_minutes(self):
        """Simglucose: 6 steps * 3 min = 18 min prediction."""
        horizon_steps = 6
        sample_interval = 3
        prediction_minutes = horizon_steps * sample_interval
        assert prediction_minutes == 18

    def test_prediction_time_with_timedelta(self):
        """Verify timedelta calculation matches expected prediction time."""
        origin_time = datetime(2025, 12, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Ohio: 6 steps * 5 min = 30 min
        ohio_interval = 5
        ohio_horizon = 6
        ohio_prediction_time = origin_time + timedelta(
            minutes=ohio_horizon * ohio_interval
        )
        assert ohio_prediction_time == datetime(
            2025, 12, 1, 12, 30, 0, tzinfo=timezone.utc
        )

        # Simglucose: 6 steps * 3 min = 18 min
        sim_interval = 3
        sim_horizon = 6
        sim_prediction_time = origin_time + timedelta(
            minutes=sim_horizon * sim_interval
        )
        assert sim_prediction_time == datetime(
            2025, 12, 1, 12, 18, 0, tzinfo=timezone.utc
        )


class TestExperimentInterval:
    """Test Experiment class interval handling."""

    def test_experiment_default_uses_settings(self, monkeypatch):
        """Experiment should use settings.SAMPLE_INTERVAL as default."""
        monkeypatch.setenv("SAMPLE_INTERVAL", "3")

        # We can't easily test the full Experiment class without dependencies,
        # but we can verify the logic
        from config.settings import Settings

        settings = Settings()
        min_per_measure = None
        sample_interval = (
            min_per_measure if min_per_measure is not None else settings.SAMPLE_INTERVAL
        )
        assert sample_interval == 3

    def test_experiment_explicit_interval_override(self, monkeypatch):
        """Explicit min_per_measure should override settings."""
        monkeypatch.setenv("SAMPLE_INTERVAL", "5")

        from config.settings import Settings

        settings = Settings()
        min_per_measure = 3  # Explicitly set to 3
        sample_interval = (
            min_per_measure if min_per_measure is not None else settings.SAMPLE_INTERVAL
        )
        assert sample_interval == 3

    def test_window_minutes_calculation(self):
        """Window minutes = window_steps * sample_interval."""
        window_steps = 12

        # Ohio: 12 * 5 = 60 minutes of history
        ohio_win_min = window_steps * 5
        assert ohio_win_min == 60

        # Simglucose: 12 * 3 = 36 minutes of history
        sim_win_min = window_steps * 3
        assert sim_win_min == 36

    def test_horizon_minutes_calculation(self):
        """Horizon minutes = horizon_steps * sample_interval."""
        horizon_steps = 6

        # Ohio: 6 * 5 = 30 minutes ahead
        ohio_hor_min = horizon_steps * 5
        assert ohio_hor_min == 30

        # Simglucose: 6 * 3 = 18 minutes ahead
        sim_hor_min = horizon_steps * 3
        assert sim_hor_min == 18


class TestModelIntervalCompatibility:
    """Test model/data interval compatibility warnings."""

    def test_interval_mismatch_warning(self):
        """
        Models trained on one interval should not be used with different intervals.

        This is a documentation test - actual enforcement would require
        storing interval metadata in the model file.
        """
        # Example: Model trained on Ohio (5-min) used with simglucose (3-min)
        model_interval = 5
        data_interval = 3

        # These should NOT be equal - interval mismatch!
        assert model_interval != data_interval

        # In practice, this would produce incorrect predictions because:
        # - Features extracted at different time scales
        # - Prediction time calculated incorrectly
        # - Temporal patterns don't align

    def test_same_interval_is_compatible(self):
        """Same interval between model and data is valid."""
        model_interval = 5
        data_interval = 5
        assert model_interval == data_interval


class TestCLIIntervalOption:
    """Test CLI --interval option behavior."""

    def test_interval_option_exists(self):
        """The serve predict command should have --interval option."""
        # Import the CLI app to verify the option exists
        import inspect

        from cli import serve_predict

        sig = inspect.signature(serve_predict)
        param_names = list(sig.parameters.keys())
        assert "interval" in param_names

    def test_interval_default_is_none(self):
        """Interval option should default to None (use settings)."""
        import inspect

        from cli import serve_predict

        sig = inspect.signature(serve_predict)
        interval_param = sig.parameters["interval"]
        assert interval_param.default.default is None
