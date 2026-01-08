"""Tests for simglucose training data generation.

Tests the ability to train models using synthetic data from simglucose.
"""

import pandas as pd
import pytest


class TestSimglucoseProviderTsfresh:
    """Tests for SimglucoseProvider.tsfresh_dataframe()."""

    def test_tsfresh_dataframe_generates_data(self):
        """tsfresh_dataframe() should generate a DataFrame with correct schema."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adult#001",
            insulin_mode="basal-bolus",
        )

        df = provider.tsfresh_dataframe(simulation_days=1)

        # Check shape (1 day = 480 readings at 3-min intervals)
        assert df.shape[0] == 480
        assert df.shape[1] == 7

    def test_tsfresh_dataframe_has_correct_columns(self):
        """tsfresh_dataframe() should have the same columns as Ohio provider."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adult#001",
            insulin_mode="basal-bolus",
        )

        df = provider.tsfresh_dataframe(simulation_days=1)

        expected_columns = [
            "date_time",
            "mock_date",
            "time_of_day",
            "part_of_day",
            "time",
            "bg_value",
            "id",
        ]
        assert list(df.columns) == expected_columns

    def test_tsfresh_dataframe_bg_values_in_range(self):
        """Blood glucose values should be in physiologically reasonable range."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adult#001",
            insulin_mode="basal-bolus",
        )

        df = provider.tsfresh_dataframe(simulation_days=1)

        # BG should be between 20 and 600 mg/dL (extreme but possible)
        assert df["bg_value"].min() >= 20
        assert df["bg_value"].max() <= 600

    def test_tsfresh_dataframe_patient_id(self):
        """Patient ID should be set correctly."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adolescent#003",
            insulin_mode="basal-bolus",
        )

        df = provider.tsfresh_dataframe(simulation_days=1)

        assert df["id"].iloc[0] == "adolescent#003"
        assert df["id"].unique().tolist() == ["adolescent#003"]

    def test_tsfresh_dataframe_truncate(self):
        """truncate parameter should limit the number of rows."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adult#001",
            insulin_mode="basal-bolus",
        )

        df = provider.tsfresh_dataframe(simulation_days=1, truncate=100)

        assert df.shape[0] == 100


class TestSimglucoseProviderGetGlycoseLevels:
    """Tests for SimglucoseProvider.get_glycose_levels()."""

    def test_get_glycose_levels_returns_list(self):
        """get_glycose_levels() should return a list of dicts."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adult#001",
            insulin_mode="basal-bolus",
        )

        # First call tsfresh_dataframe to generate data
        provider.tsfresh_dataframe(simulation_days=1)
        levels = provider.get_glycose_levels()

        assert isinstance(levels, list)
        assert len(levels) == 480

    def test_get_glycose_levels_dict_format(self):
        """Each item should have 'ts' and 'value' keys."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adult#001",
            insulin_mode="basal-bolus",
        )

        provider.tsfresh_dataframe(simulation_days=1)
        levels = provider.get_glycose_levels()

        assert "ts" in levels[0]
        assert "value" in levels[0]

    def test_get_glycose_levels_start_parameter(self):
        """start parameter should skip initial readings."""
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = SimglucoseProvider(
            patient_name="adult#001",
            insulin_mode="basal-bolus",
        )

        provider.tsfresh_dataframe(simulation_days=1)
        levels = provider.get_glycose_levels(start=100)

        assert len(levels) == 380  # 480 - 100


class TestProviderFactory:
    """Tests for the provider factory function."""

    def test_create_simglucose_provider(self):
        """create_provider('simglucose', ...) should return SimglucoseProvider."""
        from src.bgc_providers.factory import create_provider
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        provider = create_provider("simglucose", "adult#001")

        assert isinstance(provider, SimglucoseProvider)

    def test_create_provider_default_insulin_mode(self):
        """Factory should default to basal-bolus for training."""
        from src.bgc_providers.factory import create_provider

        provider = create_provider("simglucose", "adult#001")

        assert provider.insulin_mode == "basal-bolus"

    def test_create_provider_custom_insulin_mode(self):
        """Factory should accept custom insulin mode."""
        from src.bgc_providers.factory import create_provider

        provider = create_provider("simglucose", "adult#001", insulin_mode="basal")

        assert provider.insulin_mode == "basal"

    def test_get_sample_interval_ohio(self):
        """Ohio data source should have 5-minute interval."""
        from src.bgc_providers.factory import get_sample_interval

        assert get_sample_interval("ohio") == 5

    def test_get_sample_interval_simglucose(self):
        """Simglucose data source should have 3-minute interval."""
        from src.bgc_providers.factory import get_sample_interval

        assert get_sample_interval("simglucose") == 3

    def test_create_provider_invalid_source(self):
        """Invalid data source should raise ValueError."""
        from src.bgc_providers.factory import create_provider

        with pytest.raises(ValueError, match="Unknown data_source"):
            create_provider("invalid", "patient")


class TestExperimentDataSource:
    """Tests for Experiment class data_source parameter."""

    def test_experiment_default_data_source(self):
        """Experiment should default to 'ohio' data source."""
        from src.helpers.experiment import Experiment

        exp = Experiment(
            patient=559,
            window=6,
            horizon=6,
            enable_neptune=False,
        )

        assert exp.data_source == "ohio"
        assert exp.sample_interval == 5

    def test_experiment_simglucose_data_source(self):
        """Experiment should accept 'simglucose' data source."""
        from src.helpers.experiment import Experiment

        exp = Experiment(
            patient="adult#001",
            window=6,
            horizon=6,
            data_source="simglucose",
            enable_neptune=False,
        )

        assert exp.data_source == "simglucose"
        assert exp.sample_interval == 3

    def test_experiment_simglucose_prediction_time(self):
        """Simglucose experiment should calculate correct prediction time."""
        from src.helpers.experiment import Experiment

        exp = Experiment(
            patient="adult#001",
            window=6,
            horizon=6,
            data_source="simglucose",
            enable_neptune=False,
        )

        # 6 steps * 3 min = 18 min prediction window/horizon
        assert exp.win_min == 18
        assert exp.hor_min == 18

    def test_experiment_ohio_prediction_time(self):
        """Ohio experiment should calculate correct prediction time."""
        from src.helpers.experiment import Experiment

        exp = Experiment(
            patient=559,
            window=6,
            horizon=6,
            data_source="ohio",
            enable_neptune=False,
        )

        # 6 steps * 5 min = 30 min prediction window/horizon
        assert exp.win_min == 30
        assert exp.hor_min == 30

    def test_experiment_train_parameters_include_data_source(self):
        """Train parameters should include data_source."""
        from src.helpers.experiment import Experiment

        exp = Experiment(
            patient="adult#001",
            window=6,
            horizon=6,
            data_source="simglucose",
            enable_neptune=False,
        )

        assert exp.train_parameters["data_source"] == "simglucose"
        assert exp.train_parameters["patient"] == "adult#001"
        assert exp.train_parameters["simulation_days"] == 14

    def test_experiment_test_parameters_half_simulation_days(self):
        """Test parameters should use half the simulation days."""
        from src.helpers.experiment import Experiment

        exp = Experiment(
            patient="adult#001",
            window=6,
            horizon=6,
            data_source="simglucose",
            simulation_days=14,
            enable_neptune=False,
        )

        assert exp.unseen_data_parameters["simulation_days"] == 7


class TestDatasetNaming:
    """Tests for dataset file naming convention."""

    def test_ohio_dataset_name(self):
        """Ohio datasets should not have prefix."""
        from src.helpers.experiment import create_ds_name

        params = {
            "data_source": "ohio",
            "patient": "559",
            "scope": "train",
            "train_ds_size": 0,
            "window_size": 6,
            "prediction_horizon": 6,
        }

        name = create_ds_name(params)
        assert name == "dataframes/559_train_0_6_6.pkl"

    def test_simglucose_dataset_name(self):
        """Simglucose datasets should have sim_ prefix."""
        from src.helpers.experiment import create_ds_name

        params = {
            "data_source": "simglucose",
            "patient": "adult#001",
            "scope": "train",
            "train_ds_size": 0,
            "window_size": 6,
            "prediction_horizon": 6,
        }

        name = create_ds_name(params)
        assert name == "dataframes/sim_adult#001_train_0_6_6.pkl"

    def test_default_data_source_ohio(self):
        """Missing data_source should default to ohio (no prefix)."""
        from src.helpers.experiment import create_ds_name

        params = {
            "patient": "559",
            "scope": "train",
            "train_ds_size": 0,
            "window_size": 6,
            "prediction_horizon": 6,
        }

        name = create_ds_name(params)
        assert name == "dataframes/559_train_0_6_6.pkl"


class TestSimglucoseTrainingIntegration:
    """Integration tests for simglucose training pipeline.

    These tests verify the end-to-end pipeline works without running
    full model training (which would be too slow for CI).
    """

    def test_timeseries_dataframe_simglucose(self):
        """timeseries_dataframe() should work with simglucose data source."""
        from src.helpers.experiment import timeseries_dataframe

        params = {
            "data_source": "simglucose",
            "patient": "adult#001",
            "scope": "train",
            "train_ds_size": 100,  # Truncate to 100 rows for speed
            "simulation_days": 1,
        }

        df = timeseries_dataframe(params)

        assert df.shape[0] == 100
        assert "bg_value" in df.columns
        assert "id" in df.columns
        assert df["id"].iloc[0] == "adult#001"

    def test_experiment_creates_train_parameters_correctly(self):
        """Experiment should create correct train parameters for simglucose."""
        from src.helpers.experiment import Experiment

        exp = Experiment(
            patient="adult#001",
            window=6,
            horizon=6,
            data_source="simglucose",
            simulation_days=7,
            enable_neptune=False,
        )

        # Verify train parameters
        assert exp.train_parameters["data_source"] == "simglucose"
        assert exp.train_parameters["patient"] == "adult#001"
        assert exp.train_parameters["simulation_days"] == 7

        # Verify test parameters (half simulation days)
        assert exp.unseen_data_parameters["simulation_days"] == 3

    def test_load_trained_model_auto_detects_simglucose(self):
        """load_trained_model should auto-detect simglucose from patient ID."""
        from scripts.serving.load_model_and_predict import load_trained_model

        # This won't find a model (none exists), but should not error
        # and should use the correct pattern
        result = load_trained_model(
            patient_id="adult#001",
            window_steps=6,
            horizon_steps=6,
        )

        # Model won't exist, so result should be None
        assert result is None
