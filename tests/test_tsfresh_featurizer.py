"""Tests for tsfresh featurizer."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.featurizers.tsfresh import TsfreshFeaturizer


@pytest.fixture
def sample_timeseries_df():
    """Create a sample time series DataFrame for testing."""
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


class TestTsfreshFeaturizerInit:
    """Tests for TsfreshFeaturizer initialization."""

    def test_init_default_parameters(self, sample_timeseries_df):
        """Test initialization with default parameters."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
        )

        assert featurizer.chunk_size == 12
        assert featurizer.horizon == 6
        assert featurizer.hide_progressbars is True
        assert featurizer.plot_chunks is False

    def test_init_chunks_calculation(self, sample_timeseries_df):
        """Test that chunks are calculated correctly."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
        )

        # chunks = 50 - 12 + 1 - 6 = 33
        expected_chunks = 50 - 12 + 1 - 6
        assert featurizer.chunks == expected_chunks

    def test_init_minimal_features(self, sample_timeseries_df):
        """Test initialization with minimal features."""
        from tsfresh.feature_extraction import MinimalFCParameters

        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
            minimal_features=True,
        )

        assert isinstance(featurizer.parameters, MinimalFCParameters)

    def test_init_comprehensive_features(self, sample_timeseries_df):
        """Test initialization with comprehensive features."""
        from tsfresh.feature_extraction import ComprehensiveFCParameters

        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
            minimal_features=False,
        )

        assert isinstance(featurizer.parameters, ComprehensiveFCParameters)

    def test_init_extracts_required_columns(self, sample_timeseries_df):
        """Test that only required columns are extracted."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
        )

        assert list(featurizer.timeseries_df.columns) == ["time", "bg_value", "id"]


class TestTsfreshFeaturizerSliceDf:
    """Tests for slice_df method."""

    def test_slice_df_basic(self, sample_timeseries_df):
        """Test basic slice operation."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
        )

        sliced = featurizer.slice_df(sample_timeseries_df, 0, 10)

        assert len(sliced) == 10
        assert sliced.iloc[0]["time"] == 1

    def test_slice_df_middle(self, sample_timeseries_df):
        """Test slicing from middle of dataframe."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
        )

        sliced = featurizer.slice_df(sample_timeseries_df, 10, 5)

        assert len(sliced) == 5
        assert sliced.iloc[0]["time"] == 11


class TestTsfreshFeaturizerCreateTargetSeries:
    """Tests for create_target_series method."""

    def test_create_target_series(self, sample_timeseries_df):
        """Test target series creation."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
        )

        featurizer.create_target_series()

        assert len(featurizer.target_series) == featurizer.chunks
        assert not featurizer.target_series.empty

    def test_target_series_values_correct(self, sample_timeseries_df):
        """Test that target series contains correct values."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
        )

        featurizer.create_target_series()

        # First target should be value at index chunk_size - 1 + horizon
        expected_index = 12 - 1 + 6  # = 17
        expected_value = featurizer.timeseries_df.loc[expected_index].bg_value
        assert featurizer.target_series.iloc[0] == expected_value


class TestTsfreshFeaturizerCreateFeatureDataframe:
    """Tests for create_feature_dataframe method."""

    def test_create_feature_dataframe(self, sample_timeseries_df):
        """Test feature dataframe creation."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
            minimal_features=True,
        )

        featurizer.create_feature_dataframe()

        assert not featurizer.feature_dataframe.empty
        assert len(featurizer.feature_dataframe) == featurizer.chunks

    def test_feature_dataframe_has_metadata_columns(self, sample_timeseries_df):
        """Test that feature dataframe includes metadata columns."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
            minimal_features=True,
        )

        featurizer.create_feature_dataframe()

        assert "start" in featurizer.feature_dataframe.columns
        assert "end" in featurizer.feature_dataframe.columns
        assert "start_time" in featurizer.feature_dataframe.columns
        assert "end_time" in featurizer.feature_dataframe.columns
        assert "part_of_day" in featurizer.feature_dataframe.columns


class TestTsfreshFeaturizerCreateLabeledDataframe:
    """Tests for create_labeled_dataframe method."""

    def test_create_labeled_dataframe(self, sample_timeseries_df):
        """Test labeled dataframe creation."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
            minimal_features=True,
        )

        featurizer.create_labeled_dataframe()

        assert not featurizer.labeled_dataframe.empty
        assert "label" in featurizer.labeled_dataframe.columns

    def test_labeled_dataframe_creates_features_if_empty(self, sample_timeseries_df):
        """Test that labeled dataframe creation triggers feature creation."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
            minimal_features=True,
        )

        assert featurizer.feature_dataframe.empty

        featurizer.create_labeled_dataframe()

        assert not featurizer.feature_dataframe.empty

    def test_labeled_dataframe_creates_targets_if_empty(self, sample_timeseries_df):
        """Test that labeled dataframe creation triggers target creation."""
        featurizer = TsfreshFeaturizer(
            sample_timeseries_df,
            chunk_size=12,
            horizon=6,
            minimal_features=True,
        )

        assert featurizer.target_series.empty

        featurizer.create_labeled_dataframe()

        assert not featurizer.target_series.empty
