"""
Tests for DataPipeline class.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.pipeline.data_pipeline import DataPipeline


@pytest.fixture
def config():
    """Test configuration."""
    return Config()


@pytest.fixture
def mock_provider():
    """Mock data provider."""
    provider = MagicMock()
    return provider


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "bg_value": [120, 125, 130, np.nan, 140],
            "date_time": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "end_time": [1, 2, 3, 4, 5],
            "part_of_day": ["morning", "morning", "late night", "afternoon", "evening"],
            "feature with spaces": [1, 2, 3, 4, 5],
            "feature__special!!chars": [1, 2, 3, 4, 5],
        }
    )


class TestDataPipeline:
    """Tests for DataPipeline."""

    def test_init(self, mock_provider, config):
        """Test pipeline initialization."""
        pipeline = DataPipeline(mock_provider, config)
        assert pipeline.provider == mock_provider
        assert pipeline.config == config

    def test_load_timeseries(self, mock_provider, config):
        """Test loading time series data."""
        mock_df = pd.DataFrame({"bg_value": [120, 125]})
        mock_provider.tsfresh_dataframe.return_value = mock_df

        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.load_timeseries(truncate=10, show_plt=False)

        mock_provider.tsfresh_dataframe.assert_called_once_with(
            truncate=10, show_plt=False
        )
        assert result.equals(mock_df)

    def test_remove_missing_and_inf(self, mock_provider, config):
        """Test removing missing values and infinities."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4],
                "col2": [1, np.nan, 3, 4],
                "col3": [1, 2, np.inf, 4],
                "col4": [1, 2, 3, 4],
            }
        )

        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.remove_missing_and_inf(df)

        # col2 and col3 should be dropped
        assert "col1" in result.columns
        assert "col4" in result.columns
        assert "col2" not in result.columns
        assert "col3" not in result.columns

    def test_fix_column_names_spaces(self, mock_provider, config, sample_df):
        """Test fixing column names with spaces."""
        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.fix_column_names(sample_df)

        assert "feature_with_spaces" in result.columns
        assert "feature with spaces" not in result.columns

    def test_fix_column_names_special_chars(self, mock_provider, config, sample_df):
        """Test fixing column names with special characters."""
        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.fix_column_names(sample_df)

        assert "feature__specialchars" in result.columns
        assert "feature__special!!chars" not in result.columns

    def test_fix_column_names_part_of_day(self, mock_provider, config, sample_df):
        """Test fixing part_of_day values."""
        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.fix_column_names(sample_df)

        # "late night" should become "late_night"
        assert "late_night" in result["part_of_day"].values
        assert "late night" not in result["part_of_day"].values

    def test_remove_gaps_no_corrections(self, mock_provider, config):
        """Test gap removal when disabled."""
        df = pd.DataFrame(
            {
                "end_time": [1, 2, 10, 11],  # Gap between 2 and 10
                "value": [100, 110, 120, 130],
            }
        )

        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.remove_gaps(
            df, window=2, horizon=1, perform_corrections=False
        )

        # Should return unchanged DataFrame
        assert len(result) == len(df)

    def test_align_columns(self, mock_provider, config):
        """Test column alignment between train and test sets."""
        train_df = pd.DataFrame(
            {
                "feat1": [1, 2],
                "feat2": [3, 4],
                "feat3": [5, 6],
            }
        )
        test_df = pd.DataFrame(
            {
                "feat1": [1, 2],
                "feat2": [3, 4],
                "feat4": [7, 8],
            }
        )

        pipeline = DataPipeline(mock_provider, config)
        aligned_train, aligned_test = pipeline.align_columns(train_df, test_df)

        # Only common columns should remain
        assert set(aligned_train.columns) == {"feat1", "feat2"}
        assert set(aligned_test.columns) == {"feat1", "feat2"}
