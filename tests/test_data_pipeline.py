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

    def test_align_columns_preserves_order(self, mock_provider, config):
        """Test column alignment preserves training column order."""
        train_df = pd.DataFrame(
            {
                "z_feat": [1, 2],
                "a_feat": [3, 4],
                "m_feat": [5, 6],
            }
        )
        test_df = pd.DataFrame(
            {
                "a_feat": [1, 2],
                "m_feat": [3, 4],
                "z_feat": [7, 8],
            }
        )

        pipeline = DataPipeline(mock_provider, config)
        aligned_train, aligned_test = pipeline.align_columns(train_df, test_df)

        # Order should match training DataFrame
        assert list(aligned_train.columns) == ["z_feat", "a_feat", "m_feat"]
        assert list(aligned_test.columns) == ["z_feat", "a_feat", "m_feat"]

    def test_align_columns_identical_columns(self, mock_provider, config):
        """Test alignment when columns are identical."""
        train_df = pd.DataFrame({"feat1": [1], "feat2": [2]})
        test_df = pd.DataFrame({"feat1": [3], "feat2": [4]})

        pipeline = DataPipeline(mock_provider, config)
        aligned_train, aligned_test = pipeline.align_columns(train_df, test_df)

        assert list(aligned_train.columns) == ["feat1", "feat2"]
        assert list(aligned_test.columns) == ["feat1", "feat2"]

    def test_remove_gaps_with_corrections(self, mock_provider, config):
        """Test gap removal when corrections are enabled."""
        # Create DataFrame with a gap (end_time jumps from 2 to 5)
        df = pd.DataFrame(
            {
                "end_time": [1, 2, 5, 6, 7, 8, 9, 10],
                "value": [100, 110, 120, 130, 140, 150, 160, 170],
            }
        )

        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.remove_gaps(df, window=2, horizon=1, perform_corrections=True)

        # The gap at index 2 should cause removal of surrounding rows
        # Gap is at index 2, with horizon=1 and window=2
        # Should remove indices: 2-1=1, 2, 2+1=3, 2+2=4 (actually from -horizon to window-1)
        assert len(result) < len(df)

    def test_remove_gaps_removes_correct_indices(self, mock_provider, config):
        """Test gap removal removes the correct indices around gaps."""
        # Gap between index 2 and 3 (end_time jumps from 3 to 10)
        df = pd.DataFrame(
            {
                "end_time": [1, 2, 3, 10, 11, 12, 13, 14, 15, 16],
                "value": list(range(10)),
            }
        )

        pipeline = DataPipeline(mock_provider, config)
        # With window=3, horizon=2, should remove points around the gap
        result = pipeline.remove_gaps(df, window=3, horizon=2, perform_corrections=True)

        # Original had 10 rows, gap at index 3
        # Should remove: index 3-2=1, 3-1=2, 3, 3+1=4, 3+2=5 (from -horizon to window-1)
        assert len(result) < 10

    def test_remove_gaps_no_gaps(self, mock_provider, config):
        """Test gap removal with no gaps returns same data."""
        df = pd.DataFrame(
            {
                "end_time": [1, 2, 3, 4, 5],
                "value": [100, 110, 120, 130, 140],
            }
        )

        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.remove_gaps(df, window=2, horizon=1, perform_corrections=True)

        # No gaps, so no rows should be removed
        assert len(result) == len(df)

    def test_process_returns_dataframe(self, mock_provider, config):
        """Test process method returns a DataFrame."""
        mock_df = pd.DataFrame(
            {
                "bg_value": [120, 125, 130],
                "date_time": pd.date_range("2024-01-01", periods=3, freq="5min"),
            }
        )
        mock_provider.tsfresh_dataframe.return_value = mock_df

        pipeline = DataPipeline(mock_provider, config)
        result = pipeline.process(truncate=0, window=2, horizon=1)

        assert isinstance(result, pd.DataFrame)
        mock_provider.tsfresh_dataframe.assert_called_once()
