"""Tests for DataFrame helper functions."""

import os
import tempfile

import pandas as pd
import pytest

from src.helpers.dataframe import fix_column_names, read_df, save_df


class TestSaveAndReadDf:
    """Tests for save_df and read_df functions."""

    def test_save_and_read_df(self):
        """Test saving and reading a DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filename = f.name

        try:
            save_df(df, filename)
            assert os.path.exists(filename)

            loaded_df = read_df(filename)
            pd.testing.assert_frame_equal(df, loaded_df)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_read_df_returns_dataframe(self):
        """Test read_df returns a DataFrame."""
        df = pd.DataFrame({"a": [1, 2]})

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filename = f.name

        try:
            df.to_pickle(filename)
            result = read_df(filename)
            assert isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_read_df_raises_on_non_dataframe(self):
        """Test read_df raises TypeError for non-DataFrame pickles."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filename = f.name

        try:
            # Save a non-DataFrame object
            pd.Series([1, 2, 3]).to_pickle(filename)

            with pytest.raises(TypeError, match="Expected DataFrame"):
                read_df(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestFixColumnNames:
    """Tests for fix_column_names function."""

    def test_replaces_spaces_with_underscores(self):
        """Test spaces in column names are replaced."""
        df = pd.DataFrame({"column with spaces": [1, 2]})
        result = fix_column_names(df)
        assert "column_with_spaces" in result.columns
        assert "column with spaces" not in result.columns

    def test_removes_special_characters(self):
        """Test special characters are removed."""
        df = pd.DataFrame({"col!@#$%^&*()": [1, 2]})
        result = fix_column_names(df)
        assert "col" in result.columns

    def test_handles_duplicate_columns(self):
        """Test duplicate column names get unique suffixes."""
        # Create a DataFrame with columns that become duplicates after sanitization
        df = pd.DataFrame({"col!": [1, 2], "col@": [3, 4], "col#": [5, 6]})
        result = fix_column_names(df)

        # All should be unique
        assert len(result.columns) == len(set(result.columns))
        # Should have col, col_1, col_2
        assert "col" in result.columns
        assert "col_1" in result.columns
        assert "col_2" in result.columns

    def test_fixes_part_of_day_values(self):
        """Test part_of_day column values have spaces replaced."""
        df = pd.DataFrame({"part_of_day": ["late night", "early morning", "afternoon"]})
        result = fix_column_names(df)
        assert "late_night" in result["part_of_day"].values
        assert "early_morning" in result["part_of_day"].values
        assert "late night" not in result["part_of_day"].values

    def test_does_not_modify_original(self):
        """Test original DataFrame is not modified."""
        df = pd.DataFrame({"col with space": [1, 2]})
        original_cols = list(df.columns)
        fix_column_names(df)
        assert list(df.columns) == original_cols

    def test_preserves_valid_columns(self):
        """Test valid column names are preserved."""
        df = pd.DataFrame({"valid_col": [1], "AnotherValid123": [2]})
        result = fix_column_names(df)
        assert "valid_col" in result.columns
        assert "AnotherValid123" in result.columns
