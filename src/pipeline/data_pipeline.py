"""Data pipeline for loading and cleaning glucose data."""

import re
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import Config, get_config
from src.interfaces.bgc_provider_interface import BgcProviderInterface


class DataPipeline:
    """
    Pipeline for loading and cleaning glucose time series data.

    Handles:
    - Loading data from providers
    - Gap correction (removing data affected by measurement gaps)
    - Removing missing values and infinities
    - Column name sanitization for ML compatibility
    """

    def __init__(
        self,
        provider: BgcProviderInterface,
        config: Config | None = None,
    ):
        """
        Initialize the data pipeline.

        Args:
            provider: Data provider implementing BgcProviderInterface.
            config: Application configuration.
        """
        self.provider = provider
        self.config = config or get_config()

    def load_timeseries(
        self,
        truncate: int = 0,
        show_plt: bool = False,
    ) -> pd.DataFrame:
        """
        Load raw time series data from the provider.

        Args:
            truncate: Number of records to truncate (0 = no truncation).
            show_plt: Whether to display plots.

        Returns:
            Raw time series DataFrame.
        """
        logger.info("Loading time series data from provider")
        return self.provider.tsfresh_dataframe(truncate=truncate, show_plt=show_plt)

    def remove_gaps(
        self,
        df: pd.DataFrame,
        window: int,
        horizon: int,
        perform_corrections: bool = True,
    ) -> pd.DataFrame:
        """
        Remove data points affected by measurement gaps.

        When there are gaps in CGM data, surrounding data points should
        be removed as they may produce invalid features.

        Args:
            df: Input DataFrame with 'end_time' column.
            window: Window size for feature extraction.
            horizon: Prediction horizon.
            perform_corrections: Whether to actually perform corrections.

        Returns:
            DataFrame with gap-affected rows removed.
        """
        if not perform_corrections:
            return df

        df = df.copy()
        problematic_points = []
        old_value = 0

        for index, row in df.iterrows():
            if (row["end_time"] - old_value) > 1:
                problematic_points.append(index)
            old_value = row["end_time"]

        if problematic_points:
            logger.warning(
                f"Found {len(problematic_points)} gap points: {problematic_points}"
            )

        for point in problematic_points:
            for i in range(-1 * horizon, window):
                if (point + i) in df.index:
                    df.drop(point + i, inplace=True)

        logger.info(f"DataFrame shape after gap removal: {df.shape}")
        return df

    def remove_missing_and_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with missing values and infinite values.

        Args:
            df: Input DataFrame.

        Returns:
            Cleaned DataFrame with problematic columns removed.
        """
        df = df.copy()
        original_shape = df.shape

        # Remove columns with NaN values
        df.dropna(axis=1, inplace=True)
        after_nan_shape = df.shape

        # Replace infinities with NaN and remove those columns
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, inplace=True)
        final_shape = df.shape

        logger.info(
            f"Cleaned DataFrame: {original_shape} -> {after_nan_shape} (NaN) -> {final_shape} (inf)"
        )
        return df

    def fix_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize column names for ML compatibility.

        LightGBM and other models don't support special characters in
        feature names. This method:
        - Replaces spaces with underscores
        - Removes special characters
        - Handles duplicate column names

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with sanitized column names.
        """
        df = df.copy()

        # Fix values in part_of_day column (spaces to underscores)
        if "part_of_day" in df.columns:
            df["part_of_day"] = df["part_of_day"].str.replace(" ", "_")

        # First replace spaces with underscores, then remove other special characters
        new_names = {
            col: re.sub(r"[^A-Za-z0-9_]+", "", col.replace(" ", "_"))
            for col in df.columns
        }

        # Handle duplicate column names by appending index
        new_n_list = list(new_names.values())
        new_names = {
            col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }

        df = df.rename(columns=new_names)
        return df

    def process(
        self,
        truncate: int = 0,
        window: int = 12,
        horizon: int = 6,
        perform_gap_corrections: bool = True,
        show_plt: bool = False,
    ) -> pd.DataFrame:
        """
        Run the full data pipeline.

        Args:
            truncate: Number of records to truncate.
            window: Window size for gap correction.
            horizon: Prediction horizon for gap correction.
            perform_gap_corrections: Whether to remove gap-affected data.
            show_plt: Whether to display plots.

        Returns:
            Fully processed DataFrame ready for feature engineering.
        """
        logger.info("Starting data pipeline processing")

        # Load raw data
        df = self.load_timeseries(truncate=truncate, show_plt=show_plt)
        logger.info(f"Loaded {len(df)} records")

        # Note: Gap removal is typically done after featurization
        # This method provides the cleaned timeseries for feature extraction

        return df

    def align_columns(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align columns between training and test DataFrames.

        Ensures both DataFrames have the same columns by keeping only
        columns that exist in both.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.

        Returns:
            Tuple of (aligned_train_df, aligned_test_df).
        """
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)

        common_cols = train_cols & test_cols
        train_only = train_cols - test_cols
        test_only = test_cols - train_cols

        if train_only:
            logger.warning(
                f"Dropping {len(train_only)} columns only in train: {train_only}"
            )
        if test_only:
            logger.warning(
                f"Dropping {len(test_only)} columns only in test: {test_only}"
            )

        # Maintain original column order from train_df
        ordered_cols = [c for c in train_df.columns if c in common_cols]

        return train_df[ordered_cols], test_df[ordered_cols]
