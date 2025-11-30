"""Data pipeline for loading and cleaning glucose data."""

import numpy as np
import pandas as pd
from loguru import logger

from src.config import Config, get_config
from src.helpers.dataframe import fix_column_names
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
                if point + i in df.index:
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

        Delegates to the shared helper function in helpers/dataframe.py.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with sanitized column names.
        """
        return fix_column_names(df)

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

        Returns two DataFrames that contain **only** the intersecting columns,
        preserving the original column order from ``train_df``. This guarantees
        that the static type checker sees the return values as ``pd.DataFrame``
        objects (using ``.loc`` which always yields a DataFrame, even for a single
        column).

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.

        Returns:
            Tuple of (aligned_train_df, aligned_test_df).
        """
        # Determine column intersections and differences
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)

        common_cols = train_cols & test_cols
        train_only = train_cols - test_cols
        test_only = test_cols - train_cols

        # Log any mismatches for diagnostics
        if train_only:
            logger.warning(
                f"Dropping {len(train_only)} columns only present in training set: {train_only}"
            )
        if test_only:
            logger.warning(
                f"Dropping {len(test_only)} columns only present in test set: {test_only}"
            )

        # Preserve the original column order from the training DataFrame
        ordered_common_cols = [col for col in train_df.columns if col in common_cols]

        # Use .loc to ensure the result is always a DataFrame (not a Series)
        aligned_train = train_df.loc[:, ordered_common_cols]
        aligned_test = test_df.loc[:, ordered_common_cols]

        return aligned_train, aligned_test
