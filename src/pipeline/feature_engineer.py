"""Feature engineering pipeline using tsfresh."""

from os import path

import pandas as pd
from loguru import logger

from src.config import Config, get_config
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.helpers.dataframe import read_df, save_df


class FeatureEngineer:
    """
    Feature engineering component using tsfresh.

    Handles:
    - Creating tsfresh feature dataframes
    - Caching feature dataframes to disk
    - Managing feature extraction parameters
    """

    def __init__(self, config: Config | None = None):
        """
        Initialize the feature engineer.

        Args:
            config: Application configuration.
        """
        self.config = config or get_config()

    def _get_cache_path(
        self,
        patient_id: int,
        scope: str,
        truncate: int,
        window: int,
        horizon: int,
    ) -> str:
        """
        Generate cache file path for feature dataframe.

        Args:
            patient_id: Patient identifier.
            scope: Data scope (train/test).
            truncate: Truncation parameter.
            window: Window size.
            horizon: Prediction horizon.

        Returns:
            Path to cache file.
        """
        return (
            f"{self.config.dataframes_path}/{patient_id}_{scope}_{truncate}"
            f"_{window}_{horizon}.pkl"
        )

    def create_features(
        self,
        timeseries_df: pd.DataFrame,
        window: int,
        horizon: int,
        minimal_features: bool = False,
        plot_chunks: bool = False,
    ) -> pd.DataFrame:
        """
        Create tsfresh features from time series data.

        Args:
            timeseries_df: Input time series DataFrame.
            window: Window size for feature extraction.
            horizon: Prediction horizon.
            minimal_features: Use minimal feature set for speed.
            plot_chunks: Whether to plot data chunks.

        Returns:
            Feature DataFrame with labels.
        """
        logger.info(
            f"Creating features: window={window}, horizon={horizon}, "
            f"minimal={minimal_features}"
        )

        featurizer = TsfreshFeaturizer(
            timeseries_df,
            window,
            horizon,
            minimal_features=minimal_features,
            plot_chunks=plot_chunks,
        )
        featurizer.create_labeled_dataframe()

        labeled_df = featurizer.labeled_dataframe
        assert isinstance(labeled_df, pd.DataFrame), (
            "Expected labeled_dataframe to be a DataFrame"
        )

        logger.info(f"Created feature dataframe with shape: {labeled_df.shape}")
        return labeled_df

    def create_features_cached(
        self,
        timeseries_df: pd.DataFrame,
        patient_id: int,
        scope: str,
        truncate: int,
        window: int,
        horizon: int,
        minimal_features: bool = False,
        force_recreate: bool = False,
    ) -> pd.DataFrame:
        """
        Create features with disk caching.

        If a cached version exists, loads from disk. Otherwise, creates
        features and saves to cache.

        Args:
            timeseries_df: Input time series DataFrame.
            patient_id: Patient identifier.
            scope: Data scope (train/test).
            truncate: Truncation parameter.
            window: Window size.
            horizon: Prediction horizon.
            minimal_features: Use minimal feature set.
            force_recreate: Force recreation even if cache exists.

        Returns:
            Feature DataFrame with labels.
        """
        cache_path = self._get_cache_path(patient_id, scope, truncate, window, horizon)
        logger.info(f"Feature cache path: {cache_path}")

        if path.exists(cache_path) and not force_recreate:
            logger.info("Loading features from cache")
            cached_df = read_df(cache_path)
            assert isinstance(cached_df, pd.DataFrame), (
                "Expected cached data to be a DataFrame"
            )
            return cached_df

        logger.info("Creating new features (not cached or force_recreate=True)")
        feature_df = self.create_features(
            timeseries_df,
            window,
            horizon,
            minimal_features=minimal_features,
        )

        # Save to cache
        save_df(feature_df, cache_path)
        logger.info(f"Saved features to cache: {cache_path}")

        return feature_df

    def create_stream_features(
        self,
        stream_df: pd.DataFrame,
        window: int,
        horizon: int,
        minimal_features: bool = False,
    ) -> pd.DataFrame:
        """
        Create features for streaming prediction (single window).

        Args:
            stream_df: Recent measurements DataFrame.
            window: Window size.
            horizon: Prediction horizon.
            minimal_features: Use minimal feature set.

        Returns:
            Feature DataFrame for single prediction.
        """
        featurizer = TsfreshFeaturizer(
            stream_df.tail(window),
            window,
            horizon,
            plot_chunks=False,
            minimal_features=minimal_features,
        )
        featurizer.chunks = 1
        featurizer.create_feature_dataframe()

        feature_df = featurizer.feature_dataframe
        assert isinstance(feature_df, pd.DataFrame), (
            "Expected feature_dataframe to be a DataFrame"
        )
        return feature_df
