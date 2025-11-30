"""Tsfresh-based time series feature extraction.

This module provides the TsfreshFeaturizer class for extracting features
from glucose time series data using the tsfresh library.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from tqdm import trange
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters


class TsfreshFeaturizer:
    """Feature extraction from time series data using the Tsfresh library."""

    def __init__(
        self,
        timeseries_df: pd.DataFrame,
        chunk_size: int,
        horizon: int,
        hide_progressbars: bool = True,
        minimal_features: bool = True,
        display_chunks: bool = False,
        display_chunk_features: bool = False,
        plot_chunks: bool = False,
    ) -> None:
        """
        Initialize the featurizer with time series data and configuration.

        :param timeseries_df: The input dataframe containing the time series data. It should have
        columns named 'time', 'bg_value', and 'id'
        :param chunk_size: The `chunk_size` parameter determines the size of each chunk or window of
        data that will be used for feature extraction. It represents the number of consecutive data
        points that will be considered at a time
        :param horizon: The horizon parameter represents the number of time steps into the future that
        we want to predict
        :param hide_progressbars: The `hide_progressbars` parameter is a boolean flag that determines
        whether progress bars should be displayed during the execution of the code. If set to `True`,
        the progress bars will be hidden, defaults to True (optional)
        :param minimal_features: The `minimal_features` parameter is a boolean flag that determines
        whether to use comprehensive feature calculation parameters or minimal feature calculation
        parameters. If `minimal_features` is set to `True`, the minimal feature calculation parameters
        will be used. Otherwise, the comprehensive feature calculation parameters will be used, defaults
        to True (optional)
        :param plot_chunks: The `plot_chunks` parameter is a boolean flag that determines whether or not
        to plot the chunks of the time series data. If set to `True`, the chunks will be plotted. If set
        to `False`, the chunks will not be plotted, defaults to False (optional)
        """
        self.raw_timeseries_df: pd.DataFrame = timeseries_df
        self.timeseries_df: pd.DataFrame = timeseries_df.loc[
            :, ["time", "bg_value", "id"]
        ]
        self.chunk_size: int = chunk_size
        self.parameters: ComprehensiveFCParameters | MinimalFCParameters = (
            ComprehensiveFCParameters()
        )
        if minimal_features is True:
            self.parameters = MinimalFCParameters()
        self.hide_progressbars: bool = hide_progressbars
        self.plot_chunks: bool = plot_chunks
        self.display_chunks: bool = display_chunks
        self.display_chunk_features: bool = display_chunk_features
        self.horizon: int = horizon
        # Determine in how many chunks the dataset will be divided
        # It is calculated [Total Records] - ([Chunk Size] - 1) - [Horizon]
        self.chunks: int = timeseries_df.shape[0] - chunk_size + 1 - horizon
        # Initialize of parameter dataframes
        self.feature_dataframe: pd.DataFrame = pd.DataFrame()
        self.target_series: pd.Series = pd.Series(dtype=float)  # type: ignore[type-arg]
        self.labeled_dataframe: pd.DataFrame = pd.DataFrame()

    def slice_df(
        self, dataframe: pd.DataFrame, index: int, number: int
    ) -> pd.DataFrame:
        """
        Slice a portion of the dataframe starting from the given index.

        :param dataframe: The pandas DataFrame to slice
        :param index: The starting index of the slice
        :param number: The number of rows to slice from the dataframe
        :return: a slice of the dataframe starting from the specified index
        """
        return dataframe.iloc[index : index + number]

    def calculate_master_features(self) -> pd.DataFrame:
        """
        Calculate features for the whole timeseries dataframe.

        :return: the "master" dataframe with features for the entire series
        """
        master = pd.DataFrame(
            extract_features(
                self.timeseries_df,
                column_id="id",
                column_sort="time",
                default_fc_parameters=self.parameters,
                disable_progressbar=self.hide_progressbars,
            )
        )
        master["start"] = "all"  # type: ignore[reportArgumentType]
        master["end"] = "all"  # type: ignore[reportArgumentType]
        master["start_time"] = " "  # type: ignore[reportArgumentType]
        master["end_time"] = " "  # type: ignore[reportArgumentType]
        return master

    def create_feature_dataframe(self) -> None:
        """
        Create a feature dataframe using the tsfresh library for time-series data.
        """
        feature_dataframe: pd.DataFrame = pd.DataFrame()
        for i in trange(self.chunks):
            chunk: pd.DataFrame = self.slice_df(self.timeseries_df, i, self.chunk_size)
            if self.plot_chunks:
                chunk.plot("time", "bg_value")
                plt.show()
            if self.display_chunks:
                display(chunk)
            chunk_features = pd.DataFrame(
                extract_features(
                    chunk,
                    column_id="id",
                    column_sort="time",
                    default_fc_parameters=self.parameters,
                    disable_progressbar=self.hide_progressbars,
                )
            )
            chunk_features["start"] = i  # type: ignore[reportArgumentType]
            chunk_features["end"] = i + self.chunk_size - 1  # type: ignore[reportArgumentType]
            chunk_features["start_time"] = self.timeseries_df.loc[i].time  # type: ignore[reportArgumentType]
            chunk_features["end_time"] = self.timeseries_df.loc[  # type: ignore[reportArgumentType]
                i + self.chunk_size - 1
            ].time
            chunk_features["start_time_of_day"] = self.raw_timeseries_df.loc[  # type: ignore[reportArgumentType]
                i
            ].time_of_day
            chunk_features["end_time_of_day"] = self.raw_timeseries_df.loc[  # type: ignore[reportArgumentType]
                i + self.chunk_size - 1
            ].time_of_day
            chunk_features["part_of_day"] = self.raw_timeseries_df.loc[i].part_of_day  # type: ignore[reportArgumentType]
            if self.display_chunk_features:
                display(chunk_features)
                display(pd.melt(chunk_features))  # type: ignore[reportArgumentType]
            feature_dataframe = (
                pd.concat([feature_dataframe, chunk_features]) if i else chunk_features  # type: ignore[reportArgumentType]
            )

        feature_dataframe.reset_index(inplace=True)
        feature_dataframe.drop(["index"], axis=1, inplace=True)
        self.feature_dataframe = feature_dataframe

    def create_target_series(self) -> None:
        """
        Create a target vector by extracting values from the time-series dataframe.
        """
        array: list[Any] = [
            self.timeseries_df.loc[(i + self.chunk_size - 1 + self.horizon)].bg_value
            for i in range(self.chunks)
        ]
        self.target_series = pd.Series(array)

    def create_labeled_dataframe(self) -> None:
        """Create the labeled dataframe by combining features with target values."""
        if self.feature_dataframe.empty:
            self.create_feature_dataframe()
        if self.target_series.empty:
            self.create_target_series()
        f_df: pd.DataFrame = self.feature_dataframe
        t_series: pd.Series = self.target_series
        f_df["label"] = t_series  # type: ignore[reportArgumentType]
        self.labeled_dataframe = f_df
