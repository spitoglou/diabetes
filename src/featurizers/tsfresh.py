import pandas as pd
from tqdm import trange
from tsfresh import extract_features
from tsfresh.feature_extraction import (ComprehensiveFCParameters,
                                        MinimalFCParameters)
import matplotlib.pyplot as plt
from loguru import logger
from IPython.display import display


# The TsfreshFeaturizer class is used for feature extraction from time series data using the Tsfresh
# library.
class TsfreshFeaturizer():

    def __init__(self,
                 timeseries_df,
                 chunk_size,
                 horizon,
                 hide_progressbars=True,
                 minimal_features=True,
                 display_chunks=False,
                 display_chunk_features=False,
                 plot_chunks=False) -> None:
        """
        The function initializes an object with various parameters and dataframes for time series
        analysis.
        
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
        self.raw_timeseries_df = timeseries_df
        self.timeseries_df = timeseries_df[['time', 'bg_value', 'id']]
        self.chunk_size = chunk_size
        self.parameters = ComprehensiveFCParameters()
        if minimal_features is True:
            self.parameters = MinimalFCParameters()
        self.hide_progressbars = hide_progressbars
        self.plot_chunks = plot_chunks
        self.display_chunks = display_chunks
        self.display_chunk_features = display_chunk_features
        self.horizon = horizon
        # Determine in how many chunks the dataset will be divided
        # It is calculated [Total Records] - ([Chunk Size] - 1) - [Horizon]
        self.chunks = timeseries_df.shape[0] - chunk_size + 1 - horizon
        # Initialize of parameter dataframes
        self.feature_dataframe = pd.DataFrame()
        self.target_series = pd.DataFrame()
        self.labeled_dataframe = pd.DataFrame()

    def slice_df(self, dataframe, index, number):
        """
        The function `slice_df` takes a dataframe, an index, and a number as input and returns a sliced
        portion of the dataframe starting from the given index and containing the specified number of
        rows.
        
        :param dataframe: The dataframe parameter is the pandas DataFrame that you want to slice
        :param index: The starting index of the slice. It specifies the position of the first element to
        include in the slice
        :param number: The number of rows to slice from the dataframe starting from the specified index
        :return: a slice of the dataframe starting from the specified index and containing the specified
        number of rows.
        """
        return dataframe[index:index + number]

    def calculate_master_features(self):
        """
        The function calculates the features for the whole timeseries dataframe.
        :return: the "master" dataframe that will be used - if needed - in combination with the particulat chunk features.
        """
        master = extract_features(self.timeseries_df, column_id="id", column_sort="time",
                                  default_fc_parameters=self.parameters,
                                  disable_progressbar=self.hide_progressbars)
        master['start'] = 'all'
        master['end'] = 'all'
        master['start_time'] = ' '
        master['end_time'] = ' '
        return master

    def create_feature_dataframe(self):
        """
        The function `create_feature_dataframe` creates a feature dataframe using the tsfresh library
        for time-series data.
        """
        feature_dataframe = pd.DataFrame()
        for i in trange(self.chunks):
            chunk = self.slice_df(self.timeseries_df, i, self.chunk_size)
            if self.plot_chunks:
                chunk.plot('time', 'bg_value')
                plt.show()
            if self.display_chunks:
                display(chunk)
            chunk_features = extract_features(chunk,
                                              column_id="id",
                                              column_sort="time",
                                              default_fc_parameters=self.parameters,
                                              disable_progressbar=self.hide_progressbars)
            chunk_features['start'] = i
            chunk_features['end'] = i + self.chunk_size - 1
            chunk_features['start_time'] = self.timeseries_df.loc[i].time
            chunk_features['end_time'] = (
                self.timeseries_df.loc[i + self.chunk_size - 1].time)
            chunk_features['start_time_of_day'] = self.raw_timeseries_df.loc[i].time_of_day
            chunk_features['end_time_of_day'] = (
                self.raw_timeseries_df.loc[i + self.chunk_size - 1].time_of_day)
            chunk_features['part_of_day'] = self.raw_timeseries_df.loc[i].part_of_day
            if self.display_chunk_features:
                display(chunk_features)
                display(pd.melt(chunk_features))
            feature_dataframe = pd.concat([feature_dataframe, chunk_features]) if i else chunk_features

        feature_dataframe.reset_index(inplace=True)
        feature_dataframe.drop(['index'], axis=1, inplace=True)
        self.feature_dataframe = feature_dataframe

    def create_target_series(self):
        """
        The function `create_target_series` creates a target vector by extracting values from a
        time-series dataframe based on specified parameters.
        """
        array = [self.timeseries_df.loc[(
            i + self.chunk_size - 1 + self.horizon)].bg_value for i in range(self.chunks)]
        self.target_series = pd.Series(array)

    def create_labeled_dataframe(self):
        if self.feature_dataframe.empty:
            self.create_feature_dataframe()
        if self.target_series.empty:
            self.create_target_series()
        f_df = self.feature_dataframe
        t_series = self.target_series
        f_df['label'] = t_series
        self.labeled_dataframe = f_df
