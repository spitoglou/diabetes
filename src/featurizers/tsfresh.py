import pandas as pd
from tqdm import trange
from tsfresh import extract_features
from tsfresh.feature_extraction import (ComprehensiveFCParameters,
                                        MinimalFCParameters)
import matplotlib.pyplot as plt


class TsfreshFeaturizer():

    def __init__(self,
                 timeseries_df,
                 chunk_size,
                 horizon,
                 hide_progressbars=True,
                 minimal_features=True,
                 plot_chunks=False) -> None:
        self.raw_timeseries_df = timeseries_df
        self.timeseries_df = timeseries_df[['time', 'bg_value', 'id']]
        self.chunks = timeseries_df.shape[0] - chunk_size + 1 - horizon
        self.chunk_size = chunk_size
        self.parameters = ComprehensiveFCParameters()
        if minimal_features is True:
            self.parameters = MinimalFCParameters()
        self.hide_progressbars = hide_progressbars
        self.plot_chunks = plot_chunks
        self.horizon = horizon
        self.feature_dataframe = None
        self.target_series = None
        self.labeled_dataframe = None

    def slice_df(self, dataframe, index, number):
        return dataframe[index:index + number]

    def calculate_master_features(self):
        master = extract_features(self.timeseries_df, column_id="id", column_sort="time",
                                  default_fc_parameters=self.parameters,
                                  disable_progressbar=self.hide_progressbars)
        master['start'] = 'all'
        master['end'] = 'all'
        master['start_time'] = ' '
        master['end_time'] = ' '
        return master

    def create_feature_dataframe(self):
        """Creates Feaure Dataframe using tsfresh

        Parameters:
            initial_df (pandas dataframe): Initial Dataframe containing the time-series (pandas)
            size (int): the size of the time "chunks" that will be used for feature extraction
            chunks (int or 'auto)': # of chunks ('auto' for automatic calculation based on the dataframe row count)
            window (int): the prediction window
            hide_progressbars (bool): if tsfresh progress bars are to be hidden
            minimal_features (bool): if True tsfresh MinimalFCParameters() will be used
            plot_chunks (bool): if plots for every time chunk are to be plotted

        Returns:
            Feature dataframe (pandas)
        """
        master = pd.DataFrame()
        for i in trange(self.chunks):
            chunk = self.slice_df(self.timeseries_df, i, self.chunk_size)
            if self.plot_chunks:
                chunk.plot('time', 'bg_value')
                plt.show()
            # display(chunk)
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

            # display(chunk_features)
            master = pd.concat([master, chunk_features]) if i else chunk_features

            # master.loc[i] = chunk_features

        master.reset_index(inplace=True)
        master.drop(['index'], axis=1, inplace=True)
        self.feature_dataframe = master
        # return master

    def create_target_series(self):
        """Creates target vector

        Parameters:
            initial_df (pandas dataframe): Initial Dataframe containing the time-series (pandas)
            size (int): the size of the time "chunks" that will be used for feature extraction
            chunks (int or 'auto)': # of chunks ('auto' for automatic calculation based on the dataframe row count)
            window (int): the prediction window

        Returns:
            Target Values (pandas series)
        """

        # if chunks == 'auto':
        #     chunks = initial_df.shape[0] - size + 1 - window
        array = [self.timeseries_df.loc[(
            i + self.chunk_size - 1 + self.horizon)].bg_value for i in range(self.chunks)]
        self.target_series = pd.Series(array)

    def create_labeled_dataframe(self):
        if not self.feature_dataframe:
            self.create_feature_dataframe()
        if not self.target_series:
            self.create_target_series()
        df = self.feature_dataframe
        df['label'] = self.target_series
        self.labeled_dataframe = df
