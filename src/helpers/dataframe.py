import pandas as pd


def save_df(dataframe, filename='test.pkl'):
    dataframe.to_pickle(filename)


def read_df(filename='test.pkl'):
    return pd.read_pickle(filename)
