import pandas as pd
from loguru import logger


def save_df(dataframe, filename='test.pkl'):
    logger.info(f'Attempting to save file {filename}')
    dataframe.to_pickle(filename)


def read_df(filename='test.pkl'):
    logger.info(f'Attempting to read from pickle file {filename}')
    return pd.read_pickle(filename)
