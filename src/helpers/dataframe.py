from __future__ import annotations

import pandas as pd
from loguru import logger


def save_df(dataframe: pd.DataFrame, filename: str = "test.pkl") -> None:
    """Save a DataFrame to a pickle file.

    Args:
        dataframe: DataFrame to save
        filename: Path to the output pickle file
    """
    logger.info(f"Attempting to save file {filename}")
    dataframe.to_pickle(filename)


def read_df(filename: str = "test.pkl") -> pd.DataFrame:
    """Read a DataFrame from a pickle file.

    Args:
        filename: Path to the pickle file to read

    Returns:
        The loaded DataFrame
    """
    logger.info(f"Attempting to read from pickle file {filename}")
    return pd.read_pickle(filename)
