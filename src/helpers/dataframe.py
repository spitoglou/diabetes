import re

import pandas as pd
from loguru import logger


def save_df(dataframe, filename="test.pkl"):
    logger.info(f"Attempting to save file {filename}")
    dataframe.to_pickle(filename)


def read_df(filename="test.pkl"):
    logger.info(f"Attempting to read from pickle file {filename}")
    return pd.read_pickle(filename)


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame column names for ML model compatibility.

    LightGBM and other models don't support special characters in
    feature names. This function:
    - Replaces spaces with underscores
    - Removes special characters
    - Handles duplicate column names by appending index

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
        col: re.sub(r"[^A-Za-z0-9_]+", "", col.replace(" ", "_")) for col in df.columns
    }

    # Handle duplicate column names by appending index
    new_n_list = list(new_names.values())
    new_names = {
        col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
        for i, (col, new_col) in enumerate(new_names.items())
    }

    df = df.rename(columns=new_names)
    return df
