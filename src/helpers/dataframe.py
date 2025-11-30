"""DataFrame utility functions.

This module provides helper functions for DataFrame operations including
saving/loading pickle files and sanitizing column names for ML compatibility.
"""

from __future__ import annotations

import re

import pandas as pd
from loguru import logger


def save_df(dataframe: pd.DataFrame, filename: str = "test.pkl") -> None:
    """
    Save a DataFrame to a pickle file.

    Args:
        dataframe: DataFrame to save.
        filename: Output file path.
    """
    logger.info(f"Attempting to save file {filename}")
    dataframe.to_pickle(filename)


def read_df(filename: str = "test.pkl") -> pd.DataFrame:
    """
    Read a DataFrame from a pickle file.

    Args:
        filename: Input file path.

    Returns:
        Loaded DataFrame.
    """
    logger.info(f"Attempting to read from pickle file {filename}")
    result = pd.read_pickle(filename)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(result)}")
    return result


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
    new_names: dict[str, str] = {
        col: re.sub(r"[^A-Za-z0-9_]+", "", col.replace(" ", "_")) for col in df.columns
    }

    # Handle duplicate column names by appending index
    new_n_list: list[str] = list(new_names.values())
    new_names = {
        col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
        for i, (col, new_col) in enumerate(new_names.items())
    }

    df = df.rename(columns=new_names)
    return df
