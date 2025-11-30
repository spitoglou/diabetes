"""Blood glucose provider interface.

This module defines the abstract interface for blood glucose data providers,
enabling different data sources (Ohio T1DM, AIDA) to be used interchangeably.
"""

from __future__ import annotations

import abc
from collections.abc import Generator
from typing import Any

import pandas as pd


class BgcProviderInterface(metaclass=abc.ABCMeta):
    """Abstract interface for blood glucose data providers.

    Implementations must provide methods for retrieving glucose levels,
    simulating real-time data streams, and creating tsfresh-compatible DataFrames.
    """

    @abc.abstractmethod
    def get_glycose_levels(self, start: int = 0) -> Any:
        """Retrieve glucose level data starting from a given index."""

    @abc.abstractmethod
    def simulate_glucose_stream(
        self, shift: int = 0
    ) -> Generator[dict[str, Any], None, None]:
        """Generate a stream of glucose readings for simulation."""

    @abc.abstractmethod
    def tsfresh_dataframe(
        self, truncate: int = 0, show_plt: bool = False
    ) -> pd.DataFrame:
        """Create a tsfresh-compatible DataFrame from glucose data."""
