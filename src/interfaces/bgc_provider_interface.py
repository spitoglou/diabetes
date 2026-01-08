from __future__ import annotations

import abc
from typing import Any, Iterator

import pandas as pd


class BgcProviderInterface(metaclass=abc.ABCMeta):
    """Abstract interface for blood glucose concentration data providers."""

    @abc.abstractmethod
    def get_glycose_levels(self, start: int = 0) -> Any:
        """Get glucose level data starting from a specific index.

        Args:
            start: Starting index for data retrieval

        Returns:
            Glucose level data (implementation-specific format)
        """
        pass

    @abc.abstractmethod
    def simulate_glucose_stream(self, shift: int = 0) -> Iterator[dict[str, Any]]:
        """Generate a stream of glucose readings.

        Args:
            shift: Number of readings to skip from the start

        Yields:
            Dictionary containing glucose reading data
        """
        pass

    @abc.abstractmethod
    def tsfresh_dataframe(self, truncate: int = 0) -> pd.DataFrame:
        """Create a DataFrame suitable for tsfresh feature extraction.

        Args:
            truncate: Number of rows to keep (0 = no truncation)

        Returns:
            DataFrame with time series data formatted for tsfresh
        """
        pass
