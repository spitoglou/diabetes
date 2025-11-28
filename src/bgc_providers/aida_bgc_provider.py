from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from src.interfaces.bgc_provider_interface import BgcProviderInterface


class AidaBgcProvider(BgcProviderInterface):
    """Provider for AIDA dataset glucose measurements."""

    def __init__(self, aida_no: int = 6359) -> None:
        self.timeseries: int = aida_no
        self.source_file: str = "data/aida/glucose" + str(aida_no) + ".dat"

    def get_glycose_levels(self, start: int = 0) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_table(
            self.source_file, header=None, names=["time", "bg_value"]
        )
        return df.iloc[start:]

    def ts_to_datetime(self, ts: str) -> datetime:
        return datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")

    def ts_to_timestamp(self, ts: str) -> float:
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).timestamp()

    def simulate_glucose_stream(
        self, shift: int = 0
    ) -> Generator[dict[str, Any], None, None]:
        for index, glucose_event in self.get_glycose_levels().iterrows():
            logger.info(glucose_event)
            values: dict[str, Any] = {
                "time": glucose_event["time"],
                "value": float(glucose_event["bg_value"]),
                "patient": self.timeseries,
            }

            yield values
            # sleep(1)

    def tsfresh_dataframe(
        self, truncate: int = 0, show_plt: bool = False
    ) -> pd.DataFrame:
        df: pd.DataFrame = self.get_glycose_levels()
        if truncate:
            df = df.iloc[:truncate]
        df["id"] = "a"
        if show_plt:
            df.plot("time", "bg_value")
            plt.show()
        return df
