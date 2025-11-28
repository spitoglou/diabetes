from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from lxml import objectify

from src.helpers.misc import get_part_of_day
from src.interfaces.bgc_provider_interface import BgcProviderInterface


class OhioBgcProvider(BgcProviderInterface):
    """Provider for Ohio T1DM dataset glucose measurements."""

    def __init__(self, ohio_no: str | int, scope: str = "train") -> None:
        self.patient: str | int = ohio_no
        self.source_file: str = "data/ohio/{0}/{1}-ws-{0}ing.xml".format(scope, ohio_no)
        self.xml: Any = objectify.parse(open(self.source_file))
        self.root: Any = self.xml.getroot()

    def get_glycose_levels(self, start: int = 0) -> Any:
        glucose_levels_xml: Any = self.root.getchildren()[0].getchildren()
        if start > 0:
            glucose_levels_xml = glucose_levels_xml[start:]
        return glucose_levels_xml

    def ts_to_datetime(self, ts: str) -> datetime:
        return datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")

    def ts_to_timestamp(self, ts: str) -> float:
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).timestamp()

    def ts_to_iso(self, ts: str) -> str:
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).isoformat()

    def simulate_glucose_stream(
        self, shift: int = 0, verbose: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        for glucose_event in self.get_glycose_levels(shift):
            logger.info(glucose_event.attrib) if verbose else ...
            values: dict[str, Any] = {
                "timestamp": self.ts_to_timestamp(glucose_event.attrib["ts"])
            }
            values["time"] = self.ts_to_iso(glucose_event.attrib["ts"])
            values["value"] = float(glucose_event.attrib["value"])
            # TODO: This is mock
            values["patient"] = self.patient
            yield values
            # sleep(1)

    def tsfresh_dataframe(
        self, truncate: int = 0, show_plt: bool = False
    ) -> pd.DataFrame:
        """
        Create a tsfresh-compatible dataframe from glucose level data.

        :param truncate: The number of rows to keep in the resulting DataFrame.
            If 0, the DataFrame will not be truncated.
        :param show_plt: If True, display a plot of the glucose data.
        :return: A pandas DataFrame with glucose measurements and time features.
        """
        data: Any = self.get_glycose_levels()
        base_time_string: str = data[0].attrib["ts"]
        base_time: datetime = datetime.strptime(base_time_string, "%d-%m-%Y %H:%M:%S")
        # print(base_time)
        data_array: list[list[Any]] = []
        for glucose_event in self.get_glycose_levels():
            # print(glucose_event.attrib)
            dtime: datetime = datetime.strptime(
                glucose_event.attrib["ts"], "%d-%m-%Y %H:%M:%S"
            )
            time_of_day = dtime.time()
            mock_date = dtime.date()
            part_of_day: str = get_part_of_day(time_of_day.hour)
            delta = dtime - base_time
            array_time: float = abs(delta.days) * 24 + round(
                ((dtime - base_time).seconds) / 3600, 2
            )
            array_value: int = int(glucose_event.attrib["value"])
            data_array.append(
                [dtime, mock_date, time_of_day, part_of_day, array_time, array_value]
            )
        df: pd.DataFrame = pd.DataFrame(
            data=data_array,
            columns=[
                "date_time",
                "mock_date",
                "time_of_day",
                "part_of_day",
                "time",
                "bg_value",
            ],
        )
        if truncate:
            df = df[:truncate]
        df["id"] = "a"
        if show_plt:
            df.plot("time", "bg_value")
        if show_plt:
            df.plot("time", "bg_value")
            plt.show()
        return df
