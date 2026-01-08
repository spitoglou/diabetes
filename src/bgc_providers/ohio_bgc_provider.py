from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from lxml import objectify

from src.helpers.misc import get_part_of_day
from src.interfaces.bgc_provider_interface import BgcProviderInterface


class OhioBgcProvider(BgcProviderInterface):
    def __init__(self, scope="train", ohio_no="559"):
        self.patient = ohio_no
        self.source_file = "data/ohio/{0}/{1}-ws-{0}ing.xml".format(scope, ohio_no)
        self.xml = objectify.parse(open(self.source_file))
        self.root = self.xml.getroot()

    def _find_closest_time_index(self, target_time):
        """Find the index of the glucose reading closest to the target time of day.

        Uses circular time distance to handle midnight wrap-around properly.

        Args:
            target_time: datetime.time object representing the target time of day

        Returns:
            Index of the closest glucose reading
        """
        glucose_levels = self.get_glycose_levels()
        target_minutes = target_time.hour * 60 + target_time.minute

        closest_index = 0
        min_distance = float("inf")

        for i, glucose_event in enumerate(glucose_levels):
            reading_time = self.ts_to_datetime(glucose_event.attrib["ts"]).time()
            reading_minutes = reading_time.hour * 60 + reading_time.minute

            # Calculate circular distance (handles midnight wrap-around)
            direct_diff = abs(target_minutes - reading_minutes)
            circular_diff = min(direct_diff, 1440 - direct_diff)

            if circular_diff < min_distance:
                min_distance = circular_diff
                closest_index = i

        return closest_index

    def simulate_synced_glucose_stream(self, verbose=False):
        """Simulate a glucose stream with current system timestamps.

        Finds the closest time index to current time, then starts streaming from
        that point with real-time timestamps. Wraps around to the beginning of
        the dataset when reaching the end.

        Args:
            verbose: If True, log each glucose event

        Yields:
            Dict with timestamp, time (ISO), value, and patient ID
        """
        glucose_levels = self.get_glycose_levels()

        # Find starting index based on current time
        current_time = datetime.now(timezone.utc).time()
        start_index = self._find_closest_time_index(current_time)
        logger.info(f"Starting synced stream from index {start_index}")

        index = start_index
        while True:
            glucose_event = glucose_levels[index]
            logger.info(glucose_event.attrib) if verbose else ...

            # Use current system time instead of historical timestamp
            now = datetime.now(timezone.utc)
            values = {
                "timestamp": now.timestamp(),
                "time": now.isoformat(),
                "value": float(glucose_event.attrib["value"]),
                "patient": self.patient,
            }
            yield values

            # Move to next reading, wrap around if at end
            index = (index + 1) % len(glucose_levels)

    def get_glycose_levels(self, start=0):
        glucose_levels_xml = self.root.getchildren()[0].getchildren()
        if start > 0:
            glucose_levels_xml = glucose_levels_xml[start:]
        return glucose_levels_xml

    def ts_to_datetime(self, ts):
        return datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")

    def ts_to_timestamp(self, ts):
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).timestamp()

    def ts_to_iso(self, ts):
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).isoformat()

    def simulate_glucose_stream(self, shift=0, verbose=False):
        for glucose_event in self.get_glycose_levels(shift):
            logger.info(glucose_event.attrib) if verbose else ...
            values = {"timestamp": self.ts_to_timestamp(glucose_event.attrib["ts"])}
            values["time"] = self.ts_to_iso(glucose_event.attrib["ts"])
            values["value"] = float(glucose_event.attrib["value"])
            # TODO: This is mock
            values["patient"] = self.patient
            yield values
            # sleep(1)

    def tsfresh_dataframe(self, truncate=0, show_plt=False):
        """
        The function `tsfresh_dataframe` takes in glucose level data, processes it, and returns a pandas
        DataFrame with additional columns for date, time, part of day, and time difference from a base
        time.

        :param truncate: The `truncate` parameter is used to specify the number of rows to keep in the
        resulting DataFrame. If a value is provided, the DataFrame will be truncated to that number of
        rows. If no value is provided or if the value is 0, the DataFrame will not be truncated,
        defaults to 0 (optional)
        :param show_plt: The `show_plt` parameter is a boolean flag that determines whether or not to
        display a plot of the data using `matplotlib.pyplot`. If `show_plt` is set to `True`, the
        function will generate a plot of the 'bg_value' column against the 'time' column and, defaults
        to False (optional)
        :return: a pandas DataFrame object.
        """
        data = self.get_glycose_levels()
        base_time_string = data[0].attrib["ts"]
        base_time = datetime.strptime(base_time_string, "%d-%m-%Y %H:%M:%S")
        # print(base_time)
        data_array = []
        for glucose_event in self.get_glycose_levels():
            # print(glucose_event.attrib)
            dtime = datetime.strptime(glucose_event.attrib["ts"], "%d-%m-%Y %H:%M:%S")
            time_of_day = dtime.time()
            mock_date = dtime.date()
            part_of_day = get_part_of_day(time_of_day.hour)
            delta = dtime - base_time
            array_time = abs(delta.days) * 24 + round(
                ((dtime - base_time).seconds) / 3600, 2
            )
            array_value = int(glucose_event.attrib["value"])
            data_array.append(
                [dtime, mock_date, time_of_day, part_of_day, array_time, array_value]
            )
        df = pd.DataFrame(
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
