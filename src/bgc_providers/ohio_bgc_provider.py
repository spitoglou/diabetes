from lxml import objectify
from datetime import datetime, timezone
from loguru import logger
from src.interfaces.bgc_provider_interface import BgcProviderInterface
import pandas as pd
import matplotlib.pyplot as plt


def get_part_of_day(hour):
    """
    The function `get_part_of_day` returns a string indicating the part of the day based on the given
    hour.
    
    :param hour: The parameter "hour" represents the hour of the day in a 24-hour format
    :return: a string indicating the part of the day based on the given hour.
    """
    return (
        "morning" if 7 <= hour <= 11
        else
        "afternoon" if 12 <= hour <= 15
        else
        "evening" if 16 <= hour <= 20
        else
        "night" if 21 <= hour <= 23
        else
        "late night"
    )


class OhioBgcProvider(BgcProviderInterface):

    def __init__(self, scope='train', ohio_no='559'):
        self.patient = ohio_no
        self.source_file = 'data/ohio/{0}/{1}-ws-{0}ing.xml'.format(
            scope, ohio_no)
        self.xml = objectify.parse(open(self.source_file))
        self.root = self.xml.getroot()

    def get_glycose_levels(self, start=0):
        glucose_levels_xml = self.root.getchildren()[0].getchildren()
        if start > 0:
            glucose_levels_xml = glucose_levels_xml[start:]
        return glucose_levels_xml

    def ts_to_datetime(self, ts):
        return datetime.strptime(ts, '%d-%m-%Y %H:%M:%S')

    def ts_to_timestamp(self, ts):
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).timestamp()
    
    def ts_to_iso(self, ts):
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).isoformat()

    def simulate_glucose_stream(self, shift=0):
        for glucose_event in self.get_glycose_levels(shift):
            logger.info(glucose_event.attrib)
            values = {'timestamp': self.ts_to_timestamp(glucose_event.attrib['ts'])}
            values['time'] = self.ts_to_iso(glucose_event.attrib['ts'])
            values['value'] = float(glucose_event.attrib['value'])
            # TODO: This is mock
            values['patient'] = self.patient
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
        base_time_string = data[0].attrib['ts']
        base_time = datetime.strptime(base_time_string, '%d-%m-%Y %H:%M:%S')
        # print(base_time)
        data_array = []
        for glucose_event in self.get_glycose_levels():
            # print(glucose_event.attrib)
            dtime = datetime.strptime(
                glucose_event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
            time_of_day = dtime.time()
            mock_date = dtime.date()
            part_of_day = get_part_of_day(time_of_day.hour)
            delta = dtime - base_time
            array_time = (
                abs(delta.days) * 24 + round(((dtime - base_time).seconds) / 3600, 2))
            array_value = int(glucose_event.attrib['value'])
            data_array.append([dtime, mock_date, time_of_day, part_of_day, array_time, array_value])
        df = pd.DataFrame(data=data_array, columns=['date_time', 'mock_date', 'time_of_day', 'part_of_day', 'time', 'bg_value'])
        if truncate:
            df = df[:truncate]
        df['id'] = 'a'
        if show_plt:
            df.plot('time', 'bg_value')
            plt.show()
        return df
