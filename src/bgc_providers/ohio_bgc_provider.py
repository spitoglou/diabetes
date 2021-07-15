from lxml import objectify
from datetime import datetime, timezone
from loguru import logger
from src.interfaces.bgc_provider_interface import BgcProviderInterface
import pandas as pd
import matplotlib.pyplot as plt


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

    def simulate_glucose_stream(self, shift=0):
        for glucose_event in self.get_glycose_levels(shift):
            logger.info(glucose_event.attrib)
            values = {'time': self.ts_to_timestamp(glucose_event.attrib['ts'])}
            values['value'] = float(glucose_event.attrib['value'])
            # TODO: This is mock
            values['patient'] = self.patient
            yield values
            # sleep(1)

    def tsfresh_dataframe(self, truncate=0, show_plt=False):
        data = self.get_glycose_levels()
        base_time_string = data[0].attrib['ts']
        base_time = datetime.strptime(base_time_string, '%d-%m-%Y %H:%M:%S')
        # print(base_time)
        data_array = []
        for glucose_event in self.get_glycose_levels():
            # print(glucose_event.attrib)
            dtime = datetime.strptime(
                glucose_event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
            delta = dtime - base_time
            array_time = (
                abs(delta.days) * 24 + round(((dtime - base_time).seconds) / 3600, 2))
            array_value = int(glucose_event.attrib['value'])
            data_array.append([array_time, array_value])
        df = pd.DataFrame(data=data_array, columns=['time', 'bg_value'])
        if truncate:
            df = df[:truncate]
        df['id'] = 'a'
        if show_plt:
            df.plot('time', 'bg_value')
            plt.show()
        return df
