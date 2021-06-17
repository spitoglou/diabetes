from datetime import datetime, timezone
from loguru import logger
from src.interfaces.bgc_provider_interface import BgcProviderInterface
import pandas as pd
import matplotlib.pyplot as plt


class AidaBgcProvider(BgcProviderInterface):

    def __init__(self, aida_no=6359):
        self.timeseries = aida_no
        self.source_file = 'data/aida/glucose' + str(aida_no) + '.dat'

    def get_glycose_levels(self, start=0):
        return pd.read_table(self.source_file, header=None, names=['time', 'bg_value'])[start:]

    def ts_to_datetime(self, ts):
        return datetime.strptime(ts, '%d-%m-%Y %H:%M:%S')

    def ts_to_timestamp(self, ts):
        return self.ts_to_datetime(ts).replace(tzinfo=timezone.utc).timestamp()

    def simulate_glucose_stream(self, shift=0):
        for index, glucose_event in self.get_glycose_levels().iterrows():
            logger.info(glucose_event)
            values = {
                'time': glucose_event['time'],
                'value': float(glucose_event['bg_value']),
                'patient': self.timeseries,
            }

            yield values
            # sleep(1)

    def tsfresh_dataframe(self, truncate=0):
        df = self.get_glycose_levels()
        if truncate:
            df = df[:truncate]
        df['id'] = 'a'
        df.plot('time', 'bg_value')
        plt.show()
        return df
