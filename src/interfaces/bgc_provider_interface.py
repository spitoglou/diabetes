import abc


class BgcProviderInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_glycose_levels(self, start=0):
        pass

    @abc.abstractmethod
    def simulate_glucose_stream(self, shift=0):
        pass

    @abc.abstractmethod
    def tsfresh_dataframe(self, truncate=0):
        pass
