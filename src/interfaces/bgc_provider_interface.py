import abc
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class BgcProviderInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_glycose_levels(self, start: int = 0) -> Any:
        pass

    @abc.abstractmethod
    def simulate_glucose_stream(
        self, shift: int = 0
    ) -> Generator[dict[str, Any], None, None]:
        pass

    @abc.abstractmethod
    def tsfresh_dataframe(
        self, truncate: int = 0, show_plt: bool = False
    ) -> "pd.DataFrame":
        pass
