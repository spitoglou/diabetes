"""Factory function for creating blood glucose data providers.

Provides a unified interface for instantiating either Ohio (real patient)
or simglucose (synthetic) data providers based on configuration.
"""

from typing import Literal

from loguru import logger

from config.settings import settings
from src.interfaces.bgc_provider_interface import BgcProviderInterface

DataSource = Literal["ohio", "simglucose"]


def create_provider(
    data_source: DataSource,
    patient: str,
    scope: str = "train",
    simulation_days: int | None = None,
    insulin_mode: str | None = None,
) -> BgcProviderInterface:
    """Create a blood glucose data provider based on the data source.

    Args:
        data_source: Either "ohio" for real patient data or "simglucose" for synthetic.
        patient: Patient identifier.
            - For Ohio: numeric ID like "559", "570", "588", etc.
            - For simglucose: virtual patient name like "adult#001", "adolescent#002".
        scope: Data scope for Ohio provider ("train" or "test"). Ignored for simglucose.
        simulation_days: Number of days to simulate for simglucose (default: 14).
                        Ignored for Ohio provider.
        insulin_mode: Insulin mode for simglucose ("none", "basal", "basal-bolus").
                     Default: "basal-bolus" for training. Ignored for Ohio provider.

    Returns:
        BgcProviderInterface implementation for the specified data source.

    Raises:
        ValueError: If data_source is not "ohio" or "simglucose".

    Example:
        >>> provider = create_provider("ohio", "559")
        >>> df = provider.tsfresh_dataframe()

        >>> provider = create_provider("simglucose", "adult#001", simulation_days=14)
        >>> df = provider.tsfresh_dataframe()
    """
    if data_source == "ohio":
        from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider

        logger.info(f"Creating Ohio provider for patient {patient} (scope={scope})")
        return OhioBgcProvider(scope=scope, ohio_no=patient)

    elif data_source == "simglucose":
        from src.bgc_providers.simglucose_provider import SimglucoseProvider

        # Insulin mode defaults to basal-bolus for training (per design decision)
        # This differs from streaming default (basal) in settings
        mode = insulin_mode if insulin_mode is not None else "basal-bolus"

        # Simulation days defaults to 14 per design decision
        days = simulation_days if simulation_days is not None else 14

        logger.info(
            f"Creating Simglucose provider for patient {patient} "
            f"(insulin_mode={mode}, simulation_days={days})"
        )
        return SimglucoseProvider(
            patient_name=patient,
            insulin_mode=mode,
        )

    else:
        raise ValueError(
            f"Unknown data_source: {data_source}. Must be 'ohio' or 'simglucose'."
        )


def get_sample_interval(data_source: DataSource) -> int:
    """Get the CGM sample interval for a data source.

    Args:
        data_source: Either "ohio" or "simglucose".

    Returns:
        Sample interval in minutes:
        - Ohio: 5 minutes (Guardian sensor)
        - Simglucose: 3 minutes (Dexcom sensor)
    """
    if data_source == "ohio":
        return 5
    elif data_source == "simglucose":
        return 3
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
