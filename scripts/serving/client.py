"""
┌─┐┬  ┬┌─┐┌┐┌┌┬┐
│  │  │├┤ │││ │
└─┘┴─┘┴└─┘┘└┘ ┴
Author: Stavros Pitoglou
"""

from time import sleep

import requests
from loguru import logger

from config.settings import settings
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.helpers.fhir import create_fhir_json_from_reading


def stream_data(
    send_to_service: bool = True, verbose: bool = False, patient: str | None = None
):
    """Συνάρτηση ανάκτησης, προετοιμασίας και αποστολής σειράς μετρήσεων
        για το simulation και τη δοκιμή γεννήτριας μετρήσεων CGM

    Args:
        send_to_service (bool, optional): Διακόπτης τελικής αποστολής στο service. Defaults to True.
        verbose (bool, optional): Διακόπτης εκτεταμένων μηνυμάτων εκτέλεσης. Defaults to False.
        patient (str, optional): Patient ID to stream data for. Defaults to settings.OHIO_ID.
    """

    # Ορισμός της μεθόδου streaming από τον αντίστοιχο provider
    patient_id = patient or settings.OHIO_ID
    provider = OhioBgcProvider(ohio_no=patient_id)
    stream = provider.simulate_glucose_stream()
    try:
        # Εκτέλεση ατέρμονου βρόχου έως την ακύρωση από το χρήστη
        while True:
            # ανάκτηση επόμενης μέτρησης
            values = next(stream)
            logger.info(values) if verbose else ...
            # κλήση μεθόδου μετατροπής της μέτρησης σε αντικέιμενο FHIR
            payload = create_fhir_json_from_reading(values)
            logger.info(payload) if verbose else ...
            # αποστολή στο RESTful endpoint του service (εφόσον είναι ενεργοποιημένη)
            if send_to_service:
                r = requests.post(
                    f"http://localhost:{settings.PORT}/bg/reading", data=payload
                )
                logger.info(r.status_code) if verbose else ...
                logger.info(r.text) if verbose else ...
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(values)
            sleep(settings.INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted by the user")


def stream_synced_data(
    send_to_service: bool = True, verbose: bool = False, patient: str | None = None
):
    """Stream CGM data with current system timestamps.

    Similar to stream_data(), but uses simulate_synced_glucose_stream() which
    starts from the dataset reading closest to the current time of day and
    sends readings with current system timestamps instead of historical ones.

    Args:
        send_to_service: Whether to send data to the FastAPI service. Defaults to True.
        verbose: Whether to log detailed output. Defaults to False.
        patient: Patient ID to stream data for. Defaults to settings.OHIO_ID.
    """
    patient_id = patient or settings.OHIO_ID
    provider = OhioBgcProvider(ohio_no=patient_id)
    stream = provider.simulate_synced_glucose_stream(verbose=verbose)

    try:
        while True:
            values = next(stream)
            logger.info(values) if verbose else ...

            payload = create_fhir_json_from_reading(values)
            logger.info(payload) if verbose else ...

            if send_to_service:
                r = requests.post(
                    f"http://localhost:{settings.PORT}/bg/reading", data=payload
                )
                logger.info(r.status_code) if verbose else ...
                logger.info(r.text) if verbose else ...
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(values)
            sleep(settings.INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted by the user")


def stream_simglucose_data(
    send_to_service: bool = True,
    verbose: bool = False,
    patient: str | None = None,
    seed: int | None = None,
    sync: bool = True,
    insulin_mode: str | None = None,
    basal_rate: float | None = None,
    target_glucose: float | None = None,
    carb_ratio: float | None = None,
    correction_factor: float | None = None,
    pre_bolus_minutes: int | None = None,
):
    """Stream synthetic CGM data from simglucose simulation.

    Uses the simglucose library to generate realistic glucose readings from
    FDA-approved UVA/Padova virtual patient models.

    Args:
        send_to_service: Whether to send data to the FastAPI service. Defaults to True.
        verbose: Whether to log detailed output. Defaults to False.
        patient: Virtual patient name (e.g., "adult#001"). Defaults to settings.
        seed: Random seed for reproducibility. Defaults to settings.
        sync: If True, fast-forward to current time of day. Defaults to True.
        insulin_mode: Insulin delivery mode - "none", "basal", or "basal-bolus".
                     Defaults to settings.SIMGLUCOSE_INSULIN_MODE ("basal").
        basal_rate: Basal insulin rate override in U/hr. None = patient default.
        target_glucose: Target glucose for corrections in mg/dL. None = 140.
        carb_ratio: Carbohydrate ratio override in g/U. None = patient default.
        correction_factor: Correction factor override in mg/dL/U. None = patient default.
        pre_bolus_minutes: Minutes before meal to deliver bolus. None = 0.
    """
    from src.bgc_providers.simglucose_provider import SimglucoseProvider

    provider = SimglucoseProvider(
        patient_name=patient,
        seed=seed,
        insulin_mode=insulin_mode,
        basal_rate=basal_rate,
        target_glucose=target_glucose,
        carb_ratio=carb_ratio,
        correction_factor=correction_factor,
        pre_bolus_minutes=pre_bolus_minutes,
    )
    stream = provider.simulate_glucose_stream(
        sync_to_current_time=sync, verbose=verbose
    )

    try:
        while True:
            values = next(stream)

            payload = create_fhir_json_from_reading(values)
            logger.info(payload) if verbose else ...

            if send_to_service:
                r = requests.post(
                    f"http://localhost:{settings.PORT}/bg/reading", data=payload
                )
                logger.info(r.status_code) if verbose else ...
                logger.info(r.text) if verbose else ...
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(values)
            sleep(settings.SIMGLUCOSE_INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted by the user")


if __name__ == "__main__":
    stream_data(send_to_service=True, verbose=False)
