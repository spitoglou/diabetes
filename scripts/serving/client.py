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


if __name__ == "__main__":
    stream_data(send_to_service=True, verbose=False)
