"""
┌─┐┬  ┬┌─┐┌┐┌┌┬┐
│  │  │├┤ │││ │
└─┘┴─┘┴└─┘┘└┘ ┴
Author: Stavros Pitoglou

Client for streaming glucose data to the server.
"""

from time import sleep

import requests
from loguru import logger

from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.config import get_config
from src.helpers.fhir import create_fhir_json_from_reading

# Streaming interval in seconds (default 5 sec = CGM measurement interval)
INTERVAL = 5


def stream_data(send_to_service: bool = True, verbose: bool = False):
    """
    Stream glucose readings to the server.

    Retrieves readings from the Ohio dataset and sends them as FHIR
    observations to the REST API endpoint.

    Args:
        send_to_service: Whether to actually POST to the service.
        verbose: Enable verbose logging.
    """
    config = get_config()
    patient_id = config.default_patient_id

    logger.info(f"Starting glucose stream for patient {patient_id}")

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
                r = requests.post("http://localhost:8000/bg/reading", data=payload)
                logger.info(r.status_code) if verbose else ...
                logger.info(r.text) if verbose else ...
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(values)
            sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted by the user")


if __name__ == "__main__":
    stream_data(send_to_service=True, verbose=False)
