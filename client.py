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
from src.logging_config import setup_logging

config = get_config()
setup_logging(level=config.log_level)

logger.info(f"Logging initialized with level={config.log_level}, debug={config.debug}")

# Streaming interval in seconds (default 5 sec = CGM measurement interval)
INTERVAL = 5


def stream_data(send_to_service: bool = True):
    """
    Stream glucose readings to the server.

    Retrieves readings from the Ohio dataset and sends them as FHIR
    observations to the REST API endpoint.

    Args:
        send_to_service: Whether to actually POST to the service.
    """
    config = get_config()
    patient_id = config.default_patient_id

    logger.info(f"Starting glucose stream for patient {patient_id}")

    provider = OhioBgcProvider(ohio_no=patient_id)
    stream = provider.simulate_glucose_stream()
    try:
        while True:
            values = next(stream)
            if config.debug:
                logger.debug(f"Raw glucose values: {values}")

            payload = create_fhir_json_from_reading(values)
            if config.debug:
                logger.debug(f"FHIR payload: {payload}")

            if send_to_service:
                r = requests.post("http://localhost:8000/bg/reading", data=payload)
                if config.debug:
                    logger.debug(f"Response status: {r.status_code}")
                    logger.debug(f"Response body: {r.text}")
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(values)
            sleep(INTERVAL)
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")


if __name__ == "__main__":
    stream_data(send_to_service=True)
