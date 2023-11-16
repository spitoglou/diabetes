"""
┌─┐┬  ┬┌─┐┌┐┌┌┬┐
│  │  │├┤ │││ │
└─┘┴─┘┴└─┘┘└┘ ┴
Author: Stavros Pitoglou
"""
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.helpers.fhir import create_fhir_json_from_reading
from time import sleep
import requests
from loguru import logger
import json

# logger.disable('')


def stream_data(send_to_service: bool = True, verbose: bool = False):
    provider = OhioBgcProvider()
    stream = provider.simulate_glucose_stream()
    try:
        while True:
            values = next(stream)
            logger.info(values) if verbose else ...
            payload = create_fhir_json_from_reading(values)
            logger.info(payload) if verbose else ...
            if send_to_service:
                r = requests.post("http://localhost:8000/bg/reading", data=payload)
                logger.info(r.status_code) if verbose else ...
                logger.info(r.text) if verbose else ...
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(values)
            sleep(5)
    except KeyboardInterrupt:
        print("Interrupted by the user")


if __name__ == "__main__":
    stream_data(True)
