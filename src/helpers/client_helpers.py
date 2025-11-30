"""Helper functions for client scripts."""

from typing import Any, Dict

import requests
from loguru import logger

from src.config import Config
from src.helpers.fhir import create_fhir_json_from_reading


def send_reading(
    values: Dict[str, Any],
    config: Config,
) -> requests.Response | None:
    """
    Send a glucose reading to the server.

    Creates a FHIR payload and POSTs it to the configured server endpoint.

    Args:
        values: Glucose reading data with 'value', 'time', 'patient' keys.
        config: Application configuration with server_url and request_timeout.

    Returns:
        Response object if request was made, None if an error occurred.
    """
    if config.debug:
        logger.debug(f"Raw glucose values: {values}")

    payload = create_fhir_json_from_reading(values)
    if config.debug:
        logger.debug(f"FHIR payload: {payload}")

    try:
        response = requests.post(
            f"{config.server_url}/bg/reading",
            data=payload,
            timeout=config.request_timeout,
        )
        if config.debug:
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response body: {response.text}")
        if response.status_code != 200:
            logger.warning(response.text)
        return response
    except requests.RequestException as e:
        logger.error(f"Failed to send reading: {e}")
        return None
