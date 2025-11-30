"""
┌─┐┬  ┬┌─┐┌┐┌┌┬┐  ┌┬┐┬ ┬┬┌┐┌
│  │  │├┤ │││ │    │ ││││││││
└─┘┴─┘┴└─┘┘└┘ ┴    ┴ └┴┘┴┘└┘
Author: Stavros Pitoglou

Time-synchronized client for streaming glucose data as a digital twin.
Starts streaming from a point in the dataset matching the current time of day.
"""

from datetime import datetime, timezone
from time import sleep

import requests
from loguru import logger

from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.config import get_config
from src.helpers.fhir import create_fhir_json_from_reading
from src.logging_config import setup_logging

# Streaming interval in seconds (default 5 sec = CGM measurement interval)
INTERVAL = 5


def find_time_matched_index(provider: OhioBgcProvider, target_time: datetime) -> int:
    """
    Find the index of the first glucose reading closest to the target time of day.

    Args:
        provider: The BGC provider with glucose data.
        target_time: The target datetime (only hour:minute is used).

    Returns:
        Index of the closest matching reading in the dataset.
    """
    target_minutes = target_time.hour * 60 + target_time.minute
    glucose_levels = provider.get_glycose_levels()

    best_index = 0
    best_diff = float("inf")

    for i, glucose_event in enumerate(glucose_levels):
        event_time = provider.ts_to_datetime(glucose_event.attrib["ts"])
        event_minutes = event_time.hour * 60 + event_time.minute

        # Calculate difference accounting for day wrap-around
        diff = abs(event_minutes - target_minutes)
        diff = min(diff, 1440 - diff)  # 1440 = minutes in a day

        if diff < best_diff:
            best_diff = diff
            best_index = i

        # If we find an exact or very close match, use it
        if diff <= 2:  # Within 2 minutes
            break

    return best_index


def create_twin_timestamp(original_ts: str, current_date: datetime) -> str:
    """
    Replace the date in the original timestamp with the current date.

    Args:
        original_ts: Original timestamp string from dataset (DD-MM-YYYY HH:MM:SS).
        current_date: Current datetime to use for the date portion.

    Returns:
        ISO format timestamp with current date and original time.
    """
    original_dt = datetime.strptime(original_ts, "%d-%m-%Y %H:%M:%S")
    twin_dt = original_dt.replace(
        year=current_date.year,
        month=current_date.month,
        day=current_date.day,
        tzinfo=timezone.utc,
    )
    return twin_dt.isoformat()


def stream_data_twin(send_to_service: bool = True):
    """
    Stream glucose readings synchronized to the current time of day.

    Finds a starting point in the dataset matching the current time,
    then streams data with timestamps adjusted to the current date.
    This simulates a "digital twin" of the patient in real-time.

    Args:
        send_to_service: Whether to actually POST to the service.
    """
    config = get_config()
    patient_id = config.default_patient_id
    now = datetime.now()

    logger.info(f"Starting digital twin stream for patient {patient_id}")
    logger.info(f"Current time: {now.strftime('%H:%M:%S')}")

    provider = OhioBgcProvider(ohio_no=patient_id)

    # Find starting index matching current time of day
    start_index = find_time_matched_index(provider, now)
    glucose_levels = provider.get_glycose_levels()
    start_ts = glucose_levels[start_index].attrib["ts"]
    start_time = provider.ts_to_datetime(start_ts)

    logger.info(
        f"Found matching data at index {start_index}, "
        f"original time: {start_time.strftime('%H:%M:%S')}"
    )

    # Stream from the matched position
    stream = provider.simulate_glucose_stream(shift=start_index)

    try:
        for values in stream:
            # Replace with current date
            original_ts = glucose_levels[start_index].attrib["ts"]
            values["time"] = create_twin_timestamp(original_ts, now)
            values["timestamp"] = datetime.fromisoformat(values["time"]).timestamp()

            if config.debug:
                logger.debug(f"Twin glucose values: {values}")

            payload = create_fhir_json_from_reading(values)
            if config.debug:
                logger.debug(f"FHIR payload: {payload}")

            if send_to_service:
                r = requests.post(f"{config.server_url}/bg/reading", data=payload)
                if config.debug:
                    logger.debug(f"Response status: {r.status_code}")
                    logger.debug(f"Response body: {r.text}")
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(f"BG: {values['value']} mg/dL @ {values['time']}")

            start_index += 1
            sleep(INTERVAL)

    except StopIteration:
        logger.info("Reached end of dataset")
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")


def main():
    """CLI entry point for the digital twin client."""
    config = get_config()
    setup_logging(level=config.log_level)
    stream_data_twin(send_to_service=True)


if __name__ == "__main__":
    main()
