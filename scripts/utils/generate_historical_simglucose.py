"""
Generate historical simglucose data for a specified date range.

This script runs a simglucose simulation and generates CGM readings with
historical timestamps, then sends them to the FastAPI server for storage.

Usage:
    uv run python scripts/utils/generate_historical_simglucose.py

The script generates data for December 2025 by default.
"""

from datetime import datetime, timezone
from time import time

import requests
from loguru import logger

from config.settings import settings
from src.bgc_providers.simglucose_provider import SimglucoseProvider
from src.helpers.fhir import create_fhir_json_from_reading


def generate_historical_data(
    patient: str = "adult#001",
    start_date: datetime = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
    end_date: datetime = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    seed: int = 42,
    insulin_mode: str = "basal-bolus",
    send_to_server: bool = True,
    batch_size: int = 100,
) -> list[dict]:
    """
    Generate historical simglucose data for a date range.

    Args:
        patient: Virtual patient name (e.g., "adult#001")
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        seed: Random seed for reproducibility
        insulin_mode: Insulin mode ("none", "basal", "basal-bolus")
        send_to_server: Whether to send data to the FastAPI server
        batch_size: Log progress every N readings

    Returns:
        List of all generated readings
    """
    # Calculate simulation parameters
    total_minutes = (end_date - start_date).total_seconds() / 60
    sample_time = 3  # Dexcom sensor: 3 minutes
    total_readings = int(total_minutes / sample_time)

    logger.info(f"Generating historical data for {patient}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total readings to generate: {total_readings:,}")
    logger.info(f"Insulin mode: {insulin_mode}")
    logger.info(f"Seed: {seed}")

    # Create provider
    provider = SimglucoseProvider(
        patient_name=patient,
        seed=seed,
        insulin_mode=insulin_mode,
    )

    # Get the simulation stream (no sync, start from midnight)
    stream = provider.simulate_glucose_stream(sync_to_current_time=False, verbose=False)

    readings = []
    sent_count = 0
    failed_count = 0
    start_time = time()

    # Current timestamp tracker (start from start_date)
    current_time = start_date

    try:
        for i in range(total_readings):
            # Get next glucose value from simulation
            sim_reading = next(stream)

            # Replace timestamp with historical date
            reading = {
                "timestamp": current_time.timestamp(),
                "time": current_time.isoformat(),
                "value": sim_reading["value"],
                "patient": patient,
            }
            readings.append(reading)

            # Send to server
            if send_to_server:
                payload = create_fhir_json_from_reading(reading)
                try:
                    r = requests.post(
                        f"http://localhost:{settings.PORT}/bg/reading",
                        data=payload,
                        timeout=10,
                    )
                    if r.status_code == 200:
                        sent_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to send reading {i}: {r.status_code}")
                except requests.RequestException as e:
                    failed_count += 1
                    logger.warning(f"Request error for reading {i}: {e}")

            # Progress logging
            if (i + 1) % batch_size == 0:
                elapsed = time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total_readings - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {i + 1:,}/{total_readings:,} "
                    f"({100 * (i + 1) / total_readings:.1f}%) "
                    f"| Rate: {rate:.1f}/s "
                    f"| ETA: {remaining / 60:.1f} min "
                    f"| Glucose: {reading['value']:.1f} mg/dL"
                )

            # Advance time by sample interval
            current_time = datetime.fromtimestamp(
                current_time.timestamp() + sample_time * 60, tz=timezone.utc
            )

            # Stop if we've passed the end date
            if current_time > end_date:
                break

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")

    # Final summary
    elapsed = time() - start_time
    logger.success(f"Generation complete!")
    logger.info(f"Total readings: {len(readings):,}")
    logger.info(f"Sent to server: {sent_count:,}")
    logger.info(f"Failed: {failed_count:,}")
    logger.info(f"Time elapsed: {elapsed / 60:.1f} minutes")
    logger.info(f"Average rate: {len(readings) / elapsed:.1f} readings/sec")

    # Calculate statistics
    if readings:
        values = [r["value"] for r in readings]
        avg_glucose = sum(values) / len(values)
        min_glucose = min(values)
        max_glucose = max(values)
        in_range = sum(1 for v in values if 70 <= v <= 180)
        tir = 100 * in_range / len(values)

        logger.info(
            f"Glucose stats: avg={avg_glucose:.1f}, min={min_glucose:.1f}, max={max_glucose:.1f} mg/dL"
        )
        logger.info(f"Time in Range (70-180): {tir:.1f}%")

    return readings


if __name__ == "__main__":
    generate_historical_data(
        patient="adult#001",
        start_date=datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        seed=42,
        insulin_mode="basal-bolus",
        send_to_server=True,
        batch_size=500,
    )
