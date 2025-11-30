"""
Main experiment runner for glucose prediction models.

Usage:
    python final_run.py

Configure via environment variables or .env file.
"""

import itertools

from loguru import logger

from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.config import get_config
from src.logging_config import setup_logging
from src.services.experiment_orchestrator import ExperimentOrchestrator

# Experiment parameters (full run)
# PATIENTS = [559, 563, 570, 575, 588, 591]
# WINDOWS = [6, 12]  # Window sizes in 5-min intervals
# HORIZONS = [1, 6, 12]  # Prediction horizons in 5-min intervals

# Quick testing configuration (comment out and uncomment above for full run)
PATIENTS = [563]
WINDOWS = [12]
HORIZONS = [6]


def run_experiments():
    """Run experiments for all patient/window/horizon combinations."""
    config = get_config()
    setup_logging(level=config.log_level)

    logger.info(
        f"Starting experiments: {len(PATIENTS)} patients, "
        f"{len(WINDOWS)} windows, {len(HORIZONS)} horizons"
    )
    logger.info(f"Neptune tracking: {config.enable_neptune}")

    results = []

    for patient, window, horizon in itertools.product(PATIENTS, WINDOWS, HORIZONS):
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Running experiment: Patient={patient}, Window={window}, Horizon={horizon}"
        )

        try:
            # Create provider for training data
            provider = OhioBgcProvider(scope="train", ohio_no=patient)

            # Create orchestrator and run experiment
            orchestrator = ExperimentOrchestrator(provider=provider, config=config)
            result = orchestrator.run(
                patient_id=patient,
                window=window,
                horizon=horizon,
                enable_neptune=config.enable_neptune,
            )

            results.append(result)
            logger.success(f"Experiment completed in {result.execution_time:.1f}s")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception(f"Experiment failed: {e}")

    # Summary
    logger.info(f"{'=' * 60}")
    logger.info(
        f"Completed {len(results)}/{len(PATIENTS) * len(WINDOWS) * len(HORIZONS)} experiments"
    )

    return results


if __name__ == "__main__":
    run_experiments()
