"""
Real-time prediction service that watches MongoDB for new measurements
and generates glucose predictions.

Usage:
    python load_model_and_predict.py

Configure via environment variables or .env file.
"""

import pymongo
from loguru import logger

from src.config import get_config
from src.logging_config import setup_logging
from src.mongo import MongoDB
from src.repositories.measurement_repository import MeasurementRepository
from src.repositories.prediction_repository import PredictionRepository
from src.services.prediction_service import PredictionService


def create_prediction_service(config):
    """Create and initialize the prediction service with dependencies."""
    mongo = MongoDB(config)
    db = mongo.get_database()

    measurement_repo = MeasurementRepository(db, config)
    prediction_repo = PredictionRepository(db, config)

    service = PredictionService(
        measurement_repo=measurement_repo,
        prediction_repo=prediction_repo,
        config=config,
    )

    # Load model for the configured patient
    service.load_model(
        patient_id=config.default_patient_id,
        window=config.window_steps,
        horizon=config.prediction_horizon,
    )

    return service, measurement_repo


def handle_new_data(service, patient_id, config):
    """Handle new measurement data by generating and storing a prediction."""
    try:
        record_id = service.predict_and_store(
            patient_id=patient_id,
            window=config.window_steps,
            horizon=config.prediction_horizon,
        )
        logger.success(f"Prediction stored with ID: {record_id}")
    except ValueError as e:
        logger.warning(f"Insufficient data for prediction: {e}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(f"Prediction failed: {e}")


def run_watcher():
    """Main function to run the prediction watcher service."""
    config = get_config()
    setup_logging(level=config.log_level)

    patient_id = config.default_patient_id
    logger.info(f"Starting prediction service for patient {patient_id}")
    logger.info(f"Window: {config.window_steps}, Horizon: {config.prediction_horizon}")

    # Create service
    service, measurement_repo = create_prediction_service(config)

    # Watch for new measurements
    resume_token = None
    pipeline = [{"$match": {"operationType": "insert"}}]

    logger.info("Starting database watch for new measurements...")

    try:
        with measurement_repo.watch(patient_id, pipeline) as stream:
            for _ in stream:
                logger.debug("Detected new measurement")
                handle_new_data(service, patient_id, config)
                resume_token = stream.resume_token

    except pymongo.errors.PyMongoError as e:  # pyright: ignore[reportAttributeAccessIssue]
        if resume_token is None:
            logger.error(f"Change stream initialization failed: {e}")
            raise
        # Resume from last known position
        logger.warning(f"Change stream interrupted, resuming: {e}")
        with measurement_repo.watch(patient_id, pipeline) as stream:
            for _ in stream:
                handle_new_data(service, patient_id, config)


def main():
    """CLI entry point for the prediction service."""
    run_watcher()


if __name__ == "__main__":
    main()
