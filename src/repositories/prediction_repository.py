"""Repository for prediction data access."""

from datetime import datetime
from typing import Any, Dict, List

from loguru import logger
from pymongo.collection import Collection
from pymongo.database import Database

from src.config import Config, get_config


class PredictionRepository:
    """
    Repository for CRUD operations on glucose predictions.

    Encapsulates all MongoDB access for prediction data with
    patient ID validation and efficient query patterns.
    """

    def __init__(self, db: Database, config: Config | None = None):
        """
        Initialize repository with database connection.

        Args:
            db: MongoDB database instance.
            config: Application configuration (uses global if not provided).
        """
        self.db = db
        self.config = config or get_config()

    def _get_collection(self, patient_id: str) -> Collection:
        """
        Get predictions collection for a patient with validation.

        Args:
            patient_id: Patient identifier.

        Returns:
            MongoDB collection for the patient's predictions.

        Raises:
            ValueError: If patient ID is not in whitelist.
        """
        if not self.config.is_valid_patient_id(patient_id):
            raise ValueError(
                f"Invalid patient ID: {patient_id}. Valid IDs: {self.config.valid_patient_ids}"
            )
        return self.db[f"predictions_{patient_id}"]

    def save(self, patient_id: str, prediction: Dict[str, Any]) -> str:
        """
        Save a prediction to the database.

        Args:
            patient_id: Patient identifier.
            prediction: Prediction data with origin_time, prediction_time, value.

        Returns:
            Inserted document ID as string.
        """
        collection = self._get_collection(patient_id)
        result = collection.insert_one(prediction)
        logger.debug(f"Saved prediction for patient {patient_id}: {result.inserted_id}")
        return str(result.inserted_id)

    def get_recent(
        self,
        patient_id: str,
        limit: int = 50,
        projection: Dict[str, int] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent predictions for a patient.

        Args:
            patient_id: Patient identifier.
            limit: Maximum number of records to return.
            projection: Fields to include/exclude.

        Returns:
            List of prediction documents, sorted by most recent first.
        """
        collection = self._get_collection(patient_id)
        cursor = (
            collection.find(
                {},
                projection=projection,
            )
            .sort([("_id", -1)])
            .limit(limit)
        )
        return list(cursor)

    def get_by_prediction_time_range(
        self,
        patient_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Get predictions for a time range.

        Args:
            patient_id: Patient identifier.
            start_time: Start of prediction time range.
            end_time: End of prediction time range.
            limit: Maximum records to return.

        Returns:
            List of predictions within the range.
        """
        collection = self._get_collection(patient_id)
        cursor = (
            collection.find(
                {
                    "prediction_time": {
                        "$gte": start_time,
                        "$lte": end_time,
                    }
                }
            )
            .sort([("prediction_time", 1)])
            .limit(limit)
        )
        return list(cursor)

    def count(self, patient_id: str) -> int:
        """
        Count total predictions for a patient.

        Args:
            patient_id: Patient identifier.

        Returns:
            Total document count.
        """
        collection = self._get_collection(patient_id)
        return collection.count_documents({})

    def delete_all(self, patient_id: str) -> int:
        """
        Delete all predictions for a patient (use with caution).

        Args:
            patient_id: Patient identifier.

        Returns:
            Number of deleted documents.
        """
        collection = self._get_collection(patient_id)
        result = collection.delete_many({})
        logger.warning(
            f"Deleted {result.deleted_count} predictions for patient {patient_id}"
        )
        return result.deleted_count
