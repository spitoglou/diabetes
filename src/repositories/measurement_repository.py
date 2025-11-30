"""Repository for glucose measurement data access."""

from datetime import datetime
from typing import Any, Dict, List

from loguru import logger
from pymongo.collection import Collection
from pymongo.database import Database

from src.config import Config, get_config


class MeasurementRepository:
    """
    Repository for CRUD operations on glucose measurements.

    Encapsulates all MongoDB access for measurement data with
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
        Get collection for a patient with validation.

        Args:
            patient_id: Patient identifier.

        Returns:
            MongoDB collection for the patient.

        Raises:
            ValueError: If patient ID is not in whitelist.
        """
        if not self.config.is_valid_patient_id(patient_id):
            raise ValueError(
                f"Invalid patient ID: {patient_id}. Valid IDs: {self.config.valid_patient_ids}"
            )
        return self.db[f"measurements_{patient_id}"]

    def save(self, patient_id: str, measurement: Dict[str, Any]) -> str:
        """
        Save a measurement to the database.

        Args:
            patient_id: Patient identifier.
            measurement: FHIR-formatted measurement data.

        Returns:
            Inserted document ID as string.
        """
        collection = self._get_collection(patient_id)
        result = collection.insert_one(measurement)
        logger.debug(
            f"Saved measurement for patient {patient_id}: {result.inserted_id}"
        )
        return str(result.inserted_id)

    def get_recent(
        self,
        patient_id: str,
        limit: int = 100,
        projection: Dict[str, int] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent measurements for a patient.

        Args:
            patient_id: Patient identifier.
            limit: Maximum number of records to return.
            projection: Fields to include/exclude.

        Returns:
            List of measurement documents, sorted by most recent first.
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

    def get_by_date_range(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get measurements within a date range.

        Args:
            patient_id: Patient identifier.
            start_date: Start of date range.
            end_date: End of date range.
            limit: Maximum records to return.

        Returns:
            List of measurements within the range.
        """
        collection = self._get_collection(patient_id)
        cursor = (
            collection.find(
                {
                    "effectiveDateTime": {
                        "$gte": start_date.isoformat(),
                        "$lte": end_date.isoformat(),
                    }
                }
            )
            .sort([("effectiveDateTime", 1)])
            .limit(limit)
        )
        return list(cursor)

    def count(self, patient_id: str) -> int:
        """
        Count total measurements for a patient.

        Args:
            patient_id: Patient identifier.

        Returns:
            Total document count.
        """
        collection = self._get_collection(patient_id)
        return collection.count_documents({})

    def delete_all(self, patient_id: str) -> int:
        """
        Delete all measurements for a patient (use with caution).

        Args:
            patient_id: Patient identifier.

        Returns:
            Number of deleted documents.
        """
        collection = self._get_collection(patient_id)
        result = collection.delete_many({})
        logger.warning(
            f"Deleted {result.deleted_count} measurements for patient {patient_id}"
        )
        return result.deleted_count

    def watch(self, patient_id: str, pipeline: List[Dict] | None = None):
        """
        Watch for changes in the measurements collection.

        Args:
            patient_id: Patient identifier.
            pipeline: Aggregation pipeline for filtering changes.

        Returns:
            Change stream cursor.
        """
        collection = self._get_collection(patient_id)
        if pipeline is None:
            pipeline = [{"$match": {"operationType": "insert"}}]
        return collection.watch(pipeline)
