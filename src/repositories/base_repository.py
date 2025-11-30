"""Base repository with shared functionality for MongoDB data access."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from loguru import logger
from pymongo.collection import Collection
from pymongo.database import Database

from src.config import Config, get_config


class BaseRepository(ABC):
    """
    Abstract base repository providing common MongoDB operations.

    Subclasses must implement `_get_collection_name()` to specify
    the collection naming pattern.
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

    @abstractmethod
    def _get_collection_name(self, patient_id: str) -> str:
        """
        Get the collection name for a patient.

        Args:
            patient_id: Patient identifier.

        Returns:
            Collection name string.
        """
        pass

    @abstractmethod
    def _get_entity_name(self) -> str:
        """
        Get the entity name for logging purposes.

        Returns:
            Entity name (e.g., 'measurement', 'prediction').
        """
        pass

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
                f"Invalid patient ID: {patient_id}. "
                f"Valid IDs: {self.config.valid_patient_ids}"
            )
        return self.db[self._get_collection_name(patient_id)]

    def save(self, patient_id: str, document: Dict[str, Any]) -> str:
        """
        Save a document to the database.

        Args:
            patient_id: Patient identifier.
            document: Document data to save.

        Returns:
            Inserted document ID as string.
        """
        collection = self._get_collection(patient_id)
        result = collection.insert_one(document)
        logger.debug(
            f"Saved {self._get_entity_name()} for patient {patient_id}: "
            f"{result.inserted_id}"
        )
        return str(result.inserted_id)

    def get_recent(
        self,
        patient_id: str,
        limit: int = 100,
        projection: Dict[str, int] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent documents for a patient.

        Args:
            patient_id: Patient identifier.
            limit: Maximum number of records to return.
            projection: Fields to include/exclude.

        Returns:
            List of documents, sorted by most recent first.
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

    def count(self, patient_id: str) -> int:
        """
        Count total documents for a patient.

        Args:
            patient_id: Patient identifier.

        Returns:
            Total document count.
        """
        collection = self._get_collection(patient_id)
        return collection.count_documents({})

    def delete_all(self, patient_id: str) -> int:
        """
        Delete all documents for a patient (use with caution).

        Args:
            patient_id: Patient identifier.

        Returns:
            Number of deleted documents.
        """
        collection = self._get_collection(patient_id)
        result = collection.delete_many({})
        logger.warning(
            f"Deleted {result.deleted_count} {self._get_entity_name()}s "
            f"for patient {patient_id}"
        )
        return result.deleted_count
