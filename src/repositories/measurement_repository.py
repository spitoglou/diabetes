"""Repository for glucose measurement data access."""

from datetime import datetime
from typing import Any, Dict, List

from src.repositories.base_repository import BaseRepository


class MeasurementRepository(BaseRepository):
    """
    Repository for CRUD operations on glucose measurements.

    Encapsulates all MongoDB access for measurement data with
    patient ID validation and efficient query patterns.
    """

    def _get_collection_name(self, patient_id: str) -> str:
        """Get the collection name for measurements."""
        return f"measurements_{patient_id}"

    def _get_entity_name(self) -> str:
        """Get the entity name for logging."""
        return "measurement"

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
