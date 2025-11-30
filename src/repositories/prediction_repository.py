"""Repository for prediction data access."""

from datetime import datetime
from typing import Any, Dict, List

from src.repositories.base_repository import BaseRepository


class PredictionRepository(BaseRepository):
    """
    Repository for CRUD operations on glucose predictions.

    Encapsulates all MongoDB access for prediction data with
    patient ID validation and efficient query patterns.
    """

    def _get_collection_name(self, patient_id: str) -> str:
        """Get the collection name for predictions."""
        return f"predictions_{patient_id}"

    def _get_entity_name(self) -> str:
        """Get the entity name for logging."""
        return "prediction"

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
            limit: Maximum number of records to return (default 50 for predictions).
            projection: Fields to include/exclude.

        Returns:
            List of prediction documents, sorted by most recent first.
        """
        return super().get_recent(patient_id, limit, projection)

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
