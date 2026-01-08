"""Tests for FastAPI server endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestServerEndpoints:
    """Test suite for server.py endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked MongoDB."""
        with patch("scripts.serving.server.MongoDB") as mock_mongo:
            mock_instance = MagicMock()
            mock_mongo.return_value = mock_instance

            from scripts.serving.server import app

            return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns success message."""
        response = client.get("/")

        assert response.status_code == 200
        assert "message" in response.json()
        assert "running" in response.json()["message"].lower()

    def test_health_endpoint_exists(self, client):
        """Test health endpoint exists and returns expected structure."""
        with patch("scripts.serving.server.dbms") as mock_dbms:
            mock_dbms.client.admin.command.return_value = True

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "database" in data
            assert "version" in data
            assert "patient_id" in data

    def test_health_endpoint_db_connected(self, client):
        """Test health endpoint when database is connected."""
        with patch("scripts.serving.server.dbms") as mock_dbms:
            mock_dbms.client.admin.command.return_value = True

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["database"] == "connected"
            assert data["status"] == "healthy"

    def test_health_endpoint_db_disconnected(self, client):
        """Test health endpoint when database is disconnected."""
        with patch("scripts.serving.server.dbms") as mock_dbms:
            mock_dbms.client.admin.command.side_effect = Exception("Connection failed")

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["database"] == "disconnected"
            assert data["status"] == "degraded"

    def test_post_reading_valid(self, client):
        """Test POST /bg/reading with valid FHIR payload."""
        with patch("scripts.serving.server.dbms") as mock_dbms:
            mock_db = MagicMock()
            mock_collection = MagicMock()
            mock_collection.insert_one.return_value.inserted_id = "test_id_123"
            mock_db.__getitem__.return_value = mock_collection
            mock_dbms.client.__getitem__.return_value = mock_db

            payload = {
                "status": "final",
                "category": [
                    {"coding": [{"system": "http://test.com", "code": "vital-signs"}]}
                ],
                "code": {"text": "Blood Glucose"},
                "subject": {"identifier": "559"},
                "effectiveDateTime": "2024-01-08T12:00:00Z",
                "valueQuantity": {"value": 120.0, "unit": "mg/dL"},
                "device": {"displayName": "Test Device", "note": "Test"},
            }

            response = client.post("/bg/reading", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert data["message"] == "Success"

    def test_post_reading_invalid_payload(self, client):
        """Test POST /bg/reading with invalid payload."""
        payload = {"invalid": "data"}

        response = client.post("/bg/reading", json=payload)

        assert response.status_code == 422  # Validation error


class TestFhirModels:
    """Test FHIR Pydantic models."""

    def test_fhir_observation_valid(self):
        """Test FhirObservation model with valid data."""
        from scripts.serving.server import FhirObservation

        data = {
            "status": "final",
            "category": [
                {"coding": [{"system": "http://test.com", "code": "vital-signs"}]}
            ],
            "code": {"text": "Blood Glucose"},
            "subject": {"identifier": "559"},
            "effectiveDateTime": "2024-01-08T12:00:00Z",
            "valueQuantity": {"value": 120.0, "unit": "mg/dL"},
            "device": {"displayName": "Test Device", "note": "Test"},
        }

        observation = FhirObservation(**data)
        assert observation.status == "final"
        assert observation.valueQuantity.value == 120.0
        assert observation.subject.identifier == "559"

    def test_health_response_model(self):
        """Test HealthResponse model."""
        from scripts.serving.server import HealthResponse

        response = HealthResponse(
            status="healthy",
            timestamp="2024-01-08T12:00:00Z",
            database="connected",
            version="1.0.0",
            patient_id="559",
        )

        assert response.status == "healthy"
        assert response.database == "connected"
