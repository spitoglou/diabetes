"""Tests for FastAPI server."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_fhir_observation():
    """Create a sample FHIR observation payload."""
    return {
        "status": "final",
        "category": [{"coding": [{"code": "vital-signs"}]}],
        "code": {"text": "Glucose"},
        "subject": {"identifier": "559"},
        "effectiveDateTime": "2024-01-01T10:00:00",
        "valueQuantity": {"value": 120.0, "unit": "mg/dL"},
        "device": {"display": "CGM Device"},
    }


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_success(self, client):
        """Test root endpoint returns success message."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["status"] == "healthy"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_db_connected(self, client):
        """Test health check when database is connected."""
        with patch("server.MongoDB") as MockMongo:
            mock_mongo = MagicMock()
            mock_mongo.ping.return_value = True
            MockMongo.return_value = mock_mongo

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["database"] == "connected"

    def test_health_check_db_disconnected(self, client):
        """Test health check when database is disconnected."""
        with patch("server.MongoDB") as MockMongo:
            mock_mongo = MagicMock()
            mock_mongo.ping.return_value = False
            MockMongo.return_value = mock_mongo

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["database"] == "disconnected"

    def test_health_check_db_exception(self, client):
        """Test health check when database throws exception."""
        with patch("server.MongoDB") as MockMongo:
            MockMongo.side_effect = Exception("Connection error")

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"


class TestPostReadingEndpoint:
    """Tests for POST /bg/reading endpoint."""

    def test_post_reading_success(self, client, sample_fhir_observation):
        """Test successful measurement posting."""
        with patch("server.get_measurement_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.save.return_value = "test_id_123"
            mock_get_repo.return_value = mock_repo

            # Override the dependency
            app.dependency_overrides[mock_get_repo] = lambda: mock_repo

            response = client.post("/bg/reading", json=sample_fhir_observation)

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Success"
            assert "record_id" in data

    def test_post_reading_missing_required_fields(self, client):
        """Test posting with missing required fields."""
        incomplete_payload = {
            "status": "final",
            # Missing other required fields
        }

        response = client.post("/bg/reading", json=incomplete_payload)

        assert response.status_code == 422  # Validation error

    def test_post_reading_invalid_glucose_value(self, client, sample_fhir_observation):
        """Test posting with invalid glucose value."""
        sample_fhir_observation["valueQuantity"]["value"] = -10  # Invalid

        response = client.post("/bg/reading", json=sample_fhir_observation)

        assert response.status_code == 422  # Validation error


class TestFhirObservationModel:
    """Tests for FHIR observation validation."""

    def test_valid_observation(self, client, sample_fhir_observation):
        """Test valid FHIR observation is accepted."""
        with patch("server.get_measurement_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.save.return_value = "test_id"
            mock_get_repo.return_value = mock_repo
            app.dependency_overrides[mock_get_repo] = lambda: mock_repo

            response = client.post("/bg/reading", json=sample_fhir_observation)

            assert response.status_code == 200

    def test_missing_subject_identifier(self, client, sample_fhir_observation):
        """Test missing subject identifier."""
        sample_fhir_observation["subject"]["identifier"] = ""

        response = client.post("/bg/reading", json=sample_fhir_observation)

        assert response.status_code == 422

    def test_glucose_value_must_be_positive(self, client, sample_fhir_observation):
        """Test glucose value must be positive."""
        sample_fhir_observation["valueQuantity"]["value"] = 0

        response = client.post("/bg/reading", json=sample_fhir_observation)

        assert response.status_code == 422


class TestValueQuantityModel:
    """Tests for ValueQuantity model."""

    def test_default_unit(self, client, sample_fhir_observation):
        """Test default unit is mg/dL."""
        del sample_fhir_observation["valueQuantity"]["unit"]

        with patch("server.get_measurement_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.save.return_value = "test_id"
            mock_get_repo.return_value = mock_repo
            app.dependency_overrides[mock_get_repo] = lambda: mock_repo

            response = client.post("/bg/reading", json=sample_fhir_observation)

            # Should use default unit
            assert response.status_code == 200


class TestErrorResponses:
    """Tests for error response handling."""

    def test_invalid_patient_returns_400(self, client, sample_fhir_observation):
        """Test invalid patient returns 400."""
        from server import get_measurement_repo

        mock_repo = MagicMock()
        mock_repo.save.side_effect = ValueError("Invalid patient ID")

        app.dependency_overrides[get_measurement_repo] = lambda: mock_repo

        try:
            response = client.post("/bg/reading", json=sample_fhir_observation)

            assert response.status_code == 400
            assert "Invalid patient ID" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_database_error_returns_500(self, client, sample_fhir_observation):
        """Test database error returns 500."""
        from server import get_measurement_repo

        mock_repo = MagicMock()
        mock_repo.save.side_effect = Exception("Database error")

        app.dependency_overrides[get_measurement_repo] = lambda: mock_repo

        try:
            response = client.post("/bg/reading", json=sample_fhir_observation)

            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()
