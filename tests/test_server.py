"""Tests for FastAPI server."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server import app, lifespan, main


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
        from server import get_measurement_repo

        mock_repo = MagicMock()
        mock_repo.save.return_value = "test_id_123"

        app.dependency_overrides[get_measurement_repo] = lambda: mock_repo

        try:
            response = client.post("/bg/reading", json=sample_fhir_observation)

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Success"
            assert "record_id" in data
        finally:
            app.dependency_overrides.clear()

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
        from server import get_measurement_repo

        mock_repo = MagicMock()
        mock_repo.save.return_value = "test_id"

        app.dependency_overrides[get_measurement_repo] = lambda: mock_repo

        try:
            response = client.post("/bg/reading", json=sample_fhir_observation)

            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

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
        from server import get_measurement_repo

        del sample_fhir_observation["valueQuantity"]["unit"]

        mock_repo = MagicMock()
        mock_repo.save.return_value = "test_id"

        app.dependency_overrides[get_measurement_repo] = lambda: mock_repo

        try:
            response = client.post("/bg/reading", json=sample_fhir_observation)

            # Should use default unit
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()


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


class TestLifespan:
    """Tests for server lifespan events."""

    def test_lifespan_startup_success(self):
        """Test lifespan startup with successful MongoDB connection."""
        import asyncio

        with patch("server.get_config") as mock_get_config:
            with patch("server.MongoDB") as MockMongo:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config
                mock_mongo = MagicMock()
                mock_mongo.ping.return_value = True
                MockMongo.return_value = mock_mongo

                async def run_lifespan():
                    async with lifespan(app):
                        pass

                asyncio.get_event_loop().run_until_complete(run_lifespan())

                MockMongo.assert_called_once_with(mock_config)
                mock_mongo.ping.assert_called_once()

    def test_lifespan_startup_ping_fails(self):
        """Test lifespan startup when MongoDB ping fails."""
        import asyncio

        with patch("server.get_config") as mock_get_config:
            with patch("server.MongoDB") as MockMongo:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config
                mock_mongo = MagicMock()
                mock_mongo.ping.return_value = False
                MockMongo.return_value = mock_mongo

                async def run_lifespan():
                    async with lifespan(app):
                        pass

                # Should not raise, just log warning
                asyncio.get_event_loop().run_until_complete(run_lifespan())

                mock_mongo.ping.assert_called_once()

    def test_lifespan_startup_connection_error(self):
        """Test lifespan startup when MongoDB connection fails."""
        import asyncio

        with patch("server.get_config") as mock_get_config:
            with patch("server.MongoDB") as MockMongo:
                mock_config = MagicMock()
                mock_get_config.return_value = mock_config
                MockMongo.side_effect = Exception("Connection failed")

                async def run_lifespan():
                    async with lifespan(app):
                        pass

                # Should not raise, just log error
                asyncio.get_event_loop().run_until_complete(run_lifespan())


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_calls_uvicorn(self):
        """Test main function calls uvicorn.run."""
        with patch("server.uvicorn.run") as mock_run:
            main()

            mock_run.assert_called_once_with(app, host="0.0.0.0", port=8000)
