"""Tests for MongoDB connection wrapper."""

from unittest.mock import MagicMock, patch

import pytest
from pymongo.errors import ConnectionFailure

from src.config import Config
from src.mongo import MongoDB


class TestMongoDB:
    """Tests for MongoDB class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return Config(
            mongo_uri="mongodb://localhost:27017",
            database_name="test_database",
        )

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoClient."""
        with patch("src.mongo.MongoClient") as mock_client:
            yield mock_client

    def test_init_with_config(self, mock_config, mock_mongo_client):
        """Test MongoDB initialization with config."""
        mongo = MongoDB(mock_config)

        assert mongo.config == mock_config
        mock_mongo_client.assert_called_once()

    def test_init_without_config(self, mock_mongo_client):
        """Test MongoDB initialization without config uses global."""
        with patch("src.mongo.get_config") as mock_get_config:
            mock_get_config.return_value = Config(
                mongo_uri="mongodb://localhost:27017",
                database_name="test_db",
            )
            mongo = MongoDB()

            mock_get_config.assert_called_once()

    def test_init_missing_uri_raises(self, mock_mongo_client):
        """Test that missing MONGO_URI raises ValueError."""
        config = Config(mongo_uri="", database_name="test")

        with pytest.raises(ValueError, match="MONGO_URI not configured"):
            MongoDB(config)

    def test_ping_success(self, mock_config, mock_mongo_client):
        """Test successful ping."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance

        mongo = MongoDB(mock_config)
        result = mongo.ping()

        assert result is True
        mock_client_instance.admin.command.assert_called_once_with("ping")

    def test_ping_connection_failure(self, mock_config, mock_mongo_client):
        """Test ping with connection failure."""
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.side_effect = ConnectionFailure("Failed")
        mock_mongo_client.return_value = mock_client_instance

        mongo = MongoDB(mock_config)
        result = mongo.ping()

        assert result is False

    def test_ping_unexpected_exception(self, mock_config, mock_mongo_client):
        """Test ping with unexpected exception."""
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.side_effect = Exception("Unexpected")
        mock_mongo_client.return_value = mock_client_instance

        mongo = MongoDB(mock_config)
        result = mongo.ping()

        assert result is False

    def test_get_database_default(self, mock_config, mock_mongo_client):
        """Test get_database with default name."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance

        mongo = MongoDB(mock_config)
        mongo.get_database()

        mock_client_instance.__getitem__.assert_called_once_with("test_database")

    def test_get_database_custom_name(self, mock_config, mock_mongo_client):
        """Test get_database with custom name."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance

        mongo = MongoDB(mock_config)
        mongo.get_database("custom_db")

        mock_client_instance.__getitem__.assert_called_once_with("custom_db")

    def test_list_databases(self, mock_config, mock_mongo_client):
        """Test list_databases."""
        mock_client_instance = MagicMock()
        mock_client_instance.list_database_names.return_value = ["db1", "db2"]
        mock_mongo_client.return_value = mock_client_instance

        mongo = MongoDB(mock_config)
        result = mongo.list_databases()

        assert result == ["db1", "db2"]

    def test_list_collections(self, mock_config, mock_mongo_client):
        """Test list_collections."""
        mock_client_instance = MagicMock()
        mock_db = MagicMock()
        mock_db.list_collection_names.return_value = ["col1", "col2"]
        mock_client_instance.__getitem__.return_value = mock_db
        mock_mongo_client.return_value = mock_client_instance

        mongo = MongoDB(mock_config)
        result = mongo.list_collections("test_database")

        assert result == ["col1", "col2"]
