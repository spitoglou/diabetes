"""Tests for MongoDB wrapper module."""

from unittest.mock import MagicMock, patch

import pytest


class TestMongoDB:
    """Test suite for src.mongo.MongoDB class."""

    def test_mongodb_import(self):
        """Test that MongoDB class can be imported."""
        from src.mongo import MongoDB

        assert MongoDB is not None

    @patch("src.mongo.MongoClient")
    def test_mongodb_init(self, mock_client):
        """Test MongoDB initialization."""
        from src.mongo import MongoDB

        mongo = MongoDB()
        assert mongo.client is not None
        mock_client.assert_called_once()

    @patch("src.mongo.MongoClient")
    def test_mongodb_ping_success(self, mock_client):
        """Test ping method with successful connection."""
        from src.mongo import MongoDB

        mock_admin = MagicMock()
        mock_client.return_value.admin = mock_admin

        mongo = MongoDB()
        mongo.ping()  # Should not raise

        mock_admin.command.assert_called_with("ping")

    @patch("src.mongo.MongoClient")
    def test_mongodb_ping_failure(self, mock_client, capsys):
        """Test ping method with connection failure."""
        from src.mongo import MongoDB

        mock_admin = MagicMock()
        mock_admin.command.side_effect = Exception("Connection failed")
        mock_client.return_value.admin = mock_admin

        mongo = MongoDB()
        mongo.ping()  # Should print error, not raise

        captured = capsys.readouterr()
        assert "Connection failed" in captured.out

    @patch("src.mongo.MongoClient")
    def test_mongodb_list_databases(self, mock_client):
        """Test list_databases method."""
        from src.mongo import MongoDB

        mock_client.return_value.list_database_names.return_value = [
            "db1",
            "db2",
            "admin",
        ]

        mongo = MongoDB()
        dbs = mongo.list_databases()

        assert dbs == ["db1", "db2", "admin"]
        assert len(dbs) == 3

    @patch("src.mongo.MongoClient")
    def test_mongodb_list_collections(self, mock_client):
        """Test list_collections method."""
        from src.mongo import MongoDB

        mock_db = MagicMock()
        mock_db.list_collection_names.return_value = [
            "measurements_559",
            "predictions_559",
        ]
        mock_client.return_value.__getitem__.return_value = mock_db

        mongo = MongoDB()
        collections = mongo.list_collections("test_database")

        assert collections == ["measurements_559", "predictions_559"]
        assert len(collections) == 2
