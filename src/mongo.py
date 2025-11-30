"""MongoDB connection wrapper."""

from typing import Optional

from loguru import logger
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from src.config import Config, get_config


class MongoDB:
    """
    MongoDB connection wrapper.

    Provides a client connection using configuration from environment variables.
    """

    def __init__(self, config: Config | None = None):
        """
        Initialize MongoDB connection.

        Args:
            config: Application configuration (uses global if not provided).
        """
        self.config = config or get_config()

        if not self.config.mongo_uri:
            raise ValueError(
                "MONGO_URI not configured. Set it in .env file or environment."
            )

        self.client = MongoClient(self.config.mongo_uri, server_api=ServerApi("1"))

    def ping(self) -> bool:
        """
        Test MongoDB connection.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.client.admin.command("ping")
            logger.debug("Successfully connected to MongoDB")
            return True
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception(f"Unexpected error pinging MongoDB: {e}")
            return False

    def get_database(self, name: Optional[str] = None) -> Database:
        """
        Get a database instance.

        Args:
            name: Database name (uses config default if not specified).

        Returns:
            MongoDB database instance.
        """
        db_name = name or self.config.database_name
        return self.client[db_name]

    def list_databases(self):
        """List all database names."""
        db_names = self.client.list_database_names()
        logger.info(f"Found {len(db_names)} databases: {db_names}")
        return db_names

    def list_collections(self, database_name: str):
        """List all collections in a database."""
        db = self.client[database_name]
        col_names = db.list_collection_names()
        logger.info(
            f"Found {len(col_names)} collections in {database_name}: {col_names}"
        )
        return col_names


if __name__ == "__main__":
    mongo = MongoDB()
    if mongo.ping():
        mongo.list_databases()
        mongo.list_collections("test_database_1")
