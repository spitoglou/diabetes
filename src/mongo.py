from __future__ import annotations

from typing import Any

from loguru import logger
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from config.settings import settings


class MongoDB:
    """MongoDB client wrapper for diabetes prediction system."""

    def __init__(self) -> None:
        """Initialize MongoDB connection using settings."""
        uri = settings.MONGO_URI
        self.client: MongoClient[dict[str, Any]] = MongoClient(
            uri, server_api=ServerApi("1")
        )

    def ping(self) -> None:
        """Test the MongoDB connection."""
        try:
            self.client.admin.command("ping")
            logger.debug(
                "Pinged your deployment. You successfully connected to MongoDB!"
            )
        except Exception as e:
            print(e)

    def list_databases(self) -> list[str]:
        """List all database names.

        Returns:
            List of database names
        """
        db_names = self.client.list_database_names()
        logger.info(f"Found {len(db_names)} databases: {db_names}")
        return db_names

    def list_collections(self, database_name: str) -> list[str]:
        """List all collection names in a database.

        Args:
            database_name: Name of the database

        Returns:
            List of collection names
        """
        db = self.client[database_name]
        col_names = db.list_collection_names()
        logger.info(
            f"Found {len(col_names)} collections in {database_name}: {col_names}"
        )
        return col_names


if __name__ == "__main__":
    mongo = MongoDB()
    mongo.ping()
    mongo.list_databases()
    mongo.list_collections(settings.MONGO_DATABASE)
