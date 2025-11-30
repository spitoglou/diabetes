"""
╔╦╗╔╗   ╔╦╗┌─┐┌┐┌┬┌┬┐┌─┐┬─┐
 ║║╠╩╗  ║║║│ │││││ │ │ │├┬┘
═╩╝╚═╝  ╩ ╩└─┘┘└┘┴ ┴ └─┘┴└─
Author: Stavros Pitoglou

Database monitor that watches all collections for changes.
"""

from loguru import logger
from pymongo.errors import PyMongoError

from src.config import get_config
from src.logging_config import setup_logging
from src.mongo import MongoDB


def watch_database():
    """
    Watch all collections in the default database for changes.

    Uses MongoDB change streams to monitor insert and update operations
    across all collections and logs each detected change.
    """
    config = get_config()
    setup_logging(level=config.log_level)

    logger.info(f"Starting database monitor for database: {config.database_name}")

    try:
        mongo = MongoDB(config)
        if not mongo.ping():
            logger.error("Failed to connect to MongoDB")
            return

        db = mongo.get_database()
        collections = db.list_collection_names()
        logger.info(f"Monitoring {len(collections)} collections: {collections}")

        # Watch the entire database for changes
        pipeline = [{"$match": {"operationType": {"$in": ["insert", "update"]}}}]

        logger.info("Watching for changes (press Ctrl+C to stop)...")

        with db.watch(pipeline) as stream:
            for change in stream:
                operation = change.get("operationType", "unknown")
                collection = change.get("ns", {}).get("coll", "unknown")
                doc_id = change.get("documentKey", {}).get("_id", "unknown")

                if operation == "insert":
                    logger.success(
                        f"INSERT detected | collection={collection} | _id={doc_id}"
                    )
                elif operation == "update":
                    logger.info(
                        f"UPDATE detected | collection={collection} | _id={doc_id}"
                    )

    except PyMongoError as e:
        logger.error(f"MongoDB error: {e}")
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(f"Unexpected error: {e}")


def main():
    """CLI entry point for the database monitor."""
    watch_database()


if __name__ == "__main__":
    main()
