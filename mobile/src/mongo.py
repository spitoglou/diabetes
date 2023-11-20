from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from loguru import logger
import config.mongo_config as conf


class MongoDB:

    def __init__(self):
        uri = conf.MONGO_URI
        # uri = "mongodb+srv://spitoglou:spitoglou@cluster0.kybxoii.mongodb.net/?retryWrites=true&w=majority"
        # Create a new client and connect to the server
        self.client = MongoClient(uri, server_api=ServerApi('1'))

    def ping(self):
        try:
            self.client.admin.command('ping')
            logger.debug("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    def list_databases(self):
        db_names = self.client.list_database_names()
        logger.info(f'Found {len(db_names)} databases: {db_names}')
        return db_names

    def list_collections(self, database_name: str):
        db = self.client[database_name]
        col_names = db.list_collection_names()
        logger.info(f'Found {len(col_names)} collections in {database_name}: {col_names}')
        return col_names


if __name__ == '__main__':
    mongo = MongoDB()
    mongo.ping()
    mongo.list_databases()
    mongo.list_collections('test_database_1')
    mongo.list_collections('test_database_12')
