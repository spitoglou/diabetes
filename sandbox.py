import pandas as pd
from src.mongo import MongoDB
from config.server_config import DATABASE, COLLECTION

def retrieve_data(limit: int = 200):
    sensor_data = pd.DataFrame(list(mongo_collection.find(
        limit=limit,
        sort=[('_id', -1)]
    )))
    # sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp']).astype(str)
    # # ?test = test.drop(columns=['_id'])
    # sensor_data['_id'] = sensor_data['_id'].astype(pd.StringDtype())
    return sensor_data

mongo = MongoDB()
db = mongo.client[DATABASE]
mongo_collection = db[COLLECTION]
ts_df = retrieve_data()
print(ts_df)

for index, row in ts_df.iterrows():
    print(row)