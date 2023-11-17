from pycaret.regression import load_model, predict_model
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from datetime import datetime, timedelta
import pymongo
import re
import pandas as pd
import matplotlib.pyplot as plt
from src.mongo import MongoDB
from config.server_config import DATABASE, COLLECTION
import loguru as logger

mongo = MongoDB()
db = mongo.client[DATABASE]
mongo_collection = db[COLLECTION]

model = load_model(
        "models/559_6_6_1_LGBMRegressor_9983a67c-632d-4ce3-8098-d73dbe2d145f"
    )


def get_part_of_day(hour):
    return (
        "morning"
        if 7 <= hour <= 11
        else "afternoon"
        if 12 <= hour <= 16
        else "evening"
        if 16 <= hour <= 20
        else "night"
        if 21 <= hour <= 23
        else "late night"
    )


def correct_lgbm_names(df) -> pd.DataFrame:
    # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
    new_names = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in df.columns}
    new_n_list = list(new_names.values())
    # [LightGBM] Feature appears more than one time.
    new_names = {
        col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
        for i, (col, new_col) in enumerate(new_names.items())
    }
    return df.rename(columns=new_names)


def correct_features(original_df, model_features):
    corrected_df = correct_lgbm_names(original_df)

    # Add columns which (for some reason) are not present but are required by the trained model's pipeline
    # Default value "None"
    for feat in model_features:
        found_in_model = feat in corrected_df
        if not found_in_model:
            print(feat)
            if feat != "label":
                corrected_df[feat] = None
    return corrected_df


def featurize_stream_df(stream_df, window, horizon):
    featurizer = TsfreshFeaturizer(
        stream_df.tail(window),
        window,
        horizon,
        plot_chunks=False,
        minimal_features=False,
    )
    featurizer.chunks = 1
    featurizer.create_feature_dataframe()
    return featurizer.feature_dataframe


def predict_last_n(last_n: list, model):
    saved_model_features = model.feature_names_in_
    stream = pd.DataFrame(last_n).reset_index(drop=True)
    print(stream)
    features = featurize_stream_df(stream, 6, 6)
    # features
    prediction = predict_model(
        model, correct_features(features, saved_model_features)
    ).prediction_label[0]
    print(prediction)
    return prediction


def provider_predictions():
    provider = OhioBgcProvider()
    measurements = provider.tsfresh_dataframe()

    # stream = pd.DataFrame()

    max_data = 7
    meas_list = []
    predictions = []
    for counter in range(max_data):
        measurement = measurements.iloc[counter]
        print(measurement)
        meas_list.append(measurement)
        if len(meas_list) >= 6:
            last_n = meas_list[-6:]

            prediction = predict_last_n(last_n, model)

            predictions.append(
                {
                    "prediction_origin_time": measurement.date_time,
                    "prediction_time": measurement.date_time + timedelta(minutes=6 * 5),
                    "prediction_value": prediction,
                }
            )
    measurement_df = pd.DataFrame(meas_list)
    prediction_df = pd.DataFrame(predictions)

    return measurement_df, prediction_df


def retrieve_data(mongo_collection, limit: int = 200):
    return pd.DataFrame(list(mongo_collection.find(limit=limit, sort=[("_id", -1)])))
    # sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp']).astype(str)
    # # ?test = test.drop(columns=['_id'])
    # sensor_data['_id'] = sensor_data['_id'].astype(pd.StringDtype())
    # return sensor_data


def mongo_predictions():
    ts_df = retrieve_data(mongo_collection, 10)
    # reverse dataframe
    ts_df = ts_df[::-1].reset_index(drop=True)
    print(ts_df)

    meas_list = []
    predictions = []
    for index, row in ts_df.iterrows():
        print(row)
        # print(row['valueQuantity']['value'])
        # print(pd.to_datetime(row['effectiveDateTime']).timetuple())
        date_time = pd.to_datetime(row["effectiveDateTime"])
        time = date_time.time
        hour = date_time.hour
        d = {
            "bg_value": row["valueQuantity"]["value"],
            "date_time": date_time,
            "time_of_day": time,
            "part_of_day": get_part_of_day(hour),
            "time": index + 1,
            "id": "a",
        }
        measurement = pd.Series(d)
        meas_list.append(measurement)
    print(meas_list)
    last_n = meas_list[-6:]
    prediction = predict_last_n(last_n, model)
    predictions.append(
        {
            "prediction_origin_time": measurement.date_time,
            "prediction_time": measurement.date_time + timedelta(minutes=6 * 5),
            "prediction_value": prediction,
        }
    )
    print(predictions)
    measurement_df = pd.DataFrame(meas_list)
    prediction_df = pd.DataFrame(predictions)
    return measurement_df, prediction_df


def handle_new_data():
    
    # # mongo_predictions(model)

    measurement_df, prediction_df = mongo_predictions()

if __name__ == "__main__":
    
    # print(prediction_df)
    # ax = prediction_df.plot(y="prediction_value", x="prediction_time")
    # measurement_df.plot(ax=ax, y="bg_value", x="date_time")
    # plt.show()

    try:
        resume_token = None
        pipeline = [{"$match": {"operationType": "insert"}}]
        with mongo_collection.watch(pipeline) as stream:
            for _ in stream:
                handle_new_data()
                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError:
        # The ChangeStream encountered an unrecoverable error or the
        # resume attempt failed to recreate the cursor.
        if resume_token is None:
            # There is no usable resume token because there was a
            # failure during ChangeStream initialization.
            logger.error("...")
        else:
            # Use the interrupted ChangeStream's resume token to create
            # a new ChangeStream. The new stream will continue from the
            # last seen insert change without missing any events.
            with mongo_collection.watch(pipeline, resume_after=resume_token) as stream:
                for _ in stream:
                    handle_new_data()
