from pycaret.regression import load_model, predict_model
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.helpers.misc import debug_print, get_part_of_day
from datetime import datetime, timedelta
import pymongo
import re
import pandas as pd
import matplotlib.pyplot as plt
from src.mongo import MongoDB
from config.server_config import DATABASE, OHIO_ID, WINDOW_STEPS, PREDICTION_HORIZON, DEBUG
from loguru import logger
import glob
import os

mongo = MongoDB()
db = mongo.client[DATABASE]
mongo_collection = db[f'measurements_{OHIO_ID}']

try:
    model_file = glob.glob(f'models/{OHIO_ID}_{WINDOW_STEPS}_{PREDICTION_HORIZON}_1*.pkl')[0]
    model_path = os.path.splitext(model_file)[0]
    print(model_path)
    model = load_model(
        model_path
    )
except Exception as e:
    logger.error(f"Could not load model [{e}]")
    exit(1)


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


def predict_last_n(last_n: list, model, window_steps, prediction_horizon):
    saved_model_features = model.feature_names_in_
    stream = pd.DataFrame(last_n).reset_index(drop=True)
    debug_print('Streamed Dataframe used for prediction', stream) if DEBUG else ...
    features = featurize_stream_df(stream, window_steps, prediction_horizon)
    # features
    prediction = predict_model(
        model, correct_features(features, saved_model_features)
    ).prediction_label[0]
    logger.info(f'The prediction is {prediction}')
    return prediction


def retrieve_data(mongo_collection, limit: int = 200):
    return pd.DataFrame(list(mongo_collection.find(limit=limit, sort=[("_id", -1)])))


def create_measurements_list(timeseries_df):
    meas_list = []
    for index, row in timeseries_df.iterrows():
        debug_print('Measurement row', row) if DEBUG else ...
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
    debug_print('List of Measurements',meas_list) if DEBUG else ...
    # * "measurement" variable here is the last measurement from the loop above
    return meas_list, measurement


def mongo_prediction(window_steps, prediction_horizon):
    ts_df = retrieve_data(mongo_collection, 12)
    # reverse dataframe
    ts_df = ts_df[::-1].reset_index(drop=True)
    debug_print('Timeseries Dataframe', ts_df) if DEBUG else ...

    meas_list, last_measurement = create_measurements_list(ts_df)
    last_n = meas_list[-1 * window_steps :]
    prediction = predict_last_n(last_n, model, window_steps, prediction_horizon)
    prediction = {
        "prediction_origin_time": last_measurement.date_time,
        "prediction_time": last_measurement.date_time + timedelta(minutes=prediction_horizon * 5),
        "prediction_value": prediction,
    }

    debug_print('Prediction', prediction)
    measurement_df = pd.DataFrame(meas_list)
    # prediction_df = pd.DataFrame(predictions)
    return measurement_df, prediction


def handle_new_data():
    # # mongo_predictions(model)
    try:
        measurement_df, prediction = mongo_prediction(WINDOW_STEPS, PREDICTION_HORIZON)
        
        logger.info('Inserting predicition in Database')
        pred_db = db[f'predictions_{OHIO_ID}']
        rec_id = pred_db.insert_one(prediction).inserted_id
        logger.success(rec_id)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":

    try:
        resume_token = None
        pipeline = [{"$match": {"operationType": "insert"}}]
        logger.info("Starting Database Watch")
        with mongo_collection.watch(pipeline) as stream:
            for _ in stream:
                handle_new_data()
                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as e:
        # The ChangeStream encountered an unrecoverable error or the
        # resume attempt failed to recreate the cursor.
        if resume_token is None:
            # There is no usable resume token because there was a
            # failure during ChangeStream initialization.
            logger.error(e)
        else:
            # Use the interrupted ChangeStream's resume token to create
            # a new ChangeStream. The new stream will continue from the
            # last seen insert change without missing any events.
            with mongo_collection.watch(pipeline, resume_after=resume_token) as stream:
                for _ in stream:
                    handle_new_data()
