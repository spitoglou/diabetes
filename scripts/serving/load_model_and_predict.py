import glob
import os
import re
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import pymongo
from loguru import logger
from pycaret.regression import load_model, predict_model
from pymongo import errors as pymongo_errors

from config.settings import settings
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.helpers.misc import debug_print, get_part_of_day
from src.mongo import MongoDB

mongo = MongoDB()
db = mongo.client[settings.DATABASE]

# Default collection for backward compatibility
mongo_collection = db[f"measurements_{settings.OHIO_ID}"]

# Initialize model as None - will be loaded when needed
model = None
model_path = None
current_patient_id = settings.OHIO_ID


def get_collection_for_patient(patient_id: str):
    """Get MongoDB collection for a specific patient."""
    return db[f"measurements_{patient_id}"]


def get_predictions_collection_for_patient(patient_id: str):
    """Get predictions MongoDB collection for a specific patient."""
    return db[f"predictions_{patient_id}"]


def load_trained_model(
    patient_id: str | None = None,
    window_steps: int | None = None,
    horizon_steps: int | None = None,
    data_source: str | None = None,
):
    """Load the trained model if available.

    Args:
        patient_id: Patient identifier (e.g., "559" for Ohio, "adult#001" for simglucose).
        window_steps: Number of historical readings used as features.
        horizon_steps: Number of steps ahead to predict.
        data_source: Data source ("ohio" or "simglucose"). If None, auto-detects
                    based on patient_id format (# in name = simglucose).

    Returns:
        Loaded model or None if not found.

    Model naming convention:
        - Ohio: models/{patient}_{window}_{horizon}_1*.pkl
        - Simglucose: models/sim_{patient}_{window}_{horizon}_1*.pkl
    """
    global model, model_path, current_patient_id

    pid = patient_id or settings.OHIO_ID
    window = window_steps or settings.WINDOW_STEPS
    horizon = horizon_steps or settings.PREDICTION_HORIZON

    # Auto-detect data source from patient ID format if not specified
    if data_source is None:
        # Simglucose patients contain '#' (e.g., adult#001)
        data_source = "simglucose" if "#" in str(pid) else "ohio"

    # Reload if patient changed
    if model is not None and pid == current_patient_id:
        return model

    current_patient_id = pid
    model = None  # Reset to force reload

    try:
        # Build search pattern based on data source
        if data_source == "simglucose":
            pattern = f"models/sim_{pid}_{window}_{horizon}_1*.pkl"
        else:
            pattern = f"models/{pid}_{window}_{horizon}_1*.pkl"

        model_files = glob.glob(pattern)

        # Fall back to old naming convention if no match found
        if not model_files and data_source == "simglucose":
            # Try without sim_ prefix for backward compatibility
            fallback_pattern = f"models/{pid}_{window}_{horizon}_1*.pkl"
            model_files = glob.glob(fallback_pattern)
            if model_files:
                logger.info(f"Found model with legacy naming: {fallback_pattern}")

        if not model_files:
            logger.warning(f"No model files found matching pattern: {pattern}")
            return None

        model_file = model_files[0]
        model_path = os.path.splitext(model_file)[0]
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Could not load model [{e}]")
        return None


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
    if model is None:
        logger.error("Model is not loaded")
        return None

    saved_model_features = model.feature_names_in_
    stream = pd.DataFrame(last_n).reset_index(drop=True)
    debug_print(
        "Streamed Dataframe used for prediction", stream
    ) if settings.DEBUG else ...
    features = featurize_stream_df(stream, window_steps, prediction_horizon)
    # features
    prediction = predict_model(
        model, correct_features(features, saved_model_features)
    ).prediction_label[0]
    logger.info(f"The prediction is {prediction}")
    return prediction


def retrieve_data(mongo_collection, limit: int = 200):
    return pd.DataFrame(list(mongo_collection.find(limit=limit, sort=[("_id", -1)])))


def create_measurements_list(timeseries_df):
    meas_list = []
    measurement = None  # Initialize to handle empty dataframe case

    for index, row in timeseries_df.iterrows():
        debug_print("Measurement row", row) if settings.DEBUG else ...
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
    debug_print("List of Measurements", meas_list) if settings.DEBUG else ...
    # * "measurement" variable here is the last measurement from the loop above
    return meas_list, measurement


def mongo_prediction(
    window_steps,
    prediction_horizon,
    patient_id: str | None = None,
    sample_interval: int | None = None,
):
    """Make a prediction based on recent MongoDB measurements.

    Args:
        window_steps: Number of historical readings to use for prediction.
        prediction_horizon: Number of steps ahead to predict.
        patient_id: Patient ID for data retrieval. Defaults to settings.OHIO_ID.
        sample_interval: Minutes between CGM readings. Defaults to settings.SAMPLE_INTERVAL.
            - Ohio dataset: 5 minutes (Guardian sensor)
            - Simglucose: 3 minutes (Dexcom sensor)

    Returns:
        Tuple of (measurement_df, prediction_dict) or (None, None) on error.
    """
    pid = patient_id or settings.OHIO_ID
    interval = sample_interval or settings.SAMPLE_INTERVAL

    # Load model if not already loaded
    current_model = load_trained_model(pid, window_steps, prediction_horizon)
    if current_model is None:
        logger.error("Cannot make prediction: model not available")
        return None, None

    collection = get_collection_for_patient(pid)
    ts_df = retrieve_data(collection, 12)
    # reverse dataframe
    ts_df = ts_df[::-1].reset_index(drop=True)
    debug_print("Timeseries Dataframe", ts_df) if settings.DEBUG else ...

    meas_list, last_measurement = create_measurements_list(ts_df)

    # Check if we have valid measurements
    if not meas_list or last_measurement is None:
        logger.error("No measurements available for prediction")
        return None, None

    last_n = meas_list[-1 * window_steps :]
    prediction = predict_last_n(last_n, current_model, window_steps, prediction_horizon)

    # Calculate prediction time using configurable sample interval
    # prediction_horizon * sample_interval = minutes ahead
    # E.g., horizon=6, interval=5 -> 30 min; horizon=6, interval=3 -> 18 min
    prediction_minutes = prediction_horizon * interval
    prediction = {
        "prediction_origin_time": last_measurement.date_time,
        "prediction_time": last_measurement.date_time
        + timedelta(minutes=prediction_minutes),
        "prediction_value": prediction,
    }

    debug_print("Prediction", prediction)
    measurement_df = pd.DataFrame(meas_list)
    # prediction_df = pd.DataFrame(predictions)
    return measurement_df, prediction


def handle_new_data(
    patient_id: str | None = None,
    window_steps: int | None = None,
    horizon_steps: int | None = None,
    sample_interval: int | None = None,
):
    pid = patient_id or settings.OHIO_ID
    window = window_steps or settings.WINDOW_STEPS
    horizon = horizon_steps or settings.PREDICTION_HORIZON
    interval = sample_interval or settings.SAMPLE_INTERVAL

    try:
        measurement_df, prediction = mongo_prediction(window, horizon, pid, interval)

        logger.info("Inserting prediction in Database")
        pred_db = get_predictions_collection_for_patient(pid)
        rec_id = pred_db.insert_one(prediction).inserted_id
        logger.success(rec_id)
    except Exception as e:
        logger.error(e)


def run_prediction_watcher(
    patient_id: str | None = None,
    window_steps: int | None = None,
    horizon_steps: int | None = None,
    sample_interval: int | None = None,
):
    """Run the prediction watcher for a specific patient.

    Args:
        patient_id: Patient ID to watch. Defaults to settings.OHIO_ID.
        window_steps: Number of historical readings. Defaults to settings.WINDOW_STEPS.
        horizon_steps: Steps ahead to predict. Defaults to settings.PREDICTION_HORIZON.
        sample_interval: Minutes between CGM readings. Defaults to settings.SAMPLE_INTERVAL.
            - Ohio dataset: 5 minutes (Guardian sensor)
            - Simglucose: 3 minutes (Dexcom sensor)
    """
    pid = patient_id or settings.OHIO_ID
    window = window_steps or settings.WINDOW_STEPS
    horizon = horizon_steps or settings.PREDICTION_HORIZON
    interval = sample_interval or settings.SAMPLE_INTERVAL
    collection = get_collection_for_patient(pid)

    resume_token = None
    pipeline = [{"$match": {"operationType": "insert"}}]

    # Calculate prediction time for logging
    prediction_minutes = horizon * interval

    try:
        logger.info(
            f"Starting Database Watch for patient {pid} "
            f"(window={window}, horizon={horizon}, interval={interval}min, "
            f"predicting {prediction_minutes}min ahead)"
        )
        with collection.watch(pipeline) as stream:
            for _ in stream:
                handle_new_data(pid, window, horizon, interval)
                resume_token = stream.resume_token
    except pymongo_errors.PyMongoError as e:
        if resume_token is None:
            logger.error(e)
        else:
            with collection.watch(pipeline, resume_after=resume_token) as stream:
                for _ in stream:
                    handle_new_data(pid, window, horizon, interval)


if __name__ == "__main__":
    resume_token = None
    pipeline = [{"$match": {"operationType": "insert"}}]

    try:
        logger.info("Starting Database Watch")
        with mongo_collection.watch(pipeline) as stream:
            for _ in stream:
                handle_new_data()
                resume_token = stream.resume_token
    except pymongo_errors.PyMongoError as e:
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
