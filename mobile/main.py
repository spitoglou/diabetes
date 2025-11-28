"""
StreamSync web dashboard for glucose monitoring.

Displays live glucose measurements and predictions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
# Handle both direct execution and StreamSync's exec context
try:
    parent_dir = Path(__file__).parent.parent
except NameError:
    # __file__ not defined in StreamSync's exec context
    # cwd is the mobile directory, so parent is project root
    parent_dir = Path.cwd().parent

sys.path.insert(0, str(parent_dir))

import pandas as pd
import plotly.express as px
import streamsync as ss
from loguru import logger

from src.config import get_config
from src.mongo import MongoDB
from src.repositories.measurement_repository import MeasurementRepository
from src.repositories.prediction_repository import PredictionRepository

# Initialize configuration and repositories
logger.info("Application Start!")

config = get_config()
LOW = config.glucose_low
HIGH = config.glucose_high

# Initialize MongoDB and repositories
try:
    mongo = MongoDB(config)
    db = mongo.get_database()
    measurement_repo = MeasurementRepository(db, config)
    prediction_repo = PredictionRepository(db, config)
    logger.info("Database connection initialized")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    measurement_repo = None
    prediction_repo = None


def _retrieve_data(limit: int = 50):
    """Retrieve recent measurements and predictions."""
    subject = config.default_patient_id

    if measurement_repo is None or prediction_repo is None:
        return pd.DataFrame(), pd.DataFrame()

    # Get measurements
    raw_measurements = measurement_repo.get_recent(subject, limit=limit)
    measurements_df = pd.DataFrame(raw_measurements)

    if not measurements_df.empty:
        measurements_df["_id"] = measurements_df["_id"].astype(str)
        measurements_df["date_time"] = pd.to_datetime(
            measurements_df["effectiveDateTime"]
        ).astype(str)
        measurements_df["value"] = measurements_df["valueQuantity"].apply(
            lambda x: x.get("value") if isinstance(x, dict) else None
        )

    # Get predictions
    raw_predictions = prediction_repo.get_recent(subject, limit=limit)
    predictions_df = pd.DataFrame(raw_predictions)

    if not predictions_df.empty:
        predictions_df["_id"] = predictions_df["_id"].astype(str)
        predictions_df["prediction_origin_time"] = pd.to_datetime(
            predictions_df["prediction_origin_time"]
        ).astype(str)
        predictions_df["prediction_time"] = pd.to_datetime(
            predictions_df["prediction_time"]
        ).astype(str)

    return measurements_df, predictions_df


def _create_graph(measurements, predictions):
    measurements = measurements.sort_values(by="date_time")
    predictions = predictions.sort_values(by="prediction_time")

    temp_graph = px.line(
        measurements,
        x="date_time",
        y="value",
        # ?title='Room Temp'
    ).update_layout(yaxis={"range": [0, 450]})

    temp_graph.add_traces(
        list(
            px.line(
                predictions,
                x="prediction_time",
                y="prediction_value",
                range_y=[0, 450],
                # ?title='Room Temp'
            )
            .update_traces(line_color="red")
            .select_traces()
        )
    )
    # Just add lines
    # temp_graph.add_hline(y=TEMP_LOW_LIMIT)
    # temp_graph.add_hline(y=27)
    temp_graph.add_hrect(
        y0=LOW,
        y1=HIGH,
        line_width=1,
        fillcolor="green",
        opacity=0.2,
        annotation_text="In Range",
        annotation_position="top left",
    )
    return temp_graph


# Its name starts with _, so this function won't be exposed
def _update_message(state):
    is_even = state["counter"] % 2 == 0
    message = "+Even" if is_even else "-Odd"
    state["message"] = message


def _get_last_measurement(measurement):
    value = measurement["value"]
    if LOW < value < HIGH:
        marker = "+"
        note = "In Range"
    else:
        marker = "-"
        if value <= LOW:
            note = "Below Range"
        else:
            note = "Above Range"
    dt = measurement["date_time"]

    return value, dt, f"{marker}{note}"


def _get_last_prediction(prediction):
    value = prediction["prediction_value"]
    if LOW < value < HIGH:
        marker = "+"
        note = "In Range"
    else:
        marker = "-"
        if value <= LOW:
            note = "Below Range"
        else:
            note = "Above Range"
    dt = prediction["prediction_time"]

    return value, dt, f"{marker}{note}"


def decrement(state):
    state["counter"] -= 1
    _update_message(state)


def increment(state):
    state["counter"] += 1
    # Shows in the log when the event handler is run
    print("The counter has been incremented.")
    _update_message(state)


def refresh(state):
    measurements, predictions = _retrieve_data()
    state["data"]["measurements"] = measurements
    state["data"]["predictions"] = predictions
    state["graph"] = _create_graph(measurements, predictions)
    lm, lm_dt, lm_note = _get_last_measurement(measurements.iloc[0])
    state["last_measurement"] = lm
    state["last_measurement_time"] = lm_dt
    state["last_measurement_note"] = lm_note
    pr, pr_dt, pr_note = _get_last_prediction(predictions.iloc[0])
    state["last_prediction"] = pr
    state["last_prediction_time"] = pr_dt
    state["last_prediction_note"] = pr_note


# Initialise the state

# "_my_private_element" won't be serialised or sent to the frontend,
# because it starts with an underscore

# Auto-load data on startup
meas_df, pred_df = _retrieve_data()
logger.info(
    f"Loaded {len(meas_df)} measurements and {len(pred_df)} predictions on startup"
)

# Create initial graph if we have data
initial_graph = None
last_meas = 0
last_meas_time = None
last_meas_note = None
last_pred = 0
last_pred_time = None
last_pred_note = None

if not meas_df.empty and not pred_df.empty:
    initial_graph = _create_graph(meas_df, pred_df)
    last_meas, last_meas_time, last_meas_note = _get_last_measurement(meas_df.iloc[0])
    last_pred, last_pred_time, last_pred_note = _get_last_prediction(pred_df.iloc[0])

initial_state = ss.init_state(
    {
        "my_app": {"title": "SP PhD"},
        "_my_private_element": 1337,
        "message": None,
        "counter": 26,
        "graph": initial_graph,
        "data": {
            "measurements": meas_df,
            "predictions": pred_df,
        },
        "last_measurement": last_meas,
        "last_measurement_time": last_meas_time,
        "last_measurement_note": last_meas_note,
        "last_prediction": last_pred,
        "last_prediction_time": last_pred_time,
        "last_prediction_note": last_pred_note,
    }
)

_update_message(initial_state)
