import streamsync as ss
import config.mongo_config as mg_conf
from src.mongo import MongoDB
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# This is a placeholder to get you started or refresh your memory.
# Delete it or adapt it as necessary.
# Documentation is available at https://streamsync.cloud

# Shows in the log when the app starts
print("Application Start!!")

mongo = MongoDB()
db = mongo.client[mg_conf.MONGO_DATABASE]
# TODO: Find a way to remove hardcoding via state?

LOW = 70
HIGH = 180


def _retrieve_data(limit: int = 50):
    # TODO: Find a way to remove hardcoding via state?
    subject = "559"
    measurements_collection = db[f"measurements_{subject}"]
    predictions_collection = db[f"predictions_{subject}"]

    measurements_df = pd.DataFrame(
        list(measurements_collection.find(limit=limit, sort=[("_id", -1)]))
    )
    measurements_df["_id"] = measurements_df["_id"].astype(pd.StringDtype())
    measurements_df["date_time"] = pd.to_datetime(
        measurements_df["effectiveDateTime"]
    ).astype(str)
    measurements_df["value"] = None
    for index, row in measurements_df.iterrows():
        print(row["valueQuantity"]["value"])
        measurements_df.at[index, "value"] = row["valueQuantity"]["value"]
        # row['value'] = row['valueQuantity']['value']

    predictions_df = pd.DataFrame(
        list(predictions_collection.find(limit=limit, sort=[("_id", -1)]))
    )
    predictions_df["_id"] = predictions_df["_id"].astype(pd.StringDtype())
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
    ).update_layout(yaxis={'range': [0, 450]})

    temp_graph.add_traces(
        list(
            px.line(
                predictions,
                x="prediction_time",
                y="prediction_value",
                range_y=[0,450]
                # ?title='Room Temp'
            ).update_traces(line_color='red').select_traces()
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
    value = measurement['value']
    if LOW < value < HIGH:
        marker = '+'
        note = 'In Range'
    else:
        marker = '-'
        if value <= LOW:
            note = 'Below Range'
        else:
            note = 'Above Range'
    dt = measurement['date_time']
    
    return value, dt, f'{marker}{note}'

def _get_last_prediction(prediction):
    value = prediction['prediction_value']
    if LOW < value < HIGH:
        marker = '+'
        note = 'In Range'
    else:
        marker = '-'
        if value <= LOW:
            note = 'Below Range'
        else:
            note = 'Above Range'
    dt = prediction['prediction_time']
    
    return value, dt, f'{marker}{note}'

def decrement(state):
    state["counter"] -= 1
    _update_message(state)


def increment(state):
    state["counter"] += 1
    # Shows in the log when the event handler is run
    print(f"The counter has been incremented.")
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

meas_df = pd.DataFrame()
pred_df = pd.DataFrame()

initial_state = ss.init_state(
    {
        "my_app": {"title": "SP PhD"},
        "_my_private_element": 1337,
        "message": None,
        "counter": 26,
        "graph": None,
        "data": {
            "measurements": meas_df,
            "predictions": pred_df,
        },
        "graph": None,
        "last_measurement": 0,
        "last_measurement_time": None,
        "last_measurement_note": None,
        "last_prediction": 0,
        "last_prediction_time": None,
        "last_prediction_note": None,
    }
)

_update_message(initial_state)
