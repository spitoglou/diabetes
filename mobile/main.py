import streamsync as ss
import config.mongo_config as mg_conf
from src.mongo import MongoDB
import pandas as pd
import plotly.express as px

# This is a placeholder to get you started or refresh your memory.
# Delete it or adapt it as necessary.
# Documentation is available at https://streamsync.cloud

# Shows in the log when the app starts
print("Application Start!!")

mongo = MongoDB()
db = mongo.client[mg_conf.MONGO_DATABASE]
# TODO: Find a way to remove hardcoding via state?


def _retrieve_data(limit: int = 50):
    # TODO: Find a way to remove hardcoding via state?
    subject = "559"
    measurements_collection = db[f"measurements_{subject}"]
    predictions_collection = db[f"predictions_{subject}"]

    measurements_df = pd.DataFrame(
        list(measurements_collection.find(limit=limit, sort=[("_id", -1)]))
    )
    measurements_df['_id'] = measurements_df['_id'].astype(pd.StringDtype())

    predictions_df = pd.DataFrame(
        list(predictions_collection.find(limit=limit, sort=[("_id", -1)]))
    )
    predictions_df['_id'] = predictions_df['_id'].astype(pd.StringDtype())
    predictions_df['prediction_origin_time'] = pd.to_datetime(predictions_df['prediction_origin_time']).astype(str)
    predictions_df['prediction_time'] = pd.to_datetime(predictions_df['prediction_time']).astype(str)

    return measurements_df, predictions_df


def _create_temp_graph(dataframe, low, high):
    temp_graph = px.scatter(
        dataframe,
        x="timestamp",
        y="value",
        # ?title='Room Temp'
    )
    # Just add lines
    # temp_graph.add_hline(y=TEMP_LOW_LIMIT)
    # temp_graph.add_hline(y=27)
    temp_graph.add_hrect(
        y0=low,
        y1=high,
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
    state['data']['measurements'] = measurements
    state['data']['predictions'] = predictions

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
        }
    }
)

_update_message(initial_state)
