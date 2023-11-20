import streamsync as ss
import config.mongo_config as mg_conf
from classes.mongo import MongoDB
import pandas as pd
from datetime import datetime, timezone
import plotly.express as px

# Shows in the log when the app starts
print("Application Start!")

INACTIVITY_THRESHOLD = 10
TEMP_LOW_LIMIT = 20
TEMP_HIGH_LIMIT = 30
CAM_MOVE_THRESHOLD = 500

mongo = MongoDB()
db = mongo.client[mg_conf.MONGO_DATABASE]
mongo_collection = db[mg_conf.MONGO_COLLECTION]

# Its name starts with _, so this function won't be exposed


def _seconds_passed(timestamp):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()


def _retrieve_sensor_data(limit: int = 200):
    sensor_data = pd.DataFrame(list(mongo_collection.find(
        limit=limit,
        sort=[('_id', -1)]
    )))
    sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp']).astype(str)
    # ?test = test.drop(columns=['_id'])
    sensor_data['_id'] = sensor_data['_id'].astype(pd.StringDtype())
    return sensor_data


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
        annotation_position="top left"
    )
    return temp_graph


def _create_cam_mov_graph(dataframe, threshold):
    cam_move_graph = px.scatter(
        dataframe,
        x="timestamp",
        y="value",
        # ?title='Camera Movement'
    )
    cam_move_graph.add_hline(y=threshold)
    # Just add lines
    # temp_graph.add_hline(y=27)
    # graph.add_hrect(
    #     y0=low,
    #     y1=high,
    #     line_width=1,
    #     fillcolor="green",
    #     opacity=0.2,
    #     annotation_text="In Range",
    #     annotation_position="top left"
    # )
    return cam_move_graph


def refresh(state):
    sensor_data = _retrieve_sensor_data()
    temp_df = sensor_data[sensor_data['sensor'] == 'Sensor1']
    cam_mov_df = sensor_data[sensor_data['sensor'] == 'Camera.movement']
    sensor2_df = sensor_data[sensor_data['sensor'] == 'Sensor2']

    # Room Temperature
    temperature = temp_df.iloc[0]
    if _seconds_passed(temperature['timestamp']) > INACTIVITY_THRESHOLD:
        state['temperature'] = 'Inactive'
        state['temp_message'] = f'No data for more than {INACTIVITY_THRESHOLD} seconds'
    else:
        state['temperature'] = f"{str(temperature['value'])} Celsius"
        state['temp_message'] = ('+In Range' if TEMP_LOW_LIMIT < int(temperature['value']) < TEMP_HIGH_LIMIT else '-Out of Range')
    state['temp_graph'] = _create_temp_graph(temp_df, TEMP_LOW_LIMIT, TEMP_HIGH_LIMIT)

    state['data'] = {'sensor1': temp_df, 'sensor2': sensor2_df, 'camera_movement': cam_mov_df}

    # Camera Movement
    camera_movement = cam_mov_df.iloc[0]
    if _seconds_passed(camera_movement['timestamp']) > INACTIVITY_THRESHOLD:
        state['camera_movement'] = 'Inactive'
        state['cam_move_message'] = f'No data for more than {INACTIVITY_THRESHOLD} seconds'
    else:
        state['camera_movement'] = f"{str(camera_movement['value'])}"
        state['cam_move_message'] = '+Receiving'
        state['cam_move_message'] += (' Below Threshold' if int(camera_movement['value']) < CAM_MOVE_THRESHOLD else ' Above Threshold')
    state['cam_mov_graph'] = _create_cam_mov_graph(cam_mov_df, CAM_MOVE_THRESHOLD)
# Initialise the state


# "_my_private_element" won't be serialised or sent to the frontend,
# because it starts with an underscore
sensor1 = pd.DataFrame()
sensor2 = pd.DataFrame()
camera_movement = pd.DataFrame()

initial_state = ss.init_state({
    "my_app": {
        "title": "Sandalphon Dashboard"
    },
    "_my_private_element": 1337,
    "temperature": '0',
    "camera_movement": '0',
    "cam_move_message": None,
    "temp_message": None,
    "counter": 26,
    "data": {
        'sensor1': sensor1,
        'sensor2': sensor2,
        'camera_movement': camera_movement,
    },
    "temp_graph": None,
    "cam_mov_graph": None,
})

# _update_message(initial_state)
