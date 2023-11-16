from pycaret.regression import load_model, predict_model
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from datetime import datetime, timedelta
import re
import pandas as pd
import matplotlib.pyplot as plt
from src.mongo import MongoDB
from config.server_config import DATABASE, COLLECTION

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
            if feat != 'label':
                corrected_df[feat] = None
    return corrected_df


def featurize_stream_df(stream_df, window, horizon):
    featurizer = TsfreshFeaturizer(stream_df.tail(window), window, horizon, plot_chunks=False, minimal_features=False)
    featurizer.chunks = 1
    featurizer.create_feature_dataframe()
    return featurizer.feature_dataframe


if __name__ == '__main__':
    model = load_model('models/559_6_6_1_LGBMRegressor_3e73c509-6e25-421e-989a-e303e4a58c06')
    saved_model_features = model.feature_names_in_

    provider = OhioBgcProvider()
    measurements = provider.tsfresh_dataframe()

    stream = pd.DataFrame()
    
    mongo = MongoDB()
    db = mongo.client[DATABASE]
    mongo_collection = db[COLLECTION]

    max_data = 7
    meas_list = []
    predictions = []
    for counter in range(max_data):
        measurement = measurements.iloc[counter]
        print(measurement)
        meas_list.append(measurement)
        if len(meas_list) >= 6:
            last_n = meas_list[-6:]
            stream = pd.DataFrame(last_n).reset_index(drop=True)
            print(stream)
            features = featurize_stream_df(stream, 6, 6)
            # features
            prediction = predict_model(model, correct_features(features, saved_model_features)).prediction_label[0]
            print(prediction)
            predictions.append({
                'prediction_origin_time': measurement.date_time,
                'prediction_time': measurement.date_time + timedelta(minutes=6*5),
                'prediction_value': prediction,
            })
    measurement_df = pd.DataFrame(meas_list)
    prediction_df = pd.DataFrame(predictions)
    print(prediction_df)
    ax = prediction_df.plot(y='prediction_value', x='prediction_time')
    measurement_df.plot(ax=ax, y='bg_value', x='date_time')
    plt.show()