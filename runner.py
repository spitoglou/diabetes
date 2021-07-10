from src.helpers.experiment import create_tsfresh_dataframe
from src.helpers.diabetes.cega import clarke_error_grid
from src.helpers.diabetes.madex import mean_adjusted_exponent_error
from pycaret.regression import setup, create_model, compare_models, predict_model, get_config, pull
# import matplotlib.pyplot as plt
from loguru import logger
# import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
# import pandas


def run(patient, window, horizon):
    print(patient, window, horizon)

    train_parameters = {
        'ohio_no': patient,
        'scope': 'train',
        'train_ds_size': 0,
        'window_size': window,
        'prediction_horizon': horizon,
        'minimal_features': False,
    }

    unseen_data_parameters = {
        'ohio_no': patient,
        'scope': 'test',
        'train_ds_size': 0,
        'window_size': window,
        'prediction_horizon': horizon,
        'minimal_features': False,
    }

    source_df = create_tsfresh_dataframe(train_parameters)
    clean_df = source_df.drop(
        columns=['start', 'end', 'start_time', 'end_time'])
    print(clean_df.columns)

    regressor = setup(clean_df,
                      target='label',
                      feature_selection=True,
                      ignore_low_variance=True,
                      html=False,
                      silent=True,
                      verbose=False,
                      session_id=1974
                      )
    print(get_config('prep_pipe'))
    print(get_config('X').columns)

    best3 = compare_models(
        exclude=['catboost', 'xgboost', 'et', 'rf', 'ada', 'gbr'],
        sort='RMSE',
        n_select=3,
        verbose=True
    )
    comparison_df = pull()
    print(comparison_df)

    output = {}
    output['patient'] = patient
    output['window'] = f'{window}({(window*5)} minutes)'
    output['horizon'] = f'{horizon}({(horizon*5)} minutes)'
    model = create_model(best3[0])
    first_model_df = pull()
    print(first_model_df)
    model_name = model.__str__().split('(')[0]
    logger.info(f'Processing model: {model_name}')
    output['model'] = model
    output['model_name'] = model_name
    pd = predict_model(model)
    (_, res) = clarke_error_grid(pd['label'], pd['Label'], 'Test')
    # plt.show()
    output['internal_cga_analysis'] = res
    rmse = np.sqrt(mean_squared_error(pd['label'], pd['Label']))
    rmadex = np.sqrt(mean_adjusted_exponent_error(
        pd['label'], pd['Label']))
    output['internal_rmse'] = rmse
    output['internal_rmadex'] = rmadex

    unseen_df = create_tsfresh_dataframe(unseen_data_parameters)
    clean_unseen_df = unseen_df.drop(
        columns=['start', 'end', 'start_time', 'end_time'])
    unseen_pd = predict_model(model, data=clean_unseen_df)
    (_, res) = clarke_error_grid(
        unseen_pd['label'], unseen_pd['Label'], 'Test')
    # plt.show()
    output['unseen_cga_analysis'] = res
    rmse = np.sqrt(mean_squared_error(unseen_pd['label'], unseen_pd['Label']))
    rmadex = np.sqrt(mean_adjusted_exponent_error(
        unseen_pd['label'], unseen_pd['Label']))
    output['unseen_rmse'] = rmse
    output['unseen_rmadex'] = rmadex
#     logger.info(output)

    return output


if __name__ == '__main__':
    patients = [559]
    windows = [6]
    horizons = [6]
    outcome = [run(patient, window, horizon)
               for patient in patients for window in windows for horizon in horizons]
    print(outcome)
