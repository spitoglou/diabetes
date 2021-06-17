from src.helpers.experiment import create_tsfresh_dataframe
from src.helpers.diabetes.cega import clarke_error_grid
from src.helpers.diabetes.madex import mean_adjusted_exponent_error
from pycaret.regression import setup, create_model, compare_models, predict_model
from loguru import logger
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def run_experiment(train_parameters, unseen_data_parameters):

    source_df = create_tsfresh_dataframe(train_parameters)
    clean_df = source_df.drop(
        columns=['start', 'end', 'start_time', 'end_time'])

    exp_reg = setup(clean_df,
                    target='label',
                    feature_selection=True,
                    html=False,
                    silent=True
                    )

    best3 = compare_models(
        exclude=['catboost', 'xgboost'],
        sort='RMSE',
        n_select=3,
        # verbose=False
    )

    for selected_model in best3:
        output = {}
        model = create_model(selected_model)
        output['model'] = model
        pd = predict_model(model)
        (_, res) = clarke_error_grid(pd['label'], pd['Label'], 'Test')
        output['internal_cga_analysis'] = res
        rmse = np.sqrt(mean_squared_error(pd['label'], pd['Label']))
        rmadex = np.sqrt(mean_adjusted_exponent_error(
            pd['label'], pd['Label']))
        output['internal_rmese'] = rmse
        output['internal_rmadex'] = rmadex

        unseen_df = create_tsfresh_dataframe(unseen_data_parameters)
        clean_unseen_df = unseen_df.drop(
            columns=['start', 'end', 'start_time', 'end_time'])
        unseen_pd = predict_model(model, data=clean_unseen_df)
        (_, res) = clarke_error_grid(unseen_pd['label'], unseen_pd['Label'], 'Test')
        output['unseen_cga_analysis'] = res
        rmse = np.sqrt(mean_squared_error(unseen_pd['label'], unseen_pd['Label']))
        rmadex = np.sqrt(mean_adjusted_exponent_error(
            unseen_pd['label'], unseen_pd['Label']))
        output['unseen_rmese'] = rmse
        output['unseen_rmadex'] = rmadex
        logger.info(output)


if __name__ == '__main__':
    parameters = {
        'ohio_no': 559,
        'scope': 'train',
        'train_ds_size': 0,
        'window_size': 6,
        'prediction_horizon': 1,
        'minimal_features': False,
    }

    test_parameters = {
        'ohio_no': 559,
        'scope': 'test',
        'train_ds_size': 100000,
        'window_size': 6,
        'prediction_horizon': 1,
        'minimal_features': False,
    }

    run_experiment(parameters, test_parameters)
