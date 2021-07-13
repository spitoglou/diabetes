from src.helpers.experiment import create_tsfresh_dataframe
from src.helpers.diabetes.cega import clarke_error_grid
from src.helpers.diabetes.madex import mean_adjusted_exponent_error
from pycaret.regression import setup, create_model, compare_models, predict_model, get_config, pull
import matplotlib.pyplot as plt
from loguru import logger
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas
from pprint import pprint


class Experiment():

    def __init__(self, patient, window, horizon, min_per_measure: int = 5, best_models_no: int = 3, speed: int = 3, minimal_features: bool = False) -> None:
        self.patient = patient
        self.window = window
        self.horizon = horizon
        self.win_min = window * min_per_measure
        self.hor_min = horizon * min_per_measure
        self.best_models_no = best_models_no
        self.speed = speed
        self.train_parameters = {
            'ohio_no': patient,
            'scope': 'train',
            'train_ds_size': 0,
            'window_size': window,
            'prediction_horizon': horizon,
            'minimal_features': minimal_features,
        }

        self.unseen_data_parameters = {
            'ohio_no': patient,
            'scope': 'test',
            'train_ds_size': 0,
            'window_size': window,
            'prediction_horizon': horizon,
            'minimal_features': minimal_features,
        }

    def create_dataframe(self, parameters: dict):
        return create_tsfresh_dataframe(parameters)

    def create_train_dataframe(self):
        self.train_df = self.create_dataframe(self.train_parameters)

    def create_unseen_data_dataframe(self):
        self.unseen_data_df = self.create_dataframe(
            self.unseen_data_parameters)

    def setup_regressor(self):
        self.regressor = setup(self.train_df,
                               target='label',
                               feature_selection=True,
                               ignore_low_variance=True,
                               html=False,
                               silent=True,
                               verbose=False,
                               session_id=1974
                               )

    def get_regressor_param(self, param: str):
        return get_config(param)

    def log_regressor_param(self, param: str):
        logger.info(self.get_regressor_param(param))

    def compute_best_n_models(self, verbose=True):
        exc_dict = {
            1: [],
            2: ['catboost', 'xgboost'],
            3: ['catboost', 'xgboost', 'et', 'rf', 'ada', 'gbr']
        }
        self.best_models = compare_models(
            exclude=exc_dict[self.speed],
            sort='RMSE',
            n_select=self.best_models_no,
            verbose=verbose
        )
        self.models_comparison_df = pull()

    def log_best_models(self):
        for index, model in enumerate(self.best_models):
            logger.info(f'Model {(index+1)}:')
            logger.info(model)

    def calculate_prediction(self, model, custom_data=None, legend=''):
        if not model:
            model = self.best_models[0]
        pd = predict_model(model, data=custom_data)
        # print(pull())
        (fig, res) = clarke_error_grid(
            pd['label'], pd['Label'], legend)
        rmse = np.sqrt(
            mean_squared_error(pd['label'], pd['Label']))
        rmadex = np.sqrt(mean_adjusted_exponent_error(
            pd['label'], pd['Label']))
        return (fig, res, rmse, rmadex)

    def predict_holdout(self, model=None):

        legend = f'[{self.patient}]Holdout_W{(self.window*5)}_H{(self.horizon*5)}'
        (self.holdout_cega_fig, self.holdout_cega_res, self.holdout_rmse,
         self.holdout_rmadex) = self.calculate_prediction(model=model, legend=legend)

    def predict_unseen(self, model=None):
        legend = f'[{self.patient}]UnseenData_W{(self.window*5)}_H{(self.horizon*5)}'
        self.create_unseen_data_dataframe()
        (self.unseen_cega_fig, self.unseen_cega_res, self.unseen_rmse,
         self.unseen_rmadex) = self.calculate_prediction(model=model, custom_data=self.unseen_data_df, legend=legend)


if __name__ == '__main__':
    exp = Experiment(559, 12, 6)
    exp.create_train_dataframe()
    exp.setup_regressor()
    exp.log_regressor_param('prep_pipe')
    exp.compute_best_n_models(verbose=False)
    print(exp.models_comparison_df)
    exp.log_best_models()
    exp.predict_holdout()
    logger.info(exp.holdout_cega_res)
    logger.info(exp.holdout_rmse)
    logger.info(exp.holdout_rmadex)
    exp.predict_unseen()
    logger.info(exp.unseen_cega_res)
    logger.info(exp.unseen_rmse)
    logger.info(exp.unseen_rmadex)
    pprint(exp.__dict__)
