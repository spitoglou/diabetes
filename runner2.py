from src.helpers.experiment import create_tsfresh_dataframe
from src.helpers.diabetes.cega import clarke_error_grid
from src.helpers.diabetes.madex import mean_adjusted_exponent_error
from pycaret.regression import setup, create_model, compare_models, predict_model, get_config, pull
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
from sklearn.metrics import mean_squared_error
# import pandas
from pprint import pprint
import sys
import time


class Experiment():

    def __init__(self,
                 patient: int, window: int, horizon: int,
                 min_per_measure: int = 5,
                 best_models_no: int = 3,
                 speed: int = 3,
                 log_type: str = 'standard',
                 minimal_features: bool = False) -> None:

        logger.remove()
        if log_type == 'standard':
            logger.add(sys.stderr)
        if log_type == 'file':
            logger.add('logger.log')

        self.patient = patient
        self.window = window
        self.horizon = horizon
        self.win_min = window * min_per_measure
        self.hor_min = horizon * min_per_measure
        self.best_models_no = best_models_no
        self.speed = speed
        self.logger = logger
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

    def remove_gaps(self, df):
        # clean wrong values due to measure gaps
        # print(df[200:300])
        problematic_points = []
        old_value = 0
        for index, row in df.iterrows():
            # logger.debug(row['start_time'])
            if (row['end_time'] - old_value) > 1:
                problematic_points.append(index)
            old_value = row['end_time']
        logger.warning(problematic_points)
        for point in problematic_points:
            for i in range(-1 * self.horizon, self.window):
                df.drop(point + i, inplace=True)
        return df

    def create_train_dataframe(self):
        self.train_df = self.remove_gaps(
            self.create_dataframe(self.train_parameters))

    def create_unseen_data_dataframe(self):
        self.unseen_data_df = self.remove_gaps(self.create_dataframe(
            self.unseen_data_parameters))

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

        # workaround for indexing cega and madex problems
        cega_pd = pd.reset_index(drop=True)
        (fig, res) = clarke_error_grid(
            cega_pd['label'], cega_pd['Label'], legend)
        rmse = np.sqrt(
            mean_squared_error(pd['label'], pd['Label']))
        rmadex = np.sqrt(mean_adjusted_exponent_error(
            cega_pd['label'], cega_pd['Label']))
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

    def run_experiment(self):
        start_time = time.time()
        self.create_train_dataframe()
        self.setup_regressor()
        self.log_regressor_param('prep_pipe')
        self.compute_best_n_models(verbose=False)
        print(self.models_comparison_df)
        self.log_best_models()
        self.predict_holdout()
        logger.info(self.holdout_cega_res)
        logger.info(self.holdout_rmse)
        logger.info(self.holdout_rmadex)
        self.predict_unseen()
        logger.info(self.unseen_cega_res)
        logger.info(self.unseen_rmse)
        logger.info(self.unseen_rmadex)
        logger.info(f'Execution Time: {(time.time() - start_time)}')


if __name__ == '__main__':
    exp = Experiment(559, 12, 6, log_type='standard')
    # exp.create_train_dataframe()
    # exp.remove_gaps(exp.train_df)
    exp.run_experiment()
    pprint(exp.__dict__)
