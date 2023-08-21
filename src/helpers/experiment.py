from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider

# ?from src.bgc_providers.aida_bgc_provider import AidaBgcProvider
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.helpers.dataframe import save_df, read_df
from os import path
from loguru import logger
from pycaret.regression import (
    setup,
    compare_models,
    predict_model,
    get_config,
    pull,
    add_metric,
    dashboard,
    save_model
)  # create_model,
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
import time
import neptune
from src.helpers.diabetes.madex import madex, rmadex, mean_adjusted_exponent_error
from src.helpers.diabetes.cega import clarke_error_grid


def create_ds_name(parameters):
    ds_name = (
        f"dataframes/{parameters['ohio_no']}_{parameters['scope']}_{parameters['train_ds_size']}"
        f"_{parameters['window_size']}_{parameters['prediction_horizon']}.pkl"
    )
    logger.info(ds_name)
    return ds_name


def timeseries_dataframe(p, show_plt=False):
    provider = OhioBgcProvider(scope=p["scope"], ohio_no=p["ohio_no"])
    logger.info(p)
    return provider.tsfresh_dataframe(truncate=p["train_ds_size"], show_plt=show_plt)


def create_tsfresh_dataframe(p, show_plt=False):
    ds_name = create_ds_name(p)
    df = timeseries_dataframe(p, show_plt)
    if path.exists(ds_name):
        logger.info("Found existing pickle file. Continuing...")
        out = read_df(ds_name)
    else:
        ts = TsfreshFeaturizer(
            df,
            p["window_size"],
            p["prediction_horizon"],
            minimal_features=p["minimal_features"],
        )
        ts.create_labeled_dataframe()

        out = ts.labeled_dataframe
        save_df(out, ds_name)
    return out


class Experiment:
    def __init__(
        self,
        patient: int,
        window: int,
        horizon: int,
        min_per_measure: int = 5,
        best_models_no: int = 3,
        speed: int = 3,
        log_type: str = "standard",
        perform_gap_corrections: bool = True,
        minimal_features: bool = False,
        enable_neptune: bool = True,
    ) -> None:
        logger.remove()
        if log_type == "file":
            logger.add("logger.log")
        elif log_type == "standard":
            logger.add(sys.stderr)

        self.patient = patient
        self.window = window
        self.horizon = horizon
        self.win_min = window * min_per_measure
        self.hor_min = horizon * min_per_measure
        self.best_models_no = best_models_no
        self.speed = speed
        self.logger = logger
        self.perform_gap_corrections = perform_gap_corrections
        self.enable_neptune = enable_neptune
        self.train_parameters = {
            "ohio_no": patient,
            "scope": "train",
            "train_ds_size": 0,
            "window_size": window,
            "prediction_horizon": horizon,
            "minimal_features": minimal_features,
        }

        self.unseen_data_parameters = {
            "ohio_no": patient,
            "scope": "test",
            "train_ds_size": 0,
            "window_size": window,
            "prediction_horizon": horizon,
            "minimal_features": minimal_features,
        }

        if self.enable_neptune:
            self.neptune = neptune.init_run(
                project="spitoglou/intermediate",
                api_token=(
                    "eyJhcGlfYWRkcmVzcyI6Imh0dHB"
                    "zOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIj"
                    "oiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV"
                    "9rZXkiOiJlOWFkOTc3My0zZjQ3LTQ3MGMtOTQ2Zi0"
                    "3NjA5ZDgzN2IyZTIifQ=="
                ),
            )

            self.neptune["parameters"] = {
                "train parameters": self.train_parameters,
                "unseen data parameters": self.unseen_data_parameters,
                "patient": patient,
                "window": window,
                "horizon": horizon,
                "gap corrections": perform_gap_corrections,
                "speed": speed,
            }

    def create_dataframe(self, parameters: dict):
        return self.remove_missing_and_inf(create_tsfresh_dataframe(parameters))

    def remove_gaps(self, df):
        if self.perform_gap_corrections:
            # clean wrong values due to measure gaps
            problematic_points = []
            old_value = 0
            for index, row in df.iterrows():
                if (row["end_time"] - old_value) > 1:
                    problematic_points.append(index)
                old_value = row["end_time"]
            logger.warning(problematic_points)
            for point in problematic_points:
                for i in range(-1 * self.horizon, self.window):
                    df.drop(point + i, inplace=True)
        return df

    def remove_missing_and_inf(self, df):
        # sourcery skip: extract-duplicate-method
        print("Original Dataframe")
        print(df.shape)
        df.dropna(axis=1, inplace=True)
        print("Dataframe after removing cols with NaNs")
        print(df.shape)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, inplace=True)
        print("Dataframe after removing cols with infinites")
        print(df.shape)
        # LightGBMError: Do not support special JSON characters in feature name.
        import re

        # df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
        new_names = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in df.columns}
        new_n_list = list(new_names.values())
        # [LightGBM] Feature appears more than one time.
        new_names = {
            col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }
        df = df.rename(columns=new_names)
        return df

    def create_train_dataframe(self):
        self.train_df = self.remove_gaps(self.create_dataframe(self.train_parameters))

    def create_unseen_data_dataframe(self):
        self.unseen_data_df = self.remove_gaps(
            self.create_dataframe(self.unseen_data_parameters)
        )

    def setup_regressor(self):
        self.regressor = setup(
            self.train_df,
            target="label",
            feature_selection=True,
            ignore_features=[
                "start",
                "end",
                "start_time",
                "end_time",
                "start_time_of_day",
                "end_time_of_day",
            ],
            html=False,
            verbose=False,
            session_id=1974,
            #  ignore_low_variance=True,
            #  silent=True,
            #  profile=True,
        )

    def get_regressor_param(self, param: str):
        return get_config(param)

    def get_model_name(self, model):
        return model.__str__().split("(")[0]

    def log_regressor_param(self, param: str):
        if self.enable_neptune:
            self.neptune[f"model/{param}"] = self.get_regressor_param(param).__str__()
            # self.neptune[f'model/{param}_html'].upload(
            #     neptune.types.File.as_html(self.get_regressor_param(param).__str__()))
        logger.info(self.get_regressor_param(param))

    def compute_best_n_models(self, verbose=True):
        exc_dict = {
            1: [],
            2: ["catboost", "xgboost"],
            3: ["catboost", "xgboost", "et", "rf", "ada", "gbr"],
        }
        add_metric("madex", "MADEX", madex, False)
        add_metric("rmadex", "RMADEX", rmadex, False)
        self.best_models = compare_models(
            exclude=exc_dict[self.speed],
            # RMSE sort
            # sort='RMSE',
            sort="RMADEX",
            n_select=self.best_models_no,
            verbose=verbose,
        )
        self.models_comparison_df = pull()

    def log_best_models(self):
        import uuid
        for index, model in enumerate(self.best_models):
            save_model(model, f'models/{self.patient}_{self.window}_{self.horizon}_{index+1}_{self.get_model_name(model.__str__())}_{uuid.uuid4()}')
            logger.info(f"Model {(index+1)}:")
            logger.info(model)

    def calculate_prediction(self, model, custom_data=None, legend=""):
        if not model:
            model = self.best_models[0]
        if self.enable_neptune:
            self.neptune["model/name"] = self.get_model_name(model.__str__())
            self.neptune["model/details"] = model.__str__()
        pd = predict_model(model, data=custom_data)
        print(pd.head())

        # workaround for indexing cega and madex problems
        cega_pd = pd.reset_index(drop=True)
        (fig, res) = clarke_error_grid(
            cega_pd["label"], cega_pd["prediction_label"], legend
        )
        rmse = np.sqrt(mean_squared_error(pd["label"], pd["prediction_label"]))
        rmadex = np.sqrt(
            mean_adjusted_exponent_error(cega_pd["label"], cega_pd["prediction_label"])
        )
        print(res)
        res = dict(zip(["A", "B", "C", "D", "E"], res))
        return (fig, res, rmse, rmadex)

    def predict_holdout(self, model=None):
        legend = f"[{self.patient}]Holdout_W{(self.window*5)}_H{(self.horizon*5)}"
        (
            self.holdout_cega_fig,
            self.holdout_cega_res,
            self.holdout_rmse,
            self.holdout_rmadex,
        ) = self.calculate_prediction(model=model, legend=legend)
        # self.neptune[f'holdout/images/{legend}'].log(self.holdout_cega_fig)
        if self.enable_neptune:
            self.neptune["cega"].log(plt.gcf())
        # self.neptune[f'holdout/images/{legend}'].upload(neptune.types.File.as_image(self.holdout_cega_fig))

    def predict_unseen(self, model=None):
        legend = f"[{self.patient}]UnseenData_W{(self.window*5)}_H{(self.horizon*5)}"
        self.create_unseen_data_dataframe()
        (
            self.unseen_cega_fig,
            self.unseen_cega_res,
            self.unseen_rmse,
            self.unseen_rmadex,
        ) = self.calculate_prediction(
            model=model, custom_data=self.unseen_data_df, legend=legend
        )
        # self.neptune[f'images/{legend}'].log(self.unseen_cega_fig)
        if self.enable_neptune:
            self.neptune["cega"].log(plt.gcf())
        # self.neptune[f'unseen/images/{legend}'].upload(neptune.types.File.as_image(self.unseen_cega_fig))

    def run_experiment(self):
        start_time = time.time()
        self.create_train_dataframe()
        self.setup_regressor()
        self.log_regressor_param("pipeline")
        self.compute_best_n_models(verbose=True)
        print(self.models_comparison_df)
        if self.enable_neptune:
            # self.neptune['pipeline'].log(self.get_regressor_param('pipeline'))
            self.neptune["model/train_columns"] = list(get_config("X_train").columns)
            self.neptune["model/comparison"].upload(
                neptune.types.File.as_html(self.models_comparison_df)
            )
        self.log_best_models()
        self.predict_holdout()
        # plt.show()
        plt.close("all")
        if self.enable_neptune:
            # self.neptune['holdout/cega'].log(self.holdout_cega_res)
            self.neptune["holdout/cega"] = self.holdout_cega_res
            self.neptune["holdout/RMSE"] = self.holdout_rmse
            self.neptune["holdout/RMADEX"] = self.holdout_rmadex
        logger.info(self.holdout_cega_res)
        logger.info(self.holdout_rmse)
        logger.info(self.holdout_rmadex)
        self.predict_unseen()
        # plt.show()
        plt.close("all")
        logger.info(self.unseen_cega_res)
        logger.info(self.unseen_rmse)
        logger.info(self.unseen_rmadex)
        logger.info(f"Execution Time: {(time.time() - start_time)}")
        if self.enable_neptune:
            self.neptune["unseen/cega"] = self.unseen_cega_res
            self.neptune["unseen/RMSE"] = self.unseen_rmse
            self.neptune["unseen/RMADEX"] = self.unseen_rmadex
            self.neptune["Execution Time"] = time.time() - start_time
            self.neptune.stop()
