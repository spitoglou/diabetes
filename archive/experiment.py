from __future__ import annotations

import re
import sys
import time
import uuid
from os import path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
from loguru import logger
from neptune.types import File
from pycaret.regression import (
    add_metric,
    compare_models,
    get_config,
    predict_model,
    pull,
    save_model,
    setup,
)
from sklearn.metrics import mean_squared_error

from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.helpers.dataframe import read_df, save_df
from src.helpers.diabetes.cega import clarke_error_grid
from src.helpers.diabetes.madex import madex, mean_adjusted_exponent_error, rmadex

matplotlib.use("Agg")


def create_ds_name(parameters: dict[str, Any]) -> str:
    ds_name = (
        f"dataframes/{parameters['ohio_no']}_{parameters['scope']}_{parameters['train_ds_size']}"
        f"_{parameters['window_size']}_{parameters['prediction_horizon']}.pkl"
    )
    logger.info(ds_name)
    return ds_name


def timeseries_dataframe(p: dict[str, Any], show_plt: bool = False) -> pd.DataFrame:
    provider = OhioBgcProvider(scope=p["scope"], ohio_no=p["ohio_no"])
    logger.info(p)
    return provider.tsfresh_dataframe(truncate=p["train_ds_size"], show_plt=show_plt)


def create_tsfresh_dataframe(p: dict[str, Any], show_plt: bool = False) -> pd.DataFrame:
    ds_name = create_ds_name(p)
    df = timeseries_dataframe(p, show_plt)
    out: pd.DataFrame
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

        self.patient: int = patient
        self.window: int = window
        self.horizon: int = horizon
        self.win_min: int = window * min_per_measure
        self.hor_min: int = horizon * min_per_measure
        self.best_models_no: int = best_models_no
        self.speed: int = speed
        self.logger = logger
        self.perform_gap_corrections: bool = perform_gap_corrections
        self.enable_neptune: bool = enable_neptune
        self.train_parameters: dict[str, Any] = {
            "ohio_no": patient,
            "scope": "train",
            "train_ds_size": 0,
            "window_size": window,
            "prediction_horizon": horizon,
            "minimal_features": minimal_features,
        }

        self.unseen_data_parameters: dict[str, Any] = {
            "ohio_no": patient,
            "scope": "test",
            "train_ds_size": 0,
            "window_size": window,
            "prediction_horizon": horizon,
            "minimal_features": minimal_features,
        }

        # Instance attributes initialized later
        self.train_df: pd.DataFrame
        self.unseen_data_df: pd.DataFrame
        self.regressor: Any
        self.best_models: list[Any]
        self.models_comparison_df: pd.DataFrame
        self.holdout_cega_fig: Any
        self.holdout_cega_res: dict[str, int]
        self.holdout_rmse: float
        self.holdout_rmadex: float
        self.unseen_cega_fig: Any
        self.unseen_cega_res: dict[str, int]
        self.unseen_rmse: float
        self.unseen_rmadex: float
        self.neptune: Any

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

    def create_dataframe(self, parameters: dict[str, Any]) -> pd.DataFrame:
        return self.fix_names(
            self.remove_missing_and_inf(create_tsfresh_dataframe(parameters))
        )

    def remove_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.perform_gap_corrections:
            # clean wrong values due to measure gaps
            problematic_points: list[Any] = []
            old_value: float = 0
            for index, row in df.iterrows():
                end_time = float(row["end_time"])
                if (end_time - old_value) > 1:
                    problematic_points.append(index)
                old_value = end_time
            logger.warning(problematic_points)
            for point in problematic_points:
                for i in range(-1 * self.horizon, self.window):
                    df.drop(point + i, inplace=True)
        return df

    def remove_missing_and_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The function removes columns with missing values and infinite values from a dataframe, renames
        the columns to remove special characters, and returns the modified dataframe.

        :param df: The parameter `df` is a pandas DataFrame that represents the original data
        :return: the modified dataframe after removing columns with NaNs and infinites, and renaming the
        columns to remove special characters.
        """
        # sourcery skip: extract-duplicate-method
        logger.debug(f"Original Dataframe shape: {df.shape}")
        df.dropna(axis=1, inplace=True)
        logger.debug(f"Dataframe after removing cols with NaNs: {df.shape}")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, inplace=True)
        logger.debug(f"Dataframe after removing cols with infinites: {df.shape}")

        return df

    def fix_names(self, df: pd.DataFrame) -> pd.DataFrame:
        # LightGBMError: Do not support special JSON characters in feature name.

        # Fix values in part_of_day column (spaces to underscores) before pycaret one-hot encodes
        if "part_of_day" in df.columns:
            df["part_of_day"] = df["part_of_day"].str.replace(" ", "_")

        # df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
        # First replace spaces with underscores, then remove other special characters
        new_names: dict[str, str] = {
            col: re.sub(r"[^A-Za-z0-9_]+", "", col.replace(" ", "_"))
            for col in df.columns
        }
        new_n_list: list[str] = list(new_names.values())
        # [LightGBM] Feature appears more than one time.
        new_names = {
            col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
            for i, (col, new_col) in enumerate(new_names.items())
        }
        df = df.rename(columns=new_names)
        return df

    def create_train_dataframe(self) -> None:
        """
        The function creates a training dataframe by removing gaps and creating a dataframe using the
        given train parameters.
        """
        self.train_df = self.remove_gaps(self.create_dataframe(self.train_parameters))

    def create_unseen_data_dataframe(self) -> None:
        self.unseen_data_df = self.remove_gaps(
            self.create_dataframe(self.unseen_data_parameters)
        )

    def align_dataframe_columns(self) -> None:
        logger.debug("Aligning dataframe columns")
        train_df_columns: list[str] = self.train_df.columns.tolist()
        unseen_data_df_columns: list[str] = self.unseen_data_df.columns.tolist()
        for train_column in train_df_columns:
            if train_column not in unseen_data_df_columns:
                logger.debug(f"Dropping column: {train_column}")
                self.train_df.drop(train_column, axis=1, inplace=True)
        logger.debug("Column alignment complete")

    def setup_regressor(self) -> None:
        self.regressor = setup(
            self.train_df,
            target="label",
            feature_selection=True,
            normalize=True,
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

    def get_regressor_param(self, param: str) -> Any:
        return get_config(param)

    def get_model_name(self, model: Any) -> str:
        return model.__str__().split("(")[0]

    def log_regressor_param(self, param: str) -> None:
        if self.enable_neptune:
            self.neptune[f"model/{param}"] = self.get_regressor_param(param).__str__()
            # self.neptune[f'model/{param}_html'].upload(
            #     neptune.types.File.as_html(self.get_regressor_param(param).__str__()))
        logger.info(self.get_regressor_param(param))

    def compute_best_n_models(self, verbose: bool = True) -> None:
        exc_dict: dict[int, list[str]] = {
            1: [],
            2: ["catboost", "xgboost"],
            3: ["catboost", "xgboost", "et", "rf", "ada", "gbr"],
        }
        add_metric("madex", "MADEX", madex, greater_is_better=False)  # type: ignore[reportArgumentType]
        add_metric("rmadex", "RMADEX", rmadex, greater_is_better=False)  # type: ignore[reportArgumentType]
        self.best_models = compare_models(
            exclude=exc_dict[self.speed],
            # RMSE sort
            # sort='RMSE',
            sort="RMADEX",
            n_select=self.best_models_no,
            verbose=verbose,
        )
        self.models_comparison_df = pull()

    def log_best_models(self) -> None:
        for index, model in enumerate(self.best_models):
            save_model(
                model,
                f"models/{self.patient}_{self.window}_{self.horizon}_{index + 1}_{self.get_model_name(model.__str__())}_{uuid.uuid4()}",
            )
            logger.info(f"Model {(index + 1)}:")
            logger.info(model)

    def calculate_prediction(
        self, model: Any, custom_data: pd.DataFrame | None = None, legend: str = ""
    ) -> tuple[Any, dict[str, int], float, float]:
        if not model:
            model = self.best_models[0]
        if self.enable_neptune:
            self.neptune["model/name"] = self.get_model_name(model.__str__())
            self.neptune["model/details"] = model.__str__()
        pred_df = predict_model(model, data=custom_data)
        logger.debug(f"Prediction head:\n{pred_df.head()}")

        # workaround for indexing cega and madex problems
        cega_pd = pred_df.reset_index(drop=True)
        (fig, res) = clarke_error_grid(
            cega_pd["label"].tolist(), cega_pd["prediction_label"].tolist(), legend
        )
        rmse_val: float = np.sqrt(
            mean_squared_error(pred_df["label"], pred_df["prediction_label"])
        )
        rmadex_val: float = np.sqrt(
            mean_adjusted_exponent_error(
                cega_pd["label"].tolist(), cega_pd["prediction_label"].tolist()
            )
        )
        logger.debug(f"CEGA zones: {res}")
        res_dict: dict[str, int] = dict(zip(["A", "B", "C", "D", "E"], res))
        return (fig, res_dict, rmse_val, rmadex_val)

    def predict_holdout(self, model: Any = None) -> None:
        legend = f"[{self.patient}]Holdout_W{(self.window * 5)}_H{(self.horizon * 5)}"
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

    def predict_unseen(self, model: Any = None) -> None:
        legend = (
            f"[{self.patient}]UnseenData_W{(self.window * 5)}_H{(self.horizon * 5)}"
        )
        # self.create_unseen_data_dataframe()
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

    def run_experiment(self) -> None:
        start_time = time.time()
        self.create_train_dataframe()
        self.create_unseen_data_dataframe()
        self.align_dataframe_columns()
        self.setup_regressor()
        self.log_regressor_param("pipeline")
        self.compute_best_n_models(verbose=True)
        logger.info(f"Model comparison:\n{self.models_comparison_df}")
        if self.enable_neptune:
            # self.neptune['pipeline'].log(self.get_regressor_param('pipeline'))
            self.neptune["model/train_columns"] = list(get_config("X_train").columns)
            self.neptune["model/comparison"].upload(
                File.as_html(self.models_comparison_df)
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
