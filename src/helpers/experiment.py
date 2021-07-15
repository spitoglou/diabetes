from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
# from src.bgc_providers.aida_bgc_provider import AidaBgcProvider
from src.featurizers.tsfresh import TsfreshFeaturizer
from src.helpers.dataframe import save_df, read_df
from os import path
from loguru import logger


def create_ds_name(parameters):
    ds_name = (f"dataframes/{parameters['ohio_no']}_{parameters['scope']}_{parameters['train_ds_size']}"
               f"_{parameters['window_size']}_{parameters['prediction_horizon']}.pkl")
    logger.info(ds_name)
    return ds_name


def create_tsfresh_dataframe(p, show_plt=False):
    ds_name = create_ds_name(p)
    provider = OhioBgcProvider(scope=p['scope'], ohio_no=p['ohio_no'])
    logger.info(p)
    df = provider.tsfresh_dataframe(truncate=p['train_ds_size'], show_plt=show_plt)
    if path.exists(ds_name):
        logger.info('Found existing pickle file. Continuing...')
        out = read_df(ds_name)
    else:
        ts = TsfreshFeaturizer(
            df, p['window_size'], p['prediction_horizon'], minimal_features=p['minimal_features'])
        ts.create_labeled_dataframe()

        out = ts.labeled_dataframe
        save_df(out, ds_name)
    return out
