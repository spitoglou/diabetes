from src.helpers.experiment import create_tsfresh_dataframe
from run import Experiment
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.featurizers.tsfresh import TsfreshFeaturizer
import plotly.express as px
import pandas as pd
from src.helpers.experiment import Experiment
from pycaret.regression import get_leaderboard, get_config


provider = OhioBgcProvider()

tsfresh_df = provider.tsfresh_dataframe()
tsfresh_df["r_date_time"] = tsfresh_df["date_time"].dt.round("5T")
print(tsfresh_df.head())



tsfresh_df2 = tsfresh_df.set_index('r_date_time')

idx = pd.date_range(min(tsfresh_df2.index), max(tsfresh_df2.index), freq='5T')
tsfresh_df2 = tsfresh_df2.reindex(idx)

fig = px.line(tsfresh_df2[:500], y="bg_value")
fig.show()

exp = Experiment(559, 6, 6, enable_neptune=False)
exp.run_experiment()

print(get_leaderboard())