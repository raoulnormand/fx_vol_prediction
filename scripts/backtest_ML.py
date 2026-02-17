"""
Compute scores for ML forecasts.
"""

# Imports


from fxvol.backtest import backtest_results
from fxvol.data_utils import load_csv
from fxvol.ML_models import elastic_net, gb_tree, ols

# Data

log_ret = load_csv("processed", "log_returns").dropna()
eur_ret = log_ret["EUR"]

# Models

models = [
    # (ols, "ols-1-5-22", {"lags": [1, 5, 22]}),
    (ols, "ols-1-5-22-66", {"lags": [1, 5, 22, 66]}),
    # (elastic_net, "elastic_net", {"lags": [1, 5, 22, 66]}),
    (gb_tree, "gb_tree", {"lags": [1, 5, 22]}),
]

# Run backest

HORIZON = 5

backtest_results(
    log_ret=eur_ret, models=models, horizon=HORIZON, file_name="ML_models", sigfig=7
)
