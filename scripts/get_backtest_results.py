"""
Compute scores for baseline forecasts.
"""

# Imports


from fxvol.backtest import backtest_results
from fxvol.baseline_models import ewma_fc, garch11_fc, har_fc, naive_fc, rolling_mean_fc
from fxvol.data_utils import load_csv
from fxvol.ML_models import elastic_net_fc, gb_tree_fc, ols_fc

# Data

log_rets = load_csv("processed", "log_returns").dropna()

CURRENCIES = ["AUD", "CHF", "EUR", "GBP", "JPY"]

# Models

MODELS = [
    (naive_fc, "naive", {}),
    (rolling_mean_fc, "rolling5", {"window": 5}),
    (rolling_mean_fc, "rolling50", {"window": 50}),
    (ewma_fc, "ewma090", {"alpha": 0.9}),
    (ewma_fc, "ewma030", {"alpha": 0.3}),
    (har_fc, "har", {"lags": [1, 5, 22, 66]}),
    (garch11_fc, "garch11", {}),
    (ols_fc, "ols", {}),
    (elastic_net_fc, "elastic_net_1", {"alpha": 1}),
    (elastic_net_fc, "elastic_net_1e-3", {"alpha": 1e-3}),
    (gb_tree_fc, "gb_tree", {}),
]

# Run backest

HORIZON = 5
FEATURE_KWARGS = {"lags": [1, 5, 22, 66], "vol_vol": 22}

for currency in CURRENCIES:
    log_ret = log_rets[currency]
    backtest_results(
        log_ret=log_ret,
        feature_kwargs=FEATURE_KWARGS,
        models=MODELS,
        horizon=HORIZON,
        file_name=currency,
    )
