"""
Variables used by the whole project
"""

# Imports

from fxvol.baseline_models import ewma_fc, garch11_fc, har_fc, naive_fc, rolling_mean_fc
from fxvol.ML_models import elastic_net_fc, gb_tree_fc, ols_fc

# FX

TICKERS = ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "CHF=X"]
CURRENCIES = ["AUD", "CHF", "EUR", "GBP", "JPY"]
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# Models

MODELS = [
    (naive_fc, "naive", {}),
    (rolling_mean_fc, "rolling5", {"window": 5}),
    (rolling_mean_fc, "rolling50", {"window": 50}),
    (ewma_fc, "ewma080", {"alpha": 0.8}),
    (ewma_fc, "ewma030", {"alpha": 0.3}),
    (har_fc, "har", {"lags": [1, 5, 22, 66]}),
    (garch11_fc, "garch11", {}),
    (ols_fc, "ols", {}),
    (elastic_net_fc, "elastic_net_1", {"alpha": 1}),
    (elastic_net_fc, "elastic_net_1e-3", {"alpha": 1e-3}),
    (gb_tree_fc, "gb_tree", {}),
]

MODEL_NAMES = [model[1] for model in MODELS]

FEATURES = {"lags": [1, 5, 22, 66], "vol_vol": 22}

LOSS_FNS = ["MAE", "RMSE", "QLIKE"]

#Forecasting and portfolio

TARGET_VOL = 0.10
HORIZON = 5
NB_TRADING_DAYS = 252
