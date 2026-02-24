"""
Run volatility targeting straeegy for 5 assets.
"""

# Imports


from fxvol.baseline_models import ewma_fc, naive_fc
from fxvol.data_utils import load_csv, make_xy
from fxvol.ML_models import ols_fc
from fxvol.strategy import run_strategy

# Data

log_rets = load_csv("processed", "log_returns").dropna()

CURRENCIES = ["AUD", "CHF", "EUR", "GBP", "JPY"]

# Models

MODELS = [
    (naive_fc, "naive", {}),
    #     (rolling_mean_fc, "rolling5", {"window": 5}),
    #     (rolling_mean_fc, "rolling50", {"window": 50}),
    #     (ewma_fc, "ewma090", {"alpha": 0.9}),
    #     (ewma_fc, "ewma030", {"alpha": 0.3}),
    #     (har_fc, "har", {"lags": [1, 5, 22, 66]}),
    #     (garch11_fc, "garch11", {}),
    #     (ols_fc, "ols", {}),
    #     (elastic_net_fc, "elastic_net_1", {"alpha": 1}),
    #     (elastic_net_fc, "elastic_net_1e-3", {"alpha": 1e-3}),
    #     (gb_tree_fc, "gb_tree", {}),
]

# Run backest

FEATURE_KWARGS = {"lags": [1, 5, 22, 66], "vol_vol": 22}
HORIZON = 5
DATA = [
    (curr,) + make_xy(log_ret=log_rets[curr], horizon=HORIZON, **FEATURE_KWARGS)
    for curr in CURRENCIES
]
TARGET_VOL = 0.1

for model in MODELS:
    pf_log_ret = run_strategy(
    X=X,
    y=y,
    model=model,
    horizon=HORIZON,
    target_vol=TARGET_VOL,
    file_name="",
)
