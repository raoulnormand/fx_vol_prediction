"""
Run volatility targeting straeegy for 5 assets.
"""

# Imports


from fxvol.baseline_models import garch11_fc, har_fc, rolling_mean_fc
from fxvol.data_utils import load_csv, make_xy
from fxvol.ML_models import gb_tree_fc, ols_fc
from fxvol.strategy import run_strategy

# Data

log_rets = load_csv("processed", "log_returns").dropna()

CURRENCIES = ["AUD", "CHF", "EUR", "GBP", "JPY"]

# Models

MODELS = [
    # (rolling_mean_fc, "rolling50", {"window": 50}),
    # (har_fc, "har", {"lags": [1, 5, 22, 66]}),
    # (garch11_fc, "garch11", {}),
    # (ols_fc, "ols", {}),
    (gb_tree_fc, "gb_tree", {}),
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
        data=DATA,
        model=model,
        horizon=HORIZON,
        target_vol=TARGET_VOL,
        file_name=model[1],
    )
