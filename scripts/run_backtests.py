"""
Compute scores for baseline forecasts.
"""

# Imports
from fxvol.backtest import backtest_results
from fxvol.data_utils import load_csv
from fxvol.config import CURRENCIES, MODELS, HORIZON, FEATURES

# Data

log_rets = load_csv("data/processed", "log_returns")

# Run backest

for currency in CURRENCIES:
    log_ret = log_rets[currency]
    backtest_results(
        log_ret=log_ret,
        feature_kwargs=FEATURES,
        models=MODELS,
        horizon=HORIZON,
        file_name=currency,
    )
