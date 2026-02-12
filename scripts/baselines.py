"""
Compute scores for baseline forecasts.
"""

# Imports

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fxvol.backtest import run_backtest
from fxvol.data_utils import load_csv, save_csv
from fxvol.fin_comp import qlike_loss
from fxvol.models import (  # log_har_forecast,
    ewma_forecast,
    exog_har_forecast,
    garch11_forecast,
    har_forecast,
    naive_forecast,
    rolling_mean_forecast,
)

# Data

log_ret = load_csv("processed", "log_returns").dropna()
eur_ret = log_ret["EUR"]

# Models

models = [
    (naive_forecast, "naive", {}),
    (rolling_mean_forecast, "rolling5", {"window": 5}),
    (rolling_mean_forecast, "rolling20", {"window": 20}),
    (rolling_mean_forecast, "rolling50", {"window": 50}),
    (rolling_mean_forecast, "rolling100", {"window": 100}),
    (ewma_forecast, "ewma092", {}),
    (ewma_forecast, "ewma030", {"alpha": 0.3}),
    (har_forecast, "har1-5-22", {"lags": [1, 5, 22]}),
    (exog_har_forecast, "exog_har1-5-22", {"lags": [1, 5, 22]}),
    (har_forecast, "har1-5-22-66", {"lags": [1, 5, 22, 66]}),
    (garch11_forecast, "garch11", {}),
]

# Run backest

HORIZON = 5

scores = pd.DataFrame(
    index=[model[1] for model in models], columns=["RMSE", "MAE", "QLIKE"]
)

for forecast_fn, name, params in models:
    results = run_backtest(
        log_ret=eur_ret, forecast_fn=forecast_fn, horizon=HORIZON, **params
    )
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    qlike = qlike_loss(y_true, y_pred)
    scores.loc[name] = [rmse, mae, qlike]

save_csv(scores.astype(float).round(5), "results", "baselines")
