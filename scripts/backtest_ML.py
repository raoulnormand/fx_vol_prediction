"""
Compute scores for ML forecasts.
"""

# Imports

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fxvol.backtest import run_backtest
from fxvol.data_utils import load_csv, save_csv
from fxvol.fin_comp import qlie_loss
from fxvol.ML_utils import make_xy
from fxvol.ML_models import har_ols_forecast

# Data

log_ret = load_csv("processed", "log_returns").dropna()
eur_ret = log_ret["EUR"]
real_vol = 
X, y = make_xy(
    log_ret=log_ret,
    real_vol=real_vol,
    horizon=horizon,
    lags=lags,
    use_asym=use_asym,
)

# Models

models = [
    (har_ols_forecast, "har_ols", {"lags": [1, 5, 22, 66], "use_asym": True}),
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

save_csv(scores.astype(float).round(5), "results", "ML")
