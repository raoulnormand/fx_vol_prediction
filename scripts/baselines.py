"""
Compute scores for baseline forecasts.
"""

# Imports

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fxvol.backtest import run_backtest
from fxvol.data_utils import save_csv, load_csv
from fxvol.fin_comp import qlike_loss
from fxvol.models import EWMA, NaiveModel, RollingMeanModel, HAR, GARCH

# Data

log_ret = load_csv("processed", "log_returns").dropna()
eur_ret = log_ret["EUR"]

# Models

models = [
    (NaiveModel(), "naive"),
    #(RollingMeanModel(window=5), "rolling5"),
    #(RollingMeanModel(window=20), "rolling20"),
    (RollingMeanModel(window=50), "rolling50"),
    (RollingMeanModel(window=100), "rolling100"),
    (EWMA(0.9), "ewma09"),
    (EWMA(0.3), "ewma03"),
    #(HAR(), 'har'),
    (GARCH(), 'garch')
]

# Run backest

HORIZON = 5

scores = pd.DataFrame(
    index=[model[1] for model in models], columns=["RMSE", "MAE", "QLIKE"]
)

for model in models:
    results = run_backtest(eur_ret, model[0], horizon=HORIZON)
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    qlike = qlike_loss(y_true, y_pred)
    scores.loc[model[1]] = [rmse, mae, qlike]
    
save_csv(scores.astype(float).round(5), 'results', 'baselines')
