"""
Compute scores for baseline forecasts.
"""

# Imports

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fxvol.backtest import run_backtest
from fxvol.data_utils import load_csv
from fxvol.fin_comp import qlike_loss
from fxvol.models import NaiveModel, RollingMeanModel, EWMA

# Data

HORIZON = 5
log_ret = load_csv("processed", "log_returns").dropna()
eur_ret = log_ret["EUR"]
df_res = run_backtest(eur_ret, EWMA(0.5), horizon=HORIZON)
y_true = df_res["y_true"]
y_pred = df_res["y_pred"]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
qlike = qlike_loss(y_true, y_pred)

print(f"Naive forecast RMSE: {rmse:.5f}")
print(f"Naive forecast MAE:  {mae:.5f}")
print(f"Naive forecast QLIKE:  {qlike:.5f}")
