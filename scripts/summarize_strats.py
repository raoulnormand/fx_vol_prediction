"""
Get summaries of the performance of the vol targeting strategies for different
models: annualized return, annualized vol, Sharpe ratio, max drawdown.
"""

# Imports

import numpy as np
import pandas as pd

from fxvol.config import HORIZON, MODEL_NAMES, NB_TRADING_DAYS, TARGET_VOL
from fxvol.data_utils import load_csv, save_csv

# Compute and save equity log returns and equity curves

ANN_FACT = NB_TRADING_DAYS / HORIZON

log_rets = pd.DataFrame(columns=MODEL_NAMES)

for model in MODEL_NAMES:
    log_rets[model] = load_csv("results/strategy", model)

save_csv(log_rets, "results/summary", "log_rets")

equity = np.exp(log_rets.cumsum())

# Compute metrics for each model

metrics = pd.DataFrame(index=MODEL_NAMES)

# Annualized log returns
mu = np.exp(ANN_FACT * log_rets.mean()) - 1
metrics["ann_return"] = mu

# Annualized vol
sig = np.sqrt(ANN_FACT) * log_rets.std()
metrics["ann_vol"] = sig

# Difference between vol and target vol
metrics["vol_error"] = np.abs(sig - TARGET_VOL)

# Sharpe ratio assuming rates = 0
metrics["Sharpe"] = mu / sig

# Max drawdown

equity = np.exp(log_rets.cumsum())
drawdown = equity / equity.cummax() - 1
metrics["max_dd"] = drawdown.min()


save_csv(metrics, "results/summary", "metrics")
