"""
Get summaries of the performance of the vol targeting strategies for different
models: annualized return, annualized vol, Sharpe ratio, max drawdown.
"""

# Imports

import numpy as np
import pandas as pd

from fxvol.data_utils import load_csv, save_csv

# Models

MODEL_NAMES = ["rolling50", "har", "garch11", "ols", "gb_tree"]
HORIZON = 5
ANN_FACT = 252 / HORIZON

# Compute and save equity log returns and equity curves

log_rets = pd.DataFrame(columns=MODEL_NAMES)

for model in MODEL_NAMES:
    lr = load_csv("results/strategy", model)
    log_rets[model] = lr

save_csv(log_rets, "results/summary", "log_rets")

equity = np.exp(log_rets.cumsum())

save_csv(equity, "results/summary", "equity")


# Compute metrics for each model

metrics = pd.DataFrame(
    index=["ann_ret", "ann_vol", "Sharpe", "max_dd"], columns=MODEL_NAMES
)

# Annualized log returns
mu = ANN_FACT * log_rets.mean()
metrics.loc["ann_log_ret"] = mu

# Annualized vol
sig = np.sqrt(ANN_FACT) * log_rets.std()
metrics.loc["ann_vol"] = sig

# Sharpe ratio assuming rates = 0
metrics.loc["Sharpe"] = mu / sig

# Max drawdown

drawdown = equity / equity.cummax() - 1
metrics.loc["max_dd"] = drawdown.min()


save_csv(metrics, "results/summary", "metrics")
