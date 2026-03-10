"""
Run volatility targeting strategy for each model.
"""

# Imports

from fxvol.config import CURRENCIES, FEATURES, HORIZON, MODELS, TARGET_VOL
from fxvol.data_utils import load_csv, make_xy
from fxvol.strategy import run_strategy

# Data

log_rets = load_csv("data/processed", "log_returns")

# Run strategy

data = [
    (curr,) + make_xy(log_ret=log_rets[curr], horizon=HORIZON, **FEATURES)
    for curr in CURRENCIES
]

for model in MODELS:
    pf_log_ret = run_strategy(
        data=data,
        model=model,
        horizon=HORIZON,
        target_vol=TARGET_VOL,
        file_name=model[1],
    )
