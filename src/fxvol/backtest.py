"""
Rolling window backtesting.
"""

# Imports

import numpy as np
import pandas as pd

from fxvol.fin_comp import realized_vol
from fxvol.models import Model

# Backtest function


def run_backtest(
    log_ret: pd.Series,
    model: Model,
    horizon: int,
    start_date: float | str = 0.5,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Run backtests for the corresponding model.
    Start at start_date (date or fraction of total time), and jumps by stride each time.
    Computes value for the given horizon.
    """
    # Compute realized_vol for a period = horizon
    real_vol = realized_vol(log_ret, window=horizon)

    # Get index of start date
    if isinstance(start_date, float):
        end_ix = int(start_date * len(log_ret))
    else:
        end_ix = log_ret.index.get_loc(start_date)

    assert isinstance(end_ix, int)

    # Get prediction on rolling window

    results = {"Date": [], "y_true": [], "y_pred": []}

    while end_ix + horizon < len(log_ret):
        # Training data, current day included
        train_ret = log_ret.iloc[: end_ix + 1]
        train_vol = real_vol[: end_ix + 1]

        # Forecast and true value
        model.fit(log_ret=train_ret, real_vol=train_vol)
        y_pred = model.predict(horizon)
        y_true = real_vol.iloc[end_ix + horizon]

        # Store results
        results["Date"].append(log_ret.index[end_ix])
        results["y_true"].append(y_true)
        results["y_pred"].append(y_pred)

        # Next step
        end_ix += stride

    df = pd.DataFrame(results)
    df.set_index("Date", inplace=True)
    return df
