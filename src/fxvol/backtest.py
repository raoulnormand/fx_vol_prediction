"""
Rolling window backtesting.
"""

# Imports

import pandas as pd

from fxvol.data_utils import make_xy

# Backtest function


def run_backtest(
    log_ret: pd.Series,
    horizon: int,
    forecast_fn,
    start_date: float | str = 0.5,
    stride: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Run backtests for the corresponding model.
    Start at start_date (date or fraction of total time), and jumps by stride each time.
    Computes value for the given horizon.
    """
    # Compute features and target
    X, y = make_xy(log_ret=log_ret, horizon=horizon, **kwargs)

    # Get index of start date
    if isinstance(start_date, float):
        end_ix = int(start_date * len(log_ret))
    else:
        end_ix = log_ret.index.get_loc(start_date)

    assert isinstance(end_ix, int)

    # Get prediction on rolling window

    results = {"Date": [], "y_true": [], "y_pred": []}

    while end_ix + horizon < len(y):
        # Training data, current day included
        train = X.iloc[: end_ix + 1]

        # Forecast and true value
        y_pred = forecast_fn(X=train, horizon=horizon, **kwargs)
        y_true = y.iloc[end_ix]

        # Store results
        results["Date"].append(X.index[end_ix])
        results["y_true"].append(y_true)
        results["y_pred"].append(y_pred)

        # Next step
        end_ix += stride

    df = pd.DataFrame(results)
    df.set_index("Date", inplace=True)
    return df
