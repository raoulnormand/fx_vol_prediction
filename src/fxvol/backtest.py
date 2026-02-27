"""
Rolling window backtesting.
"""

# Imports

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fxvol.data_utils import make_xy, save_csv
from fxvol.fin_comp import qlike_loss

# Backtest function


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    forecast_fn,
    start_date: float | str = 0.5,
    stride: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Run backtests for the corresponding model, with design matrix X
    and target y (which may not be needed).
    Start at start_date (date or fraction of total time), and jumps by stride each time.
    Computes value for the given horizon.
    """
    # Get index of start date
    if isinstance(start_date, float):
        end_ix = int(start_date * len(y))
    else:
        end_ix = y.index.get_loc(start_date)

    assert isinstance(end_ix, int)

    # Get prediction on rolling window

    results = {"Date": [], "y_true": [], "y_pred": []}

    while end_ix + horizon < len(y):
        # Training data, current day included
        X_train = X.iloc[: end_ix + 1]
        y_train = y.iloc[: end_ix + 1]

        # Forecast and true value
        y_pred = forecast_fn(
            X_train=X_train, y_train=y_train, horizon=horizon, **kwargs
        )
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


# Store backtest results


def backtest_results(
    log_ret: pd.Series,
    feature_kwargs: dict,
    models: list,
    horizon: int,
    start_date: float | str = 0.5,
    stride: int = 1,
    file_name: str | None = None,
) -> pd.DataFrame:
    """
    Get scores for different models, and potentially saves them.
    """
    X, y = make_xy(log_ret=log_ret, horizon=horizon, **feature_kwargs)

    # Score df
    scores = pd.DataFrame(
        index=[model[1] for model in models], columns=["RMSE", "MAE", "QLIKE"]
    )

    # Run backtest for each model and get results
    for forecast_fn, name, params in models:
        results = run_backtest(
            X,
            y,
            horizon,
            forecast_fn,
            start_date=start_date,
            stride=stride,
            **params,
        )
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        qlike = qlike_loss(y_true, y_pred)
        scores.loc[name] = [rmse, mae, qlike]
        scores.sort_values(by='QLIKE', inplace=True)

    # Save results if desired
    if file_name is not None:
        save_csv(scores.astype(float), "results/backtest", file_name)

    return scores
