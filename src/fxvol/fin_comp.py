"""
Functions to compute financial quantities
"""

# Imports

import numpy as np
import pandas as pd

# Compute log returns


def log_returns(spots: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Compute log returns from spots.
    First value will become na.
    """
    returns = spots / spots.shift(1)
    return returns.apply(np.log)


# Compute realized vol


def realized_vol(
    log_ret: pd.Series | pd.DataFrame, window: int
) -> pd.Series | pd.DataFrame:
    """
    Compute the realized vol as the std of log returns
    between t - window + 1 and t.
    """
    return log_ret.rolling(window).std()


# QLIKE loss function


def qlike_loss(y_true: pd.Series, y_pred: pd.Series, eps=1e-10) -> float:
    """
    Compute QLIKE loss function.
    """
    ratio = (y_true + eps) / (y_pred + eps)
    return np.mean((ratio - np.log(ratio) - 1))  # type: ignore
