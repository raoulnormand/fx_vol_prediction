"""
Define prediction functions for different baseline models.
All functions take log_ret and real_vol as argument, even though one of them
may not be used; this is not an issue since real_vol is calculated
consistently in the basktesting function.
"""

# Imports

from typing import List

import numpy as np
import pandas as pd
from arch.univariate import HARX, arch_model

# Naive model


def naive_forecast(X: pd.DataFrame, horizon: int, **_) -> float:
    """
    Naive forecast: predict latest realized vol.
    """

    return X["rv"].iloc[-1]


# Rolling mean model


def rolling_mean_forecast(X: pd.DataFrame, horizon: int, window: int, **_) -> float:
    """
    Rolling mean forecast: predict mean of latest *window* realized vol.
    """
    return X["rv"].iloc[-window:].mean()


# Exponential weighted moving average model


def ewma_forecast(X: pd.DataFrame, horizon: int, alpha: float) -> float:
    """
    EWMA model.
    """

    var = (X["lr"] ** 2).ewm(alpha=alpha, adjust=False).mean()
    return np.sqrt(var.iloc[-1])


# HAR model


def har_forecast(
    X: pd.DataFrame,
    horizon: int,
    lags: List[int],
    scale: int = 1000,
):
    """
    Heterogeneous Autoregressive model: OLS with features = rv,
    and rolling mean over several days.
    Scale to avoid convergence issues.
    """

    scaled_vol = (scale * X["rv"]).dropna()
    model = HARX(scaled_vol, lags=lags, rescale=False).fit(disp="off")
    return model.forecast(horizon=horizon).mean.iloc[0, -1] / scale  # type: ignore


# GARCH11 model


def garch11_forecast(
    X: pd.DataFrame,
    horizon: int,
    scale: int = 100,
):
    """
    GARCH(1, 1) model
    """

    scaled_ret = scale * X["lr"]
    am = arch_model(scaled_ret, vol="GARCH", p=1, o=0, q=1, dist="normal")
    res = am.fit(disp="off", update_freq=0)
    return np.sqrt(res.forecast(horizon=horizon).variance.iloc[-1, -1]) / scale  # type: ignore
