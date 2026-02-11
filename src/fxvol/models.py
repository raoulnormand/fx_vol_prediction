"""
Define prediction functions for different models.
All functions take log_ret and real_vol as argument, even though one of them may not be used; this is not an issue since real_vol is calculated consistently in the basktesting function.
"""

# Imports

from typing import List

import numpy as np
import pandas as pd
from arch.univariate import HARX, arch_model

# Naive model


def naive_forecast(log_ret: pd.Series, real_vol: pd.Series, horizon: int) -> float:
    """
    Naive forecast: predict latest realized vol.
    """

    return real_vol.iloc[-1]


# Rolling mean model


def rolling_mean_forecast(
    log_ret: pd.Series, real_vol: pd.Series, horizon: int, window: int
) -> float:
    """
    Rolling mean forecast: predict mean of latest *window* realized vol.
    """
    return real_vol.iloc[-window:].mean()


# Exponential weighted moving average model


def ewma_forecast(
    log_ret: pd.Series, real_vol: pd.Series, horizon: int, alpha: float = 0.92
) -> float:
    """
    EWMA model. 0.92 is a recommended value for one-weeek ahead forecast.
    """

    var = (log_ret**2).ewm(alpha=alpha, adjust=False).mean()
    return np.sqrt(var.iloc[-1])


# HAR model


def har_forecast(
    log_ret: pd.Series,
    real_vol: pd.Series,
    horizon: int,
    lags: List[int],
    scale: int = 1000,
):
    """
    Heterogeneous Autoregressive model: OLS with features = rv,
    and rolling mean over several days.
    Scale to avoid convergence issues.
    """

    scaled_vol = (scale * real_vol).dropna()
    model = HARX(scaled_vol, lags=lags).fit(disp="off")
    return model.forecast(horizon=horizon).mean.iloc[0, -1] / scale  # type: ignore


# GARCH11 model


def garch11_forecast(
    log_ret: pd.Series,
    real_vol: pd.Series,
    horizon: int,
    scale: int = 100,
):
    """
    GARCH(1, 1) model
    """

    scaled_ret = scale * log_ret
    am = arch_model(scaled_ret, vol="GARCH", p=1, o=0, q=1, dist="normal")
    res = am.fit(update_freq=5, disp="off")
    return np.sqrt(res.forecast(horizon=horizon).variance.iloc[-1, -1]) / scale  # type: ignore

    # model = arch_model(scaled_ret).fit(disp="off")
    # return model.forecast(horizon=horizon).mean.iloc[0, -1] / scale  # type: ignore
