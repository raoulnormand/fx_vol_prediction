"""
Define different models.
"""

# Imports

import numpy as np
import pandas as pd
from arch.univariate import HARX, arch_model

# Model class


class Model:
    """
    Model class with fit and predict methods
    """

    def __init__(self):
        self.log_ret = pd.Series()
        self.real_vol = pd.Series()
        self.state = None

    def fit(self, *, log_ret: pd.Series, real_vol: pd.Series):
        """
        fit methods can take log returns and realized vol as arguments to avoid recomputations.
        """
        raise NotImplementedError

    def predict(self, horizon: int) -> float:
        """
        predict method only needs a horizon
        """
        raise NotImplementedError


# Naive model


class NaiveModel(Model):
    """
    Naive forecast: predict latest realized vol.
    """

    def fit(self, *, real_vol: pd.Series, **_):
        """
        fit only needs real_vol
        """
        self.real_vol = real_vol

    def predict(self, horizon: int = 1) -> float:
        """
        predict method only needs a horizon
        """

        return self.real_vol.iloc[-1]


# Rolling mean model


class RollingMeanModel(Model):
    """
    Rolling mean forecast: predict mean of latest *window* realized vol.
    """

    def __init__(self, window: int):
        super().__init__()
        self.window = window

    def fit(self, *, real_vol: pd.Series, **_):
        """
        fit only needs real_vol
        """
        self.real_vol = real_vol

    def predict(self, horizon: int = 1) -> float:
        """
        predict mean real_vol on window
        """

        return np.mean(self.real_vol.iloc[-self.window :])  # type: ignore


# Exponential weighted moving average


class EWMA(Model):
    """
    EWMA model.
    """

    def __init__(self, alpha: float):
        """
        alpha: smoothing factor (0 < alpha <= 1)
        """
        super().__init__()
        self.alpha = alpha

    def fit(self, *, log_ret: pd.Series, **_) -> None:
        """
        Compute EWMA of squared log returns.
        """

        r2 = log_ret**2
        self.state = r2.ewm(alpha=self.alpha, adjust=False).mean() ** 0.5

    def predict(self, horizon: int = 1, **_) -> float:
        """
        Forecast volatility at the end of the horizon.
        For EWMA, we just take the last value (one-step forecast).
        """
        return self.state.iloc[-1]


# HAR model


class HAR(Model):
    """
    Heterogeneous Autoregressive model: OLS with features = rv,
    and rolling mean over several days
    """

    def __init__(self, scale: int = 1000):
        """
        Scaling factor to avoid convergence issues
        """
        super().__init__()
        self.scale = scale

    def fit(self, *, real_vol: pd.Series, **_):
        """
        fit only needs real_vol
        """
        self.real_vol = (self.scale * real_vol).dropna()
        self.state = HARX(self.real_vol, lags=[1, 5, 22, 50, 100]).fit()

    def predict(self, horizon: int = 1) -> float:
        """
        predict mean real_vol on window
        """

        return self.state.forecast(horizon=horizon).mean.iloc[0, horizon - 1] / self.scale  # type: ignore


class GARCH(Model):
    """
    GARCH model
    """

    def __init__(self, scale: int = 1000):
        """
        Scaling factor to avoid convergence issues
        """
        super().__init__()
        self.scale = scale

    def fit(self, *, real_vol: pd.Series, **_):
        """
        fit only needs real_vol
        """
        self.real_vol = (self.scale * real_vol).dropna()
        self.state = arch_model(self.real_vol).fit()

    def predict(self, horizon: int = 1) -> float:
        """
        predict mean real_vol on window
        """

        return self.state.forecast(horizon=horizon).mean.iloc[0, horizon - 1] / self.scale  # type: ignore