"""
Define different models.
"""

# Imports

import numpy as np
import pandas as pd

# Model class


class Model:
    """
    Model class with fit and predict methods
    """

    def __init__(self):
        self.log_ret = None
        self.real_vol = None
        self.state = None

    def fit(
        self, *, log_ret: pd.Series | None = None, real_vol: pd.Series | None = None
    ):
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

    def fit(self, *, real_vol: pd.Series | None = None, **_):
        """
        fit only needs real_vol
        """
        self.real_vol = real_vol

    def predict(self, horizon: int = 1) -> float:
        """
        predict method only needs a horizon
        """
        assert isinstance(self.real_vol, pd.Series)

        return self.real_vol.iloc[-1]


# Rolling mean model


class RollingMeanModel(Model):
    """
    Rolling mean forecast: predict mean of latest *window* realized vol.
    """

    def __init__(self, window: int):
        super().__init__()
        self.window = window

    def fit(self, *, real_vol: pd.Series | None = None, **_):
        """
        fit only needs real_vol
        """
        self.real_vol = real_vol

    def predict(self, horizon: int = 1) -> float:
        """
        predict mean real_vol on window
        """
        assert isinstance(self.real_vol, pd.Series)

        return np.mean(self.real_vol.iloc[-self.window :])  # type: ignore


# Exponential weighted moving average


class EWMA(Model):
    """
    EWMA model.
    """

    def __init__(self, alpha: float = 0.94):
        """
        alpha: smoothing factor (0 < alpha <= 1)
        """
        super().__init__()
        self.alpha = alpha

    def fit(self, *, log_ret: pd.Series | None = None, **_) -> None:
        """
        Compute EWMA of squared log returns.
        """
        assert isinstance(log_ret, pd.Series)

        r2 = log_ret**2
        self.state = r2.ewm(alpha=self.alpha, adjust=False).mean() ** 0.5

    def predict(self, horizon: int = 1, **_) -> float:
        """
        Forecast volatility at the end of the horizon.
        For EWMA, we just take the last value (one-step forecast).
        """
        return self.state.iloc[-1]
