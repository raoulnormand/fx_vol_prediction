"""
ML-based models.
Unlike baseline models, these take X (features) and y (target) as inputs,
which avoids recomputations and guarantees consistency and lack of leakage.
"""

# Imports

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fxvol.ML_utils import make_xy

# HAR model: OLS with features = lagged rolling vol
# This should match with the arch package results.


def har_ols_forecast(
    log_ret: pd.Series,
    real_vol: pd.Series,
    horizon: int,
    lags: list[int],
    use_asym: bool = True,
) -> float:
    # Get features
    X, y = make_xy(
        log_ret=log_ret,
        real_vol=real_vol,
        horizon=horizon,
        lags=lags,
        use_asym=use_asym,
    )
    X = sm.add_constant(X)

    X_train_c = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_pred_c = X.iloc[-1]

    # Train model
    # X_train_c = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_c).fit()

    # Get predictions
    # X_pred_c = sm.add_constant(X_pred)
    return float(model.predict(X_pred_c).iloc[0])
