"""
ML-based models.
"""

# Imports

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

# HAR model: OLS with features = lagged rolling vol
# This should match with the arch package results.


def ols(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    lags: list[int],
    use_asym: bool = False,
) -> float:
    """
    OLS with HAR like features.
    Careful that this is not exactly HAR as we do h steps ahead forecasts
    directly.
    """
    # Columns for training
    train_cols = ["rv"] + [f"rv_{lag}" for lag in lags if lag != 1]

    if use_asym:
        train_cols.append("asym")

    model = LinearRegression()
    model.fit(X_train.iloc[:-1], y_train.iloc[:-1])
    return float(model.predict(X_train.iloc[[-1]]))


def elastic_net(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    lags: list[int],
    use_asym: bool = False,
) -> float:
    """
    OLS with HAR like features and elastic net regularization.
    """
    # Columns for training
    train_cols = ["rv"] + [f"rv_{lag}" for lag in lags if lag != 1]

    if use_asym:
        train_cols.append("asym")

    model = ElasticNet()
    model.fit(X_train.iloc[:-1], y_train.iloc[:-1])
    return float(model.predict(X_train.iloc[[-1]]))


def gb_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    lags: list[int],
    use_asym: bool = False,
) -> float:
    """
    Gradient Boosted tree.
    """
    # Columns for training
    train_cols = ["rv"] + [f"rv_{lag}" for lag in lags if lag != 1]

    if use_asym:
        train_cols.append("asym")

    model = GradientBoostingRegressor()
    model.fit(X_train.iloc[:-1], y_train.iloc[:-1])
    return float(model.predict(X_train.iloc[[-1]]))