"""
Functions for ML-based forecasting.
"""

# Imports

import pandas as pd

# Create features


def make_features(
    log_ret: pd.Series,
    real_vol: pd.Series,
    lags: list[int],
    use_asym: bool = False,
):
    """
    Create features for training ML models.
    """
    X = pd.DataFrame(index=real_vol.index)

    # Standard features: rolling means
    for lag in lags:
        X[f"rv_{lag}"] = real_vol.rolling(lag).mean()

    # Asymmetric feature: vol * 1_{< 0 return}

    if use_asym:
        X["asym"] = real_vol * (log_ret < 0).astype(float)

    return X


# Create aligned features and target


def make_xy(
    log_ret: pd.Series,
    real_vol: pd.Series,
    horizon: int,
    lags: list[int],
    use_asym: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build aligned features (using data up to t)
    and target (realized vol at given horizon).
    """
    # Features
    X = make_features(log_ret, real_vol, lags=lags, use_asym=use_asym)
    
    # Target = shifted real_vol
    y = real_vol.shift(-horizon).rename("y")
    
    # Deal with missing values

    df = pd.concat([y, X], axis=1).dropna()
    
    return df.drop(columns=["y"]), df["y"]
