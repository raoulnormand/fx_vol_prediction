"""
Functions to fecth and save data
"""

# Imports

from pathlib import Path

import pandas as pd
import yfinance as yf

from fxvol.fin_comp import realized_vol

# Directories

ROOT = Path(__file__).resolve().parent.parent.parent

# Fetch data


def fetch_yahoo(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance and keep close price only
    """
    df = yf.download(tickers, start=start, end=end)
    assert isinstance(df, pd.DataFrame)
    return df


# Save and load CSV


def save_csv(df: pd.DataFrame | pd.Series, folder: str, name: str) -> None:
    """
    Saves df as csv in corresponding folder
    """
    target_dir = ROOT / f"{folder}"
    target_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_dir / f"{name}.csv")


def load_csv(folder: str, name: str, index_col: str = "Date") -> pd.DataFrame:
    """
    Load csv from corresponding folder as df
    """
    origin_dir = ROOT / f"{folder}"
    return pd.read_csv(origin_dir / f"{name}.csv", index_col=index_col)


# Create features


def make_features(
    log_ret: pd.Series,
    window: int,
    lags: list[int] | None = None,
    vol_vol: int | None = None,
):
    """
    Create features for training models.
    """
    # Design matrix
    X = pd.DataFrame(index=log_ret.index)

    # X contains at least log returns and realized_vol
    X["lr"] = log_ret
    X["rv"] = realized_vol(log_ret, window=window)

    # Standard features: rolling means
    if lags is not None:
        for lag in lags:
            if lag == 1:
                pass
            else:
                X[f"rv_{lag}"] = X["rv"].rolling(lag).mean()

    # Vol of vol = std of vol -> feature to capture regime changes

    if vol_vol:
        X["vv"] = X["rv"].rolling(vol_vol).std()

    return X


# Create aligned features and target


def make_xy(
    log_ret: pd.Series,
    horizon: int,
    lags: list[int] | None = None,
    vol_vol: int | None = 22,
    **_,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build aligned features (using data up to t)
    and target (realized vol at given horizon).
    Note that realized vol over horizon days for consistency.
    """
    # Features
    X = make_features(log_ret, window=horizon, lags=lags, vol_vol=vol_vol)

    # Target = shifted real_vol
    y = X["rv"].shift(-horizon).rename("y")

    # Deal with missing values

    df = pd.concat([y, X], axis=1).dropna()

    return df.drop(columns=["y"]), df["y"]
