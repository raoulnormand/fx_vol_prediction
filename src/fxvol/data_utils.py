"""
Functions to fecth and save data
"""

# Imports

from pathlib import Path

import pandas as pd
import yfinance as yf

# Directories

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

# Fetch data


def fetch_yahoo(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance and keep close price only
    """
    df = yf.download(tickers, start=start, end=end)
    assert isinstance(df, pd.DataFrame)
    return df


# Save and load CSV


def save_csv(df: pd.DataFrame | pd.Series, folder: "str", name: str) -> None:
    """
    Saves df as csv in corresponding folder
    """
    target_dir = DATA_DIR / f"{folder}"
    target_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_dir / f"{name}.csv")


def load_csv(folder: "str", name: str, index_col: str = "Date") -> pd.DataFrame:
    """
    Load csv from corresponding folder as df
    """
    origin_dir = DATA_DIR / f"{folder}"
    return pd.read_csv(origin_dir / f"{name}.csv", index_col=index_col)
