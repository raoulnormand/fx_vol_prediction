"""
Downloads and saves data
"""

# Imports

import numpy as np
import yfinance as yf

# Tickers to get

tickers = ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "CHF=X"]

# Get raw data

df = yf.download(tickers, start="2010-01-01", end="2025-12-31")
df.to_csv("data/raw/fx_spots.csv")

# Keep only close price and rename columns

df = df["Close"]
df.rename(
    columns={
        "EURUSD=X": "EUR",
        "JPY=X": "JPY",
        "GBPUSD=X": "GBP",
        "AUDUSD=X": "AUD",
        "CHF=X": "CHF"
    },
    inplace=True
)
df.to_csv("data/processed/fx_spots_close.csv")

# Compute log returns

log_returns = np.log(df / df.shift(1)).iloc[1:]
log_returns.to_csv("data/processed/log_returns.csv")
