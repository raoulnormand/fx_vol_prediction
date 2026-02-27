"""
Downloads and saves data
"""

# Imports

from fxvol.data_utils import fetch_yahoo, save_csv

# Global variables

TICKERS = ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "CHF=X"]
NEW_TICKER_NAMES = ["EUR", "JPY", "GBP", "AUD", "CHF"]
START = "2010-01-01"
END = "2025-12-31"

# Download, keep close price, rename columns, and save data

df = fetch_yahoo(TICKERS, START, END)
df = df['Close']
name_dic = dict(zip(TICKERS, NEW_TICKER_NAMES))
df.rename(columns=name_dic, inplace=True)
save_csv(df, "data/raw", "fx_spots")
