"""
Downloads and saves data
"""

# Imports

from fxvol.config import CURRENCIES, END_DATE, START_DATE, TICKERS
from fxvol.data_utils import fetch_yahoo, save_csv

# Download, keep close price, rename columns, and save data

df = fetch_yahoo(TICKERS, START_DATE, END_DATE)
df = df["Close"]
name_dic = dict(zip(TICKERS, CURRENCIES))
df.rename(columns=name_dic, inplace=True)
save_csv(df, "data/raw", "fx_spots")
