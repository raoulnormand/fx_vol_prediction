# FX volatility forecast and regime changes

We compare different models to forecast FX realized volatility, and us them to perform a volatility targeting strategy.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). First clone the project, then run
```
uv sync
```
The whole project uses FX spot data from 2010-01-01 to 2025-12-31 for AUD, CHF, EUR, GBP, JPY vs USD. The data can be obtained from Yahoo Finance with
```
uv run scripts/get_data.py
```
Then clean data (a few rows missing) and save log returns with
```
uv run scripts/clean_data.py
```
Run the backtests for all models with
```
uv run scripts/run_backtests.py
```
Run the volatility targeting strategy and save the returns with
```
uv run scripts/comp_start_returns.py
```