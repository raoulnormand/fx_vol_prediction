# FX volatility forecast and regime changes

We compare different models to forecast FX realized volatility, and us them to perform a volatility targeting strategy.

## Summary

The whole project uses FX spot data from 2010-01-01 to 2025-12-31 for AUD, CHF, EUR, GBP, JPY vs USD. Daily close data is used. For each spot $(S_t)$, we compute the daily log returns
$$
r_t = \log \left ( \frac{S_t}{S_{t-1}} \right )
$$
then the realized volatility over horizon of $h$ days:
$$
\sigma_t = \mathrm{std} \left ( r_t, r_{t-1}, \dots, r_{t-h+1} \right ).
$$
The default value used for the following results is a week, that is $h = 5$.

We first use different models for volatility forecasting: using data $(r_t)_{t \leq T}$, we aim to give prediction $\hat{\sigma}_{t+h}$ for $\sigma_{t+h}$ (this shift avoids overlaps and means that we are using non-intersecting set of values for the features and the targets).

Then, we use the predictions to build a portfolio using a volatility targeting strategy: given a target annual vol $\sigma^*$ for our portfolio (we use $\sigma^* = 10\%$), we set weights
$$
w^{(i)}_t = \frac{\sigma^*}{\hat{\sigma}_{t+h}}
$$
for asset $i$. This is held over the period $(t, t+h]$. If the assets are uncorrelated and the vol predictions are good, we expect that the annual vol of our portfolio will be close to $\sigma^*$.

## Results

Data exploration results and graphs can be found in [this notebook](./notebooks/01_data_exploration.ipynb). All the backtesting and vol targeting results are in the [results](./results/) folder, with more explanations and figures in [this notebook](./notebooks/02_summary.ipynb). Here is a summary.

### Models

We compare the following models for volatility targeting.
- Naive model: the value predicted is the previous one.
- Rolling mean: the value predicted is the average of the $k$ previous one (we take $k=5$ and $k=50$).
- [EWMA](https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html) model: the value predicted is an average of the previous observations with exponentially decaying weights $\alpha^k$ (we take $\alpha = 0.8$ and $\alpha = 0.3$).
- [HAR](https://portfoliooptimizer.io/blog/volatility-forecasting-har-model/) model with lags 1, 5, 22, 66 (1 day, 1 week, one month, 3 months).
- [GARCH(1, 1)](https://portfoliooptimizer.io/blog/volatility-forecasting-garch11-model/) model.
- Ordinary linear regression, the features used are the same as HAR (vol with lags 1, 5, 22, 66), as the well as the "vol of vol over a month", that is the std of the volatility over the last 22 days. The expectation is that this allows to take volatility regimes into account.
- [Elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization) (OLS with $L^1$ and $L^2$ regularization), using parameters $\alpha = 1$ and $\alpha = 10^{-3}$.
- [Gradient boosted tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) with sklearn's standard parameters.

### Metrics

Assume that we have targets $(\sigma_t)_{t \in S}$ and forecasts $(\hat{\sigma}_t)_{t \in S}$. Here, $S$ is the set of values that we predict, consisting of every 5 trading days after a burn-in period (taken by default to be half of total length of our dataset). Then the metrics that we use are
- Mean Average Error:
$$
\mathrm{MAE} = \frac{1}{|S|} \sum_{s \in S} \left | \sigma_t - \hat{\sigma}_t \right |.
$$
- Root Mean Square Error:
$$
\mathrm{MAE} = \sqrt{ \frac{1}{|S|} \sum_{s \in S} \left ( \sigma_t - \hat{\sigma}_t \right )^2 }.
$$
- [QLIKE](https://public.econ.duke.edu/~ap172/Patton_robust_forecast_eval_11dec08.pdf) loss, which penalizes underestimations more than overestimations:
$$
\mathrm{QLIKE} = \frac{1}{|S|} \sum_{s \in S} \left ( \frac{\sigma_t}{\hat{\sigma}_t} - \log \left ( \frac{\sigma_t}{\hat{\sigma}_t} \right ) -1 \right ).
$$

The metric usually favored is QLIKE, and our results will be ordered by decreasing QLIKE. However, the results below show clearly that the choice of metrics makes little difference.

### Results

The losses for each model are as followed. These are averaged over all currencies, but the ordering is essentially consistent across currencies. All details are in the [notebook](./notebooks/02_summary.ipynb) or the [backtest](./results/backtest/) folder.

|     Model             |      MAE |     RMSE |    QLIKE |
|-----------------:|---------:|---------:|---------:|
|          gb_tree | 1.67e-03 | 2.32e-03 | 1.05e-01 |
|              ols | 1.68e-03 | 2.30e-03 | 1.08e-01 |
|              har | 1.70e-03 | 2.33e-03 | 1.11e-01 |
|        rolling50 | 1.71e-03 | 2.38e-03 | 1.15e-01 |
|          garch11 | 1.89e-03 | 2.44e-03 | 1.19e-01 |
|    elastic_net_1 | 2.00e-03 | 2.60e-03 | 1.38e-01 |
| elastic_net_1e-3 | 2.00e-03 | 2.60e-03 | 1.38e-01 |
|          ewma030 | 1.97e-03 | 2.71e-03 | 1.51e-01 |
|         rolling5 | 1.98e-03 | 2.72e-03 | 1.62e-01 |
|            naive | 2.15e-03 | 2.95e-03 | 2.16e-01 |
|          ewma090 | 2.73e-03 | 3.70e-03 | 6.25e-01 |

The clear winners are gradient-boosted trees and OLS. The GB tree can capture some non-linearities that OLS does not see, though the minor difference seems to indicate that this is minimal. It is also likely that hyperparameter tweaking and feature engineering may slightly improve results.

The comparison can also be visualized (using the QLIKE loss) with the following bar chart


and heat map.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). First clone the project, then run
```
uv sync
```
The data can be obtained from Yahoo Finance with
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
Finally, summary of the metrics for each model are computed and saved with
```
uv run scripts/summarize_strats.py
```
