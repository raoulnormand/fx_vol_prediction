"""
Define different models.
"""

# Imports

import pandas as pd
import numpy as np

# Naive model

def naive_forecast(log_ret: pd.Series, horizon:int) -> float:
    """
    Naive forecast: predict latest realized vol.
    """
    return log_ret.iloc[-horizon].std()