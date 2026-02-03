import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_sarima(train_series, order=(1,1,1), seasonal_order=(1,1,1,24)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    return fitted

def train_holt_winters(train_series, seasonal_periods=24):
    model = ExponentialSmoothing(train_series, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
    fitted = model.fit()
    return fitted
