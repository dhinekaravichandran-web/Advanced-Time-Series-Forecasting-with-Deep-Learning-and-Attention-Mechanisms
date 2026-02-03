import numpy as np
import pandas as pd
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n_steps = 1500
t = np.arange(n_steps)

# Trend
trend = 0.005 * t + 0.00001 * t**2

# Seasonal components
seasonal_daily = 2 * np.sin(2 * np.pi * t / 24)
seasonal_weekly = 3 * np.sin(2 * np.pi * t / (24 * 7))

# Noise
noise = np.random.normal(0, 0.5, size=n_steps)

# Features
f1 = trend + seasonal_daily + noise
f2 = 0.8 * f1 + seasonal_weekly + np.random.normal(0, 0.3, n_steps)
f3 = 0.5 * f2 + 0.3 * trend + np.random.normal(0, 0.4, n_steps)
f4 = seasonal_daily * seasonal_weekly + np.random.normal(0, 0.2, n_steps)
f5 = trend + np.random.normal(0, 0.6, n_steps)

data = pd.DataFrame({
    "feature_1": f1,
    "feature_2": f2,
    "feature_3": f3,
    "feature_4": f4,
    "feature_5": f5
})

data.to_csv("data/synthetic_data.csv", index=False)
print("Synthetic dataset saved as 'data/synthetic_data.csv'")
