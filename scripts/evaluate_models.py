import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.baselines import train_sarima, train_holt_winters
from tensorflow.keras.models import load_model

data = pd.read_csv("data/synthetic_data.csv")
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Load transformer model
transformer = load_model("models/transformer_model.h5")

# Prepare sequences for test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
SEQ_LEN = 48
HORIZON = 1

def create_sequences(data, seq_len=48, horizon=1):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+horizon, 0])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(scaled_data, SEQ_LEN, HORIZON)
y_test_flat = y_test.flatten()

# Transformer predictions
transformer_preds = transformer.predict(X_test).flatten()

# SARIMA
sarima_model = train_sarima(train_data["feature_1"])
sarima_preds = sarima_model.forecast(len(y_test_flat))

# Holt-Winters
hw_model = train_holt_winters(train_data["feature_1"])
hw_preds = hw_model.forecast(len(y_test_flat))

# Evaluation
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

print("Transformer → RMSE:", *evaluate(y_test_flat, transformer_preds))
print("SARIMA     → RMSE:", *evaluate(y_test_flat, sarima_preds))
print("Holt-Winters → RMSE:", *evaluate(y_test_flat, hw_preds))
