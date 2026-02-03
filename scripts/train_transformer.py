import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.transformer_model import build_transformer

# Load data
data = pd.read_csv("data/synthetic_data.csv")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Sequence generation
def create_sequences(data, seq_len=48, horizon=1):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+horizon, 0])  # predict feature_1
    return np.array(X), np.array(y)

SEQ_LEN = 48
HORIZON = 1
X, y = create_sequences(scaled_data, SEQ_LEN, HORIZON)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = build_transformer(input_shape=(SEQ_LEN, 5))
model.summary()

# Train model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=32)
model.save("models/transformer_model.h5")
print("Transformer model saved at 'models/transformer_model.h5'")
