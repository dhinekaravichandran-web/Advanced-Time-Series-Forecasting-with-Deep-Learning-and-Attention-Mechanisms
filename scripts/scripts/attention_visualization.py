import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from models.transformer_model import TransformerBlock

# Load data
data = pd.read_csv("data/synthetic_data.csv")
SEQ_LEN = 48
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_len=SEQ_LEN):
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
    return np.array(X)

X_test = create_sequences(scaled_data)
X_sample = X_test[:1]  # visualize first sample

# Load model
model = load_model("models/transformer_model.h5", compile=False)

# Build attention extractor
attention_layer = model.layers[2]  # Assuming transformer block is 3rd layer
attention_model = tf.keras.models.Model(inputs=model.input, outputs=attention_layer.output[1])
attn_weights = attention_model.predict(X_sample)

plt.imshow(attn_weights[0].mean(axis=0))
plt.colorbar()
plt.title("Average Attention Weights")
plt.xlabel("Input Time Steps")
plt.ylabel("Query Time Steps")
plt.show()
