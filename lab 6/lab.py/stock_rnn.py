import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
dataset = pd.read_csv("Google_Stock_Price_Train.csv")

# Use only Open price
prices = dataset["Open"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)
X = []
y = []

time_steps = 60

for i in range(time_steps, len(prices_scaled)):
    X.append(prices_scaled[i-time_steps:i, 0])
    y.append(prices_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for RNN: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)
model = Sequential()

model.add(SimpleRNN(50, return_sequences=True, input_shape=(60, 1)))
model.add(SimpleRNN(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()
model.fit(X, y, epochs=20, batch_size=32)
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

real_prices = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(real_prices, label="Real Stock Price")
plt.plot(predicted_prices, label="Predicted Stock Price")
plt.title("Stock Price Prediction using RNN")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
