import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
# Generate synthetic daily temperature data (for 2 years)
np.random.seed(42)
days = 730
temps = 20 + 10 * np.sin(np.linspace(0, 20, days)) + np.random.normal(0, 1, days)

# Create DataFrame
dataset = pd.DataFrame({"Temperature": temps})

# Visualize temperature over time
plt.figure(figsize=(12,4))
plt.plot(dataset["Temperature"])
plt.title("Daily Temperature Over Time")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.show()
scaler = MinMaxScaler(feature_range=(0,1))
temps_scaled = scaler.fit_transform(dataset["Temperature"].values.reshape(-1,1))
X = []
y = []

time_steps = 30

for i in range(time_steps, len(temps_scaled)):
    X.append(temps_scaled[i-time_steps:i, 0])
    y.append(temps_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for RNN: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)
model = Sequential()
model.add(SimpleRNN(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()
model.fit(X, y, epochs=20, batch_size=32)
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y.reshape(-1,1))
plt.figure(figsize=(12,4))
plt.plot(actual, label="Actual Temperature")
plt.plot(predicted, label="Predicted Temperature")
plt.title("Temperature Prediction using RNN")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()
