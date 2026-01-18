# nn_regression_tf.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# 1) Reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# 2) Generate 100 samples in [-3, 3] with noise
n_samples = 100
x = np.linspace(-3, 3, n_samples)
noise = np.random.normal(loc=0.0, scale=1.0, size=n_samples)   # std=1
y = x**2 + noise

# Reshape to (n,1)
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

# 3) Train/test split (helps evaluate generalization)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# 4) Build the model (1 hidden layer)
def build_model(hidden_neurons=16):
    model = keras.Sequential([
        layers.Dense(hidden_neurons, activation='relu', input_shape=(1,)),
        layers.Dense(1)   # regression (linear) output
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=['mse'])
    return model

model = build_model(hidden_neurons=16)

# 5) Train
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=200, batch_size=16, verbose=0)

# 6) Evaluate
train_mse = model.evaluate(X_train, Y_train, verbose=0)[0]
test_mse = model.evaluate(X_test, Y_test, verbose=0)[0]
print(f"Train MSE: {train_mse:.4f}  |  Test MSE: {test_mse:.4f}")

# 7) Predictions (for plotting)
X_full = X  # we'll show predictions across all x points
Y_pred = model.predict(X_full).flatten()

# 8) Plot actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(X.flatten(), Y.flatten(), label='Actual (noisy)', alpha=0.7)
# Sort for a clean prediction curve
order = np.argsort(X.flatten())
plt.plot(X.flatten()[order], Y_pred[order], linewidth=2, label='Predicted (NN)')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.title('Actual vs Predicted — NN regression')
plt.grid(True)
plt.show()

# 9) Plot loss curve
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.ylim(bottom=0)
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('Training Curve')
plt.legend(); plt.grid(True)
plt.show()
