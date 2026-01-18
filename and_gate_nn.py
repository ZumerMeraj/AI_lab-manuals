

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dataset for AND gate
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [0],
              [0],
              [1]])

# Build the neural network
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))            # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=500, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# Make predictions
predictions = model.predict(X)
predictions = [1 if p > 0.5 else 0 for p in predictions]
print("Predictions:", predictions)
print("Actual Output:", y.flatten().tolist())

# Keep terminal open
input("\nPress Enter to exit...")
