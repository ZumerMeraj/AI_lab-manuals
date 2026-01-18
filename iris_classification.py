import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# -----------------------
# 1. Load Dataset
# -----------------------
data = load_iris()
X = data.data
y = data.target  # 0 = setosa, 1 = versicolor, 2 = virginica

print("Dataset Loaded!")
print("Features:", X.shape)
print("Labels:", y.shape)

# -----------------------
# 2. One-hot encode labels
# -----------------------
y = to_categorical(y, num_classes=3)

# -----------------------
# 3. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 4. Normalize Features
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# 5. Build Neural Network
# -----------------------
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # Multi-class output
])

# -----------------------
# 6. Compile Model
# -----------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# 7. Train Model
# -----------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=5,
    validation_split=0.2,
    verbose=1
)

# -----------------------
# 8. Evaluate Model
# -----------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nFinal Test Accuracy:", accuracy)
print("Final Test Loss:", loss)

# -----------------------
# 9. Plot Accuracy & Loss
# -----------------------
epochs_range = range(1, 51)

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
