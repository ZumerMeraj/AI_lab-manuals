import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --------------------------------
# Step 1: Load MNIST Dataset
# --------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# --------------------------------
# Step 2: Preprocess Images
# Flatten 28x28 images into 784 features
# --------------------------------
X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)

# Optional: normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# --------------------------------
# Step 3: Train Decision Tree
# --------------------------------
model = DecisionTreeClassifier(criterion="entropy", max_depth=15)
model.fit(X_train, y_train)

# --------------------------------
# Step 4: Evaluate Accuracy
# --------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

# --------------------------------
# Step 5: Confusion Matrix
# --------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("MNIST Confusion Matrix - Decision Tree")
plt.show()
