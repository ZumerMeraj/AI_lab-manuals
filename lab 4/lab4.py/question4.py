import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# -----------------------------
# Step 1: Load Digits Dataset
# -----------------------------
digits = load_digits()
X = digits.data        # 64 features (8x8 image flattened)
y = digits.target

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Step 3: Train SVM with RBF Kernel
# -----------------------------
model = SVC(kernel="rbf", gamma="scale")
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Test Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# -----------------------------
# Step 5: Visualize Misclassified Samples
# -----------------------------
misclassified = np.where(y_pred != y_test)[0]

plt.figure(figsize=(10, 5))

for i, idx in enumerate(misclassified[:8]):   # show first 8 wrong predictions
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap="gray")
    plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
    plt.axis("off")

plt.suptitle("Misclassified Digits (True vs Predicted)")
plt.show()
