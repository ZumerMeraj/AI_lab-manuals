from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Iris Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)

# -----------------------------
# Step 2: Split Data (70% Train, 30% Test)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Step 3: Train Decision Tree (Entropy)
# -----------------------------
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Accuracy on Test Set
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

# -----------------------------
# Step 5: Visualize Decision Tree
# -----------------------------
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree on Iris Dataset (Entropy)")
plt.show()
