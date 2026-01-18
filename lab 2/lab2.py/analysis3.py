# Logistic Regression – Pass/Fail Classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Step 1: Create dataset
# -----------------------------
data = pd.DataFrame({
    'Hours_Study': [10, 12, 15, 20, 22, 25, 18, 28, 30],
    'Hours_Sleep': [8, 7, 6, 6, 5, 7, 5, 6, 5],
    'Pass': [0, 0, 0, 1, 1, 1, 0, 1, 1]
})

print("Dataset:\n", data)

# -----------------------------
# Step 2: Separate features and target
# -----------------------------
X = data[['Hours_Study', 'Hours_Sleep']]
y = data['Pass']

# -----------------------------
# Step 3: Train Logistic Regression model
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Step 4: Predict for given input
# -----------------------------
input_data = np.array([[30, 6]])  # Student studies 30 hours, sleeps 6 hours
prob_pass = model.predict_proba(input_data)[0][1]  # Probability of Pass (1)
prediction = model.predict(input_data)[0]

print(f"\nPredicted Probability of Passing: {prob_pass:.2f}")
print(f"Predicted Class: {'Pass' if prediction == 1 else 'Fail'}")

# -----------------------------
# Step 5: Evaluate model
# -----------------------------
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print(f"\nModel Accuracy: {acc:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# -----------------------------
# Step 6: Plot Decision Boundary
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(data['Hours_Study'], data['Hours_Sleep'], c=data['Pass'], cmap='bwr', edgecolors='k')
plt.xlabel('Hours Study')
plt.ylabel('Hours Sleep')
plt.title('Decision Boundary - Pass vs Fail')

# Create mesh grid
x_min, x_max = X['Hours_Study'].min()-2, X['Hours_Study'].max()+2
y_min, y_max = X['Hours_Sleep'].min()-1, X['Hours_Sleep'].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
plt.colorbar(label='0 = Fail, 1 = Pass')
plt.show()
