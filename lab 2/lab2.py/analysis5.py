# Q5. Comparison – Linear vs Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Step 1: Create Dataset
# -----------------------------
data = {
    'Hours_Study': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score': [40, 45, 50, 55, 60, 65, 70, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# -----------------------------
# Step 2: Linear Regression (predict exam scores)
# -----------------------------
X_linear = df[['Hours_Study']]
y_linear = df['Exam_Score']

lin_reg = LinearRegression()
lin_reg.fit(X_linear, y_linear)

# Predict exam score for 8 hours of study
pred_score = lin_reg.predict([[8]])
print(f"\nPredicted Exam Score for 8 hours study: {pred_score[0]:.2f}")

# Plot Linear Regression
plt.figure(figsize=(6,4))
plt.scatter(df['Hours_Study'], df['Exam_Score'], color='blue', label='Actual Scores')
plt.plot(df['Hours_Study'], lin_reg.predict(X_linear), color='red', label='Linear Fit')
plt.xlabel('Hours of Study')
plt.ylabel('Exam Score')
plt.title('Linear Regression - Exam Score Prediction')
plt.legend()
plt.show()

# -----------------------------
# Step 3: Logistic Regression (predict pass/fail)
# -----------------------------
X_logistic = df[['Hours_Study']]
y_logistic = df['Pass']

log_reg = LogisticRegression()
log_reg.fit(X_logistic, y_logistic)

# Predict probability of passing for 8 hours of study
prob_pass = log_reg.predict_proba([[8]])[0][1]
print(f"Probability of Passing (8 hours study): {prob_pass:.4f}")

# Predict classes for dataset
y_pred = log_reg.predict(X_logistic)
acc = accuracy_score(y_logistic, y_pred)
print(f"\nLogistic Regression Accuracy: {acc:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_logistic, y_pred))

# Plot Logistic Regression
X_test = np.linspace(2, 10, 100).reshape(-1, 1)
y_prob = log_reg.predict_proba(X_test)[:, 1]

plt.figure(figsize=(6,4))
plt.scatter(df['Hours_Study'], df['Pass'], color='blue', label='Actual (0=Fail, 1=Pass)')
plt.plot(X_test, y_prob, color='red', label='Logistic Curve')
plt.xlabel('Hours of Study')
plt.ylabel('Probability of Pass')
plt.title('Logistic Regression - Pass Prediction')
plt.legend()
plt.show()

# -----------------------------
# Step 4: Comparison Explanation
# -----------------------------
print("\n--- Comparison ---")
print("Linear Regression outputs continuous values (e.g., exam score = 75.2).")
print("It assumes a straight-line relationship and can predict values outside 0–1 range.")
print("Hence, it’s unsuitable for binary classification (Pass/Fail).")
print("Logistic Regression outputs probabilities between 0 and 1, making it ideal for classification.")
