# Q4. Logistic Regression – Diabetes Prediction (Binary Classification)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# -----------------------------
# 1. Create Dataset
# -----------------------------
data = {
    'BMI': [22, 27, 31, 26, 34, 29, 24, 33, 30],
    'Age': [25, 30, 35, 28, 45, 40, 32, 50, 42],
    'Glucose': [85, 90, 120, 100, 140, 130, 95, 160, 145],
    'Diabetic': [0, 0, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")

# -----------------------------
# 2. Split features & target
# -----------------------------
X = df[['BMI', 'Age', 'Glucose']]
y = df['Diabetic']

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 4. Fit Logistic Regression Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 6. Predict for new patient
# -----------------------------
new_patient = [[28, 45, 150]]  # BMI=28, Age=45, Glucose=150
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)[0][1]

print("\nPredicted Class (1=Diabetic, 0=Not Diabetic):", int(prediction[0]))
print("Predicted Probability of being Diabetic:", round(probability, 2))
