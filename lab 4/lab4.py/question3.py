import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load CSV File
# -----------------------------
df = pd.read_csv("students.csv")
print("Dataset:\n", df)

# -----------------------------
# Step 2: Features and Target
# -----------------------------
X = df[["study_hours", "attendance", "marks"]]
y = df["result"]

# Convert Pass/Fail to numeric
y = y.map({"Fail": 0, "Pass": 1})

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Step 4: Train Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# -----------------------------
# Step 6: Feature Importance
# -----------------------------
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.3f}")

# -----------------------------
# Step 7: Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

# -----------------------------
# Step 8: Predict New Student
# Example: study_hours=3, attendance=80, marks=60
# -----------------------------
new_student = [[3, 80, 60]]
prediction = model.predict(new_student)

print("\nNew Student Prediction:")
if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")
