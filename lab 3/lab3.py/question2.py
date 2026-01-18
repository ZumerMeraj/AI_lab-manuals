import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Create Dataset
# ---------------------------
data = {
    "StudyHours": ["Low", "Low", "Medium", "High", "High", "Medium", "Low", "High"],
    "Attendance": ["Poor", "Average", "Good", "Good", "Average", "Poor", "Good", "Good"],
    "Result": ["Fail", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass"]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# ---------------------------
# Step 2: Encode Categorical Data
# ---------------------------
le = LabelEncoder()

df["StudyHours"] = le.fit_transform(df["StudyHours"])
df["Attendance"] = le.fit_transform(df["Attendance"])
df["Result"] = le.fit_transform(df["Result"])

print("\nEncoded Data:\n", df)

# ---------------------------
# Step 3: Train Decision Tree (Entropy)
# ---------------------------
X = df[["StudyHours", "Attendance"]]
y = df["Result"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# ---------------------------
# Step 4: Visualize Tree
# ---------------------------
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=["StudyHours", "Attendance"],
    class_names=["Fail", "Pass"],
    filled=True
)
plt.title("Decision Tree - Student Performance")
plt.show()

# ---------------------------
# Step 5: Prediction
# ---------------------------
# Low = 1, Good = 1 (based on LabelEncoder)
new_student = [[1, 1]]

prediction = model.predict(new_student)

print("\nPrediction for Student (StudyHours=Low, Attendance=Good):")

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")
