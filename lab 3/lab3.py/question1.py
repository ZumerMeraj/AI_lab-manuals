# Step 1: Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 2: Create the dataset
data = {
    'Study_Hours': ['Low', 'High', 'High', 'Low', 'High'],
    'Attendance': ['Poor', 'Good', 'Poor', 'Good', 'Good'],
    'Result': ['Fail', 'Pass', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 3: Encode categorical variables
encoder = LabelEncoder()
df_encoded = df.apply(encoder.fit_transform)
print("\nEncoded Dataset:\n", df_encoded)

# Step 4: Define features (X) and target (y)
X = df_encoded[['Study_Hours', 'Attendance']]
y = df_encoded['Result']

# Step 5: Train Decision Tree Classifier with criterion='entropy'
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Step 6: Visualize the Decision Tree
plt.figure(figsize=(8,6))
plot_tree(model, feature_names=['Study_Hours', 'Attendance'], 
          class_names=encoder.classes_, filled=True)
plt.show()

# Step 7: Predict for Study_Hours=Low and Attendance=Good
# Encode input manually using same encoder mapping
sample = pd.DataFrame({'Study_Hours': ['Low'], 'Attendance': ['Good']})
sample_encoded = sample.apply(lambda col: encoder.transform(col))

prediction = model.predict(sample_encoded)
predicted_label = encoder.inverse_transform(prediction)

print("\nPrediction for Study_Hours=Low and Attendance=Good:", predicted_label[0])

