# ==============================
# Student Data Analysis Project
# ==============================

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
data = pd.read_csv("students.csv")

# Display dataset
print("===== Dataset =====")
print(data)
print("\n")

# ========================================
# Q3: Observe Dataset Information
# ========================================

# 1. Dataset structure
print("===== data.info() → Dataset structure =====")
data.info()
print("\n")

# 2. Summary statistics
print("===== data.describe() → Summary statistics =====")
print(data.describe())
print("\n")

# 3. Mean of Math marks
mean_math = data['Marks_Math'].mean()
print("Mean of Math Marks:", mean_math)
print("\n")

# 4. Maximum Science marks
max_science = data['Marks_Science'].max()
print("Maximum Science Marks:", max_science)
print("\n")

# ========================================
# Q4: Perform Some Data Analysis
# ========================================

# 1. Students with Marks_Math > 50
count = (data['Marks_Math'] > 50).sum()
print("Number of students with Math Marks > 50:", count)
print("\n")

# 2. Student with the highest Science marks
top_student = data[data['Marks_Science'] == data['Marks_Science'].max()]
print("Student with highest Science marks:")
print(top_student)
print("\n")

# 3. Correlation between Marks_Math and Marks_Science
correlation = data['Marks_Math'].corr(data['Marks_Science'])
print("Correlation between Math and Science marks:", correlation)
print("\n")

# ========================================
# Q5: Data Visualization
# ========================================

# 1. Bar Chart: Student_ID vs Marks_Math
plt.bar(data['Studen_ID'], data['Marks_Math'], color='skyblue')
plt.title("Student ID vs Marks in Math")
plt.xlabel("Student ID")
plt.ylabel("Marks in Math")
plt.show()

# 2. Histogram: Age distribution
plt.hist(data['Age'], bins=5, color='lightgreen', edgecolor='black')
plt.title("Distribution of Student Ages")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 3. Scatter Plot: Marks_Math vs Marks_Science
sns.scatterplot(x='Marks_Math', y='Marks_Science', data=data, color='orange')
plt.title("Math vs Science Marks")
plt.xlabel("Marks in Math")
plt.ylabel("Marks in Science")
plt.show()
