# Import required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Create the dataset
data = {
    'Size': [1250, 1550, 1750, 2100, 2450, 2750, 3100, 3550, 3900],
    'Bedroom': [2, 3, 3, 4, 4, 5, 5, 6, 6],
    'Age': [3, 6, 9, 11, 5, 13, 8, 16, 14],
    'Price': [310000, 360000, 400000, 460000, 510000, 560000, 620000, 670000, 710000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Separate features (X) and target (y)
X = df[['Size', 'Bedroom', 'Age']]   # Independent variables
y = df['Price']                      # Dependent variable

# Step 3: Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict the price for given values
# Size = 2000 sqft, Bedrooms = 3, Age = 10 years
predicted_price = model.predict(pd.DataFrame([[2000, 3, 10]], columns=['Size', 'Bedroom', 'Age']))


# Step 5: Print results
print("Coefficients (b1, b2, b3):", model.coef_)
print("Intercept (b0):", model.intercept_)
print(f"Predicted Price for (2000 sqft, 3 bedrooms, 10 years old): {predicted_price[0]:.2f}")


# Step 6: Interpretation
print("\nInterpretation of Coefficients:")
print(f"Size Coefficient ({model.coef_[0]:.2f}): Price increases by this amount per additional sqft.")
print(f"Bedroom Coefficient ({model.coef_[1]:.2f}): Price increases by this amount per additional bedroom.")
print(f"Age Coefficient ({model.coef_[2]:.2f}): Price decreases by this amount per extra year of age.")
