# =========================================================
# HOUSE PRICE PREDICTION
# Data Cleaning + Visualization + StandardScaler
# + Linear Regression + Evaluation
# =========================================================


# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 2. LOAD THE DATASET
# ---------------------------------------------------------

df = pd.read_csv("house_price_dirty.csv")

print("========== ORIGINAL DATASET ==========")
print(df)

print("\nShape of Dataset:")
print(df.shape)


# ---------------------------------------------------------
# 3. UNDERSTAND THE DATA
# ---------------------------------------------------------

print("\n========== DATA INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------
# 4. DATA CLEANING
# ---------------------------------------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing Area_sqft with median
df["Area_sqft"] = df["Area_sqft"].fillna(
    df["Area_sqft"].median()
)

# Fill missing Bedrooms with median
df["Bedrooms"] = df["Bedrooms"].fillna(
    df["Bedrooms"].median()
)


# ---------------------------------------------------------
# 5. CHECK DATA AFTER CLEANING
# ---------------------------------------------------------

print("\n========== CLEANED DATASET ==========")
print(df)

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES AFTER CLEANING ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------

# Area vs Price

plt.figure(figsize=(7, 5))

plt.scatter(
    df["Area_sqft"],
    df["Price_lakh"]
)

plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakh)")
plt.title("Area vs House Price")

plt.show()


# Bedrooms vs Price

plt.figure(figsize=(7, 5))

plt.scatter(
    df["Bedrooms"],
    df["Price_lakh"]
)

plt.xlabel("Number of Bedrooms")
plt.ylabel("Price (Lakh)")
plt.title("Bedrooms vs House Price")

plt.show()


# Age vs Price

plt.figure(figsize=(7, 5))

plt.scatter(
    df["Age_years"],
    df["Price_lakh"]
)

plt.xlabel("House Age (Years)")
plt.ylabel("Price (Lakh)")
plt.title("House Age vs Price")

plt.show()


# ---------------------------------------------------------
# 7. SEPARATE INPUT FEATURES AND OUTPUT
# ---------------------------------------------------------

X = df[
    [
        "Area_sqft",
        "Bedrooms",
        "Age_years"
    ]
]

y = df["Price_lakh"]


print("\n========== INPUT FEATURES (X) ==========")
print(X)

print("\n========== OUTPUT / TARGET (y) ==========")
print(y)


# ---------------------------------------------------------
# 8. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n========== TRAIN TEST SPLIT ==========")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ---------------------------------------------------------
# 9. STANDARD SCALER
# ---------------------------------------------------------

scaler = StandardScaler()

# Fit and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Only transform testing data
X_test_scaled = scaler.transform(X_test)


print("\n========== DATA BEFORE SCALING ==========")
print(X_train.head())

print("\n========== DATA AFTER SCALING ==========")
print(X_train_scaled[:5])


# ---------------------------------------------------------
# 10. CREATE LINEAR REGRESSION MODEL
# ---------------------------------------------------------

model = LinearRegression()


# ---------------------------------------------------------
# 11. TRAIN THE MODEL
# ---------------------------------------------------------

model.fit(
    X_train_scaled,
    y_train
)


print("\n========== MODEL TRAINED ==========")

print("Intercept:", model.intercept_)

print("Coefficients:", model.coef_)


# ---------------------------------------------------------
# 12. MAKE PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(
    X_test_scaled
)


print("\n========== PREDICTIONS ==========")

print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)


# ---------------------------------------------------------
# 13. MODEL EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ---------------------------------------------------------
# 14. ACTUAL VS PREDICTED VISUALIZATION
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()


# ---------------------------------------------------------
# 15. PREDICT PRICE OF A NEW HOUSE
# ---------------------------------------------------------

# New house:
# Area = 1750 sqft
# Bedrooms = 3
# Age = 5 years

new_house = [[1750, 3, 5]]


# Scale new house using the SAME scaler
new_house_scaled = scaler.transform(
    new_house
)


# Predict price
predicted_price = model.predict(
    new_house_scaled
)


print("\n========== NEW HOUSE PREDICTION ==========")

print(
    "Predicted Price:",
    predicted_price[0],
    "Lakh"
)