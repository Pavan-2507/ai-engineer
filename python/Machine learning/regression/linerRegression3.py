import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Step 1: Load Data
# -------------------------------

data = {
    "CRIM":[0.00632,0.02731,0.02729,0.03237,0.06905,0.02985,0.08829],
    "ZN":[18,0,0,0,0,0,12.5],
    "INDUS":[2.31,7.07,7.07,2.18,2.18,2.18,7.87],
    "CHAS":[0,0,0,0,0,0,0],
    "NOX":[0.538,0.469,0.469,0.458,0.458,0.458,0.524],
    "RM":[6.575,6.421,7.185,6.998,7.147,6.43,6.012],
    "AGE":[65.2,78.9,61.1,45.8,54.2,58.7,66.6],
    "DIS":[4.09,4.9671,4.9671,6.0622,6.0622,6.0622,5.5605],
    "RAD":[1,2,2,3,3,3,5],
    "TAX":[296,242,242,222,222,222,311],
    "PTRATIO":[15.3,17.8,17.8,18.7,18.7,18.7,15.2],
    "B":[396.9,396.9,392.83,394.63,396.9,394.12,395.6],
    "LSTAT":[4.98,9.14,4.03,2.94,5.33,5.21,12.43],
    "MEDV":[24,21.6,34.7,33.4,36.2,28.7,22.9]
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Select Predictor & Target
# -------------------------------

X = df[['RM']]          # predictor
y = df['MEDV']          # target

# -------------------------------
# Step 3: Scatter Plot
# -------------------------------

plt.scatter(X, y, color='blue')
plt.xlabel("RM (Average Rooms per Dwelling)")
plt.ylabel("MEDV (Median House Value)")
plt.title("Scatter Plot of RM vs MEDV")
plt.show()

# -------------------------------
# Step 4: Train-Test Split (67:33)
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# -------------------------------
# Step 5: Build Linear Regression
# -------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# Coefficient and Intercept
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# -------------------------------
# Step 6: Predictions
# -------------------------------

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

# -------------------------------
# Step 7: Model Evaluation
# -------------------------------

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse  = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2  = r2_score(y_test, y_test_pred)

print("\nTraining MSE:", train_mse)
print("Testing MSE :", test_mse)
print("Training R² :", train_r2)
print("Testing R²  :", test_r2)
