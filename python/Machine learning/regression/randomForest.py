import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------
# 1) Load Dataset
# --------------------------------------
data = pd.read_csv("nonlinear_100k.csv")

X = data.drop("target", axis=1)
y = data["target"]

# --------------------------------------
# 2) Train-Test Split
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------
# 3) Random Forest Model
# --------------------------------------
rf = RandomForestRegressor(
    n_estimators=200,        # number of trees
    max_depth=10,            # controls complexity
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1                # use all CPU cores
)

# --------------------------------------
# 4) Train Model
# --------------------------------------
rf.fit(X_train, y_train)

# --------------------------------------
# 5) Predictions
# --------------------------------------
y_train_pred = rf.predict(X_train)
y_test_pred  = rf.predict(X_test)

# --------------------------------------
# 6) Evaluation
# --------------------------------------
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Test MSE : {mean_squared_error(y_test, y_test_pred):.4f}")

print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R² : {r2_score(y_test, y_test_pred):.4f}")


# --------------------------------------
# 7) Visualization using one feature
# --------------------------------------
feature = "feature1"

plt.figure(figsize=(10,5))

# Actual data
plt.scatter(X[feature], y, s=10, alpha=0.3, label="Actual")

# Predicted values (sorted)
X_plot = X_train.copy()
X_plot["pred"] = y_train_pred
X_plot = X_plot.sort_values(feature)

plt.plot(
    X_plot[feature],
    X_plot["pred"],
    color="red",
    linewidth=2,
    label="Random Forest Prediction"
)

plt.xlabel(feature)
plt.ylabel("target")
plt.title("Random Forest Regression Visualization")
plt.legend()
plt.grid(True)
plt.show()
