import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

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
# 3) XGBoost Regressor
# --------------------------------------
xgb = XGBRegressor(
    n_estimators=300,        # number of trees
    max_depth=6,             # tree depth
    learning_rate=0.05,      # step size
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # feature sampling
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------
# 4) Train Model
# --------------------------------------
xgb.fit(X_train, y_train)

# --------------------------------------
# 5) Predictions
# --------------------------------------
y_train_pred = xgb.predict(X_train)
y_test_pred  = xgb.predict(X_test)

# --------------------------------------
# 6) Evaluation
# --------------------------------------
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Test  MSE: {mean_squared_error(y_test, y_test_pred):.4f}")

print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test  R²: {r2_score(y_test, y_test_pred):.4f}")


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
    label="XGBoost Prediction"
)

plt.xlabel(feature)
plt.ylabel("target")
plt.title("XGBoost Regression Visualization")
plt.legend()
plt.grid(True)
plt.show()

importances = xgb.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df)

plt.figure(figsize=(8,4))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.show()
