import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import scipy.stats as stats



data=pd.read_csv("boston_housing.csv")



X=data.drop(["MEDV"],axis=1)
y=data["MEDV"]


scaler=StandardScaler()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


model=LinearRegression()
model.fit(X_train_scaled,y_train)


y_train_pred=model.predict(X_train_scaled)
y_test_pred=model.predict(X_test_scaled)

mse_train=mean_squared_error(y_train,y_train_pred)
mse_test=mean_squared_error(y_test,y_test_pred)


r2_train=r2_score(y_train,y_train_pred)
r2_test=r2_score(y_test,y_test_pred)


X_new_data = np.array([
    [0.0125, 12.0, 3.0, 0, 0.510, 6.720, 62.5, 4.15, 1, 300, 15.0, 395.5, 4.20],
    [0.0750, 0.0, 7.50, 0, 0.540, 6.100, 78.3, 3.90, 4, 307, 17.0, 392.1, 9.80],
    [0.0203, 22.0, 4.95, 0, 0.460, 7.050, 45.2, 6.10, 2, 250, 14.5, 389.7, 3.95]
]).reshape(3,-1)
X_new_data_scaled=scaler.transform(X_new_data)

new_pred=model.predict(X_new_data_scaled)
j=1


# for i ,j in zip(y_test,y_test_pred):
#     print(f"{i} : {j}")
#     print(i-j)



for i in new_pred:
    print(f"house {j} :{i:.4f}")
    j=j+1

df_out = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_test_pred,
    "Residual": y_test.values - y_test_pred
})


print(f"mse_train:{mse_train:.4f}")
print(f"mse_test:{mse_test:.4f}")
print("r2_train:",r2_train)
print("r2_test:",r2_test)






# ----- VISUALIZATIONS (append after your existing code) -----
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.inspection import PartialDependenceDisplay

# Ensure y_test_pred / y_train_pred exist (they do in your code).
# Convert X_train, X_test to DataFrame if they are not (they are DataFrames in your code).
feature_names = X_train.columns.tolist()

# 1) Actual vs Predicted (Test)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, edgecolor='k', alpha=0.8)
mn = min(y_test.min(), y_test_pred.min())
mx = max(y_test.max(), y_test_pred.max())
plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted (Test set)")
plt.grid(alpha=0.3)
plt.show()

# 2) Residuals vs Predicted (Test) — check heteroscedasticity / patterns
residuals_test = y_test - y_test_pred
plt.figure(figsize=(7,4))
plt.scatter(y_test_pred, residuals_test, alpha=0.8, edgecolor='k')
plt.hlines(0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), colors='r', linestyles='dashed')
plt.xlabel("Predicted MEDV")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted (Test)")
plt.grid(True, alpha=0.3)
plt.show()

# 3) Residual distribution (hist + KDE)
plt.figure(figsize=(7,4))
sns.histplot(residuals_test, kde=True)
plt.xlabel("Residual")
plt.title("Residual Distribution (Test set)")
plt.show()

# 4) QQ-plot of residuals (normality check)
plt.figure(figsize=(6,6))
stats.probplot(residuals_test, dist="norm", plot=plt)
plt.title("QQ-plot of Residuals (Test)")
plt.show()

# 5) Coefficients: standardized (model trained on scaled X) and original-scale coefficients
coef_std = model.coef_.ravel()               # coefficients on scaled features (per 1 std)
scale = scaler.scale_
mean = scaler.mean_

# Convert to original-unit coefficients:
# X_scaled = (X - mean) / scale  =>  coef_original = coef_std / scale
coef_orig = coef_std / scale

# Intercept in original units:
intercept_orig = model.intercept_ - np.sum(coef_std * (mean / scale))

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coef_standardized': coef_std,
    'coef_original_units': coef_orig
})
coef_df['abs_coef_orig'] = coef_df['coef_original_units'].abs()
coef_df = coef_df.sort_values(by='abs_coef_orig', ascending=False)

# Plot standardized coefficients
plt.figure(figsize=(10,4))
sns.barplot(x='coef_standardized', y='feature', data=coef_df, palette='vlag')
plt.title("Standardized Coefficients (per 1 std change)")
plt.xlabel("Coefficient (std units)")
plt.tight_layout()
plt.show()



# Plot original-unit coefficients
plt.figure(figsize=(10,4))
sns.barplot(x='coef_original_units', y='feature', data=coef_df, palette='crest')
plt.title("Coefficients in Original Feature Units")
plt.xlabel("Coefficient (original units)")
plt.tight_layout()
plt.show()



print("Intercept (model on scaled X):", model.intercept_)
print("Intercept (original units):", intercept_orig)
print("\nTop coefficients (absolute) in original units:\n", coef_df[['feature','coef_original_units']].head(10))

# 6) Partial Dependence for top 2 features (approx marginal effect)
top_features = coef_df.feature.tolist()[:2]  # top two by absolute original impact
print("\nTop 2 features for partial dependence:", top_features)

# PartialDependenceDisplay expects either a DataFrame or array X; we pass scaled X.
# We need feature indices for the scaled array:
feat_idx = [feature_names.index(f) for f in top_features]

# For sklearn>=1.0 PartialDependenceDisplay.from_estimator accepts a fitted estimator and X.
PartialDependenceDisplay.from_estimator(model, X_train_scaled, features=feat_idx, feature_names=feature_names)
plt.suptitle("Partial Dependence (approx) for top 2 features")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# 7) Table: Actual vs Predicted (Test) — first few rows
compare_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': np.round(y_test_pred, 4),
    'residual': np.round(residuals_test, 4)
}, index=y_test.index).sort_index()
print("\nSample Actual vs Predicted (Test set):")
print(compare_df.head(10))

# OPTIONAL: 3D scatter of top 2 features vs MEDV (original scale)
from mpl_toolkits.mplot3d import Axes3D
f1, f2 = top_features[0], top_features[1]
f1_idx, f2_idx = feature_names.index(f1), feature_names.index(f2)
X_test_orig = scaler.inverse_transform(X_test_scaled)  # convert test X back to original units
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_orig[:, f1_idx], X_test_orig[:, f2_idx], y_test, c='b', marker='o', alpha=0.7)
ax.set_xlabel(f1)
ax.set_ylabel(f2)
ax.set_zlabel('MEDV (actual)')
ax.set_title(f"3D scatter: {f1} vs {f2} vs MEDV (test)")
plt.show()



