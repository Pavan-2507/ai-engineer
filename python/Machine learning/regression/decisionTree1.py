import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor,plot_tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt


data=pd.read_csv("nonlinear_10k.csv")
X=data.drop("target",axis=1)
y=data["target"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


model=DecisionTreeRegressor(
    max_depth=12,
    min_samples_leaf=2,
    min_samples_split=4,
    random_state=42,
    

)
model.fit(X_train,y_train)

y_train_pred=model.predict(X_train)
y_pred=model.predict(X_test)


# X_sorted = np.sort(X, axis=0)
# y_pred = model.predict(X)
X_plot = X_train.copy()
X_plot["pred"] = y_train_pred
X_plot = X_plot.sort_values("feature1")



plt.figure(figsize=(10,5))
plt.scatter(X["feature1"], y, color="blue", s=20, label="Actual Data")
plt.plot(X_plot["feature1"], X_plot["pred"], color="red", linewidth=2, label="Tree Prediction")

plt.xlabel("feature1")
plt.ylabel("target")
plt.title("Decision Tree Regression (feature1 vs target)")
plt.legend()
plt.grid(True)
plt.show()
