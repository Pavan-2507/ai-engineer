import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_csv("poly_dataset.csv")
X=data["X"].values.reshape(-1,1)
y=data["y"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

poly=PolynomialFeatures(degree=10)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

model=LinearRegression()
y_test_pred=model.predict(X_test_poly)

model.fit(X_train_poly,y_train)

y_train_pred=model.predict(X_train_poly)

X_line=np.linspace(X.min(),X.max(),500).reshape(-1,1)

X_line_poly=poly.transform(X_line)
y_line_pred=model.predict(X_line_poly)





plt.scatter(X,y, color='blue',alpha=0.3,label="Actaul Data")

plt.plot(X_line,y_line_pred,color='red',label="predicted data")
plt.title("Polynomial Regression Visualization")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()



