import pandas  as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.DataFrame({
    "age":    [18,22,35,45,60,28,55,40,30,25],
    "salary": [15000,20000,55000,80000,120000,40000,95000,70000,50000,30000],
    "bought_car": [0,0,1,1,1,1,1,1,0,0]  # Target/Label
})
X = df[['age','salary']]      # independent variables
y = df['bought_car']          # target/output variable




# X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LogisticRegression()
model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)
print(accuracy_score(y_test,y_pred))