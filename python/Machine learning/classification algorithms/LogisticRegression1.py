import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler


data=load_breast_cancer()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train) # used to train and scale 
X_test_scaled=scaler.transform(X_test) # used to scale only not train

model=LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000)

model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)

print(accuracy_score(y_test,y_pred))

y_proba = model.predict_proba(X_test_scaled)   # shape: (n_samples, 2)
print("First 5 probability predictions:\n", y_proba[:5])

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

