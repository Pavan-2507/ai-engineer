import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


iris=load_iris()

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
k=5
model=KNeighborsClassifier(
    n_neighbors=k,
    metric='euclidean',
    weights='uniform'
)

model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)

print(accuracy_score(y_test,y_pred))
