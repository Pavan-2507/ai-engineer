import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data=pd.read_csv("credit_risk.csv")

X=data.drop(["class"],axis=1)
y=data["class"]

X_encoded=pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,test_size=0.2,random_state=42,stratify=y)

model=DecisionTreeClassifier(random_state=42,min_samples_split=50,min_samples_leaf=30,max_depth=6)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(accuracy_score(y_test,y_pred))