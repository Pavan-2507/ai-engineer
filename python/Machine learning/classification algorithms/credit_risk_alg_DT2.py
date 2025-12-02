import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data=pd.read_csv("credit_risk.csv")

X=data.drop(["class"],axis=1)
y=data["class"]

cat_cols=X.select_dtypes(include=['object']).columns

column_transfer=ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(drop='first'),cat_cols)
        
    ],
    remainder="passthrough"
)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_encoded_train=column_transfer.fit_transform(X_train)
X_encoded_test=column_transfer.transform(X_test)

model=DecisionTreeClassifier(random_state=42,max_depth=6,min_samples_split=50,min_samples_leaf=30)
model.fit(X_encoded_train,y_train)
y_pred=model.predict(X_encoded_test)

print(accuracy_score(y_test,y_pred))