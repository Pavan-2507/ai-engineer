import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


data=pd.read_csv("bank.csv")

X=data.drop(["y","duration"],axis=1)
y=data["y"]

cat_cols=X.select_dtypes(include=['object']).columns
num_cols=X.select_dtypes(exclude=['object']).columns


column_transfer=ColumnTransformer(
    transformers=[
        ("cat_cols",OneHotEncoder(drop="first"),cat_cols)
    ]
)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

X_train_encoded=column_transfer.fit_transform(X_train)
X_test_encoded=column_transfer.transform(X_test)


model=DecisionTreeClassifier(random_state=42,min_samples_leaf=30,min_samples_split=50,max_depth=6)
model.fit(X_train_encoded,y_train)

y_pred=model.predict(X_test_encoded)

print(accuracy_score(y_test,y_pred))