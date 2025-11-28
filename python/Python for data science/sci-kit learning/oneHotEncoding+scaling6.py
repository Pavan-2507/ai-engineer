from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
df=pd.read_csv('car_dataset.csv')
# imputer=SimpleImputer(strategy='mean')
# df[['age']]=imputer.fit_transform(df[['age']])
# df[['salary']]=imputer.fit_transform(df[['salary']])

# cat_imputer=SimpleImputer(strategy='most_frequent')
# df[['city']]=cat_imputer.fit_transform(df[['city']])


num_feat=['age','salary']
cat_feat=['city']

preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_feat),
        ('cat',OneHotEncoder(handle_unknown='ignore'),cat_feat)
    ]
)

# df['bought_car'] = [0, 1, 0, 1, 1]  # just example labels

X=df[['age','salary','city']] # lables
y=df[['bought_car']] # target


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.pipeline import Pipeline

model=Pipeline(steps=[
    ('preprocess',preprocessor),
    ('clf',LogisticRegression(max_iter=1000)) #'clf' → classifier (Logistic Regression here)
])

model.fit(X_train,y_train.values.ravel())

y_pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))