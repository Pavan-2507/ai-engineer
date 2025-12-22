import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report

data=pd.read_csv("bank-full.csv",sep=";")
X=data.drop(["y"],axis=1)
y=data["y"]

X_encoded=pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,random_state=42,test_size=0.2,stratify=y)

model=RandomForestClassifier(
    n_jobs=-1,
    max_depth=12,
    min_samples_leaf=20,
    min_samples_split=40,
    bootstrap=True,
    verbose=2


)
