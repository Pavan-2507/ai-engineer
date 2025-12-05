import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


data=pd.read_csv("bank-full.csv",sep=";")


X=data.drop(["y"],axis=1)
y=data["y"]

X_encoded=pd.get_dummies(X,drop_first=True)

X_test,X_train,y_test,y_train=train_test_split(X_encoded,y,random_state=42,stratify=y)

model=RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    max_depth=12,
    min_samples_leaf=30,
    min_samples_split=50,
    n_estimators=100,
    max_features="sqrt",
    class_weight="balanced"
)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))



