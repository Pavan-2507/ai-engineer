import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score


data=pd.read_csv("bank-full.csv",sep=";")
X=data.drop(["y"],axis=1)
y=data["y"]

X_encoded=pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,test_size=0.2,random_state=42,stratify=y)
baseTree=DecisionTreeClassifier(
    random_state=42,
    criterion="entropy",
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=10,
    min_samples_split=20
)

bag_model=BaggingClassifier(
    estimator=baseTree,
    n_jobs=-1,
    random_state=42,
    verbose=2,
    bootstrap=True,
    n_estimators=200,
    oob_score=True,
    max_samples=0.6,



)


bag_model.fit(X_train,y_train)

y_pred=bag_model.predict(X_test)

y_prob=bag_model.predict_proba(X_test)[:,1]
y_test_binary = (y_test == "yes").astype(int)


print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Test ROC AUC:", roc_auc_score(y_test_binary, y_prob))
