import pandas as pd

from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score


data=pd.read_csv("bank-full.csv",sep=";")

X=data.drop(["y"],axis=1)
y=data["y"]

X_encoded=pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,test_size=0.2,random_state=42,stratify=y)

cv=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
rf=RandomForestClassifier(n_jobs=-1,random_state=42,class_weight="balanced")

param_grid={
    "n_estimators":[100,200,300,400,600],
    "min_samples_split":[20,40,60],
    "max_depth":[None,6,8,10,12],
    "min_samples_leaf":[10,20,30],
    "max_features":["sqrt",0.3,0.5],
    "class_weight":[None,"balanced"],
    "bootstrap":[True,False]

}

rnd_search=RandomizedSearchCV(
    n_iter=40,
    cv=cv,
    estimator=rf,
    param_distributions=param_grid,
    verbose=2,
    random_state=42,
    return_train_score=False,
    scoring="roc_auc",
    n_jobs=-1

)

rnd_search.fit(X_train,y_train)

best_rf=rnd_search.best_estimator_

y_pred=best_rf.predict(X_test)
y_prob=best_rf.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test,y_prob))

