import pandas as pd
from sklearn.model_selection import RandomizedSearchCV,train_test_split,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report,roc_auc_score

data=pd.read_csv("bank-full.csv",sep=';')


X=data.drop(["y"],axis=1)
y=data["y"]

X_encoded=pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test=train_test_split(X_encoded,y,random_state=42,stratify=y)
rf=RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced")
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

param_grid={
    "n_estimators":[100,200,300],
    "max_depth":[None,8,12,16],
   "min_samples_split":[20,40,60],
   "min_samples_leaf":[10,20,30],
   "max_features":["sqrt",0.3,0.5], # it select  only the classes  baesed on the given value
   "class_weight":[None,"balanced"], #pay attention to minority classes
   "bootstrap":[True,False] #take the same samples multiple times rather take all sample once

}


rnd_search=RandomizedSearchCV(
    estimator=rf,# classifier 
    n_iter=40, #no of random points it takes from total param_grid
    scoring="roc_auc",   #How many actual positives are correctly predicted and How many negatives are incorrectly predicted
    param_distributions=param_grid,
    cv=cv,
    random_state=42,
    n_jobs=-1,#use all the poer of cpu
    verbose=2, #how much information need to be printed
    return_train_score=False #  choose test score stored or not 

)
rnd_search.fit(X_train, y_train)

# print("Best parameters:", rnd_search.best_params_) # finds the one with the highest scoring (ROC AUC) eg: n_estimators': 300, 'max_depth': 20, 'bootstrap': True, 'class_weight': 'balanced'
print("Best CV score (roc_auc):", rnd_search.best_score_)#Shows the highest ROC AUC value it achieved during K-fold cross-validation using the best parameters. eg:(roc_auc): 0.9234
best_rf = rnd_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print(best_rf)

