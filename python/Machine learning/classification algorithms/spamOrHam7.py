import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report



data=pd.read_csv("SMSSpamCollection.csv",sep="\t",header=None,names=["label", "text"] )  
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

X=data["text"]
y=data["label_num"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

vectorizer=CountVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)


model=MultinomialNB()
model.fit(X_train_vec,y_train)

y_pred=model.predict(X_test_vec)

print(accuracy_score(y_test,y_pred))

test_df = pd.read_csv("spamandham.csv", sep="\t", header=None, names=["label", "text"])
test_df["label_num"] = test_df["label"].map({"ham": 0, "spam": 1})
y_test2=test_df["label"]

X_test_new = test_df["text"]
# y_test_new = test_df["label_num"]

X_test_new_vec = vectorizer.transform(X_test_new)
y_pred_new = model.predict(X_test_new_vec)

for msg, pred ,actual in zip(X_test_new, y_pred_new,y_test2):
    print(f"'{msg}' --> {'SPAM' if pred==1 else 'HAM'} ---->{actual}")




