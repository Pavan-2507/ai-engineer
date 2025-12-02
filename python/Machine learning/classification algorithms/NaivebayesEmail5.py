import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "Win money now",                          
    "You won a lottery prize",                
    "Claim your free gift",                 
    "Cheap loan offer just for you",          
    "Important meeting tomorrow",             
    "Let's have family dinner tonight",       
    "Are you coming to office?",              
    "Please review the attached report",      
]


labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham"
]

df=pd.DataFrame({"texts":texts,"labels":labels})

# print(df)

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df["texts"])
y=df["labels"]

# print(vectorizer.get_feature_names_out()).       ---->get the word names from the vectorizer..it converts sentence into word and count the words count ..feature names means inputs

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_pred=mnb.predict(X_test)

print(accuracy_score(y_test,y_pred))


new_messages = [
    "Win a free lottery now",
    "Project meeting schedule",
    "Get cheap loan now",
    "Family dinner tomorrow night"
]

X_new = vectorizer.transform(new_messages)
new_pred = mnb.predict(X_new)
# print(new_pred)

for msg,label in  zip(new_messages,new_pred):
    print(f"{msg} --> {label}")