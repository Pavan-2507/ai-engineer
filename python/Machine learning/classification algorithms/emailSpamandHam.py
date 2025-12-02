import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv("email_dataset_100k.csv")
X=df["raw_text"]
y=df["label"]

X=X.fillna("")
y=y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer=CountVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)


model=MultinomialNB()
model.fit(X_train_vec,y_train)

y_pred=model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))


new_emails = [
    "Subject: WINNER! Claim your reward\nClick this link to verify your account immediately.",
    "Subject: Meeting notes\nPlease find attached the minutes from today’s standup.",
]

new_vec=vectorizer.transform(new_emails)
new_pred=model.predict(new_vec)

print(new_pred)

for msg,label in zip(new_emails,new_pred):
    print(f"{msg} ---> {label}")