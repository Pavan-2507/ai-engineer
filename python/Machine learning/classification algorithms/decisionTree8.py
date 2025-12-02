import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("bank.csv")

print("\nDataset sample:")
print(df.head())
print("\nDataset shape:", df.shape)


# 2. Split into features & target
X = df.drop(["y", "duration"], axis=1)    # drop duration column
y = df["y"]                               # target column


# 3. Identify categorical & numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns


# 4. Encode categorical data using OneHot Encoding
column_transform = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), categorical_cols)],
    remainder="passthrough"
)


# 5. Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # maintain class proportion
)


# ------------------ MODEL 1 (Default Decision Tree) ------------------

model1 = Pipeline([
    ("encoder", column_transform),
    ("clf", DecisionTreeClassifier(random_state=42))
])

model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)


print("\n=============== MODEL 1: DEFAULT DECISION TREE ===============")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("\nClassification Report:\n", classification_report(y_test, y_pred1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred1))

model2 = Pipeline([
    ("encoder", column_transform),
    ("clf", DecisionTreeClassifier(
        random_state=42,
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=30,
        criterion='entropy'))   # changed from default gini
])

model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)


print("\n=============== MODEL 2: TUNED DECISION TREE ===============")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("\nClassification Report:\n", classification_report(y_test, y_pred2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred2))


# ----------- PERFORMANCE COMPARISON ----------------
print("\n================ PERFORMANCE COMPARISON ================")
print("Model 1 Accuracy:", accuracy_score(y_test, y_pred1))
print("Model 2 Accuracy:", accuracy_score(y_test, y_pred2))
print("\nCheck precision/recall in reports to compare.")
