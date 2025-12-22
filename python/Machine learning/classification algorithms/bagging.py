from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# -----------------------------
# Load dataset (Iris for simplicity)
# -----------------------------
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# -----------------------------
# Base model (weak learner)
# -----------------------------
base_tree = DecisionTreeClassifier(max_depth=None, random_state=42)

# -----------------------------
# Bagging model (multiple decision trees)
# -----------------------------
bag = BaggingClassifier(
    estimator=base_tree,
    n_estimators=5,       # Keep small number to see models clearly
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)

# Train Bagging
bag.fit(X_train, y_train)

# -----------------------------
# SEE each individual model
# -----------------------------
print("\n================= INDIVIDUAL MODELS =================")
for idx, model in enumerate(bag.estimators_):
    print(f"\nModel {idx+1}:", model)

# -----------------------------
# SHOW how each model predicts
# -----------------------------
print("\n================= INDIVIDUAL PREDICTIONS =================")
sample = X_test[0].reshape(1, -1)
print("Sample Input:", sample)

for idx, model in enumerate(bag.estimators_):
    print(f"Model {idx+1} Prediction:", model.predict(sample))

# -----------------------------
# FINAL BAGGING prediction (majority vote)
# -----------------------------
print("\n================= FINAL BAGGING OUTPUT =================")
print("Bagging Prediction:", bag.predict(sample))
