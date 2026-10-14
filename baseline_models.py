"""Traditional ML baselines required/encouraged by the course project."""
from time import perf_counter

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def get_baseline_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    }


def evaluate_baseline(model, X_train, y_train, X_test, y_test):
    start = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - start
    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "training_time": train_seconds,
        "predictions": pred,
        "model": model,
    }


def run_all_baselines(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in get_baseline_models().items():
        result = evaluate_baseline(model, X_train, y_train, X_test, y_test)
        rows.append({
            "Model": name,
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-score": result["f1"],
            "Training time (s)": result["training_time"],
        })
    return pd.DataFrame(rows)
