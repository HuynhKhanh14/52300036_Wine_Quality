"""Data loading and preprocessing for the Wine Quality project."""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]
CLASS_NAMES = ["Low", "Medium", "High"]


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_data(wine_type: str) -> pd.DataFrame:
    filename = "red.csv" if wine_type == "Red Wine" else "white.csv"
    path = project_root() / "data" / filename
    df = pd.read_csv(path, sep=";")
    df["category"] = df["quality"].apply(quality_category)
    return df


def quality_category(quality: int) -> str:
    if quality <= 5:
        return "Low"
    if quality == 6:
        return "Medium"
    return "High"


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Return scaled train/validation/test arrays and the fitted scaler.

    The scaler is fitted ONLY on the training set to avoid test/validation leakage.
    """
    X = df[FEATURES].astype("float32").values
    y = df["category"].map({"Low": 0, "Medium": 1, "High": 2}).astype("int64").values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def to_sequence(X: np.ndarray) -> np.ndarray:
    """Represent the 11 tabular features as a length-11 sequence for LSTM/CNN-1D."""
    return X.reshape((X.shape[0], X.shape[1], 1)).astype("float32")
