"""
src/preprocess.py
Downloads the Telco Churn dataset and builds a scikit-learn preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Column definitions ─────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges"
]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

TARGET = "Churn"


def load_data(path: str = "data/telco_churn.csv") -> pd.DataFrame:
    """Load dataset; download a sample if file not found."""
    if not os.path.exists(path):
        print("Dataset not found — generating synthetic sample for demo...")
        df = _make_synthetic_data(1000)
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved synthetic data to {path}")
    else:
        df = pd.read_csv(path)
    return df


def _make_synthetic_data(n: int = 1000) -> pd.DataFrame:
    """Generate a small synthetic dataset matching the Telco schema."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(1, 72, n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges": rng.uniform(18, 120, n).round(2),
        "TotalCharges": rng.uniform(18, 8500, n).round(2),
        "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning steps."""
    df = df.copy()
    # TotalCharges sometimes arrives as string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    # Drop non-feature column
    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numerics and encodes categoricals."""
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])
    return preprocessor


def get_splits(df: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test."""
    # Keep SeniorCitizen as extra numeric feature
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + ["SeniorCitizen"]
    X = df[feature_cols]
    y = (df[TARGET] == "Yes").astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


if __name__ == "__main__":
    df = load_data()
    df = clean(df)
    X_train, X_test, y_train, y_test = get_splits(df)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Churn rate (train): {y_train.mean():.2%}")
    print("Preprocessing module ready.")
