"""
src/train.py
Trains an XGBoost classifier and logs everything to MLflow.
Run: python src/train.py
"""
import mlflow
import pandas as pd
import numpy as np

# 🔥 ADD THIS LINE HERE
mlflow.set_tracking_uri("file:./mlruns")
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os

from preprocess import load_data, clean, build_preprocessor, get_splits
from xgboost import XGBClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "churn-prediction"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "scale_pos_weight": 2.7,   # handles class imbalance: ~73% No Churn / 27% Churn
    "eval_metric": "auc",
    "random_state": 42,
}


def train():
    df = load_data()
    df = clean(df)
    X_train, X_test, y_train, y_test = get_splits(df)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Churn rate in train: {y_train.mean():.2%}")

    if y_train.mean() > 0.45 or y_train.mean() < 0.05:
        print("\n⚠  WARNING: You are training on SYNTHETIC data.")
        print("   Metrics will be poor because synthetic data has no real patterns.")
        print("   For good metrics, download the real dataset:")
        print("   https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("   Save it as:  data/telco_churn.csv  then re-run.\n")

    preprocessor = build_preprocessor()
    model = XGBClassifier(**PARAMS)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")
        mlflow.log_params(PARAMS)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Use threshold 0.4 instead of default 0.5 to catch more churners
        y_pred_tuned = (y_prob >= 0.40).astype(int)

        metrics = {
            "accuracy":      accuracy_score(y_test, y_pred_tuned),
            "roc_auc":       roc_auc_score(y_test, y_prob),
            "f1_churn":      f1_score(y_test, y_pred_tuned),
            "f1_weighted":   f1_score(y_test, y_pred_tuned, average="weighted"),
        }

        mlflow.log_metrics(metrics)
        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nClassification report (threshold=0.40):")
        print(classification_report(y_test, y_pred_tuned, target_names=["No Churn", "Churn"]))

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="churn_model",
            registered_model_name="churn-xgboost",
        )

        local_path = os.path.join(MODEL_DIR, "pipeline.joblib")
        joblib.dump(pipeline, local_path)
        mlflow.log_artifact(local_path)
        print(f"Model saved to {local_path}")
        print("View in MLflow UI: run `mlflow ui` then open http://127.0.0.1:5000")

    return pipeline


if __name__ == "__main__":
    train()
