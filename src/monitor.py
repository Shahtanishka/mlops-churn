"""
src/monitor.py
Compares a reference dataset (training data) against simulated production data
and generates an Evidently AI drift report as HTML.
Run: python src/monitor.py
"""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric
import os

from preprocess import load_data, clean, get_splits, NUMERIC_COLS, CATEGORICAL_COLS

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def simulate_production_drift(reference: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """
    Create a 'production' batch with slight distribution shift:
    - MonthlyCharges shifted up (prices rose)
    - Tenure skewed shorter (newer customers)
    - Some categories shifted
    """
    rng = np.random.default_rng(99)
    prod = reference.sample(n=n, replace=True, random_state=99).copy()

    # Introduce drift
    prod["MonthlyCharges"] = prod["MonthlyCharges"] * rng.uniform(1.1, 1.3, n)
    prod["tenure"] = (prod["tenure"] * rng.uniform(0.5, 0.8, n)).astype(int).clip(1)
    prod["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.75, 0.15, 0.10]
    )
    return prod


def run_report():
    # ── Load reference data (training set) ────────────────────────────────────
    df = load_data()
    df = clean(df)
    X_train, X_test, y_train, y_test = get_splits(df)

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + ["SeniorCitizen"]
    reference = X_train[feature_cols].copy()
    reference["target"] = y_train.values

    # ── Simulate production data ───────────────────────────────────────────────
    production = simulate_production_drift(reference, n=300)
    # Production labels (in real life these arrive with a delay)
    production["target"] = np.random.choice([0, 1], len(production), p=[0.65, 0.35])

    print(f"Reference rows: {len(reference)} | Production rows: {len(production)}")

    # ── Build Evidently report ─────────────────────────────────────────────────
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(
        reference_data=reference.drop(columns=["target"]),
        current_data=production.drop(columns=["target"]),
    )

    # Save HTML report
    html_path = os.path.join(REPORT_DIR, "drift_report.html")
    report.save_html(html_path)
    print(f"\nDrift report saved → {html_path}")
    print("Open it in your browser to see which features drifted.")

    # Print quick summary
    result = report.as_dict()
    drift_detected = result["metrics"][1]["result"]["dataset_drift"]
    share = result["metrics"][1]["result"]["share_of_drifted_columns"]
    print(f"\nDataset drift detected: {drift_detected}")
    print(f"Share of drifted columns: {share:.0%}")

    if drift_detected:
        print("\n⚠ Drift detected — consider triggering a retraining run!")
    else:
        print("\n✓ No significant drift — model is still healthy.")


if __name__ == "__main__":
    run_report()
