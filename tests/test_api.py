"""
tests/test_api.py
Basic tests for the FastAPI prediction endpoint.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Make sure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_valid_response():
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_high_risk_customer():
    """A month-to-month customer with high charges should lean toward churn."""
    risky = SAMPLE_CUSTOMER.copy()
    risky["Contract"] = "Month-to-month"
    risky["MonthlyCharges"] = 110.0
    risky["tenure"] = 2
    resp = client.post("/predict", json=risky)
    assert resp.status_code == 200


def test_predict_low_risk_customer():
    """A long-tenure two-year contract customer should lean toward no churn."""
    safe = SAMPLE_CUSTOMER.copy()
    safe["Contract"] = "Two year"
    safe["tenure"] = 60
    safe["MonthlyCharges"] = 25.0
    resp = client.post("/predict", json=safe)
    assert resp.status_code == 200


def test_missing_field_returns_422():
    bad = SAMPLE_CUSTOMER.copy()
    del bad["tenure"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
